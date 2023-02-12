import os

import numpy as np
import time

import scipy.special
import torch
from pprint import pprint
from collections import OrderedDict

from peach.base import *
from peach.common import save_jsonl
from work_scruples.dataset_scruples import DilemmasDataset
from work_scruples.modeling_dilemmas import AutoModelForDilemmas
from work_scruples.utils import calibration_factor



def evaluate(
        args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
        calibration=False, calibrated_temperature=None, **kwargs,
):
    accelerator.wait_for_everyone()
    if not accelerator.is_local_main_process:
        return

    model.eval()
    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=False)
    logger.info(f"Evaluation for {eval_dataset.data_type}:")
    # Metrics
    metric = eval_dataset.get_metric()

    sup_wrong_metric, sup_rel_metric = None, None

    logits_for_cali, label_probs_for_cali = [], []

    data_for_save = defaultdict(list)

    for step, batch in enumerate(eval_dataloader):
        batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())

        with torch.no_grad():
            outputs = model(**batch)
        # predictions = outputs.logits.argmax(dim=-1)
        logits_for_cali.append(outputs.logits)
        label_probs_for_cali.append(batch["soft_labels"])

        # save for predict
        for key in batch.keys():
            data_for_save[key].extend(list(batch[key].detach().cpu().numpy()))
        for key in ["explain_cross_attentions", "explain_rel_priors", "logits"]:
            data_for_save[key].extend(list(outputs[key].detach().cpu().numpy()))

        metric.add_batch(
            predictions=outputs.logits,
            references=batch["freq_labels"],
        )

        bs, nc, ssn, ssl = batch["sup_input_ids"].size()
        sup_sent_mask = (batch["sup_attention_mask"].sum(dim=-1) > 0).to(torch.long).view(bs * nc, ssn)

        if hasattr(outputs, "distill_sup_wrong_logits"):
            sup_wrong_metric = sup_wrong_metric or load_metric("peach/metrics/soft_classification.py")
            sup_wrong_metric.add_batch(
                # predictions=outputs["distill_sup_judge_logits"].argmax(-1).view(-1)[sup_sent_mask.view(-1)==1],
                # references=batch["sup_judgment_labels"].view(-1)[sup_sent_mask.view(-1)==1],
                predictions=outputs["distill_sup_wrong_logits"].view(-1, 2)[sup_sent_mask.view(-1) == 1],
                references=outputs["sup_wrong_probs"].view(-1, 2)[sup_sent_mask.view(-1) == 1],
            )

        if hasattr(outputs, "distill_rel_logits"):
            sup_rel_metric = sup_rel_metric or load_metric("peach/metrics/soft_classification.py")
            sup_rel_metric.add_batch(
                predictions=outputs["distill_rel_logits"].view(-1, 3)[sup_sent_mask.view(-1) == 1],
                references=batch["sup_nli_probs"].view(-1, 3)[sup_sent_mask.view(-1) == 1],
            )

    calibration_metric = None
    if calibration:
        logger.info("Doing calibration ...")
        calibration_metric = load_metric("peach/metrics/soft_classification.py")
        logits_for_cali = torch.cat(logits_for_cali, dim=0).detach().cpu().numpy()
        label_probs_for_cali = torch.cat(label_probs_for_cali, dim=0).detach().cpu().numpy()
        if calibrated_temperature is None:
            logger.info("Calibrated_temperature is None, generating ...")
            calibrated_temperature = calibration_factor(logits_for_cali, label_probs_for_cali)
        logger.info(f"The calibrated_temperature is {calibrated_temperature}")
        calibration_metric.add_batch(
            predictions=logits_for_cali/calibrated_temperature,
            references=label_probs_for_cali,
        )

    if True:
        jsonl_to_save = []
        tokenizer = kwargs["tokenizer"]
        for idx in range(len(data_for_save["input_ids"])):
            cur_data = dict((k, v[idx]) for k,v in data_for_save.items())
            anchor = 0
            ex = dict()
            ex["soft_label"] = cur_data.pop("soft_labels").tolist()
            ex["label"] = int(cur_data.pop("labels"))
            cur_data.pop("freq_labels")
            pred_dist = scipy.special.softmax(cur_data.pop("logits"), axis=-1)
            ex["pred_dist"] = pred_dist.tolist()
            ex["pred"] = int(pred_dist.argmax())

            # flags
            ex["flag_correct"] = (ex["label"]==ex["pred"])
            ex["flag_complete"] = True

            ex["sents"] = []
            for sidx in range(cur_data["input_ids"].shape[0]):
                sent_ex = dict()
                sent_data = dict((k, v[sidx]) for k,v in cur_data.items())
                # 1. situation
                sent_len = int(sent_data["attention_mask"].sum())
                if sent_len == 0:
                    break
                if sent_len > args.max_length-2:
                    ex["flag_complete"] = False
                sent_ex["sent_text"] = tokenizer.decode(sent_data["input_ids"].tolist()[:sent_len], skip_special_tokens=True)
                # 2. actions
                num_actions = int((sent_data["sup_attention_mask"].sum(-1) > 0).sum())
                sent_ex["actions"] = []
                for aidx in range(num_actions):
                    act_ex = dict()
                    act_data = dict((key, sent_data[key][aidx]) for key in sent_data
                                  if key.startswith("sup_") or key.startswith("explain"))
                    act_len = int(act_data["sup_attention_mask"].sum())
                    act_ex["act_text"] = tokenizer.decode(act_data["sup_input_ids"].tolist()[:act_len], skip_special_tokens=True)
                    act_ex["act_judge"] = int(act_data["sup_judgment_labels"])
                    act_ex["s_xattn"] = float(act_data["explain_cross_attentions"])
                    act_ex["s_prior"] = float(act_data["explain_rel_priors"])
                    act_ex["s_nli"] = 1-float(act_data["sup_nli_probs"][1])
                    sent_ex["actions"].append(act_ex)

                ex["sents"].append(sent_ex)
            assert len(ex["sents"]) == 2  # for dilemmas
            jsonl_to_save.append(ex)
        save_jsonl(jsonl_to_save, os.path.join(args.output_dir, f"predictions_{eval_dataset.data_type}.jsonl"))

    if (not eval_dataset.data_type == "test") or eval_dataset.test_has_label:
        eval_metric = metric.compute()
        logger.info(f"step {global_step}: {eval_metric}")
        if sup_wrong_metric is not None:
            logger.info(f"\tsup_wrong_metric: {sup_wrong_metric.compute()}")
        if sup_rel_metric is not None:
            logger.info(f"\tsup_rel_metric: {sup_rel_metric.compute()}")

        if calibration_metric is not None:
            logger.info(f"\t\t\tcalibration_metric: {calibration_metric.compute()}")

        if calibrated_temperature is not None:
            eval_metric["calibrated_temperature"] = calibrated_temperature
        return eval_metric[eval_dataset.key_metric_name], eval_metric
    else:
        return 0., dict()


def train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=None, eval_fn=None):
    if accelerator.is_local_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    train_dataloader = setup_train_dataloader(args, train_dataset, accelerator)
    model, optimizer, lr_scheduler = setup_opt(args, model, accelerator, len(train_dataloader))

    logging_berfore_training(args, train_dataset)

    # Metrics
    # metric = load_metric("accuracy")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    step_loss = 0.
    step_loss_dict = defaultdict(float)
    best_metric = NEG_INF
    ma_dict = MovingAverageDict()
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch["training_progress"] = 1.0 * global_step / args.max_train_steps
            outputs = model(**batch)
            # calculate loss
            update_wrt_loss(args, accelerator, model, optimizer, outputs["loss"])
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item() / args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)
                progress_bar.update(1)
                global_step += 1
                # update loss for logging
                if tb_writer is not None:  # local main process
                    ma_dict(step_loss_dict)
                    for key, loss_val in step_loss_dict.items():
                        tb_writer.add_scalar(f"training-{key}", loss_val, global_step)
                step_loss_dict = defaultdict(float)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if accelerator.is_local_main_process:
                        logging.info(ma_dict.get_val_str())

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args)

                if (eval_dataset is not None and eval_fn is not None) and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    key_metric, eval_metrics = eval_fn(args, eval_dataset, model, accelerator, global_step=global_step, tb_writer=tb_writer)
                    if key_metric > best_metric:
                        best_metric = key_metric
                        save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args_to_save=args)

            if global_step >= args.max_train_steps:
                break
        # evaluation each epoch or last epoch
        if (accelerator.is_local_main_process and eval_dataset is not None and eval_fn is not None) and \
                (global_step >= args.max_train_steps or args.eval_steps <= 0):
            key_metric, eval_metrics = eval_fn(args, eval_dataset, model, accelerator, global_step=global_step, tb_writer=tb_writer)
            if key_metric > best_metric:
                best_metric = key_metric
                save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args_to_save=args)


def standard_training_and_eval_procedure(
        args, accelerator, config, tokenizer, raw_datasets,
        model_class, dataset_class, eval_fn,
        **kwargs
):

    # ====== data pre-processing ======
    train_dataset = dataset_class(args, raw_datasets, "train", tokenizer, accelerator, column_names=None)
    dev_dataset = dataset_class(args, raw_datasets, "dev", tokenizer, accelerator,
                                  column_names=train_dataset.column_names)
    if "test" in raw_datasets:
        test_dataset = dataset_class(args, raw_datasets, "test", tokenizer, accelerator,
                                       column_names=train_dataset.column_names)
    else:
        test_dataset = None

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        train(args, train_dataset, model, accelerator, tokenizer,
              eval_dataset=dev_dataset if args.do_eval else None,
              eval_fn=eval_fn if args.do_eval else None,
              )

    if args.do_eval or args.do_prediction:
        if args.do_train:
            model = model_class.from_pretrained(
                args.output_dir, config=config,
                # from_tf=bool(".ckpt" in args.model_name_or_path),
            )
        else:
            pass
        model = accelerator.prepare(model)

        if args.do_eval:
            best_dev_result, dev_eval_metric = eval_fn(args, dev_dataset, model, accelerator, global_step=None,
                                                       calibration=True, save_prediction=True, tokenizer=tokenizer)
            # args, eval_dataset, model, accelerator, global_step=global_step, tb_writter=tb_writer
        if args.do_prediction and test_dataset is not None:
            best_test_result, _ = eval_fn(args, test_dataset, model, accelerator, global_step=None,
                                          calibration=True, calibrated_temperature=dev_eval_metric["calibrated_temperature"],
                                          save_prediction=True, tokenizer=tokenizer)

        meta_best_str = f"best_dev_result: {best_dev_result}, best_test_result: {best_test_result}"

        with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
            fp.write(f"{best_dev_result}, {meta_best_str}{os.linesep}")


def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    #
    define_hparams_training(parser)
    parser.add_argument("--problem_type", default="single_label_classification", type=str)
    parser.add_argument("--enable_support", default="true", type=str)
    parser.add_argument("--enable_integration", default="true", type=str)
    parser.add_argument("--enable_distillation", default="true", type=str)
    parser.add_argument("--rel_prior_source", default="learned", type=str, choices=["learned", "nli", "ones"])

    parser.add_argument("--data_scale", default=None, type=float)

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)
    raw_datasets = load_raw_datasets(args)

    # model config
    num_labels = 1
    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            "num_labels": num_labels,
            "problem_type": args.problem_type,
        })

    config.enable_support = (args.enable_support == "true")
    config.enable_integration = (args.enable_integration == "true")
    config.enable_distillation = (args.enable_distillation == "true")
    config.rel_prior_source = args.rel_prior_source

    # add customized hyperparam here

    dataset_class = DilemmasDataset
    model_class = AutoModelForDilemmas

    # model_init -> data -> taining -> test
    standard_training_and_eval_procedure(
        args, accelerator, config, tokenizer, raw_datasets,
        model_class=model_class, dataset_class=dataset_class, eval_fn=evaluate,
    )


if __name__ == '__main__':
    main()
