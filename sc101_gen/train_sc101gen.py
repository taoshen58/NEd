import numpy as np
import torch.nn

from peach.base import *
from peach.nn_utils.general import masked_pool, zero_mask
from sc101_gen.dataset_sc101_gen import Sc101GenDataset, SYM_LABLE_DICT

from sc101_gen.models_sc101gen import AutoModelForSC101Gen

from collections import defaultdict


def eval_fn(args, eval_dataset, model, accelerator, global_step=None, tb_writer=None):
    accelerator.wait_for_everyone()
    if not accelerator.is_local_main_process:
        return
    model.eval()
    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=False)
    logger.info(f"Evaluation for {eval_dataset.data_type}:")
    # Metrics
    metrics = {
        "char": load_metric("peach/metrics/detail_classification.py")
    }
    if args.value_to_predict == "action":
        metrics["judgment"] = load_metric("peach/metrics/detail_classification.py")
        metrics["agency"] = load_metric("peach/metrics/detail_classification.py")
    else:
        pass

    ppl_list = []
    for step, batch in enumerate(eval_dataloader):
        batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())
        with torch.no_grad():
            outputs = model(**batch)
            for label_name in batch:
                if not label_name.endswith("_labels"):
                    continue
                name = label_name.split("_")[0]
                metrics[name].add_batch(
                    predictions=outputs[name + "_logits"].argmax(dim=-1),
                    references=batch[label_name],)

            # clm loss
            batch_size, seq_len, n_classes = outputs["logits"].shape
            losses_lm_2d = torch.nn.CrossEntropyLoss(reduction="none")(outputs["logits"].view(-1, n_classes), batch["labels"].view(-1))  # pooling
            losses_lm_2d = losses_lm_2d.view(batch_size, seq_len)  # bs, sl
            # use macro loss
            ppl_lm_1d = masked_pool(losses_lm_2d, batch["lm_loss_mask"], high_rank=False)
            ppl_list.append(ppl_lm_1d.detach().cpu().numpy())

    if (not eval_dataset.data_type == "test") or eval_dataset.test_has_label:
        eval_metric = {"macro_perplexity": float(np.exp(np.mean(np.concatenate(ppl_list, axis=0))))}
        eval_metric["neg_macro_perplexity"] = - eval_metric["macro_perplexity"]
        for key in metrics:
            try:
                curr_eval_metric = metrics[key].compute()
                for _k in curr_eval_metric:
                    eval_metric[key+"_"+_k] = curr_eval_metric[_k]
            except ValueError:
                logger.info(f"skipped metric {key}.")
        logger.info(f"step {global_step}: {eval_metric}")
        key_metric = "neg_macro_perplexity"
        return eval_metric[key_metric], eval_metric
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
            outputs = model(**batch)
            # calculate loss
            update_wrt_loss(args, accelerator, model, optimizer, outputs["loss"])
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item() / args.gradient_accumulation_steps
            # step_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)
                progress_bar.update(1)
                global_step += 1
                # update loss for logging
                if tb_writer is not None: # local main process
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


def main():
    parser = argparse.ArgumentParser()
    define_hparams_training(parser)
    # add task specific hyparam
    #
    parser.add_argument("--loss_components",  default="[clm]", type=str,
                        help="[clm], agree, judgment, soft1_judgment, soft2_judgment")
    parser.add_argument("--value_to_predict",  default="action", type=str, choices=["action", "rot"])

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)
    raw_datasets = load_raw_datasets(args)

    # model config
    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            # "num_labels": num_labels,
            # "problem_type": args.problem_type,
            "loss_components": args.loss_components,
            "value_to_predict": args.value_to_predict,
        })

    config.loss_components = args.loss_components
    config.value_to_predict = args.value_to_predict

    dataset_class = Sc101GenDataset
    model_class = AutoModelForSC101Gen

    # model_init -> data -> taining -> test
    # standard_training_and_eval_procedure(
    #     args, accelerator, config, tokenizer, raw_datasets,
    #     model_class=model_class, dataset_class=dataset_class, eval_fn=None,
    # )
    # eval_fn = None
    # ====== data pre-processing ======
    # # special tokens
    first_time_add_token = (not hasattr(tokenizer, "added_tokens_encoder")) or len(tokenizer.added_tokens_encoder) == 0

    if first_time_add_token:
        new_tokens = ["[|situation|]", "[|attribute|]", "[|rot|]", "[|action|]", "[|char-involved|]", ]
        for key in SYM_LABLE_DICT:
            new_tokens.append(f"[|{key}|]")
            for name, sp in SYM_LABLE_DICT[key][0].items():
                new_tokens.append(f"<|{sp}|>")
        # special token
        special_tokens_dict = {}
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<|pad|>"
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "<|eos|>"
        add_new_tokens_to_tokenizer(
            tokenizer,
            special_tokens_dict=special_tokens_dict,
            new_tokens=new_tokens, )
    else:
        new_tokens = None

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

    init_new_tokens_embeddings(model, tokenizer, new_tokens)

    if args.do_train:
        train(args, train_dataset, model, accelerator, tokenizer,
              eval_dataset=dev_dataset if args.do_eval else None,
              eval_fn=eval_fn if args.do_eval else None,
              )

    if eval_fn is not None and (args.do_eval or args.do_prediction):
        if args.do_train:
            model = model_class.from_pretrained(
                args.output_dir, config=config,
                # from_tf=bool(".ckpt" in args.model_name_or_path),
            )
        else:
            pass
        model = accelerator.prepare(model)

        if args.do_eval:
            eval_fn(args, dev_dataset, model, accelerator, global_step=None)
            # args, eval_dataset, model, accelerator, global_step=global_step, tb_writter=tb_writer
        if args.do_prediction and test_dataset is not None:
            eval_fn(args, test_dataset, model, accelerator, global_step=None)


if __name__ == '__main__':
    main()