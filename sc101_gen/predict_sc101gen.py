import torch
from peach.base import *
from sc101_gen.train_sc101gen import *
from sc101_gen.dataset_sc101_gen import *

from peach.common import load_jsonl, save_jsonl, USER_HOME
import os
from transformers import BartTokenizer, BartTokenizerFast
from datasets import Dataset, DatasetDict
import spacy
nlp = spacy.load("en_core_web_sm", )  # disable=["tagging", "parsing"]
from pprint import pprint


CHAR_LIST = ["narrator", "anybody", "others"]


# dataset
class SC101GenTextDataset(NewDataset):
    def preprocess_function(self, example):
        org_situation = example["text"]
        situation = "[|situation|] " + org_situation
        dec_text = self.tokenizer.pad_token

        processed_output = self.tokenizer(
            situation, dec_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.args.max_length * 2 // 3,
            return_token_type_ids=True,
        )
        processed_output["lm_loss_mask"] = processed_output.pop("token_type_ids")

        if isinstance(self.tokenizer, (BartTokenizer, BartTokenizerFast,)):
            boundary_idx = processed_output["lm_loss_mask"].index(1)
            processed_output["lm_loss_mask"] = processed_output["lm_loss_mask"][boundary_idx:]

            # processed_output["decoder_input_ids"] = processed_output["input_ids"][boundary_idx:]
            # processed_output["decoder_attention_mask"] = processed_output["attention_mask"][boundary_idx:]

            processed_output["input_ids"] = processed_output["input_ids"][:boundary_idx]
            processed_output["attention_mask"] = processed_output["attention_mask"][:boundary_idx]
        else:
            raise NotImplementedError

        return processed_output

    def __init__(self, *args, **kwargs):
        super(SC101GenTextDataset, self).__init__(*args, **kwargs)

    # def pre_filtering_fn(self, example):
    #     return True

    # def get_metric(self):
    #     pass

    # @property
    # def key_metric_name(self):
    #     return None

    # @property
    # def test_has_label(self):
    #     return False

    @property
    def numeric_columns(self):
        return ['input_ids', 'token_type_ids', 'attention_mask', ]


class SC101Generator(object):
    def __init__(self, model_path, gen_args):
        self.accelerator = Accelerator()
        self.model_class = AutoModelForSC101Gen
        self.dataset_class = SC101GenTextDataset
        self.model_path = model_path
        self.gen_args = gen_args

        self.args = torch.load(os.path.join(model_path, "training_args.bin"))

        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = self.model_class.from_pretrained(
            model_path, config=config,)
        model.eval()
        model = self.accelerator.prepare(model)
        self.config, self.tokenizer, self.model = config, tokenizer, model

    @property
    def value_to_predict(self):
        return self.args.value_to_predict

    def get_dataloader(self, text_list):
        raw_datasets = {"test": Dataset.from_dict({"text": text_list})}
        test_dataset = self.dataset_class(self.args, raw_datasets, "test", self.tokenizer, self.accelerator,
                                          column_names=None)
        eval_dataloader = DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_size=14)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        return eval_dataloader

    def model_prediction(
            self, batch, decoder_start_token_id=None, decoder_input_ids=None,
    ):
        if decoder_input_ids is not None:
            decoder_start_token_id = None
            extra_kwargs = {"decoder_input_ids": decoder_input_ids}
        else:
            extra_kwargs = {}

        with torch.no_grad():
            num_return_sequences = self.gen_args.num_return_sequences
            batch_size = batch["input_ids"].shape[0]

            generation_outputs = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                min_length=3,
                max_length=self.args.max_length * 2 // 3,
                do_sample=self.gen_args.do_sample,
                num_beams=self.gen_args.num_beams,
                temperature=self.gen_args.temperature,
                top_k=self.gen_args.top_k,
                top_p=self.gen_args.top_p,

                decoder_start_token_id=decoder_start_token_id,  # self.tokenizer.convert_tokens_to_ids("[|action|]"),
                # decoder_input_ids=decoder_input_ids,

                num_return_sequences=self.gen_args.num_return_sequences,

                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

                # others
                early_stopping=True,
                no_repeat_ngram_size=3,

                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,

                forced_eos_token_id=True,
                **extra_kwargs,
            )

            # re-feed to get predict
            sequences = generation_outputs["sequences"].detach()
            decoder_attention_mask = torch.eq(
                torch.eq(
                    torch.cat([sequences.new_zeros(sequences.shape[0], 1), sequences[:, :-1]], dim=1),
                    self.tokenizer.eos_token_id
                ).to(sequences.dtype).cumsum(dim=1),
                0
            ).to(sequences.dtype)

            refeed_batch = {
                "input_ids": batch["input_ids"].unsqueeze(1).repeat(1, num_return_sequences, 1).view(
                    batch_size * num_return_sequences, -1),
                "attention_mask": batch["attention_mask"].unsqueeze(1).repeat(1, num_return_sequences, 1).view(
                    batch_size * num_return_sequences, -1),
                "decoder_input_ids": sequences,
                "decoder_attention_mask": decoder_attention_mask,
            }
            model_outputs = self.model(**refeed_batch)
            model_outputs["decoder_attention_mask"] = decoder_attention_mask

            return generation_outputs, model_outputs

    def predict_action(self, sentence_list):
        eval_dataloader = self.get_dataloader(sentence_list)

        num_return_sequences = self.gen_args.num_return_sequences

        all_predicts = []
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Generating", disable=False):
            batch = dict((k, v.to(self.accelerator.device)) for k, v in batch.items())
            batch_size = batch["input_ids"].shape[0]
            # assert batch_size == 1

            generation_outputs, model_outputs = self.model_prediction(
                batch, decoder_start_token_id=self.tokenizer.convert_tokens_to_ids("[|action|]"),

            )

            nbs = batch_size * num_return_sequences  # new_batch_size
            lm_mask = model_outputs["decoder_attention_mask"]
            lm_mask = torch.cat(
                [
                    # lm_mask.new_zeros([nbs, 0]),
                    lm_mask[:, 1:],
                    lm_mask.new_zeros([nbs, 1]),
                ], dim=1
            )
            nbs_sanity, seq_len, n_classes = model_outputs["logits"].shape
            assert nbs == nbs_sanity
            lm_label = torch.cat([generation_outputs["sequences"][:, 1:],
                                  generation_outputs["sequences"].new_ones(nbs, 1) * self.tokenizer.pad_token_id],
                                 dim=-1)
            losses_lm_2d = torch.nn.CrossEntropyLoss(reduction="none")(
                model_outputs["logits"].view(-1, n_classes), lm_label.view(-1))  # pooling
            losses_lm_2d = losses_lm_2d.view(nbs, seq_len)  # bs, sl
            # use macro loss
            ppl_lm_1d = masked_pool(losses_lm_2d, lm_mask, high_rank=False).exp()

            for idx_ex in range(batch_size):
                curr_predicts = []
                for idx_s in range(num_return_sequences):
                    idx_line = idx_ex * num_return_sequences + idx_s

                    token_ids = generation_outputs["sequences"][idx_line, :].detach().cpu().numpy().tolist()
                    # token_names = tokenizer.convert_ids_to_tokens(token_ids)
                    eos_idx = token_ids.index(self.tokenizer.eos_token_id)
                    if eos_idx > -1:
                        token_ids = token_ids[:(eos_idx + 1)]  # include <eos>
                    else:
                        token_ids.append(self.tokenizer.eos_token_id)

                    if self.tokenizer.convert_ids_to_tokens(token_ids[0]) != "[|action|]":
                        continue

                    predicted_dict = {
                        "predicted_text": self.tokenizer.decode(token_ids[1:], skip_special_tokens=True).strip(),
                        # "predicted_text_type": self.args.value_to_predict,
                        "ppl": ppl_lm_1d[idx_line].item(),
                    }
                    # meta
                    for meta_name in ["char", "judgment"]:
                        _probs = torch.softmax(model_outputs[meta_name + "_logits"][idx_line], dim=-1)
                        predicted_dict[meta_name+"_probs"] = _probs.detach().cpu().numpy().tolist()
                        predicted_dict[meta_name+"_label"] = _probs.argmax(dim=-1).item()
                        if meta_name == "char":
                            predicted_dict[meta_name + "_label"] = CHAR_LIST[predicted_dict[meta_name + "_label"]]

                    curr_predicts.append(predicted_dict)
                all_predicts.append(curr_predicts)
        return all_predicts

    def predict_rot(self, sentence_list,):
        eval_dataloader = self.get_dataloader(sentence_list)
        num_return_sequences = self.gen_args.num_return_sequences

        multi_all_predicts = dict()
        for rot_cat in ROT_CATEGORIZATION_TO_STR:
            if rot_cat == "description":
                continue
            all_predicts = []
            for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader),
                                    desc="Generating-"+rot_cat,
                                    disable=False):
                batch = dict((k, v.to(self.accelerator.device)) for k, v in batch.items())
                batch_size = batch["input_ids"].shape[0]

                decoder_input_ids = self.tokenizer(f"[|rot-categorization|]<|{ROT_CATEGORIZATION_TO_STR[rot_cat]}|>[|rot|]", add_special_tokens=False)
                decoder_input_ids = torch.tensor(decoder_input_ids["input_ids"], dtype=torch.long).to(self.accelerator.device)
                decoder_input_ids = decoder_input_ids.unsqueeze(0).repeat(batch_size, 1)  # bs,3

                generation_outputs, model_outputs = self.model_prediction(
                    batch, # decoder_start_token_id=self.tokenizer.convert_tokens_to_ids("[|rot|]"),
                    decoder_input_ids=decoder_input_ids,
                )

                # clm loss
                nbs = batch_size * num_return_sequences  # new_batch_size
                lm_mask = model_outputs["decoder_attention_mask"]
                lm_mask = torch.cat(
                    [
                        lm_mask.new_zeros([nbs, 2]),
                        lm_mask[:,3:],
                        lm_mask.new_zeros([nbs, 1]),
                    ], dim=1
                )
                nbs_sanity, seq_len, n_classes = model_outputs["logits"].shape
                assert nbs == nbs_sanity
                lm_label = torch.cat([generation_outputs["sequences"][:,1:], generation_outputs["sequences"].new_ones(nbs, 1) * self.tokenizer.pad_token_id], dim=-1)
                losses_lm_2d = torch.nn.CrossEntropyLoss(reduction="none")(
                    model_outputs["logits"].view(-1, n_classes), lm_label.view(-1))  # pooling
                losses_lm_2d = losses_lm_2d.view(nbs, seq_len)  # bs, sl
                # use macro loss
                ppl_lm_1d = masked_pool(losses_lm_2d, lm_mask, high_rank=False).exp()

                for idx_ex in range(batch_size):
                    curr_predicts = []
                    for idx_s in range(num_return_sequences):
                        idx_line = idx_ex * num_return_sequences + idx_s

                        token_ids = generation_outputs["sequences"][idx_line, :].detach().cpu().numpy().tolist()
                        # token_names = tokenizer.convert_ids_to_tokens(token_ids)
                        eos_idx = token_ids.index(self.tokenizer.eos_token_id)
                        if eos_idx > -1:
                            token_ids = token_ids[:(eos_idx + 1)]  # include <eos>
                        else:
                            token_ids.append(self.tokenizer.eos_token_id)

                        predicted_dict = {
                            "predicted_text": self.tokenizer.decode(token_ids[3:], skip_special_tokens=True).strip(),
                            # "predicted_text_type": self.args.value_to_predict,
                            # "rot_categorization": rot_cat,
                            "ppl": ppl_lm_1d[idx_line].item(),
                        }

                        # meta
                        for meta_name in ["char", ]:
                            _probs = torch.softmax(model_outputs[meta_name + "_logits"][idx_line], dim=-1)
                            predicted_dict[meta_name + "_probs"] = _probs.detach().cpu().numpy().tolist()
                            predicted_dict[meta_name + "_label"] = _probs.argmax(dim=-1).item()
                            if meta_name == "char":
                                predicted_dict[meta_name + "_label"] = CHAR_LIST[predicted_dict[meta_name + "_label"]]

                        curr_predicts.append(predicted_dict)
                    all_predicts.append(curr_predicts)
            multi_all_predicts[rot_cat] = all_predicts
        # re-org
        new_all_predicts = []
        for idx_s in range(len(sentence_list)):
            ex = dict()
            for rot_cat in ROT_CATEGORIZATION_TO_STR:
                try:
                    ex[rot_cat] = multi_all_predicts[rot_cat][idx_s]
                except KeyError:
                    pass
            new_all_predicts.append(ex)
        return new_all_predicts

    def predict(self, sentence_list):
        if self.args.value_to_predict == "action":
            return self.predict_action(sentence_list)
        else:
            return self.predict_rot(sentence_list)

    def _lagacy_predict(self, sentence_list):
        raw_datasets = {"test": Dataset.from_dict({"text": sentence_list})}
        test_dataset = self.dataset_class(self.args, raw_datasets, "test", self.tokenizer, self.accelerator,
                                     column_names=None)
        eval_dataloader = DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_size=1)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        num_return_sequences = 5
        predicted_dicts = []

        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=False):
            curr_predicted_dicts = []
            batch = dict((k, v.to(self.accelerator.device)) for k, v in batch.items())
            batch_size = batch["input_ids"].shape[0]
            assert batch_size == 1
            with torch.no_grad():

                # build <|other|> token
                # decoder_input_ids = self.tokenizer.convert_tokens_to_ids(["[|char-involved|]", "<|others|>"]),
                # decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(self.accelerator.device)

                generation_outputs = self.model.generate(
                    batch["input_ids"],
                    min_length=6,
                    max_length=self.args.max_length * 2 // 3,
                    do_sample=self.gen_args.do_sample,
                    num_beams=self.gen_args.num_beams,
                    temperature=self.gen_args.temperature,
                    top_k=self.gen_args.top_k,
                    top_p=self.gen_args.top_p,

                    decoder_start_token_id=self.tokenizer.convert_tokens_to_ids("[|char-involved|]"),
                    # decoder_input_ids =decoder_input_ids,

                    num_return_sequences=self.gen_args.num_return_sequences,

                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,

                    # others
                    early_stopping=True,
                    no_repeat_ngram_size=3,

                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,

                    forced_eos_token_id=True,
                )

                # re-feed to get predict
                sequences = generation_outputs["sequences"].detach()
                sequences_scores = generation_outputs["sequences_scores"].detach()
                decoder_attention_mask = torch.eq(
                    torch.eq(
                        torch.cat([sequences.new_zeros(sequences.shape[0], 1), sequences[:, :-1]], dim=1),
                        self.tokenizer.eos_token_id
                    ).to(sequences.dtype).cumsum(dim=1),
                    0
                ).to(sequences.dtype)

                refeed_batch = {
                    "input_ids": batch["input_ids"].unsqueeze(1).repeat(1, num_return_sequences, 1).view(
                        batch_size * num_return_sequences, -1),
                    "attention_mask": batch["attention_mask"].unsqueeze(1).repeat(1, num_return_sequences, 1).view(
                        batch_size * num_return_sequences, -1),
                    "decoder_input_ids": sequences,
                    "decoder_attention_mask": decoder_attention_mask,
                }
                #  ============
                model_outputs = self.model(**refeed_batch)

                assert batch_size == 1

                judgment_logits = model_outputs["judgment_logits"]
                judgment_probs = torch.softmax(judgment_logits, dim=-1)
                judgment_label = judgment_probs.argmax(-1)

                agree_logits = model_outputs["agree_logits"]
                agree_probs = torch.softmax(agree_logits, dim=-1)
                agree_label = agree_probs.argmax(-1)

                # get result
                for idx_seq in range(num_return_sequences):
                    token_ids = sequences[idx_seq, :].detach().cpu().numpy().tolist()
                    # token_names = tokenizer.convert_ids_to_tokens(token_ids)
                    eos_idx = token_ids.index(self.tokenizer.eos_token_id)
                    if eos_idx > -1:
                        token_ids = token_ids[:(eos_idx + 1)]  # include <eos>
                    else:
                        token_ids.append(self.tokenizer.eos_token_id)

                    if len(token_ids) < 5:
                        continue
                    if token_ids[1] not in self.tokenizer.convert_tokens_to_ids(["<|narrator|>", "<|others|>", "<|anybody|>", ]):
                        continue

                    predicted_dict = {
                        "predicted_character": self.tokenizer.convert_ids_to_tokens(token_ids[1]),
                        "predicted_text_score": sequences_scores[idx_seq].item(),
                        "predicted_text": self.tokenizer.decode(token_ids[3:], skip_special_tokens=True),
                        "predicted_text_type": self.args.value_to_predict,
                    }

                    if self.args.value_to_predict == "action":
                        if token_ids[2] != self.tokenizer.convert_tokens_to_ids("[|action|]"):
                            continue
                        predicted_dict["predicted_judgment_probs"] = judgment_probs[idx_seq].detach().cpu().numpy().tolist()
                        predicted_dict["predicted_judgment_label"] = judgment_label[idx_seq].item()

                    elif self.args.value_to_predict == "rot":
                        if token_ids[2] != self.tokenizer.convert_tokens_to_ids("[|rot|]"):
                            continue
                        predicted_dict["predicted_agree_probs"] = agree_probs[
                            idx_seq].detach().cpu().numpy().tolist()
                        predicted_dict["predicted_agree_label"] = agree_label[idx_seq].item()
                    else:
                        raise NotImplementedError
                    curr_predicted_dicts.append(predicted_dict)
            predicted_dicts.append(curr_predicted_dicts)
        return predicted_dicts


def sentence_split(text, ):
    doc = nlp(text, )
    sent_list = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_list.append(sent_text)
    return sent_list

action_args = CustomArgs(
    model_path=None,
    do_sample=True,
    num_beams=2,
    temperature=1.5,
    top_k=50,
    top_p=0.9,
    num_return_sequences=5,
)