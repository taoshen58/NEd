import numpy as np
import time
import torch
from peach.base import *
from copy import deepcopy

SUP_START_POSITION_ID = 490
SUP_MAX_LENGTH = 20

SIT_PREFIX = "Context: "
ACT_PREFIX = "Action: "

MAX_SENT_LEN = 40


class DilemmasDataset(NewDataset):
    def process_one_act(self, action):


        situation = SIT_PREFIX + action["description"]
        supportive_sent = []
        supportive_judgment_label = []
        supportive_judgment_probs = []
        supportive_nli_probs = []
        for act in action["action_gen"]:
            # if (ACT_PREFIX + act["predicted_text"]) in supportive_sent:
            #     continue
            supportive_sent.append(ACT_PREFIX + act["predicted_text"])
            supportive_judgment_label.append(act["judgment_label"])
            supportive_judgment_probs.append(act["judgment_probs"])
            supportive_nli_probs.append(act["nli_probs"])

        return situation, supportive_sent, supportive_judgment_label, supportive_judgment_probs, supportive_nli_probs

    def preprocess_function(self, example):
        assert len(example["actions"]) == 2
        max_sent_num = 5
        situation1, supportive_sent1, supportive_judgment_label1, supportive_judgment_probs1, supportive_nli_probs1 = self.process_one_act(
            example["actions"][0])
        situation2, supportive_sent2, supportive_judgment_label2, supportive_judgment_probs2, supportive_nli_probs2 = self.process_one_act(
            example["actions"][1])
        sent_num1, sent_num2 = len(supportive_sent1), len(supportive_sent2)
        assert sent_num1 <= max_sent_num and sent_num2 <= max_sent_num

        # main tokenization
        processed_output = self.tokenizer(
            [situation1, situation2],
            padding=self.padding,
            truncation=True,
            max_length=self.args.max_length,
        )
        # sup tokenization
        supportive_output = self.tokenizer(
            supportive_sent1 + supportive_sent2,
            padding=self.padding,
            truncation=True,
            max_length=SUP_MAX_LENGTH,
        )
        processed_output["sup_input_ids"] = [supportive_output["input_ids"][:sent_num1],
                                             supportive_output["input_ids"][sent_num1:sent_num1+sent_num2]]
        processed_output["sup_attention_mask"] = [supportive_output["attention_mask"][:sent_num1],
                                                  supportive_output["attention_mask"][sent_num1:sent_num1+sent_num2]]
        processed_output["sup_position_ids"] = [[list(range(SUP_START_POSITION_ID, SUP_START_POSITION_ID+len(ids)))
                                                 for ids in idss]
                                                for idss in processed_output["sup_input_ids"]]
        processed_output["sup_judgment_labels"] = [supportive_judgment_label1, supportive_judgment_label2]
        processed_output["sup_judgment_probs"] = [supportive_judgment_probs1, supportive_judgment_probs2]
        processed_output["sup_nli_probs"] = [supportive_nli_probs1, supportive_nli_probs2]

        processed_output["labels"] = example["gold_label"]
        processed_output["freq_labels"] = example["gold_annotations"]
        processed_output["soft_labels"] = list(np.array(example["gold_annotations"]) / sum(example["gold_annotations"]))
        return processed_output

    def __init__(self, *args, **kwargs):
        super(DilemmasDataset, self).__init__(*args, **kwargs)

    def get_metric(self):
        return load_metric("peach/metrics/metric_scruples.py")

    @property
    def key_metric_name(self):
        return "f1_macro"

    @property
    def test_has_label(self):
        return True

    @property
    def numeric_columns(self):
        return ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'freq_labels', 'soft_labels'] + \
               ["sup_input_ids", "sup_attention_mask", "sup_judgment_labels", "sup_judgment_probs", "sup_nli_probs", ]


class AnecdotesDataset(NewDataset):
    label_order = ["NOBODY", "AUTHOR", "OTHER", "EVERYBODY", "INFO"]

    def preprocess_function(self, example):

        # title = example["title"]
        # text = example["text"]
        text_sents = deepcopy(example["text_sents"])
        meta_sents = deepcopy(example["meta_sents"])
        # text replacement
        text_sents = [
            text.replace("WIBTA", "Will I become the asshole").replace("AITA", "Am I the asshole")
            for text in text_sents]
        # prefix a token "A " for all non-first sentence
        text_sents = [
            (SIT_PREFIX if idx == 0 else "A ") + text for idx, text in enumerate(text_sents)]
        # encode
        input_ids_list = self.tokenizer(
            text_sents, add_special_tokens=False, truncation=True,
            max_length=MAX_SENT_LEN)["input_ids"]
        input_ids_list = [input_ids if idx==0 else input_ids[1:] for idx, input_ids in enumerate(input_ids_list)]

        # truncation at sentence-level for input_ids_list and meta_sents
        cur_len = len(input_ids_list[0])
        new_input_ids_list, new_meta_sents = [input_ids_list.pop(0), ], [meta_sents.pop(0), ]
        while len(input_ids_list) > 0 and cur_len + len(input_ids_list[-1]) <= self.args.max_length - 2:
            cur_len += len(input_ids_list[-1])
            new_input_ids_list.insert(1, input_ids_list.pop(-1))
            new_meta_sents.insert(1, meta_sents.pop(-1))
        input_ids_list, meta_sents = new_input_ids_list, new_meta_sents

        num_sents = len(input_ids_list)
        len_sents = [len(input_ids) for input_ids in input_ids_list]
        span_sents = np.stack([np.cumsum([1,] + len_sents[:-1]), np.cumsum([1,] + len_sents)[1:]], axis=1).tolist()
        # input ids and spans
        input_ids = [self.tokenizer.cls_token_id]
        for ele in input_ids_list:
            input_ids.extend(ele)
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        processed_output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "span_sents": span_sents,
        }

        # support sentence
        processed_output["sup_judgment_labels"] = []
        processed_output["sup_judgment_probs"] = []
        processed_output["sup_nli_probs"] = []
        processed_output["sup_input_ids"] = []
        processed_output["sup_attention_mask"] = []
        processed_output["sup_position_ids"] = []

        for meta_sent in meta_sents:
            supportive_sent = []
            supportive_judgment_label = []
            supportive_judgment_probs = []
            supportive_nli_probs = []
            for idx_a, act in enumerate(meta_sent):
                if (ACT_PREFIX + act["predicted_text"]) in supportive_sent:
                    continue
                supportive_sent.append(ACT_PREFIX + act["predicted_text"])
                supportive_judgment_label.append(act["judgment_label"])
                supportive_judgment_probs.append(act["judgment_probs"])
                supportive_nli_probs.append(act["nli_probs"])

            top_n = 3
            neutral_probs = [elem[1] for elem in supportive_nli_probs]
            top_idxs = sorted([(idx, val) for idx, val in enumerate(neutral_probs)], key=lambda d: d[1], )
            top_idxs = set(list(zip(*top_idxs))[0][:top_n])
            supportive_sent = [elem for idx, elem in enumerate(supportive_sent) if idx in top_idxs]
            supportive_judgment_label = [elem for idx, elem in enumerate(supportive_judgment_label) if idx in top_idxs]
            supportive_judgment_probs = [elem for idx, elem in enumerate(supportive_judgment_probs) if idx in top_idxs]
            supportive_nli_probs = [elem for idx, elem in enumerate(supportive_nli_probs) if idx in top_idxs]

            tkr_outputs = self.tokenizer(
                supportive_sent,
                add_special_tokens=True,
                truncation=True,
                max_length=SUP_MAX_LENGTH,
            )
            processed_output["sup_judgment_labels"].append(supportive_judgment_label)
            processed_output["sup_judgment_probs"].append(supportive_judgment_probs)
            processed_output["sup_nli_probs"].append(supportive_nli_probs)
            processed_output["sup_input_ids"].append(tkr_outputs["input_ids"])
            processed_output["sup_attention_mask"].append(tkr_outputs["attention_mask"])
            processed_output["sup_position_ids"].append(
                [list(range(SUP_START_POSITION_ID, SUP_START_POSITION_ID+len(ids))) for ids in tkr_outputs["input_ids"]])

        # labels
        # 1. traditional label
        scores = []
        for ln in self.label_order:
            scores.append(example["label_scores"][ln])
        processed_output["freq_labels"] = scores
        processed_output["soft_labels"] = list(np.array(scores) / sum(scores))
        processed_output["labels"] = int(np.argmax(scores))
        # 2. binary label
        score_dict = example["label_scores"]  # ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]
        author_scores = [score_dict["OTHER"] + score_dict["NOBODY"],  # + score_dict["INFO"],
                           score_dict["AUTHOR"] + score_dict["EVERYBODY"], ]
        processed_output["author_freq_labels"] = author_scores
        processed_output["author_soft_labels"] = list(np.array(author_scores) / sum(author_scores))
        processed_output["author_labels"] = int(np.argmax(author_scores))

        other_scores = [score_dict["AUTHOR"] + score_dict["NOBODY"],  # + score_dict["INFO"],
                        score_dict["OTHER"] + score_dict["EVERYBODY"],]
        processed_output["other_freq_labels"] = other_scores
        processed_output["other_soft_labels"] = list(np.array(other_scores) / sum(other_scores))
        processed_output["other_labels"] = int(np.argmax(other_scores))

        info_scores = [score_dict["NOBODY"]+score_dict["AUTHOR"]+score_dict["OTHER"]+score_dict["EVERYBODY"],
                       score_dict["INFO"],]
        processed_output["info_freq_labels"] = info_scores
        processed_output["info_soft_labels"] = list(np.array(info_scores) / sum(info_scores))
        processed_output["info_labels"] = int(np.argmax(info_scores))

        return processed_output

    def __init__(self, *args, **kwargs):
        super(AnecdotesDataset, self).__init__(*args, **kwargs)

    def get_metric(self):
        return load_metric("peach/metrics/metric_scruples.py")

    @property
    def key_metric_name(self):
        return "f1_macro"

    @property
    def test_has_label(self):
        return True

    @property
    def numeric_columns(self):
        return ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'freq_labels', 'soft_labels'] + \
               ["span_sents", ] + \
               ['author_labels', 'author_freq_labels', 'author_soft_labels', ] + \
               ['other_labels', 'other_freq_labels', 'other_soft_labels', ] + \
               ['info_labels', 'info_freq_labels', 'info_soft_labels', ] + \
               ["sup_input_ids", "sup_attention_mask", "sup_judgment_labels", "sup_judgment_probs", "sup_nli_probs", ]
