import numpy as np
import random
from peach.base import *
from collections import OrderedDict
from transformers import BartTokenizer, BartTokenizerFast
from scipy.stats import norm

# special tokens
AGREE_TO_STR = OrderedDict(
    [("0", "nobody-agree"), ("1", "rare-agree"), ("2", "controversial-agree"),
     ("3", "most-agree"), ("4", "all-agree")]
)
# =============== RoT =================
ROT_CATEGORIZATION_TO_STR = OrderedDict(
    [("morality-ethics", "morality-ethics-category"),
     ("social-norms", "social-norms-category"),
     ("advice", "advice-category"),
     ("description", "description-category"),]
)

ROT_MORAL_FOUNDATIONS_TO_STR = OrderedDict(
    [("care-harm", "care-harm-foundation"),
     ("fairness-cheating", "fairness-cheating-foundation"),
     ("loyalty-betrayal", "loyalty-betrayal-foundation"),
     ("authority-subversion", "authority-subversion-foundation"),
     ("sanctity-degradation", "sanctity-degradation-foundation"),]
)

# =============== Act =================
ACTION_AGENCY_TO_STR = OrderedDict(
    [("agency", "agency"),
     ("experience", "experience"),]
)

ACTION_MORAL_JUDGMENT_TO_STR = OrderedDict(
    [("-2", "very-bad-moral"), ("-1", "bad-moral"), ("0", "ok-moral"),
     ("1", "good-moral"), ("2", "very-good-moral")]
)

ACTION_LEGAL_TO_STR = OrderedDict(
    [("legal", "legal"),
     ("illegal", "illegal"),
     ("tolerated", "tolerated"),]
)

ACTION_PRESSURE_TO_STR = OrderedDict(
    [
        ("-2", "strong-against-culture"),
        ("-1", "against-culture"),
        ("0", "discretionary-culture"),
        ("1", "for-culture"),
        ("2", "strong-for-culture"),]
)

ACTION_HYPOTHETICAL_TO_STR = OrderedDict(
    [("explicit-no", "explicit-no"),
     ("probable-no", "probable-no"),
     ("hypothetical", "hypothetical"),
     ("probable", "probable"),
     ("explicit", "explicit"),]
)

CHAR_TO_STR = OrderedDict(
    [("char-0", "narrator"),
     ("char-none", "anybody"),
     ("char-1", "others"),
     ("char-2", "others"),
     ("char-3", "others"),
     ("char-4", "others"),
     ("char-5", "others"),
     ("char-6", "others"),]
)

# categorized_labels, can_be_null, is_multi_label
SYM_LABLE_DICT = {
# ====== RoTs ======
    "rot-agree": (AGREE_TO_STR, True, False), # filtering
    "rot-categorization": (ROT_CATEGORIZATION_TO_STR, False, True),  # use this for diversity
    "rot-moral-foundations": (ROT_MORAL_FOUNDATIONS_TO_STR, False, True),
    "rot-char-targeting": (CHAR_TO_STR, False, False),  # check
    # ====== Actions ======
    "action-agree": (AGREE_TO_STR, True, False),  # filtering
    "action-agency": (ACTION_AGENCY_TO_STR, True, False),  # check
    "action-moral-judgment": (ACTION_MORAL_JUDGMENT_TO_STR, True, False),  # check
    "action-legal": (ACTION_LEGAL_TO_STR, True, False),
    "action-pressure": (ACTION_PRESSURE_TO_STR, True, False),
    "action-hypothetical": (ACTION_HYPOTHETICAL_TO_STR, True, False),
    "action-char-involved": (CHAR_TO_STR, False, False),  # check
    # ===== Others ======
    # n-characters, characters
}

# ======= label smoothing
def get_gaussian_label_smoothing_mask():
    scores_list = []
    for agree in range(5):
        contradiction = 4 - agree
        scores = [0.5]
        for delta in range(1, 6):
            scores.append(norm.cdf(delta, loc=0, scale=contradiction or 0.01))
        scores = np.array(scores)
        scores = scores[1:] - scores[:-1]
        scores = scores/sum(scores)
        scores[scores< 0.2] = 0.
        scores = scores / sum(scores)
        scores_list.append(scores)
    return np.stack(scores_list, axis=0)
gaussian_label_smoothing_mask = get_gaussian_label_smoothing_mask()  # 5,5

TOKEN_SITUATION = "[|situation|]"
TOKEN_ACTION = "[|action|]"
TOKEN_ROT = "[|rot|]"


class Sc101GenDataset(NewDataset):

    def preprocess_function(self, example):

        # fix char
        example["rot-char-targeting"] = example["rot-char-targeting"] if len(example["rot-char-targeting"]) > 0 else "char-none"
        example["action-char-involved"] = example["action-char-involved"] if len(example["action-char-involved"]) > 0 else "char-none"

        rot = example
        rot_id_dict = dict()
        rot_attr_dict = dict()
        for key in SYM_LABLE_DICT:
            label2attr, can_be_null, is_multi_label = SYM_LABLE_DICT[key][:3]
            all_labels = list(label2attr)  # label2attr.keys()
            if is_multi_label:
                label_id = [0] * len(label2attr)
                label_attr = []
                for s in rot[key].strip().split("|"):
                    try:
                        label_id[all_labels.index(s)] = 1
                        label_attr.append(f"<|{label2attr[s]}|>")
                    except ValueError:
                        pass
                rot_id_dict["id_"+key] = label_id
                rot_attr_dict["attr_" + key] = label_attr
            else:
                try:
                    label_id = all_labels.index(rot[key].strip())
                    label_attr = f"<|{label2attr[rot[key]]}|>"
                    if label_attr == "<|others|>":
                        anchor = 0
                except ValueError:
                    assert rot[key].strip() == ""
                    label_id = None
                    label_attr = None

                if (key == "rot-char-targeting" or key == "action-char-involved") and label_id is not None:
                    label_id = min(label_id, 2)
                rot_id_dict["id_" + key] = label_id
                rot_attr_dict["attr_" + key] = label_attr
            # rot_id_list.append(rot_id_dict)
            # rot_attr_list.append(rot_attr_dict)

        # tokenization
        situation = "[|situation|] " + example["situation"].strip()
        if self.args.value_to_predict == "action":
            dec_text = "[|action|] " + example["action"].strip()
        elif self.args.value_to_predict == "rot":
            dec_text = "[|rot-categorization|]" + random.choice(rot_attr_dict["attr_rot-categorization"]) + \
                       "[|rot|] " + example["rot"].strip()
        else:
            raise NotImplementedError

        processed_output = self.tokenizer(
            situation, dec_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.args.max_length - 1,
            return_token_type_ids=True,
        )

        if self.args.value_to_predict == "action":
            processed_output["judgment_labels"] = rot_id_dict["id_action-moral-judgment"]
            processed_output["agency_labels"] = rot_id_dict["id_action-agency"]
            processed_output["char_labels"] = rot_id_dict["id_action-char-involved"]
        else:
            processed_output["char_labels"] = rot_id_dict["id_rot-char-targeting"]

        # add <eos> token
        processed_output["lm_loss_mask"] = processed_output.pop("token_type_ids")

        processed_output["input_ids"].append(self.tokenizer.eos_token_id)
        processed_output["attention_mask"].append(1)
        processed_output["lm_loss_mask"].append(0)  # this is use as mask to calc lm_loss, so ignore <eos>

        processed_output["labels"] = processed_output["input_ids"][1:] + [self.tokenizer.pad_token_id]

        if isinstance(self.tokenizer, (BartTokenizer, BartTokenizerFast,)):
            boundary_idx = processed_output["lm_loss_mask"].index(1)
            processed_output["lm_loss_mask"] = processed_output["lm_loss_mask"][boundary_idx:]
            processed_output["labels"] = processed_output["labels"][boundary_idx:]

            processed_output["decoder_input_ids"] = processed_output["input_ids"][boundary_idx:]
            processed_output["input_ids"] = processed_output["input_ids"][:boundary_idx]
            processed_output["decoder_attention_mask"] = processed_output["attention_mask"][boundary_idx:]
            processed_output["attention_mask"] = processed_output["attention_mask"][:boundary_idx]

        return processed_output

    def __init__(self, *args, **kwargs):
        super(Sc101GenDataset, self).__init__(*args, **kwargs)

    def pre_filtering_fn(self, example):
        if self.args.value_to_predict == "action":
            valid = all(key not in example or example[key] != "" for key in [
                "action-moral-judgment", "action-agency", self.args.value_to_predict + "-agree"])
        else:  # == "rot"
            valid = all(key not in example or example[key] != "" for key in [
                "rot-categorization", self.args.value_to_predict + "-agree"])
        if valid:  # remove low agreement example
            valid = valid and (example[self.args.value_to_predict + "-agree"] != "0" and example[self.args.value_to_predict + "-agree"] != "1")
        return valid


    def get_metric(self):
        return load_metric("bleu")

    @property
    def key_metric_name(self):
        return "accuracy"

    @property
    def test_has_label(self):
        return True

    @property
    def numeric_columns(self):
        return ['input_ids', 'token_type_ids', 'lm_loss_mask', 'attention_mask',
                "decoder_input_ids", "decoder_attention_mask",
                'labels', "judgment_labels", "agency_labels", "char_labels", ]  # "soft_labels"



