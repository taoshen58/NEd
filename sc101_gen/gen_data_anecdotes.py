import argparse

from peach.common import *
from sc101_gen.predict_sc101gen import *
from tqdm import tqdm

import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_split(text, ):
    doc = nlp(text, )
    sent_list = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_list.append(sent_text)
    return sent_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--knowmodel_dir", type=str, required=True)
    parsed_args = parser.parse_args()

    action_args.model_path = action_args.knowmodel_dir

    sc101_gen = SC101Generator(action_args.model_path, action_args)
    output_suffix = f".{sc101_gen.value_to_predict}.jsonl"

    input_path_list = [
        parsed_args.data_dir + "/anecdotes/dev.scruples-anecdotes.jsonl",
        parsed_args.data_dir + "/anecdotes/test.scruples-anecdotes.jsonl",
        parsed_args.data_dir + "/anecdotes/train.scruples-anecdotes.jsonl",
    ]

    for input_path in input_path_list:
        output_path = remove_filename_suffix(input_path) + output_suffix
        print("output_path:", output_path)

        example_list = load_jsonl(input_path)

        text_list = []
        for ex in tqdm(example_list):

            paragraphs = ex["text"].split("\n")
            paragraphs = [p for p in paragraphs if len(p) > 0]

            text_sents = [ex["title"], ]  # add title as the first
            # sents_meta = []
            for p in paragraphs:
                text_sents.extend(sentence_split(p))

            ex["text_sents"] = text_sents
            text_list.extend(text_sents)

        all_outs = sc101_gen.predict(text_list, )

        ptr = 0
        for ex in tqdm(example_list):
            meta_sents = []

            for _ in ex["text_sents"]:
                meta_sents.append(all_outs[ptr])
                ptr += 1
            ex["meta_sents"] = meta_sents
        assert len(all_outs) == ptr

        save_jsonl(example_list, output_path)

if __name__ == '__main__':
    main()