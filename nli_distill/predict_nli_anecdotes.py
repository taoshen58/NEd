import os.path

from nli_distill.pipeline_entail import nli_pipeline, add_special_token
from peach.common import *
from sc101_gen.predict_sc101gen import *
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parsed_args = parser.parse_args()

    input_path_list = [
        parsed_args.data_dir + "/anecdotes/dev.scruples-anecdotes.action.jsonl",
        parsed_args.data_dir + "/anecdotes/test.scruples-anecdotes.action.jsonl",
        parsed_args.data_dir + "/anecdotes/train.scruples-anecdotes.action.jsonl",
    ]

    for input_path in input_path_list:
        example_list = load_jsonl(input_path)
        output_path = remove_filename_suffix(input_path) + ".nli.jsonl"
        print("output_path:", output_path)

        for ex in tqdm(example_list, desc=os.path.basename(input_path)):

            input_texts = []
            for sent_text, meta_sent in zip(ex["text_sents"], ex["meta_sents"]):
                input_texts.extend([
                    add_special_token(sent_text, gen_dict["predicted_text"])
                    for gen_dict in meta_sent
                ])
            # nli model
            nli_probs_list = nli_pipeline(input_texts)
            # align back
            ptr = 0
            for meta_sent in ex["meta_sents"]:
                for gen_dict in meta_sent:
                    nli_probs = nli_probs_list[ptr]
                    gen_dict["nli_probs"] = [meta_info["score"] for meta_info in nli_probs]
                    ptr += 1
            assert ptr == len(input_texts)

        save_jsonl(example_list, output_path)


if __name__ == '__main__':
    main()
