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
        parsed_args.data_dir + "/dilemmas/dev.scruples-dilemmas.jsonl.action.jsonl",
        parsed_args.data_dir + "/dilemmas/test.scruples-dilemmas.jsonl.action.jsonl",
        parsed_args.data_dir + "/dilemmas/train.scruples-dilemmas.jsonl.action.jsonl",
    ]

    for input_path in input_path_list:
        example_list = load_jsonl(input_path)
        output_path = remove_filename_suffix(input_path) + ".nli.jsonl"
        print("output_path:", output_path)

        for ex in tqdm(example_list, desc=os.path.basename(input_path)):
            for action in ex["actions"]:
                input_texts = [add_special_token(action["description"], gen_dict["predicted_text"]) for gen_dict in action["action_gen"]]
                nli_probs_list = nli_pipeline(input_texts)
                for gen_dict, nli_probs in zip(action["action_gen"], nli_probs_list):
                    gen_dict["nli_probs"] = [meta_info["score"] for meta_info in nli_probs]

        save_jsonl(example_list, output_path)


if __name__ == '__main__':
    main()