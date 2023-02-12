import argparse

from peach.common import *
from sc101_gen.predict_sc101gen import *
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--knowmodel_dir", type=str, required=True)
    parsed_args = parser.parse_args()

    action_args.model_path = action_args.knowmodel_dir

    sc101_gen = SC101Generator(action_args.model_path, action_args)
    output_suffix = f".{sc101_gen.value_to_predict}.jsonl"

    input_path_list = [
        parsed_args.data_dir + "/dilemmas/dev.scruples-dilemmas.jsonl",
        parsed_args.data_dir + "/dilemmas/test.scruples-dilemmas.jsonl",
        parsed_args.data_dir + "/dilemmas/train.scruples-dilemmas.jsonl",
    ]

    for input_path in input_path_list:
        output_path = remove_filename_suffix(input_path) + output_suffix
        print("output_path:", output_path)

        example_list = load_jsonl(input_path)

        all_text = []
        for ex in example_list:
            for action in ex["actions"]:
                all_text.append(action["description"])

        all_outs = sc101_gen.predict(all_text, )

        ptr = 0
        for ex in example_list:
            for action in ex["actions"]:
                action[sc101_gen.value_to_predict + "_gen"] = all_outs[ptr]
                ptr += 1
        save_jsonl(example_list, output_path)


if __name__ == '__main__':
    main()