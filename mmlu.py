"""
Adapted from https://github.com/hendrycks/test/blob/master/evaluate_flan.py
"""

import os
from argparse import Namespace

import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire

# from modeling_origin import select_model
from modeling import CustomLlamaStructure
from utils.base import make_save_dir

# from llama import Llama

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    return " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2  # a number of choices
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, -1])
    return prompt


def gen_prompt(df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))

    if k == -1:
        k = df.shape[0]
    for i in range(k):
        prompt += format_example(df, i)
    return prompt


def evaluate(args, subject, model, dev_df, test_df):
    cors = []
    all_probs = []

    result_dict = {'subject': [], 'question': [], 'label': [], 'pred': []}

    ## TODO: token length of instruction prompt
    if 'inst' in args.attn_type:
        instruction = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject))

        prompt_length = model.save_prompt_length(instruction)
        # prompt_length = model.count_text_length(
        #     "The following are multiple choice questions (with answers) about {}.\n\n".format(
        #         format_subject(subject)))

    for i in range(len(test_df)):
        # get prompt and make sure if fits
        k = args.ntrain
        test_prompt = format_example(test_df, i, include_answer=False)
        dev_prompt = gen_prompt(dev_df, subject, k)
        prompt = dev_prompt + test_prompt

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            dev_prompt = gen_prompt(dev_df, subject, k)
            prompt = dev_prompt + test_prompt

        label = test_df.iloc[i, -1]
        pred = model.run(prompt)

        # save inference results
        result_dict['subject'].append(subject)
        result_dict['question'].append(test_prompt)
        result_dict['label'].append(label)
        result_dict['pred'].append(pred)

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, result_dict


def main(data_dir: str = "data/seed_42",
         ntrain: int = 5,
         model_path: str = '/home/jaeyoung/llama/llama-2-7b-hf',
         max_seq_len: int = 2048,
         max_gen_len: int = 2,
         device: str = '1',
         attn_type: str = 'origin',
         # max_batch_size: int = 1,
         **kwargs):
    args = Namespace(**locals())  # local variables

    # model = select_model(max_input_length=1024, max_output_length=2, **kwargs)
    model = CustomLlamaStructure(args.model_path,
                                 max_input_length=max_seq_len,
                                 max_output_length=max_gen_len,
                                 device=f"cuda:{device}" if device != 'cpu' else 'cpu',
                                 attn_type=attn_type,
                                 )

    print(locals())

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if f.endswith("_test.csv")
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    subjects_acc = {}
    inference_dict = {'subject': [], 'question': [], 'label': [], 'pred': []}

    for i, subject in enumerate(tqdm(subjects)):
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"), header=None)[: args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{subject}_test.csv"), header=None)

        cors, acc, probs, result_dict = evaluate(args, subject, model, dev_df, test_df)
        subcats = get_subcategories()[subject]
        subjects_acc[subject] = acc
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # save inference results
        inference_dict['subject'].extend(result_dict['subject'])
        inference_dict['question'].extend(result_dict['question'])
        inference_dict['label'].extend(result_dict['label'])
        inference_dict['pred'].extend(result_dict['pred'])

    # save_results
    if not os.path.exists('./results/mmlu/'):
        os.mkdir('./results/mmlu/')

    save_dir = make_save_dir('mmlu', args.attn_type)

    subcat_accs, cat_accs = {}, {}

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        subcat_accs[subcat] = subcat_acc

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        cat_accs[cat] = cat_acc

    subjects_result = pd.DataFrame.from_dict(subjects_acc, orient='index', columns=['acc'])
    subcat_result = pd.DataFrame.from_dict(subcat_accs, orient='index', columns=['acc'])
    cat_result = pd.DataFrame.from_dict(cat_accs, orient='index', columns=['acc'])
    inference_result = pd.DataFrame(inference_dict)

    subjects_result.to_csv(os.path.join(save_dir, 'subjects_result.csv'))
    subcat_result.to_csv(os.path.join(save_dir, 'subcat_result.csv'))
    cat_result.to_csv(os.path.join(save_dir, 'cat_result.csv'))
    inference_result.to_csv(os.path.join(save_dir, 'inference_result.csv'))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    with open(os.path.join(save_dir, 'weighted_acc.txt'), 'w') as f:
        f.write(str(weighted_acc))

    return weighted_acc


if __name__ == "__main__":
    Fire()
