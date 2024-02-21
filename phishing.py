"""
Adapted from https://github.com/hendrycks/test/blob/master/evaluate_flan.py
"""

from ast import arg
from email.quoprimime import header_check
import os
from argparse import Namespace
from re import T
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire
import string
import torch

# from modeling_origin import select_model
from modeling import CustomLlamaStructure
from utils.base import make_save_dir
from sklearn.metrics import f1_score, recall_score, precision_score

# from llama import Llama

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Phishing:
    def __init__(self, seed) -> None:
        self.seed = seed
        data_dir: str = f"data/seed_{seed}"
        self.train_df = pd.read_csv(f"{data_dir}/data_train.csv")
        self.valid_df = pd.read_csv(f"{data_dir}/data_valid.csv")
        self.test_df = pd.read_csv(f"{data_dir}/data_test.csv")

    def doc_to_text(self, doc):
        return f"Text: {doc['TEXT']}\n URL: {doc['URL']}\n Email: {doc['EMAIL']}\n Phone: {doc['PHONE']}\n Answer: Let's think step by step."

    def doc_to_target(self, doc):
        return " " + doc["LABEL"]

    def fewshot_examples(self, k):
        return self.train_df.sample(k, random_state=self.seed)

    def fewshot_context(self, doc, k):
        # Please distinguish whether the following text messages are written for phishing purposes or not.
        instruction = "You are a security officer. \nClassify the following SMS as 'Ham', 'Spam', 'Smishing' \n\n"
        labeled_examples = ""
        if k == 0:
            labeled_examples = ""
        else:
            fewshot_texts = self.fewshot_examples(k)
            labeled_examples = ("\n\n".join([
                self.doc_to_text(fewshot_texts.iloc[idx]) +
                self.doc_to_target(fewshot_texts.iloc[idx])
                for idx in range(len(fewshot_texts))
            ]) + "\n\n")
        example = self.doc_to_text(doc)
        return instruction + labeled_examples + example


def evaluate(args, task, model):

    # random shuffle
    task_docs = task.test_df
    task_docs = task_docs.sample(frac=1)  # shuffle test set
    print(f"Task: {args.data_dir}; number of docs: {len(task_docs)}")

    result_dict = {'text': [], 'label': [], 'pred': []}

    all_acc = []

    for idx in tqdm(range(len(task_docs))):
        doc = task_docs.iloc[idx]
        prompt = task.fewshot_context(doc, args.k)
        label = doc['LABEL']
        pred = model.run(prompt)

        for term in ["\n", ".", ","]:
            pred = pred.split(term)[0]
        pred = pred.strip().lower().translate(
            str.maketrans('', '', string.punctuation))

        result_dict['text'].append(prompt)
        result_dict['label'].append(label)
        result_dict['pred'].append(pred)

        acc = 1 if label == pred else 0
        all_acc.append(acc)

    acc = np.mean(all_acc)
    all_acc = np.array(all_acc)

    f1 = f1_score(result_dict['label'], result_dict['pred'], average='macro')
    recall = recall_score(result_dict['label'],
                          result_dict['pred'],
                          average='macro')
    precision = precision_score(result_dict['label'],
                                result_dict['pred'],
                                average='macro')
    print("Average accuracy {:.4f}".format(acc))
    print("F1 {:.4f}".format(f1))
    print("Recall {:.4f}".format(recall))
    print("Precision {:.4f}".format(precision))

    return all_acc, acc, f1, recall, precision, result_dict


def main(
        data_dir: str = "",
        # ntrain: int = 5,
        model_path:
    str = '/home/ailab/HardDrive/sue991/llama/models/llama-2-7b-hf',
        max_seq_len: int = 2048,
        max_gen_len: int = 8,
        attn_type: str = 'origin',
        k: int = 0,
        seed=42,
        device: int = 0,
        # max_batch_size: int = 1,
        **kwargs):
    args = Namespace(**locals())  # local variables

    print(locals())
    # model = select_model(max_input_length=1024, max_output_length=2, **kwargs)

    start = time.time()
    model = CustomLlamaStructure(
        model_path,
        max_input_length=max_seq_len,
        max_output_length=max_gen_len,
        device=f"cuda:{device}" if device != 'cpu' else 'cpu',
        # attn_type=attn_type,
    )
    print(f"Loading time: {time.time() - start:.3f} sec")

    task = Phishing(seed=seed)
    subjects_acc = {}
    inference_dict = {'label': [], 'pred': []}

    all_acc, acc, f1, recall, precision, result_dict = evaluate(
        args, task, model)

    # save_results
    if not os.path.exists('./results/phishing/'):
        os.makedirs('./results/phishing/')

    save_dir = make_save_dir('phishing', attn_type)

    df = pd.DataFrame(result_dict)
    df.to_csv(os.path.join(save_dir, f'fewshot_{k}_seed_{seed}_result.csv'),
              index=False)

    with open(os.path.join(save_dir, f'fewshot_{k}_seed_{seed}_acc.txt'),
              'w') as f:
        f.write(
            "ACC:" + str(acc) + "\tF1:" + str(f1) + "\tRecall:" + str(recall),
            "\tPrecision:" + str(precision))

    return acc


if __name__ == "__main__":
    Fire()
