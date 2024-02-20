import os
from argparse import Namespace

import numpy as np
import pandas as pd
import string
from tqdm import tqdm
from fire import Fire
import random

from datasets import load_dataset
from modeling import CustomLlamaStructure
from utils.base import make_save_dir


class Triviaqa:
    def __init__(self, data_dir, cache_dir):
        self.dataset = load_dataset(data_dir, 'rc.nocontext', cache_dir=cache_dir)

        self.training_docs = list(self.dataset['train'])
        self.validation_docs = self.dataset['validation']

    def doc_to_text(self, doc):
        return f"Question: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["answer"]["value"]

    def fewshot_examples(self, k, rnd):
        return rnd.sample(self.training_docs, k)

    def fewshot_context(self, doc, k, rnd):
        if k == 0:
            labeled_examples = ""
        else:
            fewshot_texts = self.fewshot_examples(k, rnd)
            labeled_examples = (
                    "\n\n".join(
                        [
                            self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshot_texts
                        ]
                    )
                    + "\n\n"
            )
        example = self.doc_to_text(doc)
        return labeled_examples + example


def evaluate(args, task, model):
    rnd = random.Random()
    rnd.seed(42)

    task_docs = list(task.validation_docs)
    print(f"Task: {args.data_dir}; number of docs: {len(task_docs)}")

    rnd.shuffle(task_docs)

    result_dict = {'prompt': [], 'question': [], 'alias': [], 'label': [], 'pred': [], 'em': []}
    all_em = []

    for doc_id, doc in enumerate(tqdm(task_docs)):
        prompt = task.fewshot_context(doc, args.k, rnd)


        # TODO: token length of question prompt
        if 'inst' in args.attn_type:
            prompt_length = model.save_prompt_length(prompt)


        pred = model.run(prompt)

        for term in ["\n", ".", ","]:
            pred = pred.split(term)[0]

        result = pred.strip().lower().translate(str.maketrans('', '', string.punctuation))
        list_of_candidates = [alias.lower().translate(str.maketrans('', '', string.punctuation)) for alias in
                              doc["answer"]["aliases"]]

        # exact match
        em = float(result in list_of_candidates)

        # save inference results
        result_dict['question'].append(doc['question'])
        result_dict['prompt'].append(prompt)
        result_dict['alias'].append(list_of_candidates)
        result_dict['label'].append(doc['answer']['value'])
        result_dict['pred'].append(pred)
        result_dict['em'].append(em)

        all_em.append(em)

    all_em = np.array(all_em)
    acc = np.mean(all_em)
    print("Average EM {:.3f}".format(acc))

    return acc, result_dict


def main(data_dir: str = "trivia_qa",
         # ntrain: int = 5,
         model_path: str = '/home/ailab/HardDrive/sue991/llama/models/llama-2-7b-hf',
         max_seq_len: int = 4096,
         max_gen_len: int = 128,
         device: int = 0,
         attn_type: str = 'origin',
         cache_dir='./data/trivia_qa',
         k: int = 0,
         # max_batch_size: int = 1,
         **kwargs):
    args = Namespace(**locals())  # local variables
    print(locals())

    model = CustomLlamaStructure(args.model_path,
                                 max_input_length=max_seq_len,
                                 max_output_length=max_gen_len,
                                 device=f"cuda:{device}" if device != 'cpu' else 'cpu',
                                 attn_type=attn_type,
                                 )

    # dataset = load_dataset(data_dir, 'rc.nocontext', cache_dir=cache_dir)
    # validation_dataset = dataset['validation'].map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer", "question_id"])
    # train_dataset = dataset['train']
    # validation_dataset = dataset['validation']
    # train_docs = list(train_dataset)
    # task_docs = list(validation_dataset)

    task = Triviaqa(args.data_dir, args.cache_dir)
    acc, result_dict = evaluate(args, task, model)

    # save_results
    if not os.path.exists('./results/trivia_qa/'):
        os.makedirs('./results/trivia_qa/')
    save_dir = make_save_dir('trivia_qa', f"{args.attn_type}/{args.k}")

    results = pd.DataFrame(result_dict)
    results.to_csv(os.path.join(save_dir, 'results.csv'))
    # save acc to file
    with open(os.path.join(save_dir, 'em.txt'), 'w') as f:
        f.write(str(acc))

    return acc


if __name__ == "__main__":
    Fire()
