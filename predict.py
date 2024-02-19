import os
import glob
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score


def seed_result():
    seed_list = [0, 42, 2023]

    result_dict ={}
    for seed in seed_list:
        file_list = glob.glob(f'results/phishing/output_seed_{seed}/*.csv')
        file_list = sorted(file_list, key=lambda x: int(x.split('/')[-1].split('_')[1]))
        result_dict[seed] = []
        # file in column: label, pred
        for file in file_list:
            df = pd.read_csv(file)
            acc = accuracy_score(df['label'], df['pred'])
            f1 = f1_score(df['label'], df['pred'], average='macro')
            recall = recall_score(df['label'], df['pred'], average='macro')
            result_dict[seed].append([acc, f1, recall])

    result_acc = pd.DataFrame(result_dict)
    result_acc.to_csv('results/phishing/result_dict.csv')

def zeroshot_result():
    df = pd.read_csv('results/phishing/fewshot_0_result.csv')

    # pred가 none은 틀린 경우
    df = df[df['pred'].notnull()]
    acc = accuracy_score(df['label'], df['pred'])
    f1 = f1_score(df['label'], df['pred'], average='macro')
    recall = recall_score(df['label'], df['pred'], average='macro')
    print(acc, f1, recall)

def main():
    # seed_result()
    zeroshot_result()

if __name__ == '__main__':
    main()