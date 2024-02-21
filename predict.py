import os
import glob
import pandas as pd
from fire import Fire

from sklearn.metrics import f1_score, recall_score, accuracy_score


def main(path: str):
    df = pd.read_csv(path)

    # pred가 none은 틀린 경우
    df = df[df['pred'].notnull()]
    acc = accuracy_score(df['label'], df['pred'])
    f1 = f1_score(df['label'], df['pred'], average='macro')
    recall = recall_score(df['label'], df['pred'], average='macro')
    print("ACC:", acc, "\nF1:", f1, "\nRecall:", recall)


if __name__ == '__main__':
    Fire(main)
