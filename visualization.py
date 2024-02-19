import glob
import pandas as pd

seed_list = [0, 42, 2023]

result_acc ={}
for seed in seed_list:
    file_list = glob.glob(f'results/phishing/output_seed_{seed}/*.txt')
    file_list = sorted(file_list, key=lambda x: int(x.split('/')[-1].split('_')[1]))
    result_acc[seed] = []
    for file in file_list:
        with open(file, 'r') as f:
            line = f.readline()
            acc = float(line)
            result_acc[seed].append(acc)

result_acc = pd.DataFrame(result_acc)
result_acc.to_csv('results/phishing/result_acc.csv')
