import pandas as pd
import numpy as np
import os
import glob
from math import sqrt
import csv


def evaluate_max(original_dataset, results_dataset, target_name='', bounds={}, print_res=False, columns=[]):

    result = pd.read_csv(results_dataset)
    df_original = pd.read_csv(original_dataset)

    indices = []

    for ii in range(len(result)):
        res = result.loc[ii, columns]
    
        conditions = {}

        for key, value in res.iteritems():

            if key != "target":
                conditions[key] = value

        mask = pd.Series([True] * len(df_original))

        for column, value in conditions.items():
            mask &= (df_original[column] == value)

        founds_ = df_original[mask]
        indeces_ = df_original[mask].index.tolist()

        for idx in indeces_:
            
            if round(res.loc["target"],6) == round(founds_.loc[idx][target_name],6):
                indices.append(idx)

    df = pd.read_csv(original_dataset).loc[indices].sort_values(by=target_name, ascending=False)

    for index, row in df.iterrows():

        constraints_respected = True
        
        for key, value in bounds.items():
            
            lb, ub = value

            if row[key] < lb or row[key] > ub:
                constraints_respected = False
                
        if constraints_respected:
            if print_res:
                print("\nMax feasible:")
                print(row)
            return row[target_name]
        
    return -np.inf


def list_csv_files(directory):
    csv_files = [file for file in glob.glob("%s/*.csv" %directory)]
    return csv_files


def mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def variance(lst):
    if len(lst) == 0:
        return 0.0 

    mean_val = sum(lst) / len(lst)
    
    if abs(mean_val) == np.inf:
        return np.inf

    squared_diff = [(x - mean_val) ** 2 for x in lst]
    variance_val = sum(squared_diff) / len(lst)

    return variance_val


if __name__ == "__main__":

    dataset = "ligen"
    csv_files = list_csv_files("experiments/%s/pr__ei" %dataset)

    header = ['error (%)', 'std_dev (%)', 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)', 'avg_time (sec)']

    maximum = {
        "oscarp": -0.174361111,
        "stereomatch": -40196.0,
        "query26": -4079658.0,
        "ligen": -567.312555400384
    }

    columns = {
        "oscarp": ['parallelism_ffmpeg-0', 'parallelism_librosa', 'parallelism_ffmpeg-1', 'parallelism_ffmpeg-2', 'parallelism_deepspeech', 'target'],
        "query26": ['#vm', 'ram', 'target'],
        "stereomatch": ["confidence", "hypo_step", "max_arm_length", "num_threads", 'target'],
        "ligen": ['ALIGN_SPLIT', 'OPTIMIZE_SPLIT', 'OPTIMIZE_REPS', 'CUDA_THREADS', 'N_RESTART', 'CLIPPING', 'SIM_THRESH', 'BUFFER_SIZE', 'target']
    }

    target_name = {
        "oscarp": "-cost",
        "query26": "-cost",
        "stereomatch": "-cost",
        "ligen": "-RMSD^3*TIME"
    }

    bounds = {
        "oscarp": {"total_time": [0, 300.00001]},
        "query26": {"time": [0, 205000]},
        "stereomatch": {"time": [0, 17000]},
        "ligen": {"RMSD_0.75": [0.0, 2.1]}
    }

    real_max = maximum[dataset]

    repetitions = 30

    if len(csv_files) != repetitions:
        import pdb; pdb.set_trace()

    results = []

    for file in csv_files:
        print(file)
        results.append(evaluate_max("discrete_mixed_bo/problems/data/%s.csv" %dataset, file, target_name=target_name[dataset], bounds=bounds[dataset], print_res=False, columns=columns[dataset]))

    results_cleaned = [x for x in results if x != float('inf') and x != float('-inf')]

    data = [[round(100*(mean(results) - real_max)/real_max,3), round(100*sqrt(variance(results))/abs(real_max),3), round(100*(mean(results_cleaned) - real_max)/(real_max),3), round(100*sqrt(variance(results_cleaned))/abs(real_max),3), round(100*len(results_cleaned)/repetitions,3)]]

    with open("%s_bo_pr.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  