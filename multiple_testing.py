from numpy.random import randint
import os
import time
import csv
from math import sqrt

def mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def variance(lst):
    if len(lst) == 0:
        return 0 

    mean_val = sum(lst) / len(lst)
    squared_diff = [(x - mean_val) ** 2 for x in lst]
    variance_val = sum(squared_diff) / len(lst)

    return variance_val

def analyse_single_result(filename):

    keyword = "current best obj: "

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if keyword in line:
                return -float(line.split(keyword)[1].split(".\n")[0])
    return None

repetitions = 30
seeds = randint(0,1e9,repetitions).tolist()

algorithm = "pr__ei"

for dataset in ["oscarp", "stereomatch", "query26", "ligen"]:

    print("\nDataset: %s" %dataset)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("outputs/%s" %dataset):
        os.makedirs("outputs/%s" %dataset)
    if not os.path.exists("outputs/%s/%s" %(dataset, algorithm)):
        os.makedirs("outputs/%s/%s" %(dataset, algorithm))

    maximum = {
        "oscarp": 0.174361111,
        "stereomatch": 36791.0,
        "query26": 683682.0,
        "ligen": 340.099251789615
    }

    durations = []
    results = []

    for idx in range(len(seeds)):

        seed = seeds[idx]

        print("\nSimulation %s out of %s, Seed: %s\n" %(idx+1,len(seeds),seed))

        output_filename = "outputs/%s/%s/output_%s.txt" %(dataset, algorithm, seed)

        start_time = time.time()
        os.system("python3 experiments/main.py %s %s %s > %s" %(dataset, algorithm, seed, output_filename))
        durations.append(time.time() - start_time)

        results.append(analyse_single_result(output_filename))
        print("Error: %s %%, time = %s sec" %(round(100*(results[-1]-maximum[dataset])/maximum[dataset],3), round(durations[-1],3)))

    #results_cleaned = [x for x in results if x != float('inf') and x != float('-inf')]

    header = ['error (%)', 'std_dev (%)', 'time (s)'] #, 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)']

    data = [[round(100*(mean(results) - maximum[dataset])/maximum[dataset],3), round(100*sqrt(variance(results))/abs(maximum[dataset]),3), round(mean(durations),3)]] 

    with open("%s.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  