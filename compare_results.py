'''
This script generates purity-efficiency plots from existing runs, combining
them into a single plot. The usage is as follows

python compare_results.py output input1 input2 input3 ...
'''

import sys
import glob
import csv
import os
import matplotlib.pyplot as plt

def main():
    models = sys.argv[2:]
    data = {}

    for model in models:
        base_dir = os.path.join('results', model)
        fig_dir = os.path.join(base_dir, 'figures')
        fig_pattern = os.path.join(fig_dir, '*.csv')
        figures = glob.glob(fig_pattern)
        figures.sort()
        last_figure = figures[-1]
        print(last_figure)
        
        purity_list = []
        efficiency_list = []

        with open(last_figure) as csvfile:
            reader = csv.reader(csvfile)
            titles = reader.__next__()
            for (cutoff, purity, efficiency) in reader:
                purity_list.append(float(purity))
                efficiency_list.append(float(efficiency))
        
        data[model] = {}
        data[model]['purity'] = purity_list
        data[model]['efficiency'] = efficiency_list

    for model in models:
        plt.plot(data[model]['purity'], data[model]['efficiency'], label=model)

    plt.axis([0, 1, 0, 1])
    plt.xlabel('Purity')
    plt.ylabel('Efficiency')   
    plt.legend()
    plt.savefig(os.path.join('results', sys.argv[1] + '.png'))
    plt.close()

if __name__ == "__main__":
    main()