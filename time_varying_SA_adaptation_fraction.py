import numpy as np
import pandas as pd
from SALib.analyze import sobol
import argparse

def time_varying_SA(data_dir, output_name):
    mean_output = np.load(f'{data_dir}_{output_name}_means.npy')

    # define sobol problem dictionary
    problem = {
        'num_vars': 4,
        'names': ['exposed', 'elev_eff', 'wet_eff', 'dry_eff'],
        'bounds': [[0.1, 1], [1, 5], [0.15, 0.55], [0.1, 0.85]]
    }

    # create dataframe to store outputs
    ST_indices = pd.DataFrame(np.zeros((problem['num_vars'], 400)), columns = [np.arange(1, 401)])
    for j in range(400):
        result = sobol.analyze(problem, mean_output[:, j], calc_second_order=False)
        ST_indices[j+1] = result['ST']
    ST_indices.to_csv(data_dir[:-7] + output_name + '.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model using parameter sample')
    parser.add_argument('data_dir', type=str,
                        help='experiment directory')
    parser.add_argument('output_name', type=str,
                        help='output to analyze')
    args = parser.parse_args()
    time_varying_SA(args.data_dir, args.output_name)
