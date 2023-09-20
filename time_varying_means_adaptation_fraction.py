import os
import glob
import numpy as np
import duckdb
import argparse

def time_varying_means(data_dir, output_name):
    # target glob path
    macro_glob_path = os.path.join(data_dir, 'macro_variables_*_*.parquet')

    # number of files in file query
    n_files = len(glob.glob(macro_glob_path))

    # number of records in file query
    n_runs = n_files * 10  # seeds per file

    print("Exploring {:,} model runs over {} files.".format(n_runs, n_files))

    # load parameter sample
    parameter_samples = np.load('param_values.npy')

    # create output array to analyze
    # loop through all samples and calculate value across all runs and timesteps
    mean_output = np.zeros((len(parameter_samples), 400))
    
    exposed_agents = np.load(f'{data_dir}_exposed_agents.npy')

    for i in range(len(parameter_samples)):
        sql = f"""
        SELECT
            AVG({output_name}) as avg_adapted
            ,AVG("Households Coastal region") as avg_households
        FROM
            '{macro_glob_path}'
        WHERE
            Sample = {i} AND __index_level_0__ !=0
        GROUP BY
            __index_level_0__;
        """
        # get query result as a data frame
        df = duckdb.query(sql).df()

        mean_output[i, :] = np.around(df['avg_adapted']/(df['avg_households']*exposed_agents[i,:]), decimals = 2)

    # save mean output
    np.save(f'{data_dir}_{output_name}_means.npy', mean_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model using parameter sample')
    parser.add_argument('data_dir', type=str,
                        help='experiment directory')
    parser.add_argument('output_name', type=str,
                        help='output to analyze')
    args = parser.parse_args()
    time_varying_means(args.data_dir, args.output_name)
