import os
import glob
import numpy as np
import duckdb

def time_varying_means():
    data_dir = './EU_het_2_outputs'

    # target glob path
    micro_glob_path = os.path.join(data_dir, 'micro_variables_*_*.parquet')

    # number of files in file query
    n_files = len(glob.glob(micro_glob_path))

    # number of records in file query
    n_runs = n_files * 10  # seeds per file

    print("Exploring {:,} model runs over {} files.".format(n_runs, n_files))

    # load parameter sample
    parameter_samples = np.load('param_values.npy')

    # create output array to analyze
    # loop through all samples and calculate value across all runs and timesteps x 4 for each education level
    mean_output = np.zeros((len(parameter_samples), 400, 4))

    exposed_ratio = np.zeros((len(parameter_samples), 400))


    for i in range(len(parameter_samples)):
        print(i)
        sql = f"""
        SELECT
            AVG(Damage_coeff) *  AVG(House_value) / AVG(Wage) as potential_damage
            ,At_risk
            ,Education
            ,Step
            ,COUNT(*)
        FROM
            '{micro_glob_path}'
        WHERE
            Sample = {i} AND Education != 'NaN' AND Step !=0
        GROUP BY
            Step
            ,Education
            ,At_risk;
        """
        # get query result as a data frame
        df = duckdb.query(sql).df()

        df.sort_values(by=['Education', 'Step'], inplace=True)
        for j in range(4):
            mean_output[i, :, j] = df['potential_damage'][df['Education'] == float(j + 1)][df['At_risk'] == True]

        grouped_df = df.groupby(by=['Step', 'At_risk']).sum()
        total_agents = grouped_df.groupby(level="Step").sum()['count_star()'].values
        at_risk_agents = grouped_df[grouped_df.index.isin([True], level=1)]['count_star()'].values
        exposed_ratio[i, :] = np.around(at_risk_agents/total_agents, decimals=2)

    # save mean output
    np.save(f'{data_dir}_potential_damages_means.npy', mean_output)
    np.save(f'{data_dir}_exposed_agents.npy', exposed_ratio)

if __name__ == '__main__':
    time_varying_means()
