import os
import glob
import numpy as np
import duckdb
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt

# directory with all SA outputs
data_dir = './PMT_het_2_outputs'

# directory to store figures
figs_dir = './PMT_het_2_figures/'

# target glob path
macro_glob_path = os.path.join(data_dir, 'macro_variables_*_*.parquet')

# number of files in file query
n_files = len(glob.glob(macro_glob_path))

# number of records in file query
n_runs = n_files * 10  # seeds per file

print("Exploring {:,} model runs over {} files.".format(n_runs, n_files))

sql = f"""
SELECT
    *
FROM
    '{macro_glob_path}'
WHERE
    __index_level_0__ = '400';
"""

# get query result as a data frame
df = duckdb.query(sql).df()

# define sobol problem dictionary
problem = {
    'num_vars': 4,
    'names': ['exposed', 'elev_eff', 'wet_eff', 'dry_eff'],
    'bounds': [[0.1, 1], [1, 5], [0.15, 0.55], [0.1, 0.85]]
}

# load parameter sample
parameter_samples = np.load('param_values.npy')

# outputs of interest
all_outputs = list(df.columns.values)[2:-1]

for output_name in all_outputs:
    # create output array to analyze
    # loop through all samples and calculate value across all runs
    mean_output = np.zeros(len(parameter_samples))
    var_output = np.zeros(len(parameter_samples))

    for i in range(len(parameter_samples)):
        output_values = df[df["Sample"] == i][output_name]
        mean_output[i] = np.mean(output_values)
        var_output[i] = np.var(output_values)

    mean_Si = sobol.analyze(problem, mean_output, calc_second_order=False)
    var_Si = sobol.analyze(problem, var_output, calc_second_order=False)

    labels = problem['names']

    S1_values = mean_Si['S1']
    ST_values = mean_Si['ST']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, S1_values, width, label='First order')
    rects2 = ax.bar(x + width / 2, ST_values, width, label='Total order')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Indices')
    ax.set_title('Indices for ' + output_name + ' mean across seeds')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()

    plt.savefig(figs_dir + output_name + '_mean.png', dpi=300)

    S1_values = var_Si['S1']
    ST_values = var_Si['ST']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, S1_values, width, label='First order')
    rects2 = ax.bar(x + width / 2, ST_values, width, label='Total order')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Indices')
    ax.set_title('Indices for ' + output_name + ' variance across seeds')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()

    plt.savefig(figs_dir + output_name + '_variance.png', dpi=300)
    plt.close()