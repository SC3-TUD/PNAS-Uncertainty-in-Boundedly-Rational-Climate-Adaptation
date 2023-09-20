# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: TabernaA

Script to run the model for one single run and save macro (model level) and
micro (agent level) outputs.

Copied and adjusted from single_run.py on April 8th 2022

"""

import argparse
import random
import numpy as np
import pandas as pd
from model import CRAB_Model

# Create array of seeds (all the same across all experiments)
#seeds = np.random.randint(0, high=1000000, size=100, dtype=int)
seeds = [685524, 790702, 288179, 255899, 411122, 588081, 699985, 114841,
         497201, 551814, 965721, 988709, 108969, 495174,  98283, 660761,
         944018, 584815, 940650, 459327,  13646,  93328, 965907, 594852,
         356805, 860691, 689834, 459732, 346911, 475661, 750606, 874421,
         752694, 431790, 528463, 609479, 279575, 832623, 317682, 194918,
         764233, 279684, 951910, 574739,  20711, 355481, 677800,  55980,
         585721, 301542, 637973, 413551, 916677, 156075,  59165, 764779,
         173584, 357937, 702364, 882533, 196871, 499780, 833323, 905677,
         356731, 224889, 877610, 214526, 399256, 773885, 517309, 586683,
         756865, 282232, 147552, 532326, 726461, 198064, 805099, 901006,
         303586, 246239, 507652, 149401, 141479, 781180, 973830, 835451,
         459833,  42742, 795882, 745778, 971487, 785601, 623448, 641975,
         778793, 465775, 627206, 344301]
steps = 400

parameter_samples = np.load('param_values.npy')

# directory to store SA outputs
data_dir = './EU_het_2_outputs/'

def sample_model_runs(sample, seed_chunk):
    # Read sampled parameter values for this sample
    exposed = parameter_samples[sample][0]
    elev_eff = parameter_samples[sample][1]
    wet_eff = parameter_samples[sample][2]
    dry_eff = parameter_samples[sample][3]
    micro_variables = []
    macro_variables = []

    for i, seed in enumerate(seeds[seed_chunk*10:seed_chunk*10 + 10]):
        # print(("Model initialized with " + str(H) + " Households"))
        print("Run num", seed_chunk*10+i, "with random seed", seed)
        # seed = random.randint(i, seed_value)
        # seed = 0
        model = CRAB_Model(F1=50, F2=100, F3=100, H=3000, Exp=300, T=0.03,
                           flood_schedule={-100: 3, -140: 3},
                           fraction_exposed=exposed,
                           cca_eff={"Elevation": elev_eff,
                                    "Wet_proof": wet_eff,
                                    "Dry_proof": dry_eff},
                           flood_prob=1, av_network=7,
                           social_int=True, collect_each=1,
                           cca_model="EU", attributes='Het', seed=seed)
        model.reset_randomizer(seed)

        for j in range(steps):
            print("#------------ step", j+1, "------------#")
            model.step()

        # --------
        # COMMENT:
        # Look into efficient output saving later
        # --------
        macro_variable = model.datacollector.get_model_vars_dataframe()
        macro_variable.at[0, 'Av_income_pp'] = 0
        macro_variable.insert(0, "Sample", sample)
        macro_variable.insert(1, "Run", seed_chunk*10+i)
        # Iteratively add dataframe to list
        macro_variables.append(macro_variable)

        micro_variable = model.datacollector.get_agent_vars_dataframe()
        micro_variable.insert(0, "Sample", sample)
        micro_variable.insert(1, "Run", seed_chunk*10+i)
        micro_variables.append(micro_variable)
    #TODO: need to make step a column
    # ----------------------------------------------------------------------------
    #                              Output manipulation
    # ----------------------------------------------------------------------------
    micro_df = pd.concat(micro_variables)
    micro_df.to_parquet(data_dir + f'micro_variables_{sample}_{seed_chunk*10}.parquet',
                        engine='pyarrow', compression='gzip')
    macro_df = pd.concat(macro_variables)
    macro_df.to_parquet(data_dir + f'macro_variables_{sample}_{seed_chunk*10}.parquet',
                        engine='pyarrow', compression='gzip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model using parameter sample')
    parser.add_argument('sample', type=int,
                        help='sample number')
    parser.add_argument('seed_chunk', type=int,
                        help='seed segment to run')
    args = parser.parse_args()
    sample_model_runs(args.sample, args.seed_chunk)