# PNAS_SPECIAL_ISSUE_CRAB

This repository stores the code and the analysis scripts of the evolutionary economic agent-based model named CRAB (Coastal Regional Agent-Based model). CRAB was developed to study how damages and adaptation benefits across different actors ( diverse firms and hosueholds) evolve over time in a regional economy exposed to climate-induced hazards, as these agents decide where to reside and whether or not to invest in private adaptation. This research is possible thanks to the funding from the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Program, project 'SCALAR: Scaling up behavior and autonomous adaptation for macro models of climate change damage assessment'; grant agreement number 758014.

![image](https://github.com/SC3-TUD/PNAS-Uncertainty-in-Boundedly-Rational-Climate-Adaptation/assets/83168418/9f49f519-0c9c-40c7-afb3-6a8a662fec87)


## Author contributions

*Model design*:  Alessandro Taberna, Prof. dr. Tatiana Filatova,

*Code implementation*: Alessandro Taberna, Liz Verbeek

*Sensitivity analysis*: Dr. Antonia Hadjimichael

*Model design CRAB v1.1*: Alessandro Taberna, prof. dr. Tatiana Filatova, prof. dr. Andrea Roventini, Dr. Francesco Lamperti, 

Here we provide the code files for the CRAB model v1.2  ( Original paper describing v1.1 of the model is available at this link: [https://www.sciencedirect.com/science/article/pii/S0921800922002506]





### Run the model - single processor

```single_run.py```: runs the model on a sigle processor. The parameters at the top "runs" and "steps" determine the amount of runs and how many steps per run. The parameters to be choosen are the following:

  1) F1, F2, F3, H: represent the initial agents population. F1-3 are for the number of firms for each sector (capital, goods, services), H is the number of households.
     
  2) flood_schedule: represents the time step and depth (m) of the flood.
     
  3) fraction_exposed is the fraction of the population residing in flood-prone areas.

  4) cca_eff: sets the objective efficacy of climate change adaptation (CCA) protective measures.
  5) attributes: either 'Hom' or 'Het' decide whether to initialize a homogenous or heterogenous population of household agents.
  6) cca_model: 'EU' or 'PMT', sets the behavioral heuristic for the household CCA actions.
  7) seed: change the seed of the pseudo random number generator, to regulate stochasticity and reproducibility.

Aggregate outputs are saved into ```macro_variables.csv``` while output at individual agent level are stored as ```micro_variables.csv```. Output from ```single_run.py``` can be analysed with ```output_analysis.ipynb```.



### In a nutshell the files are structured as follows:
In general the remaining files can be classified into three categories:
   ###### 1) Agents class files: the files which contain the agent classes and what do they do during a step. 

   * ```model.py```: it is the CRAB model class. Agents are spawned from here and most of the parameters are also assigned here as well.
   * ```capital_good_firm.py , consumption_good_firm.py , service_good_firm.py , households.py```: contain all the actions undertaken by firms of different sectors and         households during a single step.
   * ```government.py```: actions performed by the government agent. There is one government and sometimes it is used to track individual agents. variables
   ###### 2) Files that contains functions used by agents in a specific context:
   * ```goods_market.py```: contains all the functions that the agents use to buy and sell goods.
   * ```labor_dynamics.py```: contains all the functions that the agents use to regulate the jobs market such as hiring/firing process.
   * ```accounting.py```: contains all the functions that the firms agents use to track their resources ( costs, revenues and profits...).
   * ```vintage.py```: contains all the functions used in the capital market. 
   * ```research_and_development.py```: functions used by the capital-good firms for technological learning and innovation.
   * ```climate_dynamics.py```: function to model the flood as well as the individual damage to each agent.
   * ```data_collection.py```: contains all the function used to track variables and stored them as output. The variables stored can be selected/added in the  ```model_reporters ``` and ```agents_reporter``` in the file ```model.py```.

### To perform the sensitivity analysis:
You need access to parallel computing capabilities to perform Steps 1 and 2. If you don't have access to parallel computing, skip to Step 3, which uses summary outputs from the prior steps. 
1) Use ```SA_runs_submission.sh``` to perform the parallel runs using ```single_run_for_SA.py``` and ```param_values.npy``` (the Sobol sample). You need to perform this for each behavioral combination and point to the respective directories to store outputs (e.g., ```data_dir = './EU_het_2_outputs/```)
2) Use ```time_varying_analysis.sh``` to extract time-varying means across the 100 seeds to be used in the sensitivity analysis. This script can run python files ```time_varying_means_adaptation_fraction.py```, ```time_varying_means_damages.py```, and ```time_varying_means_damages_hom.py``` and will produce the summary outputs listed under ```SA_summary_output_figures```
3) To calculate sensitivity indices and re-produce the figures in the paper, use ```time_varying_SA_adaptation_fraction.py``` for all relevant outputs under the ```SA_summary_output_figures``` directory
4) Notebook ```time_varying_indices.ipynb``` uses the calculated indices to produce the figures for the paper
 

