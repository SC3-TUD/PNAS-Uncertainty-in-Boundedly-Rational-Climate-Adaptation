# PNAS_SPECIAL_ISSUE_CRAB

File for the CRAB model ( Original paper available at this link: [https://www.sciencedirect.com/science/article/pii/S0921800922002506]

### Run the model - single processor

```single_run.py```: runs the model on a sigle processor. The parameters at the top "runs" and "steps" determine the amount of runs and how many steps per run. The parameters to be choosen are the following:

  1) F1, F2, F3, H: represent the initial agents population. F1-3 are for the number of firms for each sector (capital, goods, services), H is the amount of households.
     
  2) flood_schedule: represents the time step and depth (m) of the flood.
     
  3) fraction_exposed is the fraction of the population living in flood-prone areas.

  4) cca_eff: sets the objective efficacy of climate change adaptation (CCA) protective measures.
  5) attributes: either 'Hom' or 'Het' decides whether to initialize a homogenous or heterogenous population.
  6) cca_model: 'EU' or 'PMT', sets the behavioral heuristic for   CCA actions.
  7) seed: change the seed of the pseudo random number generator, to regulate stochasticity and reproducibility.

Aggregate outputs are saved into ```macro_variables.csv``` while output at individual agent level are stored as ```micro_variables.csv```. Output from ```single_run.py``` can be analysed with ```output_analysis.ipynb```.



### In a nutshell the files are structure as follow:
In general the remaining files can be classified into three categories:
   ###### 1) Agents class files: the files which contain the agent classes and what do they do duruing a step. 

   * ```model.py```: it is the CRAB model class. Agents are spwaned from here and most of the parameters are also assigned here as well.
   * ```capital_good_firm.py , consumption_good_firm.py , service_good_firm.py , households.py```: contain all the actions undertaken by firms of different sectors and         households during a single step.
   * ```government.py```: actions performed by the government agent. There is one government and sometimes it is used to track individual agents. variables
   ###### 2) Files that contains functions used by agents in a specific context:
   * ```goods_market.py```: contains all the functions that the agents use to buy and sell goods.
   * ```labor_dynamics.py```: contains all the functions that the agents use to regulate the jobs market such as hiring/firing process.
   * ```accounting.py```: contains all the functions that the firms agents use to track their resources ( costs, revenus and profits...).
   * ```vintage.py```: contains all the functions used in the capital market. 
   * ```research_and_development.py```: fucntions used by the capital-good firms for technological learning and innovation.
   * ```climate_dynamics.py```: function to model the flood as well as the individual damage to each agent.
   * ```data_collection.py```: contains all the function used to track variables and stored them as output. The variables stored can be selected/added in the  ```model_reporters ``` and ```agents_reporter``` in the file ```model.py```.



 

