# PNAS_SPECIAL_ISSUE_CRAB

This repository stores the code and the analysis scripts of the evolutionary economic agent-based model named CRAB (Coastal Regional Agent-Based model). CRAB was developed to study how damages and adaptation benefits across different actors ( diverse firms and hosueholds) evolve over time in a regional economy exposed to climate-induced hazards, as these agents decide where to reside and whether or not to invest in private adaptation. This research is possible thanks to the funding from the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Program, project 'SCALAR: Scaling up behavior and autonomous adaptation for macro models of climate change damage assessment'; grant agreement number 758014.

![image](https://github.com/SC3-TUD/PNAS-Uncertainty-in-Boundedly-Rational-Climate-Adaptation/assets/83168418/9f49f519-0c9c-40c7-afb3-6a8a662fec87)


## **Author contributions:**

*Model design*: Prof. dr. Tatiana Filatova, Alessandro Taberna
*Code implementation*: Liz Verbeek, Alessandro Taberna
*Sensitivity analysis*: Prof. dr. Antonia Hadjimichael

*Model design CRAB v1.1*: Prof. dr. Tatiana Filatova, prof. dr. Andrea Roventini, prof. dr. Francesco Lamperti, Alessandro Taberna 

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



### In a nutshell the files are structured as follow:
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



 

