# PNAS_SPECIAL_ISSUE_CRAB

File for the CRAB model ( Working paper available at this link: https://www.lem.sssup.it/WPLem/2021-44.html)

Note_1: the model has been made to work with two regions, but now has been adjusted to work with only one. However, some procedures that were meant for the two-region framework have not been deleted yet (they just remain empy or are commented out).

### Run the model - single processor

```single_run.py```: runs the model on a sigle processor. The parameters at the top "runs" and "steps" determine the amount of runs and how many steps per run. The parameters to be choosen are the following:

  1) F1, F2, F3, H: represent the initial agents population. F1-3 are for the number of firms for each sector (capital, goods, services), H is the           amount of households.
     
  2) B and T: represent the value of transport cost and export demand. However, since they are relevant only for the two-region framework (see               Note_1), they will be removed/are not useful.
     
  3) S, s_pr: are for the average shock size (S) and probability (s_pr). Average because  each agent samples an individual shock from a beta                   distribution whose average values is S. More information in the link above, Section 2.5.
  
  5) seed: change the seed of the pseudo random number generator, to regulate stochasticity and reproducibility.

Note_2: there are many more parameters used in the model runs. They are just not set during the initialization, but kept fixed at the moment. Most of them are attributes of the model class, file --> ```model.py```.

Aggregate outputs are saved into ```macro_variables.csv``` while output at individual agent level are stored as ```micro_variables.csv```. Output from ```single_run.py``` can be analysed with ```output_analysis.ipynb```.

### Run the model - parallel runs and sentivitiy analysis (SA)
```SA.ipynb```: contains the code for One Facto At the Time (OFAT) SA and Sobol SA.

Note_3 if everything is kept fixed and only ```seed``` changes, the OFAT SA can serve as a Monte Carlo exercise to analyze the impact of stochasticity on model runs.

Note_4: to be tested in the SA, the parameters have first to be added at the top of the model class (in ```model.py``` - with the ones in the bullet before). At the moment we have done SA only with those (F1, F2, F3, H, B, T, S, s_pr, seed)

### In a nutshell the other file:
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
   * ```migration.py``` : dynamics related to migration but they are used only in a multi-region framework (off here).
   * ```climate_dynamics.py```: function to model the flood as well as the individual damage to each agent.
   * ```data_collection.py```: contains all the function used to track variables and stored them as output. The variables stored can be selected/added in the  ```model_reporters ``` and ```agents_reporter``` in the file ```model.py```.
   ##### 3) Visualization files, used to dislay the agents in a grid (not used at the moment)
   * ```cell.py```: creates the cell/terrain where agents are located.
   * ```visualization.py```: manage the visualization process.


Note_5: Some parts of the code have still to be re-coded in a more function and clean manner. In case any of you have suggestions do not hesitate to let me know.




 

