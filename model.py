# -*- coding: utf-8 -*-

"""

@author: TabernaA

The model class for the Climate-economy Regional Agent-Based (CRAB) model).
This class is based on the MESA Model class.
TODO: add reference to paper for model description?

"""
import time

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import climate_dynamics as cd

from sympy import I
from scipy.stats import beta
from mesa import Model
# from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
# from mesa.batchrunner import BatchRunner

from schedule import StagedActivationByType
from capital_good_firm import CapitalGoodFirm
from consumption_good_firm import ConsumptionGoodFirm
from household import Household
from government import Government
from service_good_firm import ServiceFirm
from vintage import Vintage
from data_collection import *
# from cell import Cell

seed_value = 12345678
random.seed(seed_value)
np.random.seed(seed=seed_value)



class CRAB_Model(Model):
    """Model class for the CRAB model. """

    def __init__(self, F1=50, F2=100, F3=100, H=3000, Exp=300, T=0.03,
                 flood_schedule={100: 3, 140: 3}, fraction_exposed=0.4,
                 cca_eff={"Elevation": 3, "Wet_proof": 0.4, "Dry_proof": 0.5},
                 flood_prob= 0, av_network=7, social_int=True, n_regions=1,
                 cca_model="EU", attributes ='Het', collect_each = 1, seed=seed_value,
                 # width=100, height=100
                 ):
        """Initialization of the CRAB model.

        Args:
            F1                  : Number of capital firm agents
            F2                  : Number of consumption firm agents
            F3                  : Number of service firm agents
            H                   : Number of household agents
            Exp                 : Export to rest of world (RoW)
            T                   : Transport cost
            flood_schedule      : Time and depth of the flood. Set time negative
                                  to get no floods
            fraction_exposed    : Fraction of population in flood-prone areas
            cca_eff             : Parameters of protection measures
            flood_prob          : Multiplies the perceived flood probability
                                  set 0 and cca_model = "EU" to get no adaptation
            av_network          : Average number of household neighbors
            social_int          : Boolean; enable or disable social network
            cca_model           : Household adaptation mdoel ("PMT" or "EU")
            seed                : Seed for random number generator
        """
        # Number of agents for initialization
        self.init_n_firms_cap = F1
        self.init_n_firms_cons = F2
        self.init_n_firms_serv = F3
        self.init_n_hh = H
       # print(seed, cca_model, attributes)

        # Keep track of firms to remove and add during runtime
        self.firms_to_remove = []
        self.number_out_firms = {"Capital": 0,  # Store for data collection
                                 "Cons": 0,
                                 "Service": 0}
        self.firm_subsidiaries = []

        # --------------------------------------------------------------------
        # -- Regulate stochasticity -- #
        self.seed = seed
        random.seed(seed)
        np.random.seed(int(seed))
        self.reset_randomizer(seed)


        # Initialize timestep and agent ID count
        self.time = 0
        self.current_id = 0

       # print(seed, cca_model, attributes, social_int)

        # Initialize scheduler with stages of agent activation
        # See agent class files for overview of steps for each agent type
        stage_list = ["stage0", "stage1", "stage2",
                      "stage3", "stage4", "stage5", "stage6"]
        self.schedule = StagedActivationByType(self, stage_list)

        # -- Initial conditions of agents -- #
        self.capital_output_ratio = 0.7
        self.capital_service_ratio = 1.3

        # -- Flood characteristics -- #
        self.n_floods = 0
        self.flood_schedule = flood_schedule
        self.fraction_exposed = fraction_exposed
        self.change_income_pp_list = []
        self.cum_flood_pr = flood_prob

        # -- CCA parameters -- #
        self.pmt_params = pd.read_csv("PMT_params.csv", index_col=0)
        self.cca_eff = cca_eff
        self.cca_cost =  {"Elevation_cost": 2,
                          "Wet_proof_cost": 0.4,
                          "Dry_proof_cost": 0.3}
        self.cca_model = cca_model
        self.attributes = attributes
        self.collect_each = collect_each

        # -- Transport cost -- #
        self.transport_cost = T  # /10          # regional transport cost
        self.transport_cost_RoW = 2 * T  # /10  # international transport cost
        self.running = True
          
        # --------------------------------------------------------------------
        #                           INITIALIZATION
        # --------------------------------------------------------------------
        # -- Initialize firms -- #
        init_n_firms = {CapitalGoodFirm: F1,
                        ConsumptionGoodFirm: F2,
                        ServiceFirm: F3}
        self.initialize_firms(init_n_firms=init_n_firms)

        # -- Social network -- #
        self.n_hh_connection = av_network
        self.social_int = social_int
        if cca_model == 'EU':
            self.social_int = False
        self.G = nx.watts_strogatz_graph(n=self.init_n_hh,
                                         k=self.n_hh_connection, p= 0, seed = seed)
        # Relabel nodes for consistency with agent IDs (zero- to one-indexing)
        self.G = nx.relabel_nodes(self.G, lambda x: x +
                                  self.init_n_firms_cap +
                                  self.init_n_firms_cons +
                                  self.init_n_firms_serv + 1)

        # -- Initialize households-- #
        HH_attributes = pd.read_csv("test")
        if self.attributes == 'Hom':
            for i in range(self.init_n_hh):
                attributes = HH_attributes.mean()
               # print(attributes) 
                self.add_household(attributes)
        elif self.attributes == 'Het':
            for i in range(self.init_n_hh):
                attr_idx = random.randint(0, len(HH_attributes)-1)
                attributes = HH_attributes.iloc[attr_idx]
                self.add_household(attributes)
        else:
            print('something wrong with HH value distribution')

        # Initialize government(s)
        self.governments = {}
        for region in range(n_regions):
            gov = Government(self, region, Exp)
            self.schedule.add(gov)
            self.governments[region] = gov

        # --------------------------------------------------------------------
        # Data collection (variables that are stored during the simulation)
        # -- NOTE all these functions are described data_collection.py -- #
        #          Select/add the variables that you want to collect
        # --------------------------------------------------------------------
        self.datacollector = DataCollector(

            # Collect data on model level (per step, for all agents)
            model_reporters={

                # -- MACROECONOMICS -- #

                # NOTE: ***OUTPUTS FOR ANTONIA *** #
                # -- Economic growth -- #
                "Unemployment rate coastal": regional_unemployment_rate_coastal,
                "GDP total": gdp_total,

                # -- Climate change adaptation (counting) -- #
                "Total_UG_CCA_coastal_elev": total_CCA_coastal_elev,
                "Total_UG_CCA_coastal_dry_proof": total_CCA_coastal_dry_proof,
                "Total_UG_CCA_coastal_wet_proof": total_CCA_coastal_wet_proof          },

            # Collect data on agent level (per step, per agent)
            agent_reporters={
                "Type": "type",
                "Net worth": "net_worth",  
                "Damage_coeff": "damage_coeff",
                "Monetary_damage": "monetary_damage",
                "House_value" : 'av_wage_house_value',  
                "At_risk": "at_risk", 
                "Wage": "wage", 
                "Education": "edu"
                }
        )
        self.running = True
        self.datacollector.collect(self)
        

    def step(self):
        """Defines a single model step.

           Model flow overview:
            1) Migration     : Households entering or leaving the region,
                               based on change in regional average income.
            2) Flood         : Check if flood occurs.
            3) Model step    : Staged activation of dynamics of all agents,
                               see scheduler and agent classes.
            4) Firm removal  : Remove bankrupt firms from model.
            5) Firm entry    : Create subsidfiaries of firms with consistent
                               high profits.
        """

        random.seed(self.seed)
        np.random.seed(int(self.seed))
        self.reset_randomizer(self.seed)

        # -- MIGRATION -- #
        gov = self.governments[0]
        self.change_income_pp_list.append(gov.income_pp_change)
        if self.time > 10:
            # Compute population change from average income change
            self.change_income_pp_list = self.change_income_pp_list[-4:]
            avg_income_change = (sum(self.change_income_pp_list) /
                                 len(self.change_income_pp_list))
            pop_change = int(0.5 * avg_income_change *
                             len(self.schedule.agents_by_type["Household"]))


            # If population change is positive: add households
            if pop_change > 0 and gov.unemployment_rates[0] < 0.15:
                self.migration_in(pop_change)

            # If population change is negative, remove households
            elif pop_change < 0 and gov.unemployment_rates[0] > 0.05:
                self.migration_out(pop_change)

        # -- FLOOD -- #
        # Check if flood occurs at this timestep
        self.flood_depth = cd.flood_handler(self,
                                            self.flood_schedule,
                                            self.time)
        if self.flood_depth > 0:
            self.is_flood_now = True
            self.n_floods += 1
            # self.recovery = 4
        else:
            self.is_flood_now = False

        # -- MODEL STEP -- #
        # (In stages, see agent class files for more details)
        self.schedule.step()

        # -- FIRM REMOVAL -- #
        # Store for data collection
        firm_types = list(map(type, self.firms_to_remove))
        self.number_out_firms = {"Capital": firm_types.count(CapitalGoodFirm),
                                 "Cons": firm_types.count(ConsumptionGoodFirm),
                                 "Service": firm_types.count(ServiceFirm)}
        # Remove bankrupt firms
        self.remove_firms(self.firms_to_remove)

        # -- FIRM ENTRY -- #
        self.governments[0].new_firms_resources = 0
        if self.firm_subsidiaries:
            self.add_subsidiaries(self.firm_subsidiaries)
            self.firm_subsidiaries = []

        # Collect data and go to next time step
        self.time += 1
        if self.time % self.collect_each == 0 or (self.time > list(self.flood_schedule.items())[0][0] and self.time < list(self.flood_schedule.items())[-1][0] + 50) :
           #print(self.time)
           self.datacollector.collect(self)
        
    def add_firm(self, firm_class, **kwargs):
        """Create new firm and add it to the model scheduler. """
        firm = firm_class(self, **kwargs)
        self.schedule.add(firm)
        return firm

    def initialize_firms(self, init_n_firms, region=0,
                         lower_bound: float=0.9,
                         upper_bound: float=1.1) -> None:
        """Initialize firms in coastal region

        Args:
            init_n_firms        : Dict containing number of firms
                                  to create per firm class
            lower_bound         : Lower bound of productivity and wage dist.
            upper_bound         : Upper bound of productivity and wage dist.
            region              : Firms region
        """
        for firm_class, n_firms in init_n_firms.items():
            prods = np.random.uniform(lower_bound, upper_bound, n_firms)
            wages = np.random.uniform(lower_bound, upper_bound, n_firms)
            prices = wages/prods
            for i in range(n_firms):
                prod = np.repeat(prods[i], 2)
                market_share = 1 / init_n_firms[firm_class]
                self.add_firm(firm_class, prod=prod, wage=wages[i],
                              price=prices[i], market_share=market_share,
                              region=region)

    def add_subsidiaries(self, firms):
        """Create subsidiaries for firms with consistent high profits.

        Args:
            firms       : Firms to create subsidiaries for
        """
        gov = self.governments[0]
        model_vars = self.datacollector.model_vars
        wealth = gov.av_net_worth[0]
        # Recompute best capital firm since firms have been removed recently
        brochure = gov.get_best_cap()[0].brochure_regional
        for firm in firms:
            # Initialize net worth as fraction of average net worth
            fraction_wealth = (0.9 - 0.1) * np.random.random_sample() + 0.1
            net_worth = max(wealth, 1) * fraction_wealth

            firm_class = type(firm)
            if firm_class == CapitalGoodFirm:
                # Initialize productivity as fraction of regional top productivity
                # COMMENT: make bounds, a and b model parameters? 
                x_low, x_up, a, b = (-0.05, 0.05, 2, 4)
                fraction_prod = 1 + x_low + beta.rvs(a, b, size=2) * (x_up - x_low)
                prod = np.around(gov.top_prod *
                                 fraction_prod, 3)
                # Initialize market share as fraction of total at beginning
                market_share = 1 / self.init_n_firms_cap

                # Create new firm
                self.add_firm(firm_class, prod=prod, wage=firm.wage,
                              price=firm.price, market_share=market_share,
                              net_worth=net_worth, lifecycle=0)

            elif firm_class == ConsumptionGoodFirm or firm_class == ServiceFirm:
                # Initialize productivity as productivity of best supplier
                prod = [brochure[0], brochure[0]]
                # Initialize competitiveness as average competitiveness
                comp = gov.avg_norm_comp[firm.type]
                # Initialize capital
                capital = {"Cons": gov.regional_capital_cons[2] *
                                   self.capital_output_ratio,
                           "Service": gov.regional_capital_serv[2] *
                                      self.capital_service_ratio}
                capital_amount = round(capital[firm.type])

                # Create new firm
                self.add_firm(firm_class, prod=prod, wage=firm.wage,
                              price=firm.price, market_share=0, sales=0,
                              net_worth=net_worth, init_n_machines=1,
                              init_capital_amount=capital_amount,
                              brochure=brochure, competitiveness=comp,
                              lifecycle=0)

            else:
                print("Firm type not recognized.")
            gov.new_firms_resources += gov.av_net_worth[0]

    def remove_firms(self, firms_to_remove, bailout=0) -> None:
        """Remove list of firms from the model.

        Args:
            firms_to_remove             : List of firms to remove
            bailout                     : Initial bailout cost
        """
        for firm_out in firms_to_remove:
            # Update bailout cost
            bailout += firm_out.net_worth
            # Remove firm from model and schedule
            self.schedule.remove(firm_out)

        # Reset removal lists
        self.firms_to_remove = []
        # Store new bailout cost in government
        self.governments[0].bailout_cost = bailout

    def add_household(self, attributes):
        """Create new household and add it to the model scheduler.

        Args:
            attributes      : Household attributes
                              (TODO: finish description, see also HH class)
        """
        household = Household(self, attributes)
        self.schedule.add(household)
        return household

    def migration_in(self, pop_change):
        """Create new (randomly sampled) new households, add those to the
           model, schedule and social network.

        Args:
            pop_change      : Number of households entering the region
        """
        
        HH_attributes = pd.read_csv("test")
        random.seed(self.seed)
        np.random.seed(seed=seed_value)
        #print("Hence I am adding" ,pop_change)
        for i in range(pop_change):
           
            # Get random household attributes from attributes file
            if self.attributes == 'Hom':
                attributes = HH_attributes.mean() #iloc[attr_idx]
            elif self.attributes == 'Het':
                attr_idx = random.randint(0, len(HH_attributes) - 1)
                attributes = HH_attributes.iloc[attr_idx]
            else:
                print('something wrong with new HH attributes distribution')
            # Add new household to model
            new_household = self.add_household(attributes)
            # Add household to social network and connect to neighbors
            self.G.add_node(new_household.unique_id)
            n_new = round(np.random.normal(self.n_hh_connection, 0))
            #print(n_new)
            neighbors =  random.sample(self.schedule.agents_by_type["Household"].keys(), n_new)
            for node in neighbors:
                self.G.add_edge(new_household.unique_id, node)

    def migration_out(self, pop_change):
        """Remove (randomly sampled) household from the model, scheduler
           and social network.

        Args:
            pop_change      : Number of households leaving the region
        """
        # Random sample of households to be removed
        random.seed(self.seed)
        sample_out = random.sample(self.G.nodes, abs(pop_change))
        #print("Hence I am removing" ,len(sample_out))
        for hh_out_id in sample_out:
          #  print(hh_out_id)
            hh_out = self.schedule.agents_by_type["Household"][hh_out_id]
            # Remove from employer
            if hh_out.employer_ID != None:
                employer = self.schedule._agents[hh_out.employer_ID]
                employer.employees_IDs.remove(hh_out_id)
            # Remove from schedule
            self.schedule.remove(hh_out)
        # Remove household and its connections from social network
        self.G.remove_nodes_from(sample_out)

    def run_model(self, step_count=300):
        """Run model for {step_count} number of steps."""
        for i in range(step_count):
            self.step()


if __name__ == "__main__":
    model = CRAB_Model()
