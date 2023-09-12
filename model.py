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
                #"Real_GDP_cons_coastal": real_gdp_cons_reg_0,
                #"Real_GDP_serv_coastal": real_gdp_serv_reg_0,
                #"Av_income_pp" : income_pp,
                "Unemployment rate coastal": regional_unemployment_rate_coastal,

                # -- Climate change adaptation (counting) -- #
                "Total_UG_CCA_coastal_elev": total_CCA_coastal_elev,
                "Total_UG_CCA_coastal_dry_proof": total_CCA_coastal_dry_proof,
                "Total_UG_CCA_coastal_wet_proof": total_CCA_coastal_wet_proof,

                # -- Population -- #
                #"Population Consumption firms Coastal" : regional_population_cons_coastal,
                #"Population Capital firms Coastal" : regional_population_cap_coastal,
                #"Population Service firms Coastal" : regional_population_serv_coastal,
                #"Households Coastal region" :regional_population_households_region_0,
                #"Coastal average salary": coastal_average_salary,
                #"Price_cons_coastal": price_average_cons_coastal,
                #"Price_serv_coastal": price_average_serv_coastal,
                "GDP total": gdp_total,
                #"INVESTMENT coastal": investment_coastal,
                 #"CONSUMPTION coastal": consumption_coastal,
                #"Debt": debt,

                # -- Debt (Revenues - cost) -- #
                #"Tax_revenues_coastal": tax_revenues_coastal,
                #"Unemployment_cost_coastal" :unemployment_cost_coastal,
                
                ### STOP FOR ANTONIA --###
                #"Number of floods": n_floods,
   
                # -- Productivity -- #
                #"Regional_average_productivity": productivity_firms_average,
                #"Coastal productivity average": av_productivity_coastal,
                #"Coastal productivity growth": gr_productivity_coastal,
                #"Coastal_cons_av_prod": productivity_coastal_consumption_firms,
                #"Coastal_cap_av_prod": productivity_coastal_capital_firms,
                #"Coastal_serv_av_prod": productivity_coastal_service_firms,
               # "Top_prod": top_prod,
                # "Inland productivity average": av_productivity_inland,
                # "Inland productivity growth": gr_productivity_inland,
                # "Inland_cons_av_prod": productivity_inland_consumption_firms,
                # "Inland_cap_av_prod": productivity_inland_capital_firms,
                # "Inland_serv_av_prod": productivity_inland_service_firms,
                # "Investment_units": investment_units,
                # "Capital_firms_av_prod": productivity_capital_firms_average,
                # "Capital_firms_av_prod_region_1": productivity_capital_firms_region_1_average,
                # "Consumption_firms_av_prod": productivity_consumption_firms_average,

                # -- Prices -- #
                # "Price_average": price_average,
                # "Capital_price_average": price_average_cap,
                #Price_cap_coastal": price_average_cap_coastal,
                # "Price_cons_internal": price_average_cons_internal,
                # "Price_cap_internal": price_average_cap_internal,
                # "Price_serv_internal": price_average_serv_internal,
                # "Price total": price_total,

                # -- GDPs -- #
                # "GDP": gdp,
                #"Real_GDP_cap_coastal": real_gdp_cap_reg_0,
                #"INVENTORIES": inventories,
                # "Feas prod": feas_prod,
                # "GDP_cons": gdp_cons,
                # "GDP_cap": gdp_cap,
                # "Real_GDP_cap_internal": real_gdp_cap_reg_1,
                # "Real_GDP_cons_internal": real_gdp_cons_reg_1,
                # "Real_GDP_serv_internal": real_gdp_serv_reg_1,
                # "Demand_exp_ratio": demand_export_rate,   # GDP/export ratio

                # -- INVESTMENTS -- #
               # "INVESTMENT": investment,
                # "INVESTMENT inland": investment_inland,
                #"INVESTMENT_total": investment_total,       # Returns list
                #"Capital_Region_cons": regional_capital_cons,
                #"Capital_Region_serv": regional_capital_serv,
                # "RD_CCA_INVESTMENT": RD_CCA_investment,
                # "orders_cons": quantity_ordered_cons,
                # "orders_serv": quantity_ordered_serv,
                # "orders_received": orders_received_cap,
                # "Investment_units": investment_units,

                # -- CONSUMPTION -- #
                # "CONSUMPTION": consumption,
                # "CONSUMPTION inland": consumption_inland,
                #"CONSUMPTION total": consumption_total,
                #"Aggregate_services": aggregate_serv,
                #"Test_cons_coastal": test_consumption_coastal,
                # "Test_cons_inland": test_consumption_inland,

                # -- CCA -- #
                # "Average_CCA_coeff": RD_coefficient_average,

                # -- PROFITS/DEBT/RESOURCES -- #
                # "Regional_average_profits_cons": regional_average_profits_cons,
                # "Regional_average_profits_cap": regional_average_profits_cap,
                # "Regional_profits_cons": regional_profits_cons,
                # "Capital_price_average": price_average_cap,
                #"Avg net worth Coastal": av_nw_coastal,
                # "Avg net worth migration to Coastal": av_nw_migrant_to_coastal,
                # "Avg net worth migration to Inland": av_nw_migrant_to_inland,
                #"Tax_revenues_coastal": tax_revenues_coastal,
                # "Tax_revenues_inland": tax_revenues_inlnad,
                #"Unemployment_cost_coastal": unemployment_cost_coastal,
                # "Sectoral_debt": sectoral_aggregate_debt,
                # "Sectoral_liquid_assets": sectoral_aggregate_liquid_assets,

                # -- UNEMPLOYEMENT-- #
                # "Unemployment_cost_inland": unemployment_cost_inland,
                # "Regional_average_NW": regional_average_nw,
                # "Avg net worth Inland": av_nw_inland,
                # "Aggregate_Employment": regional_aggregate_employment,
                # "Aggregate_Unemployment": regional_aggregate_unemployment,
                #"Unemployment_Regional": regional_unemployment_rate,
                # "Regional_unemployment_subsidy": regional_unemployment_subsidy,
                # "Aggregate_unemployment_rate": total_unemployment_rate,
                #"LD_cap_0": ld_cap_0,           # LD = labour demand
                #"LD_cons_0": ld_cons_0,
                #"LD_serv_0": ld_serv_0,
                # "Unemployment rate inland": regional_unemployment_rate_internal,
                # "Unemployment total": regional_unemployment_rate_SA,
                # "LD_cap_1": ld_cap_1,
                # "LD_cons_1": ld_cons_1,
                # "LD_serv_1": ld_serv_1,
                # "labor check": consumption_labor_check,    # Check labour demand and employment

                # -- SALARIES/COSTS -- #
                # "Regional_Costs": regional_costs,
                #"Average_Salary": regional_average_salary,
                #"Coastal_cons_av_salary": coastal_average_cons_salary,
                #"Coastal_cap_av_salary": coastal_average_cap_salary,
                #"Coastal_serv_av_salary": coastal_average_serv_salary,
                # "Inland average salary": inland_average_salary,
                # "Inland_cons_av_salary": inland_average_cons_salary,
                # "Inland_cap_av_salary": inland_average_cap_salary,
                # "Inland_serv_av_salary": inland_average_serv_salary,
                # "Salary differential": regional_diff_salary_cons,
                # "Regional_fiscal_balance": regional_balance
                #"Minimum_wage": regional_minimum_wage,
              #  "Top_wage": top_wage,                # highest wage in each region

                # -- MARKET DYNAMICS --#
                # "Competitiveness_Regional": regional_average_competitiveness,
              #  "Regional_sum_market_share": regional_aggregate_market_share,
              #  "Regional_sum_market_share_serv": regional_aggregate_market_share_serv,
                #"MS_exp": ms_exp,
                # "Profitability_cons_coastal": profitability_cons_coastal,
                # "Profitability_cons_coastal": profitability_cons_inland,
                #"Number_entrants_cons_coastal": entrants_cons_coastal,
                # "Profitability_serv_coastal": profitability_serv_coastal,
                # "Profitability_serv_coastal": profitability_serv_inland,
                #"Number_entrants_serv_coastal": entrants_serv_coastal,
                #"Number_exit_cons_coastal": exit_cons_coastal,
               # "Number_exit_serv_coastal": exit_serv_coastal,
                # "Number_entrants_cons_inland": entrants_cons_inland,
                # "Number_entrants_serv_inland": entrants_serv_inland,
                # "Sales_firms": sales_firms
                # "Market_share_normalized": market_share_normalized,

                # -- POPULATION -- #
                # "Population_Regional": regional_population_total,
                # "Population_Regional_Households": regional_population_households,
                # "Average size Coastal": av_size_coastal,
                # "Average size Inland": av_size_inland,
                # "Average size migration to Coastal": av_size_migrant_to_coastal,
                # "Average size migration to Inland": av_size_migrant_to_inland,
                # "Population Households Coastal": regional_population_households_region_0,
                # "Population_Regional_Cons_Firms": regional_population_cons,
                # "Population_Region_0_Cons_Firms": regional_population_cons_region_0,
                # "Population Households Inland": regional_population_households_region_1,
                # "Population_Region0_Households": regional_population_households_region_0,
                # "Cons_regional_IDs": cons_ids_region,
                # "Firms_regions": firm_region,

                # -- ACCOUNTING -- #
                #"Real_demand_cap": real_demand_cap,
                #"Real_demand_cons": real_demand_cons,
                #"Real_demand_services": real_demand_services,
               # "Feasible_prod_services_coastal": feasible_prod_serv_coastal,
                # "Feasible_prod_services_inland": feasible_prod_serv_inland,
                # "Des_prod_services_coastal": deas_prod_serv_coastal,
                # "Ordered_quantities": ordered_quantities,
                # "Total_offers_cons_coastal": total_offers_cons_coastal,
                # "Total_offers_serv_coastal": total_offers_serv_coastal,
                # "Des_prod_services_inland": deas_prod_serv_inland,
                # "Desired_services": desired_services,
                # "Ordered_services": ordered_services,
                # "Total_offers_cons_inland": total_offers_cons_inland,
                # "Total_offers_serv_inland": total_offers_serv_inland
                },

            # Collect data on agent level (per step, per agent)
            agent_reporters={
                "Type": "type",
                "Net worth": "net_worth",  
                "Damage_coeff": "damage_coeff",
                "Monetary_damage": "monetary_damage",
                "House_value" : 'av_wage_house_value',  
                "At_risk": "at_risk", 
                "Wage": "wage", 
                "Education": "edu",    # Liquid resources per firm
               # "House_wage_ratio": "house_quarter_income_ratio", # lambda x: x.house_wage_ratio if x.type == "Household" else None,
                #"Total_damage": "total_damage",  # lambda x: x.total_damage if x.type == "Household" else None,
                #"Total_savings": "total_savings_pre_flood",  # lambda x: x.total_savings if x.type == "Household" else None,
                #"Recovery_time": "recovery_time",
                #"Repair_exp": "repair_exp",  # lambda x: x.repair_exp if x.type == "Household" else None,
               # "Cons": "consumption",  # lambda x: x.consumption if x.type == "Household" else None ,
               # "Employer_id": lambda x:x.employer_ID if x.type == 'Household' else None, 
                # "Edu": lambda x: x.edu if x.type == "Household" else None ,
                #"Flooded":"flooded",  # lambda x: x.flooded if x.type == "Household" else None ,
             
                #"Worry" : "worry",
        
                # "Risk_perc": "risk_perc",
                #"SE_wet": "SE_wet",
                #"RE_wet": "RE_wet",
                # "Price": "price",                                                         #- Price
                # "Prod": "productivity",
         # lambda x: x.monetary_damage if x.type == "Household" else None,
               # "Elevated" : 'elevated',
         
                # "Size": lambda x: len(x.employees_IDs) if x.type == "Service" else None,
                # "Ms": lambda x: x.market_share if x.type == "Service" else None,
                # "Regional demand": lambda x: x.regional_demand_serv if x.type == "Service" else None,
                # "Past demand": lambda x: x.past_sales if (x.type == "Cons") else None,
                # "Past demand": lambda x: x.past_demand if (x.type == "Cap") else None,
                # "Region": lambda x: x.region if x.type == "Service" else None,
                # "Feasible prod": lambda x: x.feasible_production if x.type == "Service" else None, #and x.feasible_production == 0) else None,
                # "Deas prod": lambda x: x.desired_production if x.type == "Service" else None,
                # "Migration prob": lambda x: x.migration_pr if x.type == "Service" else None,
                # "Number out": lambda x: x.number_out if x.type == "Service" else None, # and sum(x.regional_demand_serv) == 0) else None,

                # "Region": lambda x: x.region,
                # "Lifecycle": lambda x: x.lifecycle if x.type == "Cap" else None,
                # "Real_demand_cap": lambda x: x.real_demand_cap if x.type == "Cap" else None,
                # "Competitiveness": lambda x: x.competitiveness if x.type == "Cons" else None,
                # "Feasible prod": lambda x: x.feasible_production if x.type == "Cons" else None,
                # "Number out": lambda x: x.number_out if x.type == "Cons" else None,
                # "Migration prob": lambda x: x.migration_pr if x.type == "Cons" else None,
                # "Regional demand": lambda x: x.regional_demand if x.type == "Cons" else None,
                # "Markup": lambda x: x.markup if x.type == "Cons" else None
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

          #  print("pop change is " ,pop_change)

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
                # COMMENT: TODO: change this to government function,
                #                --> implies using current top_prod,
                #                instead of previous, check if possible
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
            #print('removing', firm_out.unique_id)
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
