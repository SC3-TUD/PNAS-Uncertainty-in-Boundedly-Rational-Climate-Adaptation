# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for datacollection of all model and agent variables
that are of interest.

"""
import numpy as np

from math import log
# from scipy.stats import beta

# from model.classes.capital_good_firm import CapitalGoodFirm
# from model.classes.consumption_good_firm import ConsumptionGoodFirm
# from model.classes.household import Household

seed_value = 12345678
np.random.seed(seed=seed_value)

# --------
# COMMENT:
# TODO: merge similar functions (return as lists or dicts, or depending on
# input arguments (for example func(region): returns variable[region]))
# --------


# -------------------------------
# CLIMATE CHANGE ADAPTATION (CCA)
# -------------------------------
def total_CCA_coastal_be(model):
    return model.governments[0].CCAs[0]


def total_CCA_coastal_elev(model):
    return model.governments[0].CCAs[1]


def total_CCA_coastal_des_elev(model):
    return model.governments[0].CCAs[2]


def total_CCA_coastal_dry_proof(model):
    return model.governments[0].CCAs[3]


def total_CCA_coastal_wet_proof(model):
    return model.governments[0].CCAs[4]


def feas_prod(model):
    return model.governments[0].feasible_productions


# -----------------
# GOVERNMENT BUDGET
# -----------------
def unemployment_cost_coastal(model):
    return model.governments[0].unemployment_cost[0]


def unemployment_cost_inland(model):
    return model.governments[0].unemployment_cost[1]


def tax_revenues_coastal(model):
    return model.governments[0].tax_revenues[0]


def tax_revenues_inlnad(model):
    return model.governments[0].tax_revenues[1]


# ------------
# PRODUCTIVITY
# ------------
# -- Regional -- #
def productivity_firms_average(model):
    return model.governments[0].regional_av_prod


def av_productivity_coastal(model):
    return model.governments[0].regional_av_prod[0]


def av_productivity_inland(model):
    return model.governments[0].regional_av_prod[1]


def gr_productivity_coastal(model):
    return model.governments[0].regional_av_prod[2]


def gr_productivity_inland(model):
    return model.governments[0].regional_av_prod[3]


# -- Regional by sector -- #
def productivity_coastal_consumption_firms(model):
    return model.governments[0].regional_av_prod_cons[0]


def productivity_inland_consumption_firms(model):
    return model.governments[0].regional_av_prod_cons[1]


def productivity_coastal_capital_firms(model):
    return model.governments[0].regional_av_prod_cap[0]


def productivity_inland_capital_firms(model):
    return model.governments[0].regional_av_prod_cap[1]


def productivity_coastal_service_firms(model):
    return model.governments[0].regional_av_prod_serv[0]


def productivity_inland_service_firms(model):
    return model.governments[0].regional_av_prod_serv[1]


# COMMENT: not used
def productivity_capital_firms_average(model):
    return model.governments[0].cap_av_prod


# -----------------
#      OTHER
# -----------------
def n_floods(model):
    return model.n_floods


# --------
# COMMENT: moved to goverment
# --------
def top_prod(model):
    """Get highest productivity per region."""
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    max0 = [max(a.productivity[0] for a in cap_firms if a.region == 0),
            max(a.productivity[1] for a in cap_firms if a.region == 0)]
    try:
        max1 = [max(a.productivity[0] for a in cap_firms if a.region == 1),
                max(a.productivity[1] for a in cap_firms if a.region == 1)]
    except:
        # If no second region is present, set maximum values to zero
        max1 = [0, 0]
    return [max0, max1]


# -------------------
# INVESTMENT AND R&D
# -------------------
def investment_units(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    replacements0 = round(sum(firm.replacements for firm in cons_firms
                              if firm.region == 0), 3)
    expansion0 = round(sum(firm.expansion for firm in cons_firms
                           if firm.region == 0), 3)
    replacements1 = round(sum(firm.replacements for firm in cons_firms
                              if firm.region == 1), 3)
    expansion1 = round(sum(firm.expansion for firm in cons_firms
                           if firm.region == 1), 3)

    return [replacements0, expansion0, replacements1, expansion1]


def investment(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)
    investment0 = round(sum(firm.investment_cost for firm in firms
                            if firm.region == 0), 3)
    investment1 = round(sum(firm.investment_cost for firm in firms
                            if firm.region == 1), 3)

    return [investment0, investment1, investment0 + investment1]

# ---------
# COMMENT: Also included in function above?
# ---------
def investment_coastal(model):
    """TODO: write description. """
    
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)
    # COMMENT: not for service firms??
    investment = round(sum(firm.investment_cost for firm in firms
                           if firm.region == 0), 3)
    return investment


def investment_inland(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)

    investment = round(sum(firm.investment_cost for firm in firms
                           if firm.region == 1), 3)

    return investment


def investment_total(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)

    investment = sum(firm.investment_cost for firm in firms)
    return investment

# # --------
# # NOT USED
# # --------
# def RD_CCA_investment(model):
#     """TODO: write description. """
#     cap_firms = model.schedule.agents_by_type["Cap"].values()
#     cons_firms = model.schedule.agents_by_type["Cons"].values()
#     serv_firms = model.schedule.agents_by_type["Service"].values()
#     firms = list(cap_firms) + list(cons_firms) + list(serv_firms)

#     RD_CCA = round(sum(firm.CCA_RD_budget for firm in firms))
#     return RD_CCA


def RD_coefficient_average(model):
    """TODO: write description. """

    # COMMENT: returned list of the same thing twice??
    #          --> Changed to CCA_resilience[0] and [1], instead of [1] and [1]
    #              (version 18/04/2022)
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cap_firms) + list(cons_firms) + list(serv_firms)

    RD0 = sum(firm.CCA_resilience[0] for firm in firms if firm.region == 0)
    RD1 = sum(firm.CCA_resilience[1] for firm in firms if firm.region == 0)
    return [RD0/len(firms), RD1/len(firms)]

# # --------
# # NOT USED
# # --------
# def RD_total(model):
#     """TODO: write description. """
#     cap_firms = model.schedule.agents_by_type["Cap"].values()
#     RD0 = round(sum(firm.RD_budget for firm in cap_firms if firm.region == 0), 5)
#     RD1 = round(sum(firm.RD_budget for firm in cap_firms if firm.region == 1), 5)
#     RD_total = [RD0, RD1, RD0 + RD1]
#     return RD_total


# --------------
# CLIMATE SHOCKS (Not used now)
# --------------
def climate_shock_generator(model, a=1, b=100):
    return beta.rvs(a, b)


# --------------
# (UN)EMPLOYMENT
# --------------
def regional_unemployment_rate(model):
    return model.governments[0].unemployment_rates


def regional_unemployment_rate_coastal(model):
    return model.governments[0].unemployment_rates[0]


def regional_unemployment_rate_internal(model):
    return model.governments[0].unemployment_rates[1]


def total_unemployment_rate(model):
    return model.governments[0].unemployment_rates[4]


def regional_aggregate_employment(model):
    return model.governments[0].aggregate_employment


def regional_aggregate_unemployment(model):
    return model.governments[0].aggregate_unemployment


def ld_0(model):
    return model.governments[0].labour_demands[0]


def ld_cap_0(model):
    return model.governments[0].labour_demands[2]


def ld_serv_0(model):
    return model.governments[0].labour_demands[4]


def ld_cons_0(model):
    return model.governments[0].labour_demands[6]


def ld_cap_1(model):
    return model.governments[0].labour_demands[3]


def ld_serv_1(model):
    return model.governments[0].labour_demands[5]


def ld_cons_1(model):
    return model.governments[0].labour_demands[7]


# ---------------------------------
# MARKET SHARES AND COMPETITIVENESS
# COMMENT: make functions more efficient
# ---------------------------------
def regional_aggregate_market_share(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    RAMS0 = round(sum(firm.market_share[0] for firm in cons_firms), 4)
    RAMS1 = round(sum(firm.market_share[1] for firm in cons_firms), 4)
    RAMS2_0 = round(sum(firm.market_share[2] for firm in cons_firms
                        if firm.region == 0), 4)
    RAMS2_1 = round(sum(firm.market_share[2] for firm in cons_firms
                        if firm.region == 1), 4)
    return [RAMS0, RAMS1, RAMS2_0 + RAMS2_1, RAMS2_0, RAMS2_1]


def regional_aggregate_market_share_serv(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Service"].values()

    RAMS0 = round(sum(firm.market_share[0] for firm in cons_firms), 4)
    RAMS1 = round(sum(firm.market_share[1] for firm in cons_firms), 4)
    RAMS2_0 = round(sum(firm.market_share[2] for firm in cons_firms
                        if firm.region == 0), 4)
    RAMS2_1 = round(sum(firm.market_share[2] for firm in cons_firms
                        if firm.region == 1), 4)
    return [RAMS0, RAMS1, RAMS2_0 + RAMS2_1, RAMS2_0, RAMS2_1]


def regional_average_competitiveness(model):
    avg_comp = model.governments[0].avg_norm_comp
    return [np.around(avg_comp["Cons"], 6), np.around(avg_comp["Service"], 6)]


def ms_exp(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)

    MSE0 = round(sum(firm.market_share[2] for firm in firms
                     if firm.region == 0), 5)
    MSE1 = round(sum(firm.market_share[2] for firm in firms
                     if firm.region == 1), 5)
    return [MSE0, MSE1]


def ms_region(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    ms0 = [sum(a.market_share) for a in cons_firms if a.region == 0]
    ms1 = [sum(a.market_share) for a in cons_firms if a.region == 1]
    return [ms0, ms1]


# ----------------------
# DEMAND AND CONSUMPTION
# ----------------------
def sales_firms(model):
    return [model.governments[0].net_sales_cons_firms]


def demand_export_rate(model):
    """TODO: write description. """
    if model.schedule.time > 1:
        consumption_data = model.datacollector.model_vars["CONSUMPTION"]
        old_exp_C = consumption_data[int(model.schedule.time)][2]
    else:
        old_exp_C = 1

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    exp_rate0 = round(sum(firm.regional_demand[2] for firm in cons_firms
                          if firm.region == 0)/old_exp_C, 5)
    exp_rate1 = round(sum(firm.regional_demand[2] for firm in cons_firms
                          if firm.region == 1)/old_exp_C, 5)

    demand_export_rate = [exp_rate0, exp_rate1]
    return demand_export_rate


# def quantity_ordered(model):
#     firms0 = firms1 = 1e-4
#     firms = model.schedule.agents_by_type["Cons"].values()
#     for firm in firms:
#         if firm.region == 0:
#             firms0 += firm.quantity_ordered
#         elif firm.region == 1:
#             firms1 += firm.quantity_ordered

#     demand_0 = demand_1 = 0
#     cap_agents = model.schedule.agents_by_type["Cap"].values()
#     for j in range(len(cap_agents)):
#         firm = cap_agents[j]
#         print(firm)
#         if firm.region == 0:
#             demand_0 += sum(firm.real_demand_cap)
#         if firm.region == 1:
#             demand_1 += sum(firm.real_demand_cap)
#             # firms1 += a.quantity_mad
#     return [abs(round(firms0, 5)), abs(round(firms1, 5)),
#             demand_0, demand_1,
#             firms0 + firms1 - demand_0 - demand_1]  # / firms0, price1/firms1]


def consumption_total(model):
    gov = model.governments[0]
    return gov.aggregate_cons[2] + gov.aggregate_serv[2]


def consumption(model):
    return model.governments[0].aggregate_cons


def consumption_coastal(model):
    gov = model.governments[0]
    return gov.aggregate_cons[0] + gov.aggregate_serv[0]


def test_consumption_coastal(model):
    return model.governments[0].test_cons[0]


def consumption_inland(model):
    gov = model.governments[0]
    return gov.aggregate_cons[1] + gov.aggregate_serv[1]


def test_consumption_inland(model):
    return model.governments[0].test_cons[1]

def income_pp(model):
    return model.governments[0].income_pp_change


def consumption_labor_check(model):
    """TODO: write description. """

    households = model.schedule.agents_by_type["Household"].values()
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    HE0 = sum(1 for hh in households
              if hh.region == 0 and hh.employer_ID is not None)
    HE1 = sum(1 for hh in households
              if hh.region == 1 and hh.employer_ID is not None)
    COE0 = sum(len(firm.employees_IDs) for firm in cons_firms
               if firm.region == 0)
    COE1 = sum(len(firm.employees_IDs) for firm in cons_firms
               if firm.region == 1)
    CAE0 = sum(len(firm.employees_IDs) for firm in cap_firms
               if firm.region == 0)
    CAE1 = sum(len(firm.employees_IDs) for firm in cap_firms
               if firm.region == 1)

    return [[HE0, HE1, HE1 + HE0],
            [COE0, COE1, COE1 + COE0],
            [CAE0, CAE1, CAE1 + CAE0]]


# ---------------
# WAGES AND COSTS
# ---------------
def regional_costs(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    RC0 = round(sum(firm.cost for firm in cons_firms if firm.region == 0), 4)
    RC1 = round(sum(firm.cost for firm in cons_firms if firm.region == 1), 4)

    return [RC0, RC1]


def regional_minimum_wage(model):
    return model.governments[0].min_wage[0]

def regional_unemployment_subsidy(model):
    return model.governments[0].unemployment_subsidy


def regional_average_salary(model):
    return model.governments[0].average_wages


def coastal_average_salary(model):
    return model.governments[0].average_wages[0]


def inland_average_salary(model):
    return model.governments[0].average_wages[1]


def coastal_average_cons_salary(model):
    return model.governments[0].salaries_cons[0]


def inland_average_cons_salary(model):
    return model.governments[0].salaries_cons[1]


def coastal_average_cap_salary(model):
    return model.governments[0].salaries_cap[0]


def inland_average_cap_salary(model):
    return model.governments[0].salaries_cap[1]


def coastal_average_serv_salary(model):
    return model.governments[0].salaries_serv[0]


def inland_average_serv_salary(model):
    return model.governments[0].salaries_serv[1]


def regional_average_salary_cap(model):
    return model.governments[0].salaries_cap


def regional_average_salary_cons(model):
    return model.governments[0].salaries_cons


# Top paying consumption firms
def top_wage(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()
    top_wage0 = max(firm.wage for firm in cons_firms if firm.region == 0)
    try:
        top_wage1 = max(firm.wage for firm in cons_firms if firm.region == 1)
    except:
        # If no second region is present, set maximum values to zero
        top_wage1 = 0

    return [top_wage0, top_wage1]


# ----------
# POPULATION
# ----------
def regional_population_total(model):
    """TODO: write description. """
    agents = model.schedule.agents

    r0 = sum(1 for agent in agents if agent.region == 0)
    r1 = sum(1 for agent in agents if agent.region == 1)

    return [r0, r1]


def regional_population_households(model):
    return model.governments[0].regional_pop_hous


def regional_population_households_region_0(model):
    return model.governments[0].regional_pop_hous[0]


def regional_population_households_region_1(model):
    return model.governments[0].regional_pop_hous[1]


def regional_population_cons_region_0(model):
    return model.governments[0].firms_pop[2]


def regional_population_cons(model):
    firm_pop = model.governments[0].firms_pop
    return firm_pop[2:4]


def regional_population_cap(model):
    firm_pop = model.governments[0].firms_pop
    return firm_pop[:2]


def regional_population_cap_coastal(model):
    return model.governments[0].firms_pop[0]


def regional_population_cap_inland(model):
    return model.governments[0].firms_pop


def regional_population_serv_coastal(model):
    return model.governments[0].firms_pop[4]


def regional_population_serv_inland(model):
    return model.governments[0].firms_pop[5]


# COMMENT: check name, this function occurs twice (see above)
def regional_population_cons_coastal(model):
    return model.governments[0].firms_pop[2]


def regional_population_cons_inland(model):
    return model.governments[0].firms_pop[3]


def av_size_coastal(model):
    return model.governments[0].av_size[0]


def av_size_inland(model):
    return model.governments[0].av_size[1]


def av_size_migrant_to_inland(model):
    return model.governments[1].av_migrant


def av_size_migrant_to_coastal(model):
    return model.governments[0].av_migrant


def regional_capital(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    r0 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 0), 2)
    r1 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 1), 2)

    data = model.datacollector.model_vars["Population_Regional_Cons_Firms"]
    avg_regional_cons_firm = data[int(model.schedule.time)]
    firm_capital_amount0 = round(0.4 * r0 /
                                 max(1, avg_regional_cons_firm[0]), 2)
    firm_capital_amount1 = round(0.4 * r1 /
                                 max(1, avg_regional_cons_firm[1]), 2)

    return [max(1, r0), max(1, r1),
            firm_capital_amount0, firm_capital_amount1]


# Ids of firms in each region
def cons_ids_region(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cons_firms) + list(serv_firms)

    region0_IDs = [firm.unique_id for firm in firms if firm.region == 0]
    region1_IDs = [firm.unique_id for firm in firms if firm.region == 1]

    # Remove firm with highest ID from the list
    # COMMENT: why??
    last0 = max(region0_IDs)
    region0_IDs.remove(last0)
    if region1_IDs:
        last1 = max(firm.unique_id for firm in firms if firm.region == 1)
        region1_IDs.remove(last1)

    return [region0_IDs, region1_IDs]


# All firms
def firm_region(model):
    """TODO: write description. """
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    region0_cap = [[a, a.unique_id] for a in cap_firms if a.region == 0]
    region1_cap = [[a, a.unique_id] for a in cap_firms if a.region == 1]
    region0_cons = [[a, a.unique_id] for a in cons_firms if a.region == 0]
    region1_cons = [[a, a.unique_id] for a in cons_firms if a.region == 1]

    return [[region0_cap, region0_cons], [region1_cap, region1_cons]]


def regional_capital_cons(model):
    """TODO: write description. """
    a = 0.1
    b = 0.9
    fraction = (b - a) * np.random.random_sample() + a
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    r0 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 0), 2)
    r1 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 1), 2)
    firm_capital_amount0 = min(5, fraction * r0 / len(cons_firms))
    firm_capital_amount1 = min(5, fraction * r1 / len(cons_firms))

    return [r0, r1, firm_capital_amount0, firm_capital_amount1]


def regional_capital_serv(model):
    """TODO: write description. """
    a = 0.1
    b = 0.9
    fraction = (b - a) * np.random.random_sample() + a
    cons_firms = model.schedule.agents_by_type["Service"].values()

    r0 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 0), 2)
    r1 = round(sum(sum(vintage.amount for vintage in firm.capital_vintage)
                   for firm in cons_firms if firm.region == 1), 2)
    firm_capital_amount0 = min(5, fraction * r0 / len(cons_firms))
    firm_capital_amount1 = min(5, fraction * r1 / len(cons_firms))

    return [r0, r1, firm_capital_amount0, firm_capital_amount1]


# -----
# PRICE
# -----
def price_average(model):
    return model.governments[0].av_price[0]


def price_average_cons_coastal(model):
    return model.governments[0].av_price_cons[0]


def price_average_cons_internal(model):
    return model.governments[0].av_price_cons[1]


def price_average_cap_coastal(model):
    return model.governments[0].av_price_cap[0]


def price_average_cap_internal(model):
    return model.governments[0].av_price_cap[1]


def price_average_serv_coastal(model):
    return model.governments[0].av_price_serv[0]


def price_average_serv_internal(model):
    return model.governments[0].av_price_serv[1]


def price_average_cap(model):
    return model.governments[0].av_price_cap


# ---------------
#       GDP
# ---------------
def gdp_total(model):
    """TODO: write description. """
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cap_firms) + list(cons_firms) + list(serv_firms)

    GDP = sum( firm.production_made for firm in firms)

    return GDP


def gdp_cap(model):
    """TODO: write description. """
    cap_firms = model.schedule.agents_by_type["Cap"].values()

    GDP0 = round(sum(firm.price * firm.production_made for firm in cap_firms
                     if firm.region == 0), 3)
    GDP1 = round(sum(firm.price * firm.production_made for firm in cap_firms
                     if firm.region == 1), 3)

    return [GDP0, GDP1, GDP0 + GDP1]


def gdp_cons(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    GDP0 = round(sum(firm.price * firm.production_made for firm in cons_firms
                     if firm.region == 0), 3)
    GDP1 = round(sum(firm.price * firm.production_made for firm in cons_firms
                     if firm.region == 1), 3)

    return [GDP0, GDP1, GDP0 + GDP1]


def real_gdp_cap_reg_0(model):
    return round(model.governments[0].total_productions[0])


def real_gdp_cap_reg_1(model):
    return round(model.governments[0].total_productions[1])


def real_gdp_cons_reg_0(model):
    return round(model.governments[0].total_productions[2])


def real_gdp_cons_reg_1(model):
    return round(model.governments[0].total_productions[3])


def real_gdp_serv_reg_0(model):
    return round(model.governments[0].total_productions[4])


def real_gdp_serv_reg_1(model):
    return round(model.governments[0].total_productions[5])


def real_gdp_cons(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    GDP0 = round(sum(firm.production_made for firm in cons_firms
                     if firm.region == 0), 3)
    GDP1 = round(sum(firm.production_made for firm in cons_firms
                     if firm.region == 1), 3)

    return [GDP0, GDP1, GDP0 + GDP1]


def gdp(model):
    """TODO: write description. """
    cap_firms = model.schedule.agents_by_type["Cap"].values()
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    serv_firms = model.schedule.agents_by_type["Service"].values()
    firms = list(cap_firms) + list(cons_firms) + list(serv_firms)

    GDP0 = round(sum(firm.price * firm.production_made for firm in firms
                     if firm.region == 0), 3)
    GDP1 = round(sum(firm.price * firm.production_made for firm in firms
                     if firm.region == 1), 3)

    return [GDP0, GDP1, GDP0 + GDP1]


# -----------
# INVENTORIES
# -----------
def inventories(model):
    """TODO: write description. """

    cons_firms = model.schedule.agents_by_type["Cons"].values()

    INV0 = - sum(firm.inventories * firm.price for firm in cons_firms
                 if firm.region == 0)
    INV1 = - sum(firm.inventories * firm.price for firm in cons_firms
                 if firm.region == 1)

    return [INV0, INV1, INV0 + INV1]


def price_average_cons(model):
    """TODO: write description. """
    price0 = price1 = 0
    firms0 = firms1 = 1e-4
    cons_firms = model.schedule.agents_by_type["Cons"].values()
    for firm in cons_firms:
        if firm.region == 0:
            price0 += firm.price * firm.production_made
            firms0 += firm.production_made
        elif a.region == 1:
            price1 += firm.price * firm.production_made
            firms1 += firm.production_made
    price0 = sum(firm.price * firm.production_made for firm in firms
                 if firm.region == 0)
    price1 = sum(firm.price * firm.production_made for firm in firms
                 if firm.region == 1)
    firms0 = sum(firm.production_made for firm in firms if firm.region == 0)
    firms1 = sum(firm.production_made for firm in firms if firm.region == 1)
    avg_price0 = round(price0 / (firms0 + 1e-4), 4)
    avg_price1 = round(price1 / (firms1 + 1e-4), 4)

    return [avg_price0, avg_price1]


# -------
# PROFITS
# -------
def regional_average_profits_cons(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    profits0 = [firm.profits for firm in cons_firms if firm.region == 0]
    profits1 = [firm.profits for firm in cons_firms if firm.region == 1]
    avg_profits0 = round(sum(profits0)/len(profits0), 4)
    if profits1:
        avg_profits1 = round(sum(profits1)/len(profits1), 4)
    else:
        # If no second region exists, set average profits to zero
        avg_profits1 = 0

    avg_profits = [avg_profits0, avg_profits1]
    return avg_profits


def regional_average_profits_cap(model):
    """TODO: write description. """
    cap_firms = model.schedule.agents_by_type["Cap"].values()

    profits0 = [firm.profits for firm in cap_firms if firm.region == 0]
    profits1 = [firm.profits for firm in cap_firms if firm.region == 1]
    avg_profits0 = round(sum(profits0)/len(profits0), 4)
    if profits1:
        avg_profits1 = round(sum(profits1)/len(profits1), 4)
    else:
        # If no second region exists, set average profits to zero
        avg_profits1 = 0

    avg_profits = [avg_profits0, avg_profits1]
    return avg_profits


def regional_profits_cons(model, x2=0.15):
    """TODO: write description. """

    # COMMENT: TODO: check if can be removed or optimized.
    profits_old = [0, 0]
    prof_difference0 = prof_difference1 = 0
    profits = model.governments[0].net_sales_cons_firms
    profit0 = profit1 = 0
    if profits[0] > 0:
        profit0 = log(profits[0])
    if profits[1] > 0:
        profit1 = log(profits[1])

    if model.schedule.time > 0:
        data = model.datacollector.model_vars["Regional_profits_cons"]
        profits_old = data[int(model.schedule.time) - 1]

    profit0_old = profits_old[0]
    profit1_old = profits_old[1]
    if profit0_old == 0:
        profit0_old = profits[0]

    profit1_old = profits_old[1]

    if profit1_old == 0:
        profit1_old = profits[1]

    profitability0 = max(-0.15, min(0.15, profit0 - profit0_old))
    profitability1 = max(-0.15, min(0.15, profit1 - profit1_old))
    if profitability0 < profitability1 and profit1 > profit0:
        prof_difference0 = max(-0.5,
                               (profitability0 -
                                profitability1)/abs(profitability0 + 0.001))

    if profitability1 < profitability0 and profit0 > profit1:
        prof_difference1 = max(-0.5,
                               (profitability1 -
                                profitability0)/abs(profitability1 + 0.001))
    profits_cons = [profit0, profit1,
                    prof_difference0, prof_difference1,
                    profitability0, profitability1]

    return profits_cons


# def sectoral_aggregate_liquid_assets(model):
#     LA_cap0 = [a.net_worth for a in model.schedule.agents
#                if a.type == "Cap" and a.region == 0]
#     LA_cap1 = [a.net_worth for a in model.schedule.agents
#                if a.type == "Cap" and a.region == 1]
#     LA_cons0 = [a.net_worth for a in model.schedule.agents
#                 if a.type == "Cons" and a.region == 0]
#     LA_cons1 = [a.net_worth for a in model.schedule.agents
#                 if a.type == "Cons" and a.region == 1]
#     return [[sum(LA_cap0), sum(LA_cons0)],[sum(LA_cap1), sum(LA_cons1)]]


# def sectoral_aggregate_debt(model):
#     debt_cap = 0
#     debt_cons0 = [a.debt for a in model.schedule.agents
#                   if a.type == "Cons" and a.region == 0]
#     debt_cons1 = [a.debt for a in model.schedule.agents
#                   if a.type == "Cons" and a.region == 1]
#     return [[debt_cap, sum(debt_cons0)], [debt_cap, sum(debt_cons1)]]


# ------------------
# DEBT AND NET WORTH
# ------------------
def debt(model):
    """TODO: write description. """
    cons_firms = model.schedule.agents_by_type["Cons"].values()

    debt = sum(firm.debt for firm in cons_firms)
    return debt


def av_nw_coastal(model):
    return model.governments[0].av_net_worth[0]


def av_nw_inland(model):
    return model.governments[0].av_net_worth[1]


def av_nw_migrant_to_inland(model):
    return model.governments[1].av_migrant_nw


def av_nw_migrant_to_coastal(model):
    return model.governments[0].av_migrant_nw


# ----------
# ACCOUNTING
# ----------
def total_offers_cons_coastal(model):
    return model.governments[0].total_offers[0]


def total_offers_cons_inland(model):
    return model.governments[0].total_offers[1]


def total_offers_serv_coastal(model):
    return model.governments[0].total_offers[2]


def total_offers_serv_inland(model):
    return model.governments[0].total_offers[3]


def desired_services(model):
    return model.governments[0].orders[0]


def ordered_services(model):
    return model.governments[0].orders[1]


def ordered_quantities(model):
    return model.governments[0].orders


def real_demand_cap(model):
    return model.governments[0].real_demands[0]


def real_demand_cons(model):
    return model.governments[0].real_demands[1]


def real_demand_services(model):
    return model.governments[0].real_demands[2]


def aggregate_serv(model):
    return model.governments[0].aggregate_serv[2]


def feasible_prod_serv_coastal(model):
    return model.governments[0].feasible_productions[2]


def feasible_prod_serv_inland(model):
    return model.governments[0].feasible_productions[3]


def deas_prod_serv_coastal(model):
    return model.governments[0].desired_productions[2]


def deas_prod_serv_inland(model):
    return model.governments[0].desired_productions[3]


def regional_balance(model):
    """Return fiscal balance per regional goverment. """
    d0 = d1 = 0
    for gov in model.governments:
        if gov.region == 0:
            d0 = gov.fiscal_balance
        elif gov.region == 1:
            d1 = gov.fiscal_balance
    return [d0, d1]


# -------------
# ENTRY PROCESS
# -------------
def profitability_cons_coastal(model):
    return model.governments[0].profitability_cons_coastal


def profitability_cons_inland(model):
    return model.governments[0].profitability_cons_inland


def entrants_cons_coastal(model):
    return model.governments[0].entrants_cons_coastal


def entrants_cons_inland(model):
    return model.governments[0].entrants_cons_inland


def profitability_serv_coastal(model):
    return model.governments[0].profitability_serv_coastal


def profitability_serv_inland(model):
    return model.governments[0].profitability_serv_inland


def entrants_serv_coastal(model):
    return model.governments[0].entrants_serv_coastal


def entrants_serv_inland(model):
    return model.governments[0].entrants_serv_inland


def exit_serv_coastal(model):
    return model.number_out_firms["Service"]


def exit_serv_inland(model):
    return model.serv_out_inland


def exit_cons_coastal(model):
    return model.number_out_firms["Cons"]


def exit_cons_inland(model):
    return model.cons_out_inland


def exit_cap_coastal(model):
    return model.number_out_firms["Capital"]
