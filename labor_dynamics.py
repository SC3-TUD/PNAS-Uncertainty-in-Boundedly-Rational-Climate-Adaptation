# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for labor dynamics of Household and Firm agents.

"""
import time

import numpy as np

from random import seed, sample
from math import ceil

seed_value = 12345678
np.random.seed(seed=seed_value)
seed(seed_value)


def labor_search(household, open_vacancies):
    """ Labor search performed by Household agents.

        Unemployed households search through available suitable employers
        in all sectors and pick the firm offering the highest wage.

    Args:
        household               : Household object
        open_vacancies      : List of firms with open vacancies

    Returns:
        employer_ID             : ID of new employer
    """

    # Check if there are firms with open vacancies
    if open_vacancies:
        # Choose from subset of firm (bounded rationality)
        potential_employers = sample(open_vacancies,
                                     ceil(len(open_vacancies)/3))
 
       # if household.unique_id == 251:
        #    print('open vacancies', [open_vacancies[i].unique_id for i in range(len(open_vacancies))])
         #   print('Potential employers', [potential_employers[i].unique_id for i in range(len(potential_employers))])
        # Choose firm with highest wage
        wage = 0
        for firm in potential_employers:
            if firm.wage > wage:
                #if household.unique_id == 251:
                 #   print('H 251 opportunity', firm.unique_id, firm.wage)
                employer = firm
                wage = employer.wage
        #if employer.unique_id == 61:
         #  print( 'I got the 102', household.unique_id, employer.wage)
        employer.employees_IDs.append(household.unique_id)

        # Close vacancies if firm has enough employees
        if employer.desired_employees == len(employer.employees_IDs):
            employer.open_vacancies = False
            open_vacancies.remove(employer)

        return employer.unique_id

    else:
        return None


def labor_demand(firm):
    """Labor demand determined by Firm agents.

    Args:
        firm                : Firm object (ConsumptionGood or Service Firm)

    Returns:
        labor_demand        : Labor needed to satisfy production
        avg_prod            : Average productivity of firm machines
    """

    # Find most productive machines to satisfy feasible production
    # Loop through machine stock backwards
    # COMMENT: stock was not always sorted by productivity,
    #          TODO: should be changed??
    Q = 0
    machines_used = []
    for vintage in firm.capital_vintage[::-1]:
        # Stop when desired amount is reached
        if Q < firm.feasible_production:
            machines_used.append(vintage)
            Q += vintage.amount
            vintage.amount = int(vintage.amount)

    # Weighted average productivity of chosen machines
    avg_prod = round(sum(vintage.amount * vintage.productivity
                                     for vintage in machines_used) /
                                 sum(vintage.amount
                                     for vintage in machines_used), 3)

    # Compute labor needed to satisfy feasible production
    labor_demand = max(0, ceil(firm.feasible_production / avg_prod))

    return labor_demand, avg_prod


def hire_and_fire(firm):
    """Hire and fire employees.

    Args:
        firm                : Firm object
                              (CapitalGood, ConsumptionGood or Service Firm)
    """

    desired_employees = round(firm.labor_demand)
    n_employees = len(firm.employees_IDs)

    # If number of desired employees currently employed: do nothing
    if desired_employees == n_employees:
        firm.open_vacancies = False
    # If more employees desired than employed: open vacancies
    elif desired_employees > n_employees:
        firm.open_vacancies = True
    # If more employees employed than desired: (possibly) fire employees
    elif desired_employees < n_employees:
        firm.open_vacancies = False
        # Fire employees if profits are too low
        if firm.profits < firm.wage:
            for i in range(n_employees - desired_employees):
                j = firm.employees_IDs[0]
                employee = firm.model.schedule.agents_by_type["Household"][j]
                employee.employer_ID = None
                del firm.employees_IDs[0]

    firm.desired_employees = desired_employees


def set_firm_wage(firm, minimum_wage, regional_av_prod):
    """Set new firm wage, based on average regional wage
       and regional minimum wage.

    Args:
        firm                : Firm object
        minimum_wage        : Regional minimum wage
        regional_av_prod    : Average production within region
    """

    # --------
    # COMMENT: TODO: remove unused arguments
    # --------

    # # Delta price not used for now (not influential in wage determination)
    # con_price_avg = data["Consumption_price_average"]
    # current_average_price = con_price_avg[int(self.model.schedule.time)][firm.region]
    # prev_average_price = con_price_avg[int(self.model.schedule.time)-1][firm.region]
    # delta_price_average = ((current_average_price - prev_average_price) /
    #                        previous_average_price)

    # Keep change in productivity
    prev_prod = firm.productivity[0]
    current_prod = firm.productivity[1]

    delta_my_productivity = max(-0.25, min(0.25, (current_prod - prev_prod) /
                                                 prev_prod))

    # # Delta unemployment, not used for now
    # if (previous_unemployment_rate_my_region or
    #     current_unemployment_rate_my_region) < 0.01:
    #     delta_unemployment = 0
    # else:
    # delta_unemployment = max(-0.025,
    #                          min(0.025,
    #                              (current_unemployment_rate_my_region -
    #                               previous_unemployment_rate_my_region) /
    #                              max(previous_unemployment_rate_my_region,
    #                                  current_unemployment_rate_my_region)))

    delta_unemployment = 0
    # unemploy_vars = data["Unemployment_Regional"]
    # delta_unemployment = unemploy_vars[int(self.model.schedule.time)][firm.region + 2]

    # Regional productivity change, calculated by government
    delta_productivity_average = regional_av_prod[firm.region + 2]

    # Bound wage by minimum wage (determined by government)
    # TODO: check correctness of function, 1 + b not between brackets?
    #       and why (-0.0)???
    b = 0.1
    firm.wage = max(minimum_wage,
                    round(firm.wage * (1 + b * delta_my_productivity +
                          (1 - b) * delta_productivity_average +
                          (-0.0) * delta_unemployment + 0.0), 3))


def update_employees_wage(firm):
    """Update wages for all firm employees.
    
    Args:
        firm            : Firm object
                          (CapitalGood, ConsumptionGood or Service Firm)
    """
    households = firm.model.schedule.agents_by_type["Household"]
    for i in firm.employees_IDs:
        employee = households[i]
        employee.wage = firm.wage
