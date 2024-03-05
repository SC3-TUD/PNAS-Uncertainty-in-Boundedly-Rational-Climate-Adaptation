# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains accounting functions for all Firm agents.

"""

from scipy.stats import bernoulli


def individual_demands(size, lifecycle, past_sales, market_shares,
                       total_demand, price, productivity):
    """TODO: write description.

    Args:
        size                :
        lifecycle           : Firm lifetime
        past_sales          :
        market_shares       :
        total_demand        :
        price               : Firm price
        productivity        :

    Returns:
        monetary_demand     :
        regional_demand     :
        real_demand         :
        production_made     :
        past_sales          :

    COMMENT: also here, work from firm object where possible
             adjust directly there, or not at all.
             TODO: check function, more calculations than
                   description mentioned.
                   Past_sales --> reassigned in function, confusing?

    """
    # Get demand for this firm in both regions
    regional_demand = [round(total_demand[0] * market_shares[0], 3),
                       round(total_demand[1] * market_shares[1], 3),
                       round(total_demand[3] * market_shares[2], 3)]
    # Calculate monetary and real demand from this regional demand
    monetary_demand = round(sum(regional_demand), 3)
    real_demand = round(monetary_demand / productivity, 3)

    # Actual production made is constrained by productivity
    production_made = size * productivity

    return (monetary_demand, regional_demand, real_demand,
            production_made, past_sales)


def production_filled_unfilled(production_made, inventories,
                               real_demand, lifecycle):
    """Calculate part of demand that is filled and part that is unfilled.

    Args:
        production_made     :
        inventories         :
        real_demand         :
        lifecycle           : Firm lifetime

    Returns:
        demand_filled       : Part of demand that can be filled
        unfilled_demand     : Part of demand that can not be filled
        inventories         :
    """
    stock_available = production_made + inventories
    demand_filled = min(stock_available, real_demand)

    if lifecycle > 3:
        unfilled_demand = max(0, real_demand - stock_available)
        inventories = max(0, stock_available - real_demand)
    else:
        unfilled_demand = 0
        inventories = 0

    return demand_filled, unfilled_demand, inventories


def create_subsidiaries(firm):
    """Create subsidiaries for firms with profits higher than twice
       their current wage. After 3 timesteps where this occurs, a
       new firm (subsidiary) is created.

    Args:
        firm        : Firm object
    """
    if len(firm.employees_IDs) > 1:
        if firm.profits > firm.wage * 2:
            firm.subsidiary += 1
            if firm.subsidiary > 5:
                firm.model.firm_subsidiaries.append(firm)
                firm.subsidiary = 0
        else:
            firm.subsidiary = 0


def remove_employees(firm):
    """Remove employees from firm.

    Args:
        firm        : Firm object (ConsumptionGood or Service Firm)
        
    COMMENT: again work from firm object rather than returning empty list
    COMMENT: use this function everywhere, now also typed out at many places
    """
    for employee_id in firm.employees_IDs:
        employee = firm.model.schedule._agents[employee_id]
        employee.employer_ID = None

    firm.employees_IDs = []


def remove_offers(firm):
    """Remove firm offers.

    Args:
        firm        : Firm object (ConsumptionGood or Service Firm)
    """
    for offer in firm.offers:
        supplier_ID = offer[2]
        supplier = firm.model.schedule._agents[supplier_ID]
        supplier.client_IDs.remove(firm.unique_id)
