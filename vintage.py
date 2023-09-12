# -*- coding: utf-8 -*-

"""

@author: TabernaA

Class for Vintage objects. This non-agent class represents the machines
produces by CapitalGoodFirms.

"""
import time

import random
import math
import numpy as np

from scipy.stats import bernoulli, beta
from random import choice, seed

seed_value = 12345678
seed(seed_value)
np.random.seed(seed=seed_value)


class Vintage():
    """Class representing a Vintage object. """

    def __init__(self, prod, amount=1):
        """Initialization of a Vintage object.

        Args:
            prod            : Productivity of the machine
            amount          : Number of machines (a Vintage object can represent
                              more than one machine).
        """
        self.productivity = prod
        self.amount = amount
        self.age = 0
        self.lifetime = 15 + random.randint(1, 10)


# --------
# COMMENT: Check way these functions are used, not part of class?
#          --> why not put in Firm files?
# --------
def get_best_offer(firm):
    """Get best offer (optimal productivity/price ratio) from list of offers

    Args:
        offers              : List of [productivity, price, index];
                              current offers, can be empty list.
    """
    if firm.offers:
        # Pick machine with the best product/price ratio from offers
        offer_dict = {idx: (prod, price) for prod, price, idx in firm.offers}
        ratios = {idx: prod/price for idx, (prod, price) in offer_dict.items()}
        best_idx = max(ratios, key=ratios.get)
        new_prod, new_price = offer_dict[best_idx]
    else:
        # If there are no offers: pick random capital good firms as supplier
        cap_firms = firm.model.schedule.agents_by_type["Cap"].values()
        supplier = random.choice(list(cap_firms))
        firm.supplier_id = supplier.unique_id
        #if firm.supplier_id == 2:
         #   print('get best offer', firm.unique_id)
        supplier.client_IDs.append(firm.unique_id)
        if supplier.region == firm.region:
            new_prod, new_price = supplier.brochure_regional[:2]
        else:
            new_prod, new_price = supplier.brochure_export[:2]

    return new_prod, new_price


def update_capital(firm):
    """Update capital:
       1) Handle flood damage to inventories
       2) Get ordered machines
       3) Remove old machines

    Args:
        firm            : Firm object (ConsumptionGood or Service Firm)
    """
    if firm.damage_coeff > 0:
        for vintage in firm.capital_vintage:
            # If Bernoulli is successful: vintage is destroyed
            if bernoulli.rvs(firm.damage_coeff) == 1:
                firm.capital_vintage.remove(vintage)
                del vintage

    # Handle orders if they are placed
    if firm.supplier_id is not None and firm.quantity_ordered > 0:
        # Add new vintage with amount ordered and supplier productivity
        supplier = firm.model.schedule.agents_by_type["Cap"][firm.supplier_id]
        new_machine = Vintage(prod=round(supplier.productivity[0], 3),
                              amount=round(firm.quantity_ordered))
        firm.capital_vintage.append(new_machine)
        # Reset ordered quantity
        firm.quantity_ordered = 0

        # Replace according to the replacement investment
        # COMMENT: TODO: fix this integer/rounding problem at source instead of here
        firm.scrapping_machines = round(firm.scrapping_machines)
        while firm.scrapping_machines > 0:
            vintage = firm.capital_vintage[0]
            if firm.scrapping_machines < vintage.amount:
                vintage.amount -= firm.scrapping_machines
                firm.scrapping_machines = 0
            else:
                firm.scrapping_machines -= vintage.amount
                firm.capital_vintage.remove(vintage)
                del vintage

    # Remove machines that are too old
    for vintage in firm.capital_vintage:
        vintage.age += 1
        if vintage.age > vintage.lifetime:
            firm.capital_vintage.remove(vintage)
            del vintage
    firm.investment_cost = 0


def capital_investments(firm, inventories_frac=0.1):
    """Capital investment, choose supplier, make order.

    Args:
        firm            : Firm object (ConsumptionGood or Service Firm)
    """

    # Take mean of last three demands to form adaptive demand expectations
    if len(firm.past_demands) > 1:
        firm.past_demands = firm.past_demands[-2:]
    firm.past_demands.append(firm.real_demand)
    expected_production = math.ceil(sum(firm.past_demands)/len(firm.past_demands))

    # Set desired level of inventories (fraction of expected production)
    desired_level_inventories = inventories_frac * expected_production

    # Used to let the model start smoothly
    # NOTE: to be removed with model calibration
    if firm.model.time < 10:
        firm.inventories = desired_level_inventories
        firm.unfilled_demand = 0

    # Compute how many units firm wants to produce and how many it can produce
    firm.desired_production = max(0, expected_production +
                                  desired_level_inventories - firm.inventories)
    firm.feasible_production = math.ceil(min(firm.desired_production,
                                             firm.capital_amount /
                                             firm.capital_output_ratio))

    # If capital stock is too low: expand firm (buy more capital)
    if firm.feasible_production < firm.desired_production:
        firm.expansion = (math.ceil(firm.desired_production -
                                    firm.feasible_production) *
                          firm.capital_output_ratio)
    else:
        firm.expansion = 0


def calc_replacement_investment(firm):
    """Calculate the amount of machines that will be replaced.

    Args:
        firm            : Firm object
    """
    # Set new machine that serves as replacement
    new_prod, new_price = get_best_offer(firm)

    replacements = 0
    for vintage in firm.capital_vintage:
        # Unit cost advantage of new machines (UCA)
        # COMMENT: remove use of firm.wage here --> does not matter for ratio
        UCA = firm.wage * (1/vintage.productivity - 1/new_prod)
        # Payback rule
        # Don't consider if productivity is equal, prevent division by zero
        if (UCA > 0 and new_price / UCA <= 3):
            replacements += vintage.amount
    firm.replacements = replacements


def place_order(firm):
    """Choose supplier and place the order of machines.

    Args:
        firm            : Firm object (ConsumptionGood or Service Firm)
    """
    firm.investment_cost = 0
    firm.quantity_ordered = 0
    # Choose based on highest productivity / price ratio
    if firm.offers:
        ratios = [prod/price for prod, price, u_id in firm.offers]
        # Get supplier ID and price from brochure with best prod/price ratio
        firm.supplier_id = firm.offers[ratios.index(max(ratios))][2]
        supplier_price = firm.offers[ratios.index(max(ratios))][1]
        supplier = firm.model.schedule._agents[firm.supplier_id]
    else:
        # If there are no offers: pick random capital good firms as supplier
        cap_firms = firm.model.schedule.agents_by_type["Cap"].values()
        supplier = random.choice(list(cap_firms))
        supplier_price = supplier.price
        supplier.client_IDs.append(firm.unique_id)
        #if supplier.unique_id == 2:
         #   print('Place order', firm.unique_id)
        
        if supplier.region == firm.region:
            firm.offers.append(supplier.brochure_regional)
        else:
            firm.offers.append(supplier.brochure_export)

   # if firm.unique_id == 139:
        #print(firm.offers)
        #print('suplier', firm.supplier_id)
    # Calculate how many machines can be bought based on desired amount
    # and affordable amount.
    total_number_machines_wanted = firm.expansion + firm.replacements
    total_quantity_affordable_own = max(0, firm.net_worth // supplier_price)
    quantity_bought = min(total_number_machines_wanted,
                          total_quantity_affordable_own)

    # If more machines desired than can be bought, make debt to invest
    if quantity_bought < total_number_machines_wanted and firm.net_worth > 0:
        # Compute affordable debt and adjust number of machines bought
        debt_affordable = firm.sales * firm.debt_sales_ratio
        maximum_debt_quantity = debt_affordable // supplier_price
        quantity_bought = min(total_number_machines_wanted,
                              total_quantity_affordable_own +
                              maximum_debt_quantity)

        # Set debt based on bought machines
        firm.debt = (quantity_bought -
                     total_quantity_affordable_own) * supplier_price
        if firm.debt >= debt_affordable:
            firm.credit_rationed = True

    firm.quantity_ordered = int(np.ceil(quantity_bought))
    # Machines that will be replaced (expansion investments have priority)
    firm.scrapping_machines = max(0, quantity_bought - firm.expansion)

    # Add order to suppliers list
    if firm.quantity_ordered > 0:
        # Convert quantity into cost
        firm.investment_cost = firm.quantity_ordered * supplier_price
        if supplier.region == firm.region:
            supplier.regional_orders.append([firm.quantity_ordered,
                                             firm.unique_id])
        else:
            supplier.export_orders.append([firm.quantity_ordered,
                                           firm.unique_id])
    elif firm.quantity_ordered == 0:
        firm.supplier_id = None
    else:
        print("ERROR: Quantity ordered is negative", firm.quantity_ordered)
