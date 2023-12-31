# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for the dynamics of the goods market.

"""
import time

import numpy as np



def calc_competitiveness(price, region, trade_cost,
                         trade_cost_exp, unfilled_demand):
    """Calculate firm competitiveness in both regions (Coastal and Inland).

    Args:
        price               : Firm unit price
        region              : Firm region
        trade_cost          : Regional transport cost
        trade_cost_exp      : Export transport cost
        unfilled_demand     :

    """
    if region == 0:
        comp = (np.array([1, 1 + trade_cost, 1 + trade_cost_exp]) *
                -np.array(price) - np.array(unfilled_demand))
    elif region == 1:
        comp = (np.array([1 + trade_cost, 1, 1 + trade_cost_exp]) *
                -np.array(price) - np.array(unfilled_demand))
    return comp


def compete_and_sell(firm, v=0.05):
    """Calculates firm cost, markup and price.

    Args:
        firm                : Firm object (ConsumptionGood or Service Firm)
        v                   : Markup/market_share ratio
    """

    # Cost calculation
    if firm.productivity[1] > 0:
        firm.cost = firm.wage / firm.productivity[1]
    else:
        # Avoid division by zero
        firm.cost = firm.wage

    # Markup calculation
    if len(firm.market_share_history) < 10:
        # Keep markup fixed for the first 10 timesteps (for smooth start)
        firm.markup = 0.125
    else:
        # Calculate markup from market share history,
        # bounded between 0.05 and 0.4
        firm.markup = max(0.01,
                          min(0.4,
                              round(firm.markup *
                                    (1 + v * ((firm.market_share_history[-1] -
                                               firm.market_share_history[-2]) /
                                               firm.market_share_history[-2])),
                                     5)))

    # Adjust price based on new cost and markup, bounded
    # between 0.7 and 1.3 times the old price to avoid large oscillations
    firm.price = max(0.7 * firm.price,
                     min(1.3 * firm.price,
                         round((1 + firm.markup) * firm.cost, 8)))


def calc_market_share(firm, comp_avg, K, K_total, chi=0.5):
    """Calculates market share of consumption good firms.

    Args:
        firm            : Firm object (ConsumptionGood or Service Firm)
        comp_avg        : Average sector competitiveness
        K               : Capital stock
        K_total         : Total capital amount in economy
        chi             : Scaling factor for level of competitiveness
    """
    if firm.lifecycle == 0:
        # Initial market shares
        firm.market_share = np.repeat(max(K/K_total[0], 1e-4), 3)
    else:
        comp = np.array(firm.competitiveness)
        comp_avg = np.array(comp_avg)
        firm.market_share = np.around(np.array(firm.market_share) *
                                      (1 + chi * (comp - comp_avg) /
                                      comp_avg), 8)




