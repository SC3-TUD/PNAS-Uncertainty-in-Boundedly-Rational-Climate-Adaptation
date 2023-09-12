# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for the research and development (R&D) process perfomed
by the Firm agents in the model.

"""

import random
import math
import bisect
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import beta

seed_value = 12345678
random.seed(seed_value)
np.random.seed(seed=seed_value)


def calculate_RD_budget(sales, net_worth, rd_fraction=0.04, in_fraction=0.5):
    """Calculate total research and development budget (RD),
       innovation budget (IN) and imitation budget (IM).

    Args:
        sales           : Sales in previous period
        net_worth       : Available firm money (net worth)
        rd_fraction     : Fraction of previous sales used for RD
        in_fraction     : Fraction of budget used for innovation (IN),
                          rest is used for imitation (IM)

    Returns:
        in_budget       : Budget used for innovation
        im_budget       : Budget used for imitation
    """

    # Determine total RD budget based on sales or net worth

    # COMMENT: function only called when sales > 0 (in CapitalGoodFirm class)
    #          --> change or remove this check from function, now rest does nothing
    if sales > 0:
        rd_budget = rd_fraction * sales
    elif net_worth < 0:
        rd_budget = 0
    else:
        rd_budget = rd_fraction * net_worth

    # Divide RD budget over innovation and imitation budget
    in_budget = in_fraction * rd_budget
    im_budget = rd_budget - in_budget

    return in_budget, im_budget


def innovate(in_budget, prod, Z=0.3, a=2, b=4, x_low=-0.05, x_up=0.05):
    """ Innovation process perfomed by Firm agents.

    Args:
        in_budget       : Firms innovation budget
        prod            : Firms productivity
    Params:
        Z               : Budget scaling factor for Bernoulli draw
        a               : Alpha parameter for Beta distribution
        b               : Beta parameter for Beta distribution
        x_low           : Lower bound
        x_up            : Upper bound
    """

    p = 1 - math.exp(-Z * in_budget)
    # Bernoulli draw to determine success of innovation
    if bernoulli.rvs(p) == 1:
        # Draw change in productivity from beta distribution
        prod_change = 1 + x_low + beta.rvs(a, b, size=2) * (x_up - x_low)
        in_productivity = prod_change * prod
    else:
        in_productivity = [0, 0]

    return in_productivity


def imitate(firm, Z=0.3, e=5):
    """ Imitation process performed by Firm agents.

    Args:
        firm            : Firm object (CapitalGoodFirm)

    Parameters:
        Z               : Budget scaling factor for Bernoulli draw
        e               : Distance scaling factor for firms in other region
    """

    # Bernoulli draw to determine success of imitation
    if bernoulli.rvs(1 - math.exp(-Z * firm.IM)) == 1:
        # Compute technological distances for all other capital firms
        firms = list(firm.model.schedule.agents_by_type[firm.type].values())
        IM_prob = []
        for other_firm in firms:
            distance = (math.sqrt(pow(firm.productivity[0] -
                                      other_firm.productivity[0], 2) +
                                  pow(firm.productivity[1] -
                                      other_firm.productivity[1], 2)))
            if distance == 0:
                IM_prob.append(0)
            # elif other_firm.region != firm.region:
            #     # Add geographical distance if firm is in other region
            #     # COMMENT: when readding second region: shouldn't this
            #     #          be (1/ (e*distance))?
            #     IM_prob.append(1/e * distance)
            else:
                IM_prob.append(1 / distance)

        if sum(IM_prob) > 0:
            # Pick firm to imitate from normalized cumulative imitation prob
            IM_prob = np.cumsum(IM_prob)/np.cumsum(IM_prob)[-1]
            j = bisect.bisect_right(IM_prob, random.uniform(0, 1))
            firm_to_imitate = firms[j]
            im_productivity = firm_to_imitate.productivity
    else:
        im_productivity = [0, 0]

    return im_productivity


# # --------
# # COMMENT: same as above?
# # --------
# def calculateRDBudgetCCA(sales, net_worth, v=0.005, e=0.5):
#     if sales > 0:
#         rd_budget = v * sales
#     elif net_worth < 0:
#         rd_budget = 0
#     else:
#         rd_budget = v * net_worth

#     in_budget = e * rd_budget
#     im_budget = (1 - e) * rd_budget

#     return rd_budget, in_budget, im_budget


# # --------
# # COMMENT: same as above?
# # --------
# def innovate_CCA(IN, R, Z=0.3, a=3, b=3, x_low=-0.10, x_up=0.10):
#     """
#     RD : CCA resilience coefficient
#     """
#     in_R = [0, 0]

#     # Bernoulli draw to determine success (1) or failure (0)
#     p = 1 - math.exp(-Z * IN / 2)
#     # Labor productivity resilience
#     if bernoulli.rvs(p) == 1:
#         # New resilience coefficient from innovation
#         in_R[0] = R[0] * (1 + x_low + beta.rvs(a, b) * (x_up - x_low))

#     # Capital stock resilience
#     if bernoulli.rvs(p) == 1:
#         in_R[1] = R[1] * (1 + x_low + beta.rvs(a, b) * (x_up - x_low))

#     return in_R


# # --------
# # COMMENT: same as above?
# # --------
# def imitate_CCA(IM, firm_ids, agents, R, reg, Z=0.3, e=1.5):
#     """CCA RD: imitation
#     """
#     im_R = [0, 0]

#     # Bernoulli draw to determine success (1) or failure (0)
#     p = 1 - math.exp(-Z * IM)
#     if bernoulli.rvs(p) == 1:
#         # Compute inverse Euclidean distances
#         imiProb = []
#         imiProbID = []
#         for id in firm_ids:
#             firm = agents[id]
#             distance = (math.sqrt(pow(R[0] - firm.CCA_resilience[0], 2) +
#                         pow(R[0] - firm.CCA_resilience[0], 2)))
#             if distance == 0:
#                 imiProb.append(0)
#             else:
#                 # Increase distance if the firm is in another region
#                 if firm.region != reg:
#                     imiProb.append(1/e*distance)
#                 else:
#                     imiProb.append(1/distance)
#             imiProbID.append(firm.unique_id)

#         # Cumulative probability
#         if (sum(imiProb) > 0):
#             acc = 0
#             for i in range(len(imiProb)):
#                 acc += imiProb[i] / sum(imiProb)
#                 imiProb[i] = acc

#             # Randomly pick a firm to imitate (index j)
#             rnd = random.uniform(0, 1)
#             j = bisect.bisect_right(imiProb, rnd)
#             # Copy that firm's technology
#             if j < len(imiProb):
#                 firm = agents[imiProbID[j]]
#                 im_R[0] = firm.CCA_resilience[0]
#                 im_R[1] = firm.CCA_resilience[1]

#     return im_R
