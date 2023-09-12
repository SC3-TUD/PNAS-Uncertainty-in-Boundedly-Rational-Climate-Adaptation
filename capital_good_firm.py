# -*- coding: utf-8 -*-

"""

@author: TabernaA

Class for CapitalGoodFirm agents, based on the MESA Agent class.
This class contains the intialization of capital good firms and all stages
used in the scheduler described by the StagedActivationByType class.

"""
import time

import numpy as np
import bisect

import research_and_development as rd
import labor_dynamics as ld
import climate_dynamics as cd
import accounting as ac
# import migration

from math import exp, sqrt
from random import sample, seed, uniform, shuffle
from scipy.stats import bernoulli
from scipy.stats import beta
from mesa import Agent

seed_value = 12345678
seed(seed_value)
np.random.seed(seed=seed_value)


class CapitalGoodFirm(Agent):
    """Class representing a capital good firm. """

    def __init__(self, model, prod, wage, price, market_share, net_worth=2,
                 lifecycle=1, region=0):
        """Initialization of a CapitalGoodFirm agent.

        Args:
            model           : Model object that contains the agent
            region          : Firm region
        """
        super().__init__(model.next_id(), model)

        # Please note that variables that change per region
        # (i.e. market share) are in lists, where element 0 refers
        # to the Coastal region and element 1 refers to the Inland region
        # TODO: change to dicts, keep in mind that region 1 might be
        #       added again later.

        # -- General capital good firm attributes -- #
        self.type = "Cap"
        self.region = region
        self.lifecycle = lifecycle
        self.height = 5 if uniform(0, 1) > self.model.fraction_exposed else 0

        # Only for datacollection at agent level: include other firm
        # and household attributes
        # COMMENT: TODO: see how this can be changed
        self.house_quarter_income_ratio = self.total_savings_pre_flood = None
        self.consumption = self.av_wage_house_value = self.elevated = None
        self.monetary_damage = self.total_damage = self.repair_exp = None
        self.flooded = self.at_risk = None
        self.worry = self.risk_perc = self.SE_wet = self.RE_wet = self.edu = None

        # -- Labor market -- #
        self.wage = wage
        self.employees_IDs = []

        # -- Initial resources -- #
        self.net_worth = net_worth

        # -- Capital goods market -- #
        # prod = uniform(0.9, 1.1)
        self.productivity = prod
        self.price = price
        self.sales = self.production_made = 0
        self.market_share = np.repeat(market_share, 2)

        self.regional_orders = self.export_orders = []
        self.brochure_regional = [self.productivity[0],
                                  self.price,
                                  self.unique_id]
        self.brochure_export = [self.productivity[0],
                                self.price * (1 + self.model.transport_cost),
                                self.unique_id]

        self.real_demand_cap = [0, 0]
        self.client_IDs = []
        self.productivity_list = []
        self.RD_budget = self.IM = self.IN = 0
        self.subsidiary = 0

        # -- Migration -- #
        self.distances_mig = []
        self.region_history = []
        self.pre_shock_prod = 0

        # -- Climate change -- #
        self.recovery_time = None
        self.damage_coeff = 0
        # self.CCA_resilience = [1, 1]
        # self.CCA_RD_budget = 0
        # self.migration = False

    def RD(self):
        """Research and development: Productivity.
           TODO: write more extensive description.
        """
        # -- Determine RD budget -- #
        # Split budget (if any) between innovation (IN) and imitation (IM)
        if self.sales > 0:
            self.IN, self.IM = rd.calculate_RD_budget(self.sales,
                                                      self.net_worth)
            self.RD_budget = self.IN + self.IM

        # -- INNOVATION -- #
        # OLD VERSION --> compute here
        # # Bernoulli draw to determine success of innovation
        # if bernoulli.rvs(1 - exp(-Z * self.IN)) == 1:
        #     # New production productivity (B) from innovation

        #     a_0 = (1 + x_low + beta.rvs(a, b) * (x_up - x_low))
        #     in_productivity[1] = self.productivity[0] * a_0
        #     # New machine productivity (A) from innovation
        #     a_1 = (1 + x_low + beta.rvs(a, b) * (x_up - x_low))
        #     in_productivity[0] = self.productivity[1] * a_1

        # NEW VERSION --> use RD function (version 20/4/2022)
        in_productivity = rd.innovate(self.IN, self.productivity)

        # -- IMITATION -- #
        # OLD VERSION --> compute here
        # Z = 0.3
        # e = 5        # Geographical distance
        # # Bernoulli draw to determine success of imitation
        # if bernoulli.rvs(1 - exp(-Z * self.IM)) == 1:
        #     # Store imitation probabilities and the corresponding firms
        #     IM_prob = []
        #     # Compute technological distances for all other capital firms
        #     for firm in firms:
        #         distance = (sqrt(pow(self.productivity[0] -
        #                              firm.productivity[0], 2) +
        #                          pow(self.productivity[0] -
        #                              firm.productivity[0], 2)))
        #         if distance == 0:
        #             IM_prob.append(0)
        #         # elif firm.region != self.region:
        #         #     # Add geographical distance if firm is in other region
        #         #     # COMMENT: when readding second region: shouldn't this
        #         #     #          be (1/ (e*distance))?
        #         #     IM_prob.append(1/e * distance)
        #         else:
        #             IM_prob.append(1 / distance)

        #     if sum(IM_prob) > 0:
        #         # Pick firm to imitate based on normalized
        #         # cumulative imitation probabilities
        #         IM_prob = np.cumsum(IM_prob)/np.cumsum(IM_prob)[-1]
        #         j = bisect.bisect_right(IM_prob, uniform(0, 1))
        #         firm = firms[j]
        #         im_productivity = firm.productivity
        # else:
        #     im_productivity = [0, 0]

        # NEW VERSION --> use RD function (version 20/4/2022)
        im_productivity = rd.imitate(self)

        # Recovering lab productivity after disaster
        if self.pre_shock_prod != 0:
            # If there was a shock, firm has recorded pre-shock productivity
            self.productivity[1] = self.pre_shock_prod
            # set to zero otherwise every period the firm will do this
            self.pre_shock_prod = 0

        # -- ADOPTING NEW TECHNOLOGIES -- #
        # Take highest productivity of innovation and imitation outcomes
        self.productivity[0] = round(max(self.productivity[0],
                                         in_productivity[0],
                                         im_productivity[0], 1), 3)
        self.productivity[1] = round(max(self.productivity[1],
                                         in_productivity[1],
                                         im_productivity[1], 1), 3)
        #if self.unique_id == 2:
         #   print(self.productivity,in_productivity  ,im_productivity)

    def calculateProductionCost(self):
        """Calculates the unit cost of production. """

        # # This is for CCA so it is off at the moment
        # if self.flooded:
        #     damages = min(self.model.S/self.CCA_resilience[0],
        #                   self.model.S)

        if self.damage_coeff > 0:
            # Store pre-flood productivity
            self.pre_shock_prod = self.productivity[1]
            self.productivity[1] = cd.destroy_fraction(self.productivity[1],
                                                       self.damage_coeff)
        self.cost = self.wage / self.productivity[1]

    def calculatePrice(self, markup=0.15):
        """Calculate unit price. """
        self.price = (1 + markup) * self.cost

    def advertise(self):
        """Advertise products to consumption-good firms. """

        # Create brochures
        # --------
        # COMMENT: convert to dicts
        # --------
        self.brochure_regional = [self.productivity[0],
                                  self.price,
                                  self.unique_id]
        trade_cost = self.model.governments[0].transport_cost
        self.brochure_export = [self.productivity[0],
                                self.price * (1 + trade_cost),
                                self.unique_id]

        # Choose potential clients (PC) to advertise to
        cons_firms_ids = self.model.schedule.agents_by_type["Cons"].keys()
        serv_firms_ids = self.model.schedule.agents_by_type["Service"].keys()
        total_pool = sorted(list(cons_firms_ids) + list(serv_firms_ids))

        if len(self.client_IDs) > 1:
            new_clients = sample(total_pool,
                                 1 + round(len(self.client_IDs) * 0.2))
        else:
            new_clients = sample(total_pool, 10)

        # Add potential clients to own clients, remove duplicates
        self.client_IDs = set(self.client_IDs + new_clients)
        

        # Send brochure to chosen firms
        for firm_id in self.client_IDs:
            client = self.model.schedule._agents[firm_id]
            if client.region == self.region:
                client.offers.append(self.brochure_regional)
            elif client.region == 1 - self.region:
                client.offers.append(self.brochure_export)

        self.client_IDs = list(self.client_IDs)
        #if self.unique_id ==2:
         #   print(self.client_IDs)
        return self.client_IDs

    def set_firm_wage(self):
        """Set new firm wage, based on average regional wage
           and regional minimum wage.
        """
        # Get minimum wage in this region (determined by government)
        gov = self.model.governments[0]
        minimum_wage = gov.min_wage[self.region]
        
        # Get consumption firm top wages in region
        # COMMENT: TODO: change this to not directly use datacollector
        #                (make model dynamics independent of output collection)
       # top_wages = self.model.datacollector.model_vars["Top_wage"]
        top_wage = gov.top_wage #top_wages[int(self.model.schedule.time)][self.region]
        
        # Set wage to max of min wage and top paying wage
        self.wage = max(minimum_wage, top_wage)

    def hire_and_fire_cap(self):
        """Open vacancies and fire employees based on demand. """
        # --------
        # COMMENT: move first part to separate function? --> handles demand
        # --------
        self.past_demand = self.real_demand_cap
        self.real_demand_cap = [0, 0]
        if self.regional_orders:
            demand_int = 0
            for order in self.regional_orders:
                demand_int += order[0]
            self.real_demand_cap[0] = demand_int
        if self.export_orders:
            demand_exp = 0
            for order in self.export_orders:
                demand_exp += order[0]
            self.real_demand_cap[1] = demand_exp

        self.labor_demand = (sum(self.real_demand_cap) /
                             self.productivity[1] +
                             (self.RD_budget / self.wage))

        # Hire or fire employees based on labor demand
       # if self.unique_id == 2:
            
           # print('labor demand is', self.labor_demand, self.employees_IDs)
           # print('orders', self.regional_orders)
        ld.hire_and_fire(self)

    def accounting_orders(self):
        """Checks if firm has satisfied all orders, otherwise the remaining
           orders have to be canceled.
        """
        total_orders = sum(self.real_demand_cap)
        self.production_made = max(0, round(min(len(self.employees_IDs),
                                                total_orders /
                                                self.productivity[1]) *
                                            self.productivity[1]))

        if self.damage_coeff > 0:
             self.production_made = cd.destroy_fraction(self.production_made,
                                                        self.damage_coeff)
        # if self.model.schedule.time == self.model.shock_time:
        #     damages = min(self.model.S / self.CCA_resilience[1],
        #                   self.model.S)
        #     self.production_made = (1 - damages) * self.production_made

        # Cancel orders if necessary
        self.orders_filled = min(total_orders, self.production_made)
        if self.orders_filled < total_orders:
            orders = self.regional_orders + self.export_orders
            amount_to_cancel = total_orders - self.orders_filled
            shuffle(orders)

            # Delete or reduce orders based on amount to cancel
            for order in orders:
                if amount_to_cancel > 0:
                    buyer = self.model.schedule._agents[order[1]]
                    # If more should be canceled: remove full order
                    if order[0] <= amount_to_cancel:
                        canceled_amount = order[0]
                        buyer.order_canceled = True
                    # If order is bigger than amount to cancel: reduce order
                    else:
                        canceled_amount = min(order[0], amount_to_cancel)
                        buyer.order_reduced = canceled_amount
                    amount_to_cancel -= canceled_amount

        # -- ACCOUNTING -- #
        self.sales = self.orders_filled * self.price
        self.total_costs_cap = self.cost * self.orders_filled
        self.profits = self.sales - self.total_costs_cap - self.RD_budget
        self.net_worth += self.profits

        # --------
        # COMMENT: TODO: Change these lists to dicts
        # --------
        self.regional_orders = []
        self.export_orders = []

    def market_share_normalized(self):
        """Update my market share so that it's normalized.
           The government does it at central level so the firm
           just retrieve its value (linked to its unique_id).

        COMMENT: not really necessary to have separate function only
                 for retrieving these values
        """
        gov = self.model.governments[0]
        market_share = gov.norm_market_shares_cap[self.unique_id]
        self.market_share = np.around(market_share, 8)

    # ------------------------------------------------------------------------
    #                CAPITALGOODFIRM STAGES FOR STAGED ACTIVATION
    # ------------------------------------------------------------------------
    def stage0(self):
        """Stage 0:
           TODO: write short description for all stages
        """

        # # CCA DYNAMICS
        # Handle flood if it occurs
        if self.model.is_flood_now:
            self.damage_coeff = cd.depth_to_damage(self.model.flood_depth,
                                                   self.height, self.type)
        else:
            self.damage_coeff = 0

        self.RD()
        self.calculateProductionCost()
        self.calculatePrice()
        self.advertise()

    def stage1(self):
        """Stage 1:
           TODO: write short description for all stages
        """
        self.set_firm_wage()
        self.hire_and_fire_cap()

    def stage2(self):
        pass

    def stage3(self):
        """Stage 3:
           TODO: write short description for all stages
        """
        ld.update_employees_wage(self)
        self.accounting_orders()

    def stage4(self):
        pass

    def stage5(self):
        pass

    def stage6(self):
        """Stage 6:
           TODO: write short description for all stages
        """
        self.market_share_normalized()

        if self.lifecycle > 2:
            ac.create_subsidiaries(self)

            # Remove bankrupt firm from model
            if self.net_worth <= 0 and sum(self.real_demand_cap) < 1:
                # Check that enough firms of this type still exist
                if len(self.model.schedule.agents_by_type[self.type]) > 10:
                    self.model.firms_to_remove.append(self)
                # Fire employees
                ac.remove_employees(self)

        self.lifecycle += 1