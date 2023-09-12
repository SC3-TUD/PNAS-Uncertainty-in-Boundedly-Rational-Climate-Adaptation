# -*- coding: utf-8 -*-

"""

@author: TabernaA

Class for ServiceFirm agents, based on the MESA Agent class.
This class contains the intialization of service firms and all stages
used in the scheduler described by the StagedActivationByType class.

"""

import random
import numpy as np
# import statistics
# import math

import labor_dynamics as ld
import goods_market as gm
import climate_dynamics as cd
import accounting as ac
import vintage as vin
# import research_and_development as rd
# import migration

from scipy.stats import beta
from random import choice, seed, uniform
from sympy import bernoulli
from mesa import Agent

seed_value = 12345678
seed(seed_value)
np.random.seed(seed=seed_value)


class ServiceFirm(Agent):
    """Class representing a service firm. """

    def __init__(self, model, prod, wage, price, market_share, sales=10,
                 net_worth=2, competitiveness=[1,1], init_n_machines=5,
                 init_capital_amount=3, brochure=None, lifecycle=1, region=0):
        """Initialization of a ServiceFirm agent.

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

        # -- General service firm attributes -- #
        self.type = "Service"
        self.region = region
        self.lifecycle = lifecycle
        self.height = 5 if uniform(0, 1) > self.model.fraction_exposed else 0

        # Only for datacollection at agent level: include other firm
        # and household attributes
        # COMMENT: TODO: see how this can be changed
        self.house_quarter_income_ratio = self.total_savings_pre_flood = None
        self.consumption = None
        self.monetary_damage = self.total_damage = self.repair_exp = None
        self.flooded = self.at_risk = self.av_wage_house_value = self.elevated = None
        self.worry = self.risk_perc = self.SE_wet = self.RE_wet = self.edu = None

        # -- Initial capital -- #
        self.capital_vintage = [vin.Vintage(prod[0], init_capital_amount)
                                for i in range(init_n_machines)]
        self.capital_output_ratio = 1.3
        self.capital_amount = (init_n_machines * init_capital_amount)

        # -- Labor market -- #
        self.wage = wage
        self.employees_IDs = []

        # -- Initial resources -- #
        self.net_worth = net_worth
        self.debt = 0
        self.debt_sales_ratio = 4

        # -- Production -- #
        self.productivity = prod
        self.price = price
        self.normalized_price = 0
        self.inventories = 0
        self.feasible_production = 0
        self.production_made = 1
        self.interest_rate = 0.05

        self.order_canceled = False
        self.order_reduced = 0
        self.investment_cost = 0
        self.quantity_ordered = 0
        self.past_demands = [1, 1]

        # -- Capital supplier -- #
        capital_firm_dict = self.model.schedule.agents_by_type["Cap"]
        if brochure:
            self.supplier_id = brochure[2]
            self.offers = [brochure]
        else:
            # At initialization: pick random supplier from capital firms
            self.supplier_id = random.choice(list(capital_firm_dict))
            self.offers = [capital_firm_dict[self.supplier_id].brochure_regional]
        capital_firm_dict[self.supplier_id].client_IDs.append(self.unique_id)

        # -- Goods market -- #
        self.competitiveness = competitiveness
        self.market_share = np.repeat(market_share, 3)
        self.market_share_history = []

        self.real_demand = 1
        self.regional_demand = [0, 0, 0]
        self.unfilled_demand = 0
        self.sales = sales
        self.subsidiary = 0

        # -- Migration -- #
        self.distances_mig = []
        self.region_history = []

        # -- Climate change -- #
        self.recovery_time = None
        self.damage_coeff = 0
        # self.CCA_resilience = [1, 1]  # Placeholder value for now
        # self.CCA_RD_budget = 0

    def hire_and_fire_serv(self):
        """Open vacancies and fire employees based on demand.

        COMMENT: Combine with function for other firms?
                 Maybe create superclass "Firm" to handle such things.
        """
        # Keep track of old productivity
        self.productivity[0] = self.productivity[1]

        # Determine labor demand and hire employees
        if self.feasible_production > 0:
            self.labor_demand, self.productivity[1] = ld.labor_demand(self)
        else:
            self.labor_demand = 0
            self.productivity[1] = self.productivity[0]
        # # If we want to add the coast of research once we add the CCA
        # self.labor_demand += math.floor(self.CCA_RD_budget / self.wage)

        # Hire or fire employees based on labor demand
        ld.hire_and_fire(self)

    def market_share_calculation(self):
        """Compute firm market share.
           COMMENT: now stored in government, here only retrieved from there.
        """
        # Retrieve competitiveness from central calculations of government
        gov = self.model.governments[0]
        # --------
        # COMMENT: why noise addition?
        # --------
        self.competitiveness = [comp + 1e-7 for comp in
                                gov.serv_comp_normalized[self.unique_id]]

        # Compute market share from competitiveness
        # Make the market more stable at the beginning, for a smooth start
        a = 0.75
        if self.lifecycle == 0:
            K_total = gov.regional_capital_serv
        else:
            K_total = False

        avg_comp = gov.avg_norm_comp["Service"]
        capital_stock = self.capital_amount / self.capital_output_ratio
        gm.calc_market_share(self, avg_comp, capital_stock, K_total, a)

    def price_demand_normalized(self):
        """Retrieves normalized price and unfilled demand
           from the government, that does it at central level
        """
        gov = self.model.governments[0]
        price = gov.norm_price_unf_dem_serv[self.unique_id]
        self.normalized_price = round(price[0], 8)
        self.unfilled_demand = round(price[1], 8)

    def market_share_normalized(self):
        """Retrieves normalized market shares from the government. """
        gov = self.model.governments[0]
        self.market_share = gov.norm_market_shares_serv[self.unique_id]

    def market_share_trend(self):
        """TODO: write description
           (Or remove function, it is only one line.
            Or add to function above?).
        """
        self.market_share_history.append(round(sum(self.market_share), 5))

    def accounting(self):
        """Calculates individual demand, compares to
           production made and accounting costs, sales and profits.
        """
        # -- Accounting -- #
        # NOTE: CCA budget = 0 now
        self.total_costs = round(self.production_made * self.cost, 3)
        self.sales = round(self.demand_filled * self.price, 3)

        # Cancel orders that cannot be fulfilled by supplier
        if self.order_canceled:
            self.scrapping_machines = 0
            if self.debt > 0:
                self.debt = max(0, self.debt - self.investment_cost)
            self.investment_cost = 0
            self.quantity_ordered = 0
            self.order_canceled = False

        # Reduce orders that cannot be fulfilled by supplier
        if self.order_reduced > 0 and self.supplier_id is not None:
            supplier = self.model.schedule._agents[self.supplier_id]
            self.quantity_ordered = max(0, self.quantity_ordered -
                                        self.order_reduced)
            self.scrapping_machines -= self.order_reduced
            self.investment_cost = self.quantity_ordered * supplier.price
            if self.debt > 0:
                self.debt = max(0, self.debt -
                                self.order_reduced * supplier.price)
            self.order_reduced = 0

        # Compute profits
        self.profits = round(self.sales - self.total_costs -
                             self.debt * (1 + self.interest_rate), 3)

        # If profits are positive: pay taxes
        if self.profits > 0:
            gov = self.model.governments[0]
            tax = gov.tax_rate * self.profits
            self.profits = self.profits - tax
            gov.tax_revenues[self.region] += tax

        # Add earnings to net worth
        self.net_worth += self.profits - self.investment_cost
        # If new worth is positive firm is not credit constrained
        if self.net_worth > 0:
            self.credit_rationed = False

    # ------------------------------------------------------------------------
    #                   SERVICEFIRM STAGES FOR STAGED ACTIVATION
    # ------------------------------------------------------------------------
    def stage0(self):
        """Stage 0:
           TODO: write short description for all stages
        """

        # --------
        # COMMENT: why set to 0 here?
        # ANSWER: because otherwise the step after the flood they will
        #         keep damage_coeff > 0 (what can we do here is to do
        #         that at central/model level at the end of
        #         each step with a flood)
        # COMMENT ANSWER (Liz): Yes, that would in my opinion make more sense.
        #                       TODO: change this.
        # --------
        self.damage_coeff = 0
        if self.model.is_flood_now:
            self.damage_coeff = cd.depth_to_damage(self.model.flood_depth,
                                                   self.height, self.type)

        if self.lifecycle > 0:
            # Handle flood damage; get orders and remove old machines
            vin.update_capital(self)
            self.debt = 0
            self.capital_amount = sum(vintage.amount for vintage in
                                      self.capital_vintage)

            # Invest in capital, based on changes in demand
            vin.capital_investments(self)

            # Handle inventory damage
            if self.damage_coeff > 0:
                self.inventories = cd.destroy_fraction(self.inventories,
                                                       self.damage_coeff)

            # Calculate replacement investment
            vin.calc_replacement_investment(self)
            if self.net_worth > (len(self.employees_IDs) * self.wage):
                if self.replacements > 0 or self.expansion > 0:
                    vin.place_order(self)

    def stage1(self):
        """Stage 1:
           TODO: write short description for all stages
        """
        # # Not for newborn firms
        # if self.lifecycle > 0:
        gov = self.model.governments[0]
        ld.set_firm_wage(self, gov.min_wage[self.region],
                         gov.regional_av_prod_serv)
        self.hire_and_fire_serv()

    def stage2(self):
        """Stage 2:
           TODO: write short description for all stages
        """
        if self.damage_coeff > 0:
            self.productivity[1] = cd.destroy_fraction(self.productivity[1],
                                                       self.damage_coeff)
        gm.compete_and_sell(self)

    def stage3(self):
        """Stage 3:
           TODO: write short description for all stages
        """
        # if s.lifecycle > 0:
        households = self.model.schedule.agents_by_type["Household"]
        ld.update_employees_wage(self)
        self.price_demand_normalized()
        trade_cost = self.model.governments[0].transport_cost
        trade_cost_exp = self.model.governments[0].transport_cost_RoW

        self.competitiveness = gm.calc_competitiveness(self.normalized_price,
                                                       self.region,
                                                       trade_cost,
                                                       trade_cost_exp,
                                                       self.unfilled_demand)

    def stage4(self):
        """Stage 4:
           TODO: write short description for all stages
        """
        # if self.lifecycle > 0:
        self.market_share_calculation()

    def stage5(self):
        """Stage 5:
           TODO: write short description for all stages
        """
        # TODO: comment
        # if self.lifecycle > 0:
        self.market_share_normalized()
        self.market_share_trend()

        # TODO: comment
        # if len(self.employees_IDs) > 0 or self.lifecycle < 10:
        total_demand = self.model.governments[0].aggregate_serv
        new_attr = ac.individual_demands(len(self.employees_IDs),
                                         self.lifecycle,
                                         self.regional_demand,
                                         self.market_share,
                                         total_demand,
                                         self.price,
                                         self.productivity[1])
        self.monetary_demand = new_attr[0]
        self.regional_demand = new_attr[1]
        self.real_demand = new_attr[2]
        self.production_made = new_attr[3]
        self.past_sales = new_attr[4]

        # TODO: comment
        new_attr = ac.production_filled_unfilled(self.production_made,
                                                 self.inventories,
                                                 self.real_demand,
                                                 self.lifecycle)
        self.demand_filled, self.unfilled_demand, self.inventories = new_attr
        self.accounting()

    def stage6(self):
        """Stage 6:
           TODO: write short description for all stages
        """
        if self.lifecycle > 1:
            ac.create_subsidiaries(self)

            # Remove bankrupt firm from model
            if (self.market_share[self.region] < 1e-6 or
                    sum(self.past_demands) < 1):
                # Check that enough firms of this type still exist
                if len(self.model.schedule.agents_by_type[self.type]) > 10:
                    self.model.firms_to_remove.append(self)
                # Fire employees and remove offers from suppliers
                ac.remove_employees(self)
                ac.remove_offers(self)

        # Update offer list and lifecycle
        self.offers = []
        self.lifecycle += 1
