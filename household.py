# -*- coding: utf-8 -*-

"""

@author: TabernaA

Class for Household agents, based on the MESA Agent class.
This class contains the intialization of households and all stages used
in the scheduling determined by the StagedActivationByType class.

"""

import random
import numpy as np
# import pymc3 as pm

import labor_dynamics as ld
import climate_dynamics as cd
# import migration

from mesa import Agent
from enum import unique
from scipy.stats import bernoulli
from pytest import param

seed_value = 12345678
random.seed(seed_value)
np.random.seed(seed=seed_value)


class Household(Agent):
    """Class representing a household agent. """

    def __init__(self, model, attributes, region=0):
        """Initialization of a Household agent.

        Args:
            model               : Model object that contains the agent
            attributes          : Household attributes
                                  (TODO: describe these; determined by survey ...
                                  (from HH_attributes file))
        """
        super().__init__(model.next_id(), model)

        # -- General household attributes -- #
        self.region = region
        self.type = "Household"
        self.lifecycle = 0
        self.consumption = 1
        self.savings = 0
        self.employer_ID = None
        self.wage = 1
        self.net_worth = attributes["savings_norm"]
        self.total_savings = attributes["savings_norm"]
        self.edu = attributes["edu"]
        self.adaptation_model = self.model.cca_model

        # Only for datacollection at agent level: include firm attributes
        # COMMENT: TODO: see how this can be changed
        self.employees_IDs = self.capital_vintage = self.price = None
        self.productivity = self.market_share = self.bankrupt = None
        self.migration_pr = None

        # -- Damage attributes -- #
        if random.uniform(0, 1) > self.model.fraction_exposed:
            self.height = 5
            self.at_risk = False
        else:
            self.height = 0
            self.at_risk = True
        # --------
        # COMMENT: does this initialization below mean that there is always
        #          risk and therefore adaptation? Cannot turn CCA off now?
        depth = list(self.model.flood_schedule.items())[0][1]
        self.damage_coeff = cd.depth_to_damage(depth, self.height, self.type)
        self.house_quarter_income_ratio = attributes["HH_sell_norm"]
        self.monetary_damage = 0
        self.flood_pr = self.model.cum_flood_pr * 0.3
        self.repair_exp = 0
        self.recovery_time = 0
        self.flooded = False
        self.total_savings_pre_flood = 0

        cca_eff = self.model.cca_eff
        cca_cost = self.model.cca_cost
        # -- Elevation -- #
        self.elevation = cca_eff["Elevation"]
        self.damage_coeff_elevation = cd.depth_to_damage(depth,
                                                         (self.height +
                                                          self.elevation),
                                                         self.type)
        self.resp_eff_el = attributes["S_RE1"]
        self.SE_elev = attributes["S_SE1"]
        self.PC_elev = attributes["S_cost1"]
        self.elevation_cost = cca_cost["Elevation_cost"]
        self.elevated = 0

        # -- Dry_proofing -- #
        self.damage_reduction_dry = cca_eff["Dry_proof"]
        self.resp_eff_dry = attributes["S_RE_dry"]
        self.SE_dry = attributes["S_SE_dry"]
        self.PC_dry = attributes["S_cost_dry"]
        self.dry_proofing_cost = cca_cost["Dry_proof_cost"]
        self.dry_proofed = 0

        # -- Wet_proofing -- #
        self.damage_reduction_wet = cca_eff["Wet_proof"]
        self.RE_wet = attributes["S_RE_wet"]
        self.SE_wet = attributes["S_SE_wet"]
        self.PC_wet = attributes["S_cost_wet"]
        self.wet_proofing_cost = cca_cost["Wet_proof_cost"]
        self.wet_proofed = 0
        
        # -- PMT attributes -- #
        self.perc_p = attributes["fl_30_prob"] * self.model.cum_flood_pr
        self.worry = attributes["worry"]
        self.fl_exp = attributes["fl_exp"]
        self.CCA = 0
        self.total_damage = 0
        self.av_wage_house_value = self.house_quarter_income_ratio

        # -- Social interactions -- #
        self.social_exp = attributes["soc_exp"]

    # ------------------------------------------------------------------------
    #                   HOUSEHOLD STAGES FOR STAGED ACTIVATION
    # ------------------------------------------------------------------------
    def stage0(self):
        """Stage 0:
           TODO: write short description for all stages
        """

        # Calculate total damage for this household when flood occurs
        if self.model.is_flood_now:
            avg_wage = self.model.governments[0].average_wages[0]
            value_house = avg_wage * self.house_quarter_income_ratio
            self.monetary_damage = value_house * self.damage_coeff
            self.total_damage = self.monetary_damage
            self.flooded = True
            self.fl_exp = 1
            self.total_savings_pre_flood = self.total_savings 

        # Calculate remaining damage for this timestep and invest
        # in repairing this damage based on households net worth
        if self.monetary_damage > 0:
            self.monetary_damage = max(0, self.monetary_damage -
                                       self.net_worth)
            self.net_worth = max(0, self.net_worth - self.monetary_damage)

    def stage1(self):
        pass
        
    def stage2(self):
        """Stage 2:
           TODO: write short description for all stages
        """

        # Perform labor search if household is unemployed
        gov = self.model.governments[self.region]
        if self.employer_ID is None:
            self.employer_ID = ld.labor_search(self, gov.open_vacancies_all)

        # If still unemployed: give household unemployment subsidy
        if self.employer_ID is None:
            self.wage = gov.unempl_subsidy[self.region]

    def stage3(self):
        """Stage 3:
           TODO: write short description for all stages
        """
        # -- Consumption -- #
        if self.employer_ID is not None:
            self.consumption = 0.9 * self.wage
        else:
            self.consumption = self.wage

        
        avg_wage = self.model.governments[0].average_wages[0]
        self.av_wage_house_value = avg_wage  * self.house_quarter_income_ratio

        gov = self.model.governments[0]  # should be [self.region]?
        if self.monetary_damage > 0:
            self.repair_exp = max(0, self.consumption -
                                   gov.unempl_subsidy[self.region])
            # self.repair_exp -= gov.unempl_subsidy[self.region]
            gov.repair_exp += self.repair_exp
            self.monetary_damage = max(0, self.monetary_damage -
                                       self.repair_exp)
            self.consumption = max(0, self.consumption - self.repair_exp)
            self.recovery_time += 1
        else:
            self.repair_exp = 0
        self.savings = self.wage - self.consumption
        tax = gov.tax_rate * self.savings
        self.savings = self.savings - tax
        gov.tax_revenues[self.region] += tax
        self.net_worth += self.savings

        # -- Adaptation -- #
        self.total_savings += self.savings
        if self.damage_coeff > 0 and self.lifecycle % 4 == 0:
            if self.model.social_int:
                # Get IDs of household neigbors
                neighbor_nodes = list(self.model.G.neighbors(self.unique_id))
                # Get corresponding household agents
                households = self.model.schedule.agents_by_type["Household"]
                neighbor_agents = [households[node]
                                   for node in neighbor_nodes]

               # print(len(neighbor_agents))
                # Get number of neighbors with adaptation measures
                ratio_dry_proofed = sum([neighbor.dry_proofed
                                         for neighbor in neighbor_agents])
                ratio_wet_proofed = sum([neighbor.wet_proofed
                                         for neighbor in neighbor_agents])
                ratio_elevated = sum([neighbor.elevated
                                      for neighbor in neighbor_agents])
            else:
                ratio_dry_proofed = 0
                ratio_elevated = 0
                ratio_wet_proofed = 0

            if self.dry_proofed == 0:
                if self.adaptation_model == "PMT":
                    if self.worry > -1:
                        self.compute_PMT((1 - self.damage_reduction_dry),
                                         self.damage_coeff, self.perc_p,
                                         self.dry_proofing_cost, self.PC_dry,
                                         self.resp_eff_dry, self.SE_dry,
                                         ratio_dry_proofed,
                                         self.wet_proofed, "UG_wet_proof_bi",
                                         self.elevated, "S_UG1",
                                         self.model.pmt_params, "Dry_proof",
                                         self.model.social_int)
                elif self.adaptation_model == "EU":
                     self.compute_EU(self.dry_proofing_cost,
                                     self.damage_coeff *
                                     (1 - self.damage_reduction_dry),
                                     "Dry_proof")
                else:
                    print("Theory not recognized")
            else:
                self.dry_age += 1
                if self.dry_age >= 80:
                    self.dry_proofed = 0
                    self.dry_age = 0
                    if self.elevated == 0:
                        self.damage_coeff = self.damage_coeff_old

            if self.wet_proofed == 0:
                if self.adaptation_model == "PMT":
                    if self.worry > -1:
                        self.compute_PMT((1 - self.damage_reduction_wet), 
                                         self.damage_coeff, self.perc_p,
                                         self.wet_proofing_cost, self.PC_wet,
                                         self.RE_wet, self.SE_wet,
                                         ratio_wet_proofed,
                                         self.elevated, "S_UG1",
                                         self.dry_proofed, "UG_dry_proof_bi",
                                         self.model.pmt_params, "Wet_proof", 
                                         self.model.social_int)
                elif self.adaptation_model == "EU":
                    self.compute_EU(self.wet_proofing_cost,
                                    self.damage_coeff *
                                    (1 - self.damage_reduction_wet),
                                    "Wet_proof")
                else:
                    print("Theory not recognized")

            if self.elevated == 0:
                if self.adaptation_model == "PMT":
                    if self.worry > -1:
                        self.compute_PMT(self.damage_coeff_elevation,
                                         self.damage_coeff, self.perc_p,
                                         self.elevation_cost, self.PC_elev,
                                         self.resp_eff_el, self.SE_elev,
                                         ratio_elevated,
                                         self.wet_proofed, "UG_wet_proof_bi",
                                         self.dry_proofed, "UG_dry_proof_bi",
                                         self.model.pmt_params, "Elevation",
                                         self.model.social_int)
                elif self.adaptation_model == "EU":
                     self.compute_EU(self.elevation_cost,
                                     self.damage_coeff_elevation,
                                     "Elevation")
                else:
                    print("Theory not recognized")

    def stage4(self):
        pass

    def stage5(self):
        pass

    def stage6(self):
        self.lifecycle += 1
        pass

    def compute_EU(self, cost, damage_coeff, measure):
        """Computes expected utility (EU) of CCA measures.

        Args:
            cost            :
            damage_coeff    :
            measure         :
        """
        monetary_cost_elevation = cost
        if self.net_worth > monetary_cost_elevation:
            avg_wage = max(1, self.model.governments[0].average_wages[0])
            # NOTE check change in household value
            expected_value_house = (avg_wage * self.house_quarter_income_ratio *
                                    (1 + 0.005)**120)
            EU_elevation = (self.flood_pr *
                            np.log(expected_value_house -
                                   (expected_value_house * damage_coeff) -
                                   monetary_cost_elevation) +
                            (1 - self.flood_pr) *
                            np.log(expected_value_house - monetary_cost_elevation))
            EU_no_action = (self.flood_pr *
                            np.log(expected_value_house -
                                   (expected_value_house *
                                    self.damage_coeff)) +
                            (1 - self.flood_pr) *
                            np.log(expected_value_house))

            if EU_elevation > EU_no_action:
                self.net_worth -= monetary_cost_elevation
                self.damage_coeff_old = self.damage_coeff
                self.damage_coeff = damage_coeff
                self.model.governments[0].repair_exp += monetary_cost_elevation
                #self.consumption += 
                if measure == "Elevation":
                    self.height += self.elevation
                    self.elevated = 1
                if measure == "Wet_proof":
                     self.wet_proofed = 1
                if measure == "Dry_proof":
                     self.dry_proofed = 1
                     self.dry_age = 1

    def compute_PMT(self, damage_red, perc_damage, perc_p, cost, perc_cost,
                    RE, SE, social_net, ug_1, ug_1_beta, ug_2, ug_2_beta,
                    pmt_params, measure, social_int=True):
        """TODO: write description.
           TODO: reference to article (Noll, 2022, 2021?) on PMT parameters,
                 and make reference (or explain here) to definition of these
                 parameters

        Args:
            damage_red          :
            perc_damage         :
            perp_p              :
            cost                :
            perc_cost           :
            RE                  :
            SE                  :
            social_net          :
            ug_1                :
            ug_1_beta           :
            ug_2                :
            ug_2_beta           :
            pmt_params          : Protection Motivation Theory parameters
            measure             : Adaptation measure that is being considered
            social_int          : Boolean; True if social network is present,
                                           False otherwise

        """
        if self.net_worth > cost:
            params = pmt_params[measure]
            intercept = params["Intercept"]     # Constant
            beta_d = params["fl_dam"]           # Perceived damage
            beta_p = params["fl_30_prob"]       # Perceived probability (in 30 years)
            beta_w = params["worry"]            # Worry
            beta_w_d = params["fl_dam:worry"]   # Interaction of fl. dam. and worry
            beta_re = params["RE"]              # Response efficacy
            beta_se = params["SE"]              # Self-efficacy
            beta_pc = params["PC"]              # Perceived cost
            beta_fe = params["fl_exp"]          # HasExperiencedFlood?
            beta_ug_1 = params[ug_1_beta]       # HasUnderGoneOtherMeasure1?
            beta_ug_2 = params[ug_2_beta]       # HasUnderGoneOtherMeasure2?

            if social_int:
                # Social network (how many friends/neighbors have
                # taken adaptation measures)
                beta_soc_n = params["soc_net"]
                # Social expectation (see survey)
                beta_soc_e = params["soc_exp"]
                # Interaction social network/social expectation
                beta_inter_eff_soc = params["soc_exp:soc_net"]
                social_exp = self.social_exp
                social_inter_eff = social_net * social_exp
            else:
                beta_soc_n = beta_soc_e = 0
                beta_inter_eff_soc = 0 
                social_exp = 0
                social_inter_eff = 0

            # --------
            # COMMENT: TODO: vectorize this computation (weights * ...)
            # --------
            y_hat = (intercept
                     + (beta_p * perc_p)
                     + (beta_d * perc_damage)
                     + (beta_w * self.worry)
                     + (beta_w_d * self.worry * perc_damage)
                     + (beta_re * RE)
                     + (beta_se * SE)
                     + (beta_pc * perc_cost)
                     + (beta_fe * self.fl_exp)
                     + (beta_ug_1 * ug_1)
                     + (beta_ug_2 * ug_2)
                     + (beta_soc_n * social_net)
                     + (beta_soc_e * social_exp)
                     + (beta_inter_eff_soc * social_inter_eff))

            def inv_logit(p):
                return np.exp(p) / (1 + np.exp(p))

            # Get adaptation probability
            y_hat = round(min(1, inv_logit(y_hat)), 2)
            #print(y_hat)
            if y_hat > 0:
                if bernoulli.rvs(y_hat/4) == 1:
                    self.net_worth -= cost
                    self.damage_coeff_old = self.damage_coeff
                    self.damage_coeff = perc_damage * damage_red
                    self.model.governments[0].repair_exp += cost
                    #self.consumption += cost
                    if measure == "Elevation":
                        self.height += self.elevation
                        self.elevated = 1
                    if measure == "Wet_proof":
                        self.wet_proofed = 1
                    if measure == "Dry_proof":
                        self.dry_proofed = 1
                        self.dry_age = 1
