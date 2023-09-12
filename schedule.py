# -*- coding: utf-8 -*-

"""
@author: TabernaA

The StagedActivationByType class. This class is a scheduler based on the
StagedActivation class from the MESA library.
Each timestep consists of 6 stages that are executed consecutively
for all agents.

"""

from collections import defaultdict
from mesa.time import StagedActivation


class StagedActivationByType(StagedActivation):
    """A custom activation class"""

    def __init__(self, model, stage_list, shuffle=True,
                 shuffle_between_stages=False):
        """Initialization of StagedActivationByType class.

        Args:
            model                   : Model object of CRAB_model class
            stage_list              : List of stage names
            shuffle                 : Boolean; if True, shuffle order of agents
                                      each step
            shuffle_between_stages  : Boolean; if True, shuffle agents after
                                      each stage
        """
        super().__init__(model, stage_list, shuffle, shuffle_between_stages)
        self.agents_by_type = defaultdict(dict)

    def add(self, agent):
        """Add new agent to schedule. """
        self.agents_by_type[agent.type][agent.unique_id] = agent
        self._agents[agent.unique_id] = agent

    def remove(self, agent):
        """Remove agent from schedule. """
        del self._agents[agent.unique_id]
        del self.agents_by_type[agent.type][agent.unique_id]

    def step(self):
        """Single model step.

        Each step consists of several stages to provide the right order
        of events and dynamics. The stages are executed consecutively,
        so that all agents execute one stage before moving to the next.
        """
        for stage in self.stage_list:
            for agent_type in self.agents_by_type:
                agent_keys = list(self.agents_by_type[agent_type].keys())
                # Shuffle firms per type
                if self.shuffle and agent_type != "Household":
                    self.model.random.shuffle(agent_keys)
                # Sort households by education (highest education first)
                elif agent_type == "Household":
                    agent_keys = sorted(agent_keys, key=lambda x:
                                        self.agents_by_type["Household"][x].edu,
                                        reverse=True)

                for agent_key in agent_keys:
                    getattr(self._agents[agent_key], stage)()
            # if self.shuffle_between_stages:
            #     self.model.random.shuffle(agent_keys)
            self.time += self.stage_time
        self.steps += 1
