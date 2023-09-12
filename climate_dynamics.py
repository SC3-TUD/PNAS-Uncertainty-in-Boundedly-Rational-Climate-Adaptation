# -*- coding: utf-8 -*-

"""

@author: TabernaA

This script is part of the Climate-economy Regional Agent-Based (CRAB) model
and contains functions for the climate dynamics.

"""

from scipy.stats import bernoulli


def flood_handler(model, flood_schedule: dict, current_time: int):
    """Determines if flood happens and how high the severity/depth is.

    Args:
        model           : CRAB_model object
        flood_schedule  : dict(timestamp: severity)
            timestamp   : Timesteps at which floods will occur
            severity    : Depth of flood in meter
        current_time    : Current timestep

    Returns:
        flood_depth     : Flood depth in meters
    """
    # Check if flood occurs
    if current_time in flood_schedule:
        flood_depth = flood_schedule[current_time]
        # flood_map = model.map_as_array.copy()  # Prevent overwriting map
        # flood_map = flood_map - current_S
        # flooded_cells = flood_map <= 0
        # flood_map = flood_map * -1
    else:
        flood_depth = 0
    return flood_depth


def depth_to_damage(flood_height, terrain_height, agent_type, max_depth=5):
    """TODO: write description.
       COMMENT: put research references here in header comment
       COMMENT: values could also be stored in separate file?

    Args:
        flood_height        : Height of flood
        terrain_height      : Height of terrain
        firm_type           : Firm type {"Cap", "Cons" or "Service"}
        max_depth           : Cap value for flood depth
    """

    # Compute flood depth
    flood_depth = flood_height - terrain_height
    flood_depth = min(max_depth, flood_depth)
    if agent_type == "Cap" or agent_type == "Cons" or agent_type == "Service":
        # From the Industry figure (F6) for The Netherlands
        if flood_depth <= 0:
            damage_factor = 0
        elif 0 < flood_depth <= 0.5:
            damage_factor = 0.24
        elif 0.5 < flood_depth <= 1:
            damage_factor = 0.37
        elif 1 < flood_depth <= 1.5:
            damage_factor = 0.47
        elif 1.5 < flood_depth <= 2:
            damage_factor = 0.55
        elif 2 < flood_depth <= 3:
            damage_factor = 0.69
        elif 3 < flood_depth <= 4:
            damage_factor = 0.82
        elif 4 < flood_depth <= 5:
            damage_factor = 0.91
        elif flood_depth > 5:
            damage_factor = 1
        else:
            print("Error in damage factor for this firm")
        return damage_factor

    elif agent_type == "Household":
        # since higher height is the same y value as x=6
        if flood_depth <= 0:
            damage_factor = 0
        elif 0 < flood_depth <= 0.5:
            damage_factor = 0.44
        elif 0.5 < flood_depth <= 1:
            damage_factor = 0.58
        elif 1 < flood_depth <= 1.5:
            damage_factor = 0.68
        elif 1.5 < flood_depth <= 2:
            damage_factor = 0.78
        elif 2 < flood_depth <= 3:
            damage_factor = 0.85
        elif 3 < flood_depth <= 4:
            damage_factor = 0.92
        elif 4 < flood_depth <= 5:
            damage_factor = 0.96
        elif flood_depth > 5:
            damage_factor = 1
        else:
            print('error in damage factor household')
        return damage_factor
    else:
        raise ValueError("Unrecognized firm in the flood damage curve lookup")       


def destroy_fraction(variable, fraction):
    return (1 - fraction) * variable
