import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


def create_env():
    """
    sets up the MetaDrive game with the exact rules we want for training
    """
    config = {
        "num_scenarios": 1,         
        "start_seed": 42,            
        "use_render": False,        # turn ON graphics if GPU is available (for teammates)
        "crash_vehicle_done": True, # end the game immediately if the car crashes into something
        "out_of_route_done": True   # end the game immediately if the car drives off the road
    }
    return MetaDriveEnv(config)



def flatten_obs(obs):
    """
    translates the complex game data into a flat list of numbers for the AI
    """
    return np.array(obs, dtype=np.float32).flatten()

import numpy as np

def discrete_to_continuous_action(action_idx):
    """
    translates the AI's simple choice (0-5) into real joystick movements [Steering, Gas/Brake].
    updated mapping provides smooth cornering and traction control to prevent spin-outs.
    """
    action_map = {
        0: [-0.5,  0.5], # Soft Left + Half Gas (Smooth lane keeping)
        1: [-1.0, -0.5], # Hard Left + Brake (Emergency cornering)
        2: [ 0.0, -1.0], # Straight + Hard Brake
        3: [ 0.0,  1.0], # Straight + Full Gas (Straightaway speed)
        4: [ 0.5,  0.5], # Soft Right + Half Gas (Smooth lane keeping)
        5: [ 1.0, -0.5]  # Hard Right + Brake (Emergency cornering)
    }
    
    # return the mapped action. If something goes wrong, default to [0.0, 0.0] (do nothing)
    return np.array(action_map.get(action_idx, [0.0, 0.0]))