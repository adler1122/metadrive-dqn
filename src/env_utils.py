import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


def create_env():
    """
    sets up the MetaDrive game with the exact rules we want for training.
    """
    config = {
        "num_scenarios": 1,         
        "start_seed": 42,            
        "use_render": False,        # turn ON graphics if GPU is available 
        "crash_vehicle_done": True, # end the game immediately if the car crashes into something
        "out_of_route_done": True   # end the game immediately if the car drives off the road
    }
    return MetaDriveEnv(config)



def flatten_obs(obs):
    """
    translates the complex game data into a flat list of numbers for the AI.
    """
    return np.array(obs, dtype=np.float32).flatten()

def discrete_to_continuous_action(action_idx):
    """
    translates the AI's simple choice (0-5) into real joystick movements [Steering, Gas/Brake].
    -1.0 means full left or full brake. 1.0 means full right or full gas.
    """
    action_map = {
        0: [-1.0, -1.0], # left + brake
        1: [-1.0,  1.0], # left + forward
        2: [ 0.0, -1.0], # straight + brake
        3: [ 0.0,  1.0], # straight + forward
        4: [ 1.0, -1.0], # right + brake
        5: [ 1.0,  1.0]  # right + forward
    }
    
    # Return the mapped action. If something goes wrong, default to [0.0, 0.0] (do nothing)
    return np.array(action_map.get(action_idx, [0.0, 0.0]))