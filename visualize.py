import time
import argparse
import torch
from metadrive import MetaDriveEnv

from src.env_utils import flatten_obs, discrete_to_continuous_action
from src.agent import DQNAgent

def watch_agent_drive(view_mode="3D", num_episodes=5, start_seed=42, num_scenarios=5):
    # configure environment for dynamic map rotation
    config = {
        "num_scenarios": num_scenarios,
        "start_seed": start_seed,
        "use_render": True if view_mode == "3D" else False, 
    }
    env = MetaDriveEnv(config)
    
    # initialize and load the trained brain
    state_size = 259
    action_size = 6
    hidden_size = 512
    agent = DQNAgent(state_size, hidden_size, action_size)
    agent.policy_net.load_state_dict(torch.load("models/dqn_trained.pt", map_location='cpu'))
    agent.policy_net.eval()
    

    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        print(f"\n[info] starting episode {episode} on a new map layout...")
        
        for step in range(2000): 
            state = flatten_obs(obs)
            action_idx = agent.act(state, epsilon=0.0)
            continuous_action = discrete_to_continuous_action(action_idx)
            
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            
            if view_mode == "top_down":
                env.render(mode="top_down")
                time.sleep(0.02) 
                
            if terminated or truncated:
                success = info.get("arrive_dest", False)
                terminal_state = "success" if success else "crash/out of bounds"
                print(f"[info] episode {episode} ended. result: {terminal_state}")
                
                # pause briefly so you can see the final position before the map changes
                time.sleep(1.0)
                break
            
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize the agent across multiple maps")
    parser.add_argument("--mode", type=str, default="3D", choices=["3D", "top_down"])
    parser.add_argument("--episodes", type=int, default=5, help="number of attempts to watch")
    parser.add_argument("--seed", type=int, default=100, help="starting map seed for testing")
    parser.add_argument("--scenarios", type=int, default=5, help="number of unique maps to cycle through")
    args = parser.parse_args()
    
    watch_agent_drive(args.mode, args.episodes, args.seed, args.scenarios)