import time
import argparse
import torch
from metadrive import MetaDriveEnv

from src.env_utils import flatten_obs, discrete_to_continuous_action
from src.agent import DQNAgent

def watch_agent_drive(view_mode="3D", num_episodes=5):
    # configure metadrive environment for visual output
    config = {
        "num_scenarios": 1,
        "start_seed": 42,
        # activate the 3d panda3d rendering engine if requested
        "use_render": True if view_mode == "3D" else False, 
    }
    
    env = MetaDriveEnv(config)
    
    # initialize agent architecture to match the training state
    state_size = 259 
    action_size = 6
    agent = DQNAgent(state_size, action_size)
    
    # load trained weights and lock network into evaluation mode
    agent.policy_net.load_state_dict(torch.load("models/dqn_trained.pt"))
    agent.policy_net.eval()
    
    print("==============================================")
    print(f"booting up {view_mode} visualization engine")
    print(f"running for {num_episodes} consecutive episodes.")
    print("close the graphical window or press ctrl+c in terminal to stop.")
    print("==============================================")
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        print(f"\n[info] starting episode {episode}...")
        
        # run a single continuous episode for visual observation
        for step in range(2000): 
            state = flatten_obs(obs)
            
            # strict policy enforcement; epsilon forced to 0.0
            action_idx = agent.act(state, epsilon=0.0)
            continuous_action = discrete_to_continuous_action(action_idx)
            
            # execute simulated physics step
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            
            # handle 2d pygame rendering if top_down mode is active
            if view_mode == "top_down":
                env.render(mode="top_down")
                time.sleep(0.02) 
                
            if terminated or truncated:
                # determine and log terminal state
                success = info.get("arrive_dest", False)
                terminal_state = "success" if success else "crash/out of bounds"
                print(f"[info] episode {episode} ended. result: {terminal_state}")
                
                # pause for 1 second before the next map reset
                time.sleep(1.0)
                break
            
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize the trained dqn agent in metadrive")
    parser.add_argument("--mode", type=str, default="3D", choices=["3D", "top_down"], 
                        help="select rendering engine: '3D' (panda3d) or 'top_down' (pygame)")
    parser.add_argument("--episodes", type=int, default=5, help="number of episodes to watch")
    
    args = parser.parse_args()
    watch_agent_drive(args.mode, args.episodes)