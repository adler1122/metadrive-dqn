import os
import argparse
import numpy as np
import torch

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.network import DQNNetwork
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent

def train(num_episodes=3000, batch_size=128, sync_target_freq=5):
    """
    the main training loop 
    """
    # 1. setup the game and data structures
    env = create_env()
    
    # MetaDrive natively gives us 259 numbers, and we have 6 possible actions
    state_size = 259 
    action_size = 6
    
    # initialize the memory and the driver
    memory = ReplayBuffer(capacity=10000)
    agent = DQNAgent(state_size, action_size)
    
    # exploration parameters (Epsilon)
    epsilon = 1.0          # Start by taking 100% random actions to explore
    epsilon_min = 0.05     # Never go below 5% random actions
    epsilon_decay = 0.998  # Slowly reduce random actions over time
    
    print(f"start training for {num_episodes} episodes") # a quick print to check everything is working before using google colab to train the model
    
    # 2. the training loop 
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)
        
        total_reward = 0
        steps = 0
        crashed = False
        
        while True:
            # step 1: driver picks an action
            action_idx = agent.act(state, epsilon)
            continuous_action = discrete_to_continuous_action(action_idx)
            
            # step 2: execute action in the game
            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            next_state = flatten_obs(next_obs)
            
            # check if we crashed 
            if info.get("crash", False) or info.get("crash_vehicle", False):
                crashed = True
            
            # step 3: save the memory
            memory.add(state, action_idx, reward, next_state, done)
            
            # step 4: learn from past mistakes 
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                agent.learn(batch)
                
            # move to the next frame
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # 3. end of episode bookkeeping
        # decay epsilon so the car relies more on its brain and less on random guesses
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        # update the frozen target network periodically
        if episode % sync_target_freq == 0:
            agent.update_target_network()
            
        print(f"Episode: {episode}/{num_episodes} | Steps: {steps} | Reward: {total_reward:.2f} | Crashed: {crashed} | Epsilon: {epsilon:.2f}")

    # 4. save the model 
    env.close()
    
    
    save_path = "models/dqn_trained.pt"
    
    # save the weights of the policy network
    torch.save(agent.policy_net.state_dict(), save_path)
    print("training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a DQN agent in MetaDrive")
    parser.add_argument("--episodes", type=int, default=3000, help="number of episodes to train")
    
    args = parser.parse_args()
    
    train(num_episodes=args.episodes)