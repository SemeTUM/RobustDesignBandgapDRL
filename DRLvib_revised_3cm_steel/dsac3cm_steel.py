import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import random
import math
import numpy as np
from collections import namedtuple, deque
import torch.autograd as grad
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

sys.path.append("..")

from envs.DRLComEnv3cm_steel import VibEnv3D
from envs.WalkingEnv import GridEnv, comsol_reward_function, build_p4m_structure_simple,strictly_enforce_4_connectivity, check_4_connectivity_single_component
from utils.ReplayBuffer import ReplayBuffer
from src.dsac import SAC

#Trainining main
N_EPISODES = 500
TOTAL_STEPS = 500*500
BUFFER_SIZE = 500*500
BATCH_SIZE = 128
MAX_STEPS_PER_EPISODE = 500

STATE_DIM = 2
ACTION_DIM = 4
device = torch.device("cpu")
UPDATES_PER_EPISODE = 5
GRID_SIZE = 30

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

p_loss, a_loss, b_err1, b_err2, cur_alpha = [], [], [], [], []

checkpoint3000 = torch.load("../DRLvib_revised_4cm_wo_bias/SAC_DISCRETE_WO_frame_agent_checkpoint_ep2000.pth", map_location=torch.device("cpu"))

def main():
    env = GridEnv(GRID_SIZE)
    env0 = VibEnv3D()

    agent = SAC(STATE_DIM,  ACTION_DIM,   device=device)

    agent.critic1.load_state_dict(checkpoint3000['critic_state_dict'])
    agent.critic2.load_state_dict(checkpoint3000['critic2_state_dict'])
    agent.critic1_target.load_state_dict(checkpoint3000['critic1_target_state_dict'])
    agent.critic2_target.load_state_dict(checkpoint3000['critic2_target_state_dict'])
    agent.actor_local.load_state_dict(checkpoint3000['policy_state_dict'])
    agent.critic1_optimizer.load_state_dict(checkpoint3000['critic1_optimizer_state_dict'])
    agent.critic2_optimizer.load_state_dict(checkpoint3000['critic2_optimizer_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint3000['actor_optimizer_state_dict'])
    agent.alpha_optimizer.load_state_dict(checkpoint3000['alpha_optimizer_state_dict'])
    
    replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, device=device)
    
    training_history = []
    total_steps = 0
    total_reward = 0
    print("start training")    
    for episode in range(N_EPISODES):
        state = env.reset()
        episode_buffer = []
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1
        
        fund_grid = env.get_fundamental_grid()
        full_grid = build_p4m_structure_simple(fund_grid)
        checked_grid = strictly_enforce_4_connectivity(full_grid)
        is_connected, _ = check_4_connectivity_single_component(checked_grid)
        if is_connected:
            com_reward, bngp = comsol_reward_function(env, env0)
            final_reward = 1.5*com_reward if com_reward > 0 else com_reward
        else:
            final_reward= -20 
            bngp= 0

        episode_rewards_for_save = []
        for (s, a, r, ns, d) in episode_buffer:
            reward_to_store = r+ final_reward
            replay_buffer.add(s, a, reward_to_store, ns, d)
            episode_rewards_for_save.append(reward_to_store)

        history_entry = {
            'episode': episode + 1,
            'final_bandgap_reward': final_reward,
            'states': np.array([exp[0] for exp in episode_buffer]),
            'actions': np.array([exp[1] for exp in episode_buffer]).flatten(),
            'rewards': np.array(episode_rewards_for_save),
            'bandgap': np.array(np.real(bngp)),
        }
        training_history.append(history_entry)

        if len(replay_buffer)> BATCH_SIZE:
#            for _ in range(UPDATES_PER_EPISODE):
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(step, replay_buffer.sample(), gamma=0.99)       
            print(f"Episode {episode+1}/{N_EPISODES}, Reward: {reward_to_store:.2f}, ploss: {policy_loss}, aloss:{alpha_loss}, berr1:{bellmann_error1}, berr2:{bellmann_error2}")
        print(f" {episode+1}/{N_EPISODES},EPReward: {reward_to_store:.2f}")
        
        p_loss.append(policy_loss)
        a_loss.append(alpha_loss)
        b_err1.append(bellmann_error1)
        b_err2.append(bellmann_error2)
        cur_alpha.append(float(current_alpha))

    np.save('Updated_SAC_DISCRETE_3cm_steel_wo_bias.npy', np.array(training_history, dtype=object), allow_pickle=True)
    print("Training complete. History saved to training_data_history.npy")

if __name__ == "__main__":
    main()
