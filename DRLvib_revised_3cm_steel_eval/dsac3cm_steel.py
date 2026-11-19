import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import random

import numpy as np
from collections import namedtuple, deque
import torch.autograd as grad
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

sys.path.append("..")

from envs.DRLComEnv3cm_steel_40 import VibEnv3D
from envs.WalkingEnv import GridEnv, comsol_reward_function
from utils.utils import check_4_connectivity_single_component, build_p4m_structure_simple,strictly_enforce_4_connectivity

from utils.PER import PER
from src.dsac import SAC

#Trainining main
N_EPISODES = 500
TOTAL_STEPS = 2.5e5
BUFFER_SIZE = 100000
BATCH_SIZE = 128
MAX_STEPS_PER_EPISODE = 500
EXPLORATION_STEPS = 100000
STATE_DIM =402  # position (r, c) + 20x20 fundamental grid walked so far
ACTION_DIM = 4
device = torch.device("cpu")
UPDATES_PER_EPISODE = 5
GRID_SIZE = 40

# PER
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPS = 1e-2


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

env0 = VibEnv3D()
env = GridEnv(GRID_SIZE)

def get_state(pos):
    # the design (fundamental grid) is part of the state, so the final reward is a function of the final state
    return np.concatenate([pos, env.get_fundamental_grid().ravel()]).astype(np.float32)

p_loss, a_loss, b_err1, b_err2, cur_alpha = [], [], [], [], []
checkpoint = torch.load("../DRLvib_revised_4cm_wo_bias/SAC_DISCRETE_PER_WO_frame_agent_checkpoint_ep2000.pth", map_location= device)
def main():
    agent = SAC(STATE_DIM,  ACTION_DIM,   device=device)
    replay_buffer = PER(state_size=STATE_DIM, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, device=device,
                        eps=PER_EPS, alpha=PER_ALPHA, beta=PER_BETA_START)

    training_history = []
    total_steps = 0
    
    agent.critic1.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
    agent.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
    agent.actor_local.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
    agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    
    agent.actor_local.eval()
    agent.critic1.eval()
    agent.critic2.eval()
    agent.critic1_target.eval()
    agent.critic2_target.eval()
    
    print("start evaluation")
    tot_rew = []
    for episode in range(N_EPISODES):
        state = get_state(env.reset())
        episode_buffer = []
        episode_rewards = []
        for step in range(501):

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = get_state(next_state)

            episode_rewards.append(reward)
            episode_buffer.append((state, action, reward, next_state, done))
            total_steps += 1
            state = next_state
            
            if done:
   
                fund_grid = env.get_fundamental_grid()
                full_grid = build_p4m_structure_simple(fund_grid)
                checked_grid = strictly_enforce_4_connectivity(full_grid)
                is_connected, _ = check_4_connectivity_single_component(checked_grid)

                if is_connected:
                   com_reward, bngp = comsol_reward_function(env, env0)
                   final_reward = 1.5*com_reward if com_reward > 0 else com_reward
                else:
                   final_reward= -20
                   com_reward= np.nan
                   bngp= 0
                total_reward=np.sum(episode_rewards) + final_reward
                tot_rew.append(total_reward)
                break

        training_history.append({
            'episode': episode + 1,
            'connected': is_connected,
            'com_reward': com_reward,            # raw COMSOL reward (nan if not connected)
            'bandgap': np.array(np.real(bngp)),
            'final_reward': final_reward,
            'total_reward': total_reward,
            'fund_grid': fund_grid,              # 20x20 design
        })
        print(f"Episode {episode + 1}/{N_EPISODES} | connected: {is_connected} | "
              f"com_reward: {com_reward} | bandgap: {np.real(bngp)} | total: {total_reward:.2f}")

    np.save('eval_new_material_results.npy', np.array(training_history, dtype=object), allow_pickle=True)
    print("Evaluation complete. Results saved to eval_new_material_results.npy")

 
if __name__ == "__main__":
    main()
