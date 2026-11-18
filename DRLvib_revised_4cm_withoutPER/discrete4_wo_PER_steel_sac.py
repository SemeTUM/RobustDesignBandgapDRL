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

from envs.DRLComEnv import VibEnv3D
from envs.WalkingEnv import GridEnv, comsol_reward_function, build_p4m_structure_simple,strictly_enforce_4_connectivity
from utils.utils import check_4_connectivity_single_component
from utils.PER import PER
from src.dsac import SAC

#Trainining main
N_EPISODES = 2000
TOTAL_STEPS = 1e6
BUFFER_SIZE = 500000
BATCH_SIZE = 128
MAX_STEPS_PER_EPISODE = 500
EXPLORATION_STEPS = 500000
STATE_DIM =402 # 2 + 20*20   # position (r, c) + 20x20 fundamental grid walked so far
ACTION_DIM = 4
device = torch.device("cpu")
UPDATES_PER_EPISODE = 5
GRID_SIZE = 40

# PER
PER_ALPHA = 0.6
PER_BETA_START = 0.4   # annealed to 1.0 over training
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
def main():
    agent = SAC(STATE_DIM,  ACTION_DIM,   device=device)
    replay_buffer = PER(state_size=STATE_DIM, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, device=device,
                        eps=PER_EPS, alpha=PER_ALPHA, beta=PER_BETA_START)

    training_history = []
    total_steps = 0

    print("start training")
    for episode in range(N_EPISODES):
        state = get_state(env.reset())
        episode_buffer = []
        episode_rewards = []

        for step in range(1000000):

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = get_state(next_state)

            episode_rewards.append(reward)
            episode_buffer.append((state, action, reward, next_state, done))
            total_steps += 1
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
                    bngp= 0
                break
            state = next_state

        # final (design) reward goes on the last transition only (done=True); earlier steps get it through bootstrapping
        episode_rewards_for_save = []
        for (s, a, r, ns, d) in episode_buffer:
            reward_to_store = r + final_reward if d else r
            replay_buffer.add(s, a, reward_to_store, ns, d)
            episode_rewards_for_save.append(reward_to_store)

        history_entry = {
            'episode': episode + 1,
            'final_bandgap_reward': final_reward,
            'states': np.array([exp[0][:2] for exp in episode_buffer]),   # positions only
            'actions': np.array([exp[1] for exp in episode_buffer]).flatten(),
            'rewards': np.array(episode_rewards_for_save),
            'bandgap': np.array(np.real(bngp)),
        }
        training_history.append(history_entry)

        if len(replay_buffer)> BATCH_SIZE:
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * episode / (N_EPISODES - 1)
#            for _ in range(UPDATES_PER_EPISODE):
            batch, is_weights, sample_idxs = replay_buffer.sample(beta=beta)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(step, batch, gamma=0.99, weights=None, return_td_errors=False)
            print(f"Episode {episode+1}/{N_EPISODES}, Reward: {reward_to_store:.2f}, ploss: {policy_loss}, aloss:{alpha_loss}, berr1:{bellmann_error1}, berr2:{bellmann_error2}")
        print(f" {episode+1}/{N_EPISODES},EPReward: {reward_to_store:.2f}")

        p_loss.append(policy_loss)
        a_loss.append(alpha_loss)
        b_err1.append(bellmann_error1)
        b_err2.append(bellmann_error2)
        cur_alpha.append(float(current_alpha))

        if (episode + 1) % 100 == 0:
            checkpoint = {
                'critic_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict':agent.critic2.state_dict(),
                'critic1_target_state_dict': agent.critic1_target.state_dict(),
                'critic2_target_state_dict': agent.critic2_target.state_dict(),
                'policy_state_dict': agent.actor_local.state_dict(),
                'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'log_alpha': agent.log_alpha.detach().cpu(),
                'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),

            }
            torch.save(checkpoint, f'SAC_DISCRETE_without_PER_checkpoint_ep{episode+1}.pth')
            print(f"Saved checkpoint at episode {episode+1}")

    np.save('SAC_DISCRETE_without_PER_history.npy', np.array(training_history, dtype=object), allow_pickle=True)
    print("Training complete. History saved to Updated_SAC_DISCRETE_without_PER_history.npy")

if __name__ == "__main__":
    main()
