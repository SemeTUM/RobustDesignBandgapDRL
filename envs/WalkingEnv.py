#implementation for non-framed structure where the agent also learns to form connected shapes
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
import sys

sys.path.append("..")
from utils.utils import build_p4m_structure_simple,  strictly_enforce_4_connectivity

class WalkingMan:
    def __init__(self, cube_size):
        self.cube_size = cube_size
        self.half_size = cube_size // 2

    def initial_position(self):
        return (random.randint(0, self.half_size - 1), 0)#(self.half_size - 1, 0)  # lower-left corner

    def step(self, pos, action):
        r, c = pos
        if action == 0: r += 1
        elif action == 1: r -= 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        # Stay inside the fundamental triangle
        if 0 <= r < self.half_size and 0 <= c <= r:
            return (r, c)
        return pos

# ---------------- Environment ----------------
class GridEnv:
    def __init__(self, cube_size):
        self.cube_size = cube_size
        self.half_size = cube_size // 2
        self.agent = WalkingMan(cube_size)
        self.reset()

    def reset(self):
        self.current_pos = self.agent.initial_position()
        self.path = {self.current_pos}
        self.targets = {(i, i) for i in range(self.half_size)}
        self.step_count= 0

        self.done = False
        return self._get_obs()

    def _get_obs(self):
        r, c = self.current_pos
        return np.array([r, c], dtype=np.float32)

    def step(self, action):
        prev_pos = self.current_pos     
        new_pos = self.agent.step(prev_pos, action)
        moved = (new_pos != prev_pos)

        self.current_pos = new_pos

        prev_path_size = len(self.path)
        self.path.add(new_pos)
        path_growth_reward = 0.5 if len(self.path) > prev_path_size else -0.5

        if new_pos[0] in np.arange(1,self.half_size-1) and new_pos[1] == 0:
            start_reward = 0.4
        else:
            start_reward= -0.4

        step_reward = 0.1 if moved else -0.1 


        if new_pos in self.targets:
            add_reward = 0.1
        else:
            add_reward= -0.1

        self.step_count+=1

        total_reward = step_reward + add_reward + path_growth_reward #start_reward
        done = (self.step_count >= 500)

        return self._get_obs(), total_reward, done, {}

    def get_fundamental_grid(self):
        grid = np.ones((self.half_size, self.half_size), dtype=int)
        for (r, c) in self.path:
            if c <= r:
                grid[r, c] = 0
        return grid

    def get_valid_moves(self, pos):
        r, c = pos
        valid = []
        # Action 0: down (r+1)
        if 0 <= r+1 < self.half_size and c <= r+1:
            valid.append(0)
        # Action 1: up (r-1)
        if 0 <= r-1 and c <= r-1:
            valid.append(1)
        # Action 2: left (c-1)
        if 0 <= c-1:
            valid.append(2)
        # Action 3: right (c+1)
        if c+1 <= r:       # must remain inside triangle
            valid.append(3)
        return valid

    def get_final_structure_checked(self):
        fund_grid = self.get_fundamental_grid()
        full_grid = build_p4m_structure_simple(fund_grid)
        checked_grid = strictly_enforce_4_connectivity(full_grid)
        return checked_grid
   
    def conv2com(self):
        final_grid = self.get_final_structure_checked()
        solid_coords = np.argwhere(final_grid == 1)
        N = self.half_size
        mask = (solid_coords[:,0] < N) & (solid_coords[:,1] < N)
        quarter_coords = solid_coords[mask]
        b1 = [f"arr1({r+1},{c+1})" for (r,c) in quarter_coords]
        b2 = [f"arr2({N-r},{c+1})" for (r,c) in quarter_coords]
        b3 = [f"arr3({r+1},{N-c})" for (r,c) in quarter_coords]
        b4 = [f"arr4({N-r},{N-c})" for (r,c) in quarter_coords]
        b_path = np.concatenate([b1,b2,b3,b4])
        return b_path.tolist()

def comsol_reward_function(env, env0):
    path0 = env.conv2com()
    #print(" ".join(path0))
    bandgap = env0.comsolEnv(path0)
    bandgap1, reward0 = env0.getValue(bandgap)    
    comsol_reward = reward0

    return comsol_reward , bandgap1
