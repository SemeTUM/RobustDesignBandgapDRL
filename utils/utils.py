import numpy as np
import sys, os
from scipy.ndimage import label, generate_binary_structure

# Symmetry Functions
def build_p4m_structure_simple(fund_region):
    half_size = fund_region.shape[0]
    QLL = np.ones((half_size, half_size), dtype=int)#zeros
    QLL[:] = fund_region[:]
    for r in range(half_size):
        for c in range(r):
            QLL[c, r] = fund_region[r, c]
    TL = QLL; TR = np.fliplr(QLL)
    BL = np.flipud(QLL); BR = np.flipud(np.fliplr(QLL))
    return np.block([[TL, TR], [BL, BR]])

def strictly_enforce_4_connectivity(grid):
    clean_grid = grid.copy()
    rows, cols = clean_grid.shape
    for r in range(rows-1):
        for c in range(cols-1):
            block = clean_grid[r:r+2, c:c+2]
            if block.sum() == 2:
                if block[0,0]==1 and block[1,1]==1:
                    clean_grid[r,c]=0; clean_grid[r+1,c+1]=0
                elif block[0,1]==1 and block[1,0]==1:
                    clean_grid[r,c+1]=0; clean_grid[r+1,c]=0
    return clean_grid

def check_4_connectivity_single_component(grid):  
    s_4 = generate_binary_structure(2, 1)  
    solid_mask = np.where(grid == 0, 1, 0) 
    labeled_solid, num_solid_components = label(solid_mask, structure=s_4)
    if solid_mask.sum() == 0:
        return False, 0
    is_single_component = (num_solid_components == 1)  
    #print("is_singlecomponent", is_single_component)  
    return is_single_component, num_solid_components
    

def get_fundamental_grid(half_size, path):
    grid = np.ones((half_size, half_size), dtype=int)
    for (r, c) in path:
        if c <= r:
            grid[r, c] = 0
    return grid
