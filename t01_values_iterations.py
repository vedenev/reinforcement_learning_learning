# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:25:49 2019

@author: vedenev
"""

import numpy as np

# 6  7  8
# 4     5   
# 0  1  2  3

p_right = 0.1
p_left = 0.1
p_straigth = 0.8
gamma = 1.0
N_iters = 50
Rs = -0.04

U = np.zeros(9, np.float32)

for iter_count in range(N_iters):
    U0_variants = np.asarray([p_straigth
            ], dtype=np.float32)
    U[0] = Rs + gamma* np.max()


