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
N_iters = 250
Rs = -0.04
Nx = 4
Ny = 3


R = np.full((Ny+2, Nx+2), Rs, dtype=np.float32)
R[2, 4] = -1.0
R[3, 4] = 1.0

movable = np.ones((Ny+2, Nx+2), np.bool)
movable[0, :] = False
movable[-1, :] = False
movable[:, 0] = False
movable[:, -1] = False
movable[2, 2] = False

not_terminate = np.copy(movable)
not_terminate[2, 4] = False
not_terminate[3, 4] = False

U = np.full((Ny+2, Nx+2), np.NaN, dtype=np.float32)
U[not_terminate] = 0.0

U_extended = np.copy(U)
U_extended[2, 4] = -1.0
U_extended[3, 4] = 1.0

directions_x = np.zeros((4,3), np.int64)
directions_y = np.zeros((4,3), np.int64)


# y=0, x=1
directions_x[0,0] = 0
directions_y[0,0] = -1

directions_x[0,1] = 1
directions_y[0,1] = 0

directions_x[0,2] = 0
directions_y[0,2] = 1




# y=1, x=0
directions_x[1,0] = 1
directions_y[1,0] = 0

directions_x[1,1] = 0
directions_y[1,1] = 1

directions_x[1,2] = -1
directions_y[1,2] = 0




# y=0, x=-1
directions_x[2,0] = 0
directions_y[2,0] = 1

directions_x[2,1] = -1
directions_y[2,1] = 0

directions_x[2,2] = 0
directions_y[2,2] = -1



# y=-1, x=0
directions_x[3,0] = -1
directions_y[3,0] = 0

directions_x[3,1] = 0
directions_y[3,1] = -1

directions_x[3,2] = 1
directions_y[3,2] = 0

p_vector = np.asarray([p_right, p_straigth, p_left])


not_terminate_y, not_terminate_x = np.where(not_terminate)


for iter_count in range(N_iters):
    for not_terminate_count in range(not_terminate_y.size):
        xt = not_terminate_x[not_terminate_count]
        yt = not_terminate_y[not_terminate_count]
        under_max_vector = np.zeros(4, np.float32)
        for dir_count in range(4):
            dirs_x_t = directions_x[dir_count, :]
            dirs_y_t = directions_y[dir_count, :]
            for triplet_count in range(3):
                dir_x_t = dirs_x_t[triplet_count]
                dir_y_t = dirs_y_t[triplet_count]
                xtt = xt + dir_x_t
                ytt = yt + dir_y_t
                p_tt = p_vector[triplet_count]
                if movable[ytt, xtt]:
                    under_max_vector[dir_count] += U_extended[ytt, xtt] * p_tt
                else:
                    under_max_vector[dir_count] += U_extended[yt, xt] * p_tt
        U[yt, xt] = R[yt, xt] + gamma * np.max(under_max_vector)
        U_extended[yt, xt] = U[yt, xt]

U_cut_fliped = np.flipud(U[1:-1,1:-1])
U_cut_fliped_rounded = np.round(U_cut_fliped, decimals=3)
print(U_cut_fliped_rounded)
            



