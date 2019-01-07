# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:25:49 2019

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2




p_right = 0.1
p_left = 0.1
p_straigth = 0.8
gamma = 1.0
N_iters = 20
Rs = -0.04
Nx = 4
Ny = 3

fps = 0.5
fln_video_out = './values_iterations_animation.mp4'

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


    

def redraw_figure(U, optimal_dir_all, iter_count):
    global directions_x
    global directions_y
    #plt.close('all')
    ax = plt.gca()
    U_cut = U[1:-1,1:-1]
    plt.imshow(U_cut)
    plt.colorbar()
    plt.clim(0.0,1.0)
    ax.invert_yaxis()
    
    optimal_dir_all_cut = optimal_dir_all[1:-1,1:-1]
    x_text_shift = -0.2
    y_text_shift = -0.3
    arrow_length = 0.3
    head_width = 0.05
    head_length = 0.1
    for y in range(U_cut.shape[0]):
        for x in range(U_cut.shape[1]):
            U_cut_t = U_cut[y, x]
            if not np.isnan(U_cut_t):
                U_cut_t_rounded = np.round(U_cut_t, decimals=3)
                plt.text(x + x_text_shift, y + y_text_shift, str(U_cut_t_rounded))
                optimal_dir_all_cut_t = optimal_dir_all_cut[y, x]
                dir_x_t = directions_x[optimal_dir_all_cut_t, 1]
                dir_y_t = directions_y[optimal_dir_all_cut_t, 1]
                plt.arrow(x, y, dir_x_t * arrow_length, dir_y_t * arrow_length, head_width=head_width, head_length=head_length, color='k')
    
    xticks = np.arange(1, U_cut.shape[1]+1)
    ax.set_xticks(xticks-1)
    ax.set_xticklabels(xticks)
    yticks = np.arange(1, U_cut.shape[0]+1)
    ax.set_yticks(yticks-1)
    ax.set_yticklabels(yticks)
    
    rect = patches.Rectangle((0.5, 0.5), 1.0, 1.0, linewidth=1, edgecolor='k', facecolor='k')
    ax.add_patch(rect)
    
    plt.text(2.9, 1.9, '+1')
    rect2 = patches.Rectangle((2.8, 1.85), 0.4, 0.23, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect2)
    
    plt.text(2.9, 1.9-1, '-1')
    rect2 = patches.Rectangle((2.8, 1.85-1), 0.4, 0.23, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect2)
    
    for x in range(5):
        ax.axvline(x-0.5, linestyle='-', color='k')
    for y in range(4):
        ax.axhline(y-0.5, linestyle='-', color='k')
    
    plt.title('iteration: ' + str(iter_count))
        
    
    
    



#import sys
#sys.exit()

for iter_count in range(N_iters):
    optimal_dir_all = np.zeros(U.shape, np.int64)
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
        optimal_dir = np.argmax(under_max_vector)
        optimal_dir_all[yt, xt] = optimal_dir
        U[yt, xt] = R[yt, xt] + gamma * under_max_vector[optimal_dir]
        U_extended[yt, xt] = U[yt, xt]
        
    redraw_figure(U, optimal_dir_all, iter_count)
    plt.pause(0.5)
    
    # Now we can save it to a numpy array.
    data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    
    plt.close('all')
    
    if iter_count == 0:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fln_video_out, fourcc, fps, (img.shape[1], img.shape[0]))
    
    img_bgr = img[:, :, ::-1]
    out.write(img_bgr)

U_cut_fliped = np.flipud(U[1:-1,1:-1])
U_cut_fliped_rounded = np.round(U_cut_fliped, decimals=3)
print(U_cut_fliped_rounded)
#redraw_figure(U, optimal_dir_all, iter_count)
out.release()

    
            



