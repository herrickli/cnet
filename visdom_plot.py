import math

import numpy as np
import visdom
import torch

vis = visdom.Visdom(env='faster-rcnn')

def draw_loss_cruve(loss_name, i_iter, value):
    for i in range(len(loss_name)):
        vis.line(np.array([value[i]/40.]), np.array([i_iter]), win=loss_name[i], update='append', opts={'title':loss_name[i]})