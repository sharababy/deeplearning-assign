#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 g <g@ABCL>
#
# Distributed under terms of the MIT license.

import numpy as np

par1 = np.load('./autoencoder1.npy')
[W_e_1, b_1, W_d_1, c_1] = par1


def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5*(fx-y)**2
    return err


par2 = np.load('./autoencoder2.npy')
[W_e_2, b_2, W_d_2, c_2] = par2
__import__('pdb').set_trace()
