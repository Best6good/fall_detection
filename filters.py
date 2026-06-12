#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号滤波模块
包含时序平滑滤波器，用于骨架估计和点云处理
"""

import math


class OneEuroFilter:
    """
    1€ Filter - 自适应低通滤波器
    用于实时噪声滤波，在低速时减少抖动，高速时减少延迟

    参考论文: Casiez, G., Roussel, N. and Vogel, D. (2012).
    1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
    CHI '12, pp. 2527-2530.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.7, d_cutoff: float = 1.0):
        """
        :param min_cutoff: 最小截止频率(Hz)，越低越平滑但延迟越大
        :param beta: 速度系数，越大高速时延迟越小
        :param d_cutoff: 导数滤波的截止频率
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = float(x)
            return float(x)

        dx = float(x) - self.x_prev

        r_d = 2 * math.pi * self.d_cutoff
        alpha_d = r_d / (r_d + 1)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        r = 2 * math.pi * cutoff
        alpha = r / (r + 1)
        x_hat = alpha * float(x) + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
