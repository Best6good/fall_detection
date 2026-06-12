#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骨架估计模块
从点云数据推断人体骨架结构

包含：
- 多帧融合（融合3-5帧降低误差）
- 基于点云统计的关节位置推断（使用中位数抗离群点）
- 行走状态手臂/腿摆动
- 摔倒状态旋转倾斜（基于fall_progress）
- 骨骼长度约束
- 时序平滑滤波
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from config import HumanState
from filters import OneEuroFilter


# 骨架连接定义
# 关节索引: 0-head, 1-neck, 2-chest, 3-waist,
#          4-left_shoulder, 5-right_shoulder,
#          6-left_elbow, 7-right_elbow,
#          8-left_hand, 9-right_hand,
#          10-left_hip, 11-right_hip,
#          12-left_knee, 13-right_knee,
#          14-left_foot, 15-right_foot
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (2, 4), (2, 5),
    (4, 6), (5, 7), (6, 8), (7, 9),
    (3, 10), (3, 11),
    (10, 12), (11, 13), (12, 14), (13, 15),
]

JOINT_NAMES = [
    'head', 'neck', 'chest', 'waist',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_hand', 'right_hand',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_foot', 'right_foot',
]

# 骨骼长度约束（米），基于 GB/T 10000-2023 中国成年人人体尺寸
BONE_CONSTRAINTS = {
    (0, 1): 0.12,   # head-neck
    (1, 2): 0.15,   # neck-chest
    (2, 3): 0.20,   # chest-waist
    (2, 4): 0.18,   # chest-left_shoulder
    (2, 5): 0.18,   # chest-right_shoulder
    (4, 6): 0.30,   # left_shoulder-left_elbow
    (5, 7): 0.30,   # right_shoulder-right_elbow
    (6, 8): 0.25,   # left_elbow-left_hand
    (7, 9): 0.25,   # right_elbow-right_hand
    (3, 10): 0.15,  # waist-left_hip
    (3, 11): 0.15,  # waist-right_hip
    (10, 12): 0.40, # left_hip-left_knee
    (11, 13): 0.40, # right_hip-right_knee
    (12, 14): 0.35, # left_knee-left_foot
    (13, 15): 0.35, # right_knee-right_foot
}

# 各状态的默认骨架（16×3）
STANDING_SKELETON = np.array([
    [0.0, 0.0, 1.95],   # head
    [0.0, 0.0, 1.82],   # neck
    [0.0, 0.0, 1.70],   # chest
    [0.0, 0.0, 1.50],   # waist
    [-0.18, 0.0, 1.75], # left_shoulder
    [0.18, 0.0, 1.75],  # right_shoulder
    [-0.30, 0.0, 1.60], # left_elbow
    [0.30, 0.0, 1.60],  # right_elbow
    [-0.40, 0.0, 1.45], # left_hand
    [0.40, 0.0, 1.45],  # right_hand
    [-0.12, 0.0, 1.30], # left_hip
    [0.12, 0.0, 1.30],  # right_hip
    [-0.10, 0.0, 1.00], # left_knee
    [0.10, 0.0, 1.00],  # right_knee
    [-0.10, 0.0, 0.75], # left_foot
    [0.10, 0.0, 0.75],  # right_foot
])


def get_default_skeleton(state: str) -> np.ndarray:
    """获取指定状态的默认骨架"""
    if state == HumanState.FALLEN:
        fallen = STANDING_SKELETON.copy()
        fallen[:, [0, 2]] = STANDING_SKELETON[:, [2, 0]] * [1, -1]
        fallen[:, 2] = 0.08 + fallen[:, 2] * 0.1
        return fallen
    return STANDING_SKELETON.copy()


def apply_bone_constraints(skeleton: np.ndarray, elasticity: float = 0.10) -> np.ndarray:
    """
    应用骨骼长度约束
    :param skeleton: (16, 3) 关节坐标
    :param elasticity: 弹性范围（±10%）
    :return: 调整后的骨架
    """
    for (i, j), expected_len in BONE_CONSTRAINTS.items():
        vec = skeleton[j] - skeleton[i]
        current_len = np.linalg.norm(vec)
        if current_len > 0.01:
            ratio = current_len / expected_len
            min_r, max_r = 1.0 - elasticity, 1.0 + elasticity
            if ratio < min_r or ratio > max_r:
                target = expected_len * np.clip(ratio, min_r, max_r)
                skeleton[j] = skeleton[i] + vec * (target / current_len)
    return skeleton


class SkeletonEstimator:
    """
    骨架估计器
    从点云数据推断人体骨架，支持多帧融合和时序平滑
    """

    def __init__(self, buffer_size: int = 5, bone_elasticity: float = 0.05):
        self.buffer_size = buffer_size
        self.bone_elasticity = bone_elasticity
        self.frame_buffer: List[np.ndarray] = []
        self.fused_centroid: Optional[Tuple[float, float, float]] = None
        self.walk_phase: float = 0.0

        # 16关节 × 3坐标 的滤波器
        # min_cutoff=0.3: 静止时更强平滑，减少抖动
        # beta=0.1: 降低速度系数，减少静态时的跳变
        self.filters = [
            [OneEuroFilter(min_cutoff=0.3, beta=0.1, d_cutoff=1.0) for _ in range(3)]
            for _ in range(16)
        ]

    def reset(self):
        self.frame_buffer.clear()
        self.fused_centroid = None
        self.walk_phase = 0.0
        for joint_filters in self.filters:
            for f in joint_filters:
                f.reset()

    def estimate(self, points: np.ndarray, state: str,
                 fall_progress: float = 0.0, walk_phase: float = 0.0) -> np.ndarray:
        """
        从点云估计骨架
        :param points: (N, 4/5) 点云
        :param state: 当前状态
        :param fall_progress: 摔倒进度 (0-1)
        :param walk_phase: 行走相位 (0-2π)
        :return: (16, 3) 关节坐标
        """
        if points.size == 0:
            return get_default_skeleton(state)

        # 多帧融合
        self.frame_buffer.append(points.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        fused = np.vstack(self.frame_buffer) if len(self.frame_buffer) > 1 else points

        # 统计分析 — 使用中位数代替均值，对离群点更鲁棒
        x_median, y_median, z_median = np.median(fused[:, :3], axis=0)
        z_min = np.min(fused[:, 2])
        z_max = np.max(fused[:, 2])
        z_95 = np.percentile(fused[:, 2], 95)

        self.fused_centroid = (float(x_median), float(y_median), float(z_median))

        skeleton = get_default_skeleton(state)

        # 基于状态的骨架推断
        if state == HumanState.STANDING:
            skeleton = self._estimate_standing(skeleton, z_95)
        elif state == HumanState.WALKING:
            skeleton = self._estimate_walking(skeleton, z_95, walk_phase)
        elif state == HumanState.FALLING:
            skeleton = self._estimate_falling(skeleton, fall_progress, z_min, z_max)
        else:  # FALLEN
            skeleton[:, 2] = z_min + 0.10
            skeleton[0, 2] = z_min + 0.12

        # 水平位置对齐到点云中心（使用中位数）
        skeleton[:, 0] += x_median - np.mean(skeleton[2:4, 0])
        skeleton[:, 1] += y_median - np.mean(skeleton[2:4, 1])

        # 时序平滑
        for i in range(16):
            for j in range(3):
                skeleton[i, j] = self.filters[i][j](skeleton[i, j])

        # 骨骼约束
        skeleton = apply_bone_constraints(skeleton, self.bone_elasticity)

        return skeleton

    def _estimate_standing(self, skeleton: np.ndarray, z_95: float) -> np.ndarray:
        """站立状态：直立骨架"""
        head_h = z_95 - 0.02
        neck_h = head_h - 0.12
        chest_h = head_h - 0.20
        waist_h = chest_h - 0.18
        hip_h = waist_h - 0.15
        knee_h = hip_h - 0.32
        foot_h = knee_h - 0.28

        skeleton[0, 2] = head_h
        skeleton[1, 2] = neck_h
        skeleton[2, 2] = chest_h
        skeleton[3, 2] = waist_h
        skeleton[4, 2] = chest_h + 0.03
        skeleton[5, 2] = chest_h + 0.03
        skeleton[6, 2] = chest_h - 0.08
        skeleton[7, 2] = chest_h - 0.08
        skeleton[8, 2] = chest_h - 0.18
        skeleton[9, 2] = chest_h - 0.18
        skeleton[10, 2] = hip_h
        skeleton[11, 2] = hip_h
        skeleton[12, 2] = knee_h
        skeleton[13, 2] = knee_h
        skeleton[14, 2] = foot_h
        skeleton[15, 2] = foot_h
        return skeleton

    def _estimate_walking(self, skeleton: np.ndarray, z_95: float,
                          walk_phase: float) -> np.ndarray:
        """
        行走状态：在站立骨架基础上添加手臂/腿摆动
        参考: Victor C. Chen "Micro-Doppler Effect in Radar" 2011
        手臂与腿交替摆动（对侧模式），步频1.5-2.5Hz
        """
        # 先用站立骨架作为基础
        skeleton = self._estimate_standing(skeleton, z_95)

        # 手臂摆动（Y方向前后摆动）
        # 对侧模式：左臂与右腿同相，右臂与左腿同相
        arm_swing = 0.12 * np.sin(walk_phase)
        skeleton[6, 1] += arm_swing         # left_elbow 前后
        skeleton[7, 1] -= arm_swing         # right_elbow 前后（反相）
        skeleton[8, 1] += arm_swing * 0.8   # left_hand
        skeleton[9, 1] -= arm_swing * 0.8   # right_hand

        # 腿部摆动（Y方向前后摆动，与手臂反相）
        leg_swing = 0.08 * np.sin(walk_phase + np.pi)
        skeleton[12, 1] += leg_swing        # left_knee
        skeleton[13, 1] -= leg_swing        # right_knee
        skeleton[14, 1] += leg_swing * 0.6  # left_foot
        skeleton[15, 1] -= leg_swing * 0.6  # right_foot

        # 身体上下微动（Z方向，振幅约2-3cm）
        body_bob = 0.025 * np.sin(walk_phase * 2)  # 两倍频率（一步两次起伏）
        skeleton[:, 2] += body_bob

        return skeleton

    def _estimate_falling(self, skeleton: np.ndarray, fall_progress: float,
                          z_min: float, z_max: float) -> np.ndarray:
        """
        摔倒状态：使用fall_progress进行平滑旋转倾斜
        从直立逐渐倾斜到水平（绕X轴旋转，以腰部为中心）

        fall_progress: 0=直立, 1=完全倒地
        旋转角度: 0° → 90°
        """
        # 先用站立骨架作为基础
        z_95 = np.percentile([z_max], 95)  # 简化：用z_max近似
        skeleton = self._estimate_standing(skeleton, z_max)

        # 缓动函数：先慢后快再慢（余弦插值）
        t = (1 - np.cos(fall_progress * np.pi)) / 2

        # 旋转角度（0°→90°）
        angle = t * np.pi / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 以腰部为旋转中心
        pivot_z = skeleton[3, 2]  # waist Z
        pivot_y = skeleton[3, 1]  # waist Y

        # 绕X轴旋转（向前倒：Y和Z变化）
        for i in range(16):
            dz = skeleton[i, 2] - pivot_z
            dy = skeleton[i, 1] - pivot_y
            # 旋转：Z' = pivot + dz*cos - dy*sin, Y' = pivot + dy*cos + dz*sin
            skeleton[i, 2] = pivot_z + dz * cos_a - dy * sin_a
            skeleton[i, 1] = pivot_y + dy * cos_a + dz * sin_a

        # 随着progress增加，整体高度下降（模拟身体下沉）
        height_range = z_max - z_min
        height_drop = fall_progress * height_range * 0.7
        skeleton[:, 2] -= height_drop

        # 确保关节不低于地面
        skeleton[:, 2] = np.maximum(skeleton[:, 2], z_min + 0.05)

        return skeleton
