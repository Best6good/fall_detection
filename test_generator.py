#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成器
生成用于评估的模拟测试序列

支持：
- 站立→行走→摔倒→倒地 完整序列
- 可配置噪声水平和序列参数
- 批量生成测试场景
"""

import numpy as np
from typing import List, Tuple
from config import HumanState


class TestSequenceGenerator:
    """
    测试序列生成器
    
    生成包含状态转换的点云序列，用于评估检测算法
    
    使用方法：
        gen = TestSequenceGenerator()
        sequence = gen.generate_fall_sequence(noise_level=0.1)
        for state, points in sequence:
            is_fall, info = detector.detect(...)
    """

    def __init__(self, frame_rate: int = 10, points_per_frame: int = 50):
        self.frame_rate = frame_rate
        self.points_per_frame = points_per_frame

    def generate_standing_sequence(self, num_frames: int = 30,
                                    noise_level: float = 0.1) -> List[Tuple[str, np.ndarray]]:
        """生成站立序列（应无报警）"""
        return self._generate_static_state(HumanState.STANDING, num_frames, noise_level,
                                            center_z=1.35, height_range=(0.8, 1.9))

    def generate_walking_sequence(self, num_frames: int = 50,
                                   noise_level: float = 0.1) -> List[Tuple[str, np.ndarray]]:
        """生成行走序列（应无报警）
        微多普勒特征: 躯干0.8-1.5m/s, 腿尖4-6m/s, 臂摆±2-4m/s
        参考: Victor C. Chen "Micro-Doppler Effect in Radar" 2011
        """
        frames = []
        step_freq = 2.0  # 步频2Hz
        for i in range(num_frames):
            phase = i * 2 * np.pi * step_freq / self.frame_rate
            dx = 0.02 * np.sin(phase)
            center_x = dx * i * 0.1
            # 行走时躯干垂直速度约0.1-0.3 m/s（上下起伏分量）
            torso_vz = 0.15 * np.cos(phase)
            points = self._make_body_points(
                center=(center_x, 0.0, 1.35),
                height_range=(0.6, 1.9),
                noise_level=noise_level,
                velocity_mean=torso_vz,
            )
            frames.append((HumanState.WALKING, points))
        return frames

    def generate_fall_sequence(self, noise_level: float = 0.1,
                                stand_frames: int = 20,
                                walk_frames: int = 20,
                                fall_frames: int = 15,
                                fallen_frames: int = 20) -> List[Tuple[str, np.ndarray]]:
        """
        生成完整摔倒序列：站立→行走→摔倒→倒地
        """
        frames = []
        frames.extend(self._generate_static_state(HumanState.STANDING, stand_frames,
                                                   noise_level, 1.35, (0.8, 1.9)))

        # 行走
        for i in range(walk_frames):
            phase = i * 0.1
            cx = 0.02 * np.sin(phase) * i * 0.05
            points = self._make_body_points(
                center=(cx, 0.0, 1.35), height_range=(0.6, 1.9),
                noise_level=noise_level, velocity_mean=0.03 * np.sin(phase),
            )
            frames.append((HumanState.WALKING, points))

        # 摔倒（三阶段模型，参考UP-Fall dataset）
        # 峰值速度1.5-3.0m/s，撞击后速度中断
        peak_velocity = 2.5  # m/s
        for i in range(fall_frames):
            t = i / fall_frames
            height_center = 1.35 * (1 - t) + 0.15 * t
            height_max = 1.9 * (1 - t) + 0.3 * t
            # 三阶段速度
            if t < 0.5:
                velocity = -peak_velocity * (0.3 + t * 1.4)
            elif t < 0.7:
                velocity = -peak_velocity
            else:
                # 撞击减速
                impact_t = (t - 0.7) / 0.3
                velocity = -peak_velocity * (1.0 - impact_t * 0.85)
            points = self._make_body_points(
                center=(0.0, 0.0, height_center),
                height_range=(0.05, height_max),
                noise_level=noise_level * (1 + t * 0.3),
                velocity_mean=velocity,
            )
            frames.append((HumanState.FALLING, points))

        # 倒地
        frames.extend(self._generate_static_state(HumanState.FALLEN, fallen_frames,
                                                   noise_level * 1.3, 0.15, (0.05, 0.3)))
        return frames

    def _generate_static_state(self, state: str, num_frames: int, noise_level: float,
                                center_z: float, height_range: Tuple[float, float]
                                ) -> List[Tuple[str, np.ndarray]]:
        frames = []
        for _ in range(num_frames):
            points = self._make_body_points(
                center=(0.0, 0.0, center_z),
                height_range=height_range,
                noise_level=noise_level,
                velocity_mean=0.0,
            )
            frames.append((state, points))
        return frames

    def _make_body_points(self, center: Tuple[float, float, float],
                           height_range: Tuple[float, float],
                           noise_level: float,
                           velocity_mean: float = 0.0) -> np.ndarray:
        """
        生成模拟人体点云
        使用真实雷达的身体部位分布权重（躯干50%，腿23%，臂20%，头7%）
        参考: van Dorp & Groen 2003, Victor C. Chen 2006

        :param center: (x, y, z) 中心
        :param height_range: (min_z, max_z) 高度范围
        :param noise_level: 噪声水平
        :param velocity_mean: 平均垂直速度
        :return: (N, 5) [x, y, z, velocity_z, intensity]
        """
        n = self.points_per_frame
        cx, cy, cz = center

        # 身体部位分布权重（真实雷达数据）
        # 躯干50%，头部7%，双臂20%，双腿23%
        points = []

        # 躯干（50%）- 主散射体，RCS最大
        n_torso = int(n * 0.50)
        torso_z = np.random.uniform(cz - 0.3, cz + 0.3, n_torso)
        torso_x = np.random.normal(cx, 0.12, n_torso)
        torso_y = np.random.normal(cy, 0.08, n_torso)
        points.append(np.column_stack([torso_x, torso_y, torso_z]))

        # 头部（7%）- 小目标，RCS较低
        n_head = max(1, int(n * 0.07))
        head_z = np.random.uniform(height_range[1] - 0.15, height_range[1], n_head)
        head_x = np.random.normal(cx, 0.08, n_head)
        head_y = np.random.normal(cy, 0.08, n_head)
        points.append(np.column_stack([head_x, head_y, head_z]))

        # 手臂（20%）- 细长目标，RCS低
        n_arms = int(n * 0.20)
        arm_z = np.random.uniform(cz - 0.4, cz + 0.1, n_arms)
        arm_x = np.random.normal(cx, 0.30, n_arms)  # 手臂横向分布更宽
        arm_y = np.random.normal(cy, 0.15, n_arms)
        points.append(np.column_stack([arm_x, arm_y, arm_z]))

        # 腿部（23%）- 中等RCS
        n_legs = int(n * 0.23)
        leg_z = np.random.uniform(height_range[0], cz - 0.2, n_legs)
        leg_x = np.random.normal(cx, 0.15, n_legs)
        leg_y = np.random.normal(cy, 0.10, n_legs)
        points.append(np.column_stack([leg_x, leg_y, leg_z]))

        # 杂波（剩余点）
        n_clutter = n - n_torso - n_head - n_arms - n_legs
        if n_clutter > 0:
            clutter_x = np.random.uniform(-1.5, 1.5, n_clutter)
            clutter_y = np.random.uniform(-1.5, 1.5, n_clutter)
            clutter_z = np.random.uniform(0.05, 1.8, n_clutter)
            points.append(np.column_stack([clutter_x, clutter_y, clutter_z]))

        all_points = np.vstack(points)

        # 添加噪声（受污染高斯模型：85%高斯 + 15%离群点）
        n_pts = len(all_points)
        is_outlier = np.random.random(n_pts) < 0.15
        noise = np.zeros_like(all_points)
        # Inlier: 标准高斯
        n_inlier = (~is_outlier).sum()
        if n_inlier > 0:
            noise[~is_outlier] = np.random.normal(0, noise_level * 0.05, (n_inlier, 3))
        # Outlier: Cauchy（重尾）
        if is_outlier.any():
            noise[is_outlier] = np.random.standard_cauchy((is_outlier.sum(), 3)) * noise_level * 0.2
        all_points += noise

        # 高度限制
        all_points[:, 2] = np.clip(all_points[:, 2], 0.0, 2.0)

        # 速度列（受污染高斯）
        velocity = np.random.normal(velocity_mean, 0.10, n_pts)
        if is_outlier.any():
            velocity[is_outlier] = np.random.uniform(-0.5, 0.5, is_outlier.sum())

        # 强度列（基于RCS和距离，躯干最强）
        # 躯干点强度最高，四肢较低
        intensity = np.random.uniform(0.3, 0.8, n_pts)
        intensity[:n_torso] = np.random.uniform(0.6, 1.0, n_torso)  # 躯干
        if n_head > 0:
            intensity[n_torso:n_torso+n_head] = np.random.uniform(0.2, 0.5, n_head)  # 头
        # 离群点强度更低
        if is_outlier.any():
            intensity[is_outlier] *= np.random.uniform(0.1, 0.4, is_outlier.sum())
        intensity = np.clip(intensity + np.random.normal(0, 0.03, n_pts), 0.0, 1.0)

        return np.column_stack([all_points, velocity, intensity])
