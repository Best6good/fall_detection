#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达点云模拟器模块
生成模拟的4D毫米波雷达点云数据（x, y, z, velocity_z）
优化版：基于人体骨架结构生成点云
"""

import numpy as np
from typing import Tuple, Optional, List
from config import SIMULATOR_CONFIG
from utils import logger


class RadarPointCloudSimulator:
    """
    毫米波雷达点云模拟器（优化版）
    支持4种人体状态：站立(standing)、行走(walking)、摔倒(falling)、倒地(fallen)
    点云分布基于人体骨架结构，更贴近实际雷达探测效果
    """

    def __init__(self):
        """初始化模拟器"""
        self.reset()
        # 人体骨架关键点定义（相对于中心位置的偏移）
        self._init_human_skeleton()

    def _init_human_skeleton(self):
        """初始化人体骨架关键点定义"""
        # 站立状态的人体骨架（基于标准人体比例）
        # 关键点: head, neck, chest, waist, left_shoulder, right_shoulder,
        #         left_elbow, right_elbow, left_hand, right_hand,
        #         left_hip, right_hip, left_knee, right_knee, left_foot, right_foot
        # 所有坐标是绝对坐标（相对于地面）
        self.skeleton_standing = {
            'head': (0.0, 0.0, 1.85),           # 头部高度约1.85m
            'neck': (0.0, 0.0, 1.75),           # 颈部高度约1.75m
            'chest': (0.0, 0.0, 1.65),          # 胸部高度约1.65m
            'waist': (0.0, 0.0, 1.45),          # 腰部高度约1.45m
            'left_shoulder': (-0.18, 0.0, 1.68), # 左肩
            'right_shoulder': (0.18, 0.0, 1.68), # 右肩
            'left_elbow': (-0.30, 0.0, 1.55),    # 左肘
            'right_elbow': (0.30, 0.0, 1.55),    # 右肘
            'left_hand': (-0.40, 0.0, 1.40),     # 左手
            'right_hand': (0.40, 0.0, 1.40),     # 右手
            'left_hip': (-0.12, 0.0, 1.30),      # 左髋
            'right_hip': (0.12, 0.0, 1.30),      # 右髋
            'left_knee': (-0.10, 0.0, 1.00),     # 左膝
            'right_knee': (0.10, 0.0, 1.00),     # 右膝
            'left_foot': (-0.10, 0.0, 0.70),     # 左脚
            'right_foot': (0.10, 0.0, 0.70),     # 右脚
        }

        # 倒地状态的骨架（平躺姿势）
        # 身体水平放置，从站立到倒地时，身体绕Y轴旋转90度
        # 所有坐标是绝对坐标，倒地时高度约0.15m
        self.skeleton_fallen = {
            'head': (0.35, 0.0, 0.15),           # 头部（前方）
            'neck': (0.15, 0.0, 0.15),           # 颈部
            'chest': (0.0, 0.0, 0.15),            # 胸部
            'waist': (-0.20, 0.0, 0.15),         # 腰部
            'left_shoulder': (0.08, -0.18, 0.15), # 左肩（向左）
            'right_shoulder': (0.08, 0.18, 0.15), # 右肩（向右）
            'left_elbow': (-0.05, -0.30, 0.15),   # 左肘
            'right_elbow': (-0.05, 0.30, 0.15),   # 右肘
            'left_hand': (-0.20, -0.40, 0.15),    # 左手
            'right_hand': (-0.20, 0.40, 0.15),    # 右手
            'left_hip': (-0.25, -0.12, 0.15),     # 左髋
            'right_hip': (-0.25, 0.12, 0.15),     # 右髋
            'left_knee': (-0.55, -0.10, 0.15),    # 左膝
            'right_knee': (-0.55, 0.10, 0.15),    # 右膝
            'left_foot': (-0.85, -0.10, 0.15),    # 左脚
            'right_foot': (-0.85, 0.10, 0.15),    # 右脚
        }

        # 各部位的点云密度权重
        self.part_weights = {
            'head': 1.2,      # 头部反射较强
            'neck': 0.5,      # 颈部点云较少
            'chest': 1.5,     # 胸部面积大，点云多
            'waist': 1.2,     # 腰部
            'left_shoulder': 0.8,
            'right_shoulder': 0.8,
            'left_elbow': 0.6,
            'right_elbow': 0.6,
            'left_hand': 0.5,
            'right_hand': 0.5,
            'left_hip': 0.8,
            'right_hip': 0.8,
            'left_knee': 0.6,
            'right_knee': 0.6,
            'left_foot': 0.5,
            'right_foot': 0.5,
        }

        # 各部位的点云分布标准差
        self.part_std_dev = {
            'head': (0.06, 0.06, 0.06),        # 头部较小
            'neck': (0.04, 0.04, 0.05),        # 颈部细长
            'chest': (0.15, 0.08, 0.12),       # 胸部较宽
            'waist': (0.10, 0.06, 0.10),       # 腰部较窄
            'left_shoulder': (0.06, 0.05, 0.06),
            'right_shoulder': (0.06, 0.05, 0.06),
            'left_elbow': (0.05, 0.05, 0.05),
            'right_elbow': (0.05, 0.05, 0.05),
            'left_hand': (0.04, 0.04, 0.04),
            'right_hand': (0.04, 0.04, 0.04),
            'left_hip': (0.08, 0.06, 0.08),
            'right_hip': (0.08, 0.06, 0.08),
            'left_knee': (0.06, 0.06, 0.06),
            'right_knee': (0.06, 0.06, 0.06),
            'left_foot': (0.05, 0.08, 0.04),   # 脚较长
            'right_foot': (0.05, 0.08, 0.04),
        }

    def reset(self) -> None:
        """重置到初始状态"""
        self.state = "standing"  # 当前状态: standing, walking, falling, fallen
        self.position = np.array([0.0, 0.0, 1.6])  # 人体中心位置（站立时约1.6m）
        self.fall_progress = 0.0  # 摔倒进度 0-1
        self.trajectory = []  # 轨迹记录
        self.walk_direction = np.random.uniform(0, 2*np.pi)  # 行走方向
        self.walk_step = 0.0  # 行走步幅相位
        logger.info("雷达模拟器已重置")

    def set_state(self, state: str) -> None:
        """
        设置人体状态
        :param state: 状态名称 (standing/walking/falling/fallen)
        """
        if state in ["standing", "walking", "falling", "fallen"]:
            self.state = state
            if state == "falling":
                self.fall_progress = 0.0
            logger.info(f"人体状态已切换为: {state}")
        else:
            logger.warning(f"无效的状态: {state}")

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state

    def trigger_fall(self) -> None:
        """手动触发摔倒事件"""
        if self.state == "standing" or self.state == "walking":
            self.state = "falling"
            self.fall_progress = 0.0
            logger.info("已触发摔倒事件")

    def _get_noise_scale(self) -> float:
        """获取当前噪声水平"""
        noise = SIMULATOR_CONFIG["noise_level"]
        if isinstance(noise, (list, np.ndarray)):
            noise = float(noise[0]) if len(noise) > 0 else 0.1
        return float(noise)

    def _generate_points_for_body_part(self, base_pos: np.ndarray, part_name: str,
                                       noise: float, velocity_z: float = 0.0) -> np.ndarray:
        """
        为特定身体部位生成点云（优化版，更贴合毫米波雷达特性）
        :param base_pos: 基础位置
        :param part_name: 部位名称
        :param noise: 噪声水平
        :param velocity_z: 垂直速度
        :return: 点云数组
        """
        total_points = SIMULATOR_CONFIG["point_num_per_frame"]
        weight = self.part_weights[part_name]
        std_x, std_y, std_z = self.part_std_dev[part_name]

        # 根据权重分配点数
        total_weight = sum(self.part_weights.values())
        part_points = int(total_points * weight / total_weight)
        if part_points < 1:
            part_points = 1

        # 添加噪声缩放
        noise_factor = 0.5 * noise + 0.5
        std_x *= noise_factor
        std_y *= noise_factor
        std_z *= noise_factor

        # 毫米波雷达点云特性优化：
        # 1. 使用T分布生成点云（尾部更长，更稀疏）
        # 2. 添加边缘效应（某些点偏离中心更远）

        # 生成基础点云（使用T分布近似）
        df = 3 + noise * 2  # 自由度随噪声增加
        x = base_pos[0] + np.random.standard_t(df, part_points) * std_x
        y = base_pos[1] + np.random.standard_t(df, part_points) * std_y
        z = base_pos[2] + np.random.standard_t(df, part_points) * std_z

        # 毫米波雷达的RCS特性：某些部位反射更强
        # 添加稀疏性：随机丢弃一些点（模拟雷达检测不到的区域）
        detection_mask = np.random.random(part_points) > (0.2 * noise)
        x = x[detection_mask]
        y = y[detection_mask]
        z = z[detection_mask]

        if len(x) == 0:
            return np.array([]).reshape(0, 4)

        # 速度场：靠近地面的点速度受阻力影响
        ground_height = 0.05
        height_above_ground = z - ground_height
        resistance_factor = np.exp(-height_above_ground * 2)  # 越接近地面阻力越大

        # 速度噪声
        v_z = velocity_z * (1 - resistance_factor * 0.8) + np.random.normal(0, 0.02, len(x))

        # 摔倒时的旋转效应（身体倾斜）
        if velocity_z < -0.1:  # 正在下落
            rotation_angle = np.clip(abs(velocity_z) * 0.3, 0, 0.5)
            # 绕X轴旋转（前后倾斜）
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            z_new = z * cos_a - (z - ground_height) * sin_a + ground_height
            z = z_new * 0.7 + z * 0.3  # 混合旋转和非旋转

        return np.column_stack((x, y, z, v_z))

    def _get_skeleton_for_state(self) -> dict:
        """获取当前状态对应的骨架"""
        if self.state == "fallen":
            return self.skeleton_fallen
        # 摔倒过程中插值
        elif self.state == "falling":
            return self._interpolate_skeleton(self.fall_progress)
        else:
            return self.skeleton_standing

    def _interpolate_skeleton(self, progress: float) -> dict:
        """
        在站立和倒地骨架之间插值
        :param progress: 摔倒进度 0-1
        :return: 插值后的骨架
        """
        interpolated = {}

        # 应用缓动函数使运动更自然
        t = (1 - np.cos(progress * np.pi)) / 2  # cosine缓动

        for key in self.skeleton_standing.keys():
            standing_pos = np.array(self.skeleton_standing[key])
            fallen_pos = np.array(self.skeleton_fallen[key])

            interpolated[key] = tuple(standing_pos * (1 - t) + fallen_pos * t)

        return interpolated

    def _generate_standing_points(self) -> np.ndarray:
        """生成站立状态的点云（基于人体骨架）"""
        noise = self._get_noise_scale()
        skeleton = self.skeleton_standing
        all_points = []

        for part_name in skeleton.keys():
            # 使用骨架的绝对坐标，加上水平偏移（position的x,y分量）
            skeleton_pos = np.array(skeleton[part_name])
            base_pos = np.array([
                skeleton_pos[0] + self.position[0],
                skeleton_pos[1] + self.position[1],
                skeleton_pos[2]
            ])

            # 生成该部位的点云
            part_points = self._generate_points_for_body_part(base_pos, part_name, noise)
            all_points.append(part_points)

        return np.vstack(all_points) if all_points else np.array([])

    def _generate_walking_points(self) -> np.ndarray:
        """生成行走状态的点云（带有明显步态周期和身体动态）"""
        noise = self._get_noise_scale()
        skeleton = self.skeleton_standing.copy()

        # 行走步频（每秒约1.0步，更自然）
        self.walk_step += 0.0628  # 约1步/秒 (2*pi/100)
        step_length = 0.025 * noise + 0.02  # 步长更明显

        # 沿当前方向移动（累积移动）
        step_progress = np.sin(self.walk_step)
        dx = np.cos(self.walk_direction) * step_length * step_progress
        dy = np.sin(self.walk_direction) * step_length * step_progress
        self.position[0] += dx
        self.position[1] += dy

        # 限制移动范围（更大的活动空间）
        self.position[0] = max(-1.8, min(1.8, self.position[0]))
        self.position[1] = max(-1.8, min(1.8, self.position[1]))

        # 随机改变方向（模拟自然行走）
        if np.random.random() < 0.015:
            self.walk_direction += np.random.uniform(-0.3, 0.3)
            # 限制方向变化，避免急转弯
            if self.walk_direction < 0:
                self.walk_direction += 2 * np.pi
            elif self.walk_direction > 2 * np.pi:
                self.walk_direction -= 2 * np.pi

        # ===== 步态效果增强 =====

        # 手臂摆动（更大幅度）
        arm_swing = np.sin(self.walk_step) * 0.12  # 增大到0.12
        arm_swing_z = np.cos(self.walk_step) * 0.05  # 手臂上下摆动
        skeleton['left_hand'] = (-0.38, arm_swing, 1.35 + arm_swing_z)
        skeleton['right_hand'] = (0.38, -arm_swing, 1.35 - arm_swing_z)
        skeleton['left_elbow'] = (-0.28, arm_swing * 0.6, 1.52 + arm_swing_z * 0.5)
        skeleton['right_elbow'] = (0.28, -arm_swing * 0.6, 1.52 - arm_swing_z * 0.5)

        # 腿部摆动（更大幅度）
        leg_swing = np.sin(self.walk_step + np.pi) * 0.08  # 增大到0.08
        leg_swing_forward = np.cos(self.walk_step) * 0.15  # 前进方向摆动
        skeleton['left_foot'] = (-0.10, leg_swing_forward, 0.65 + leg_swing)
        skeleton['right_foot'] = (0.10, -leg_swing_forward, 0.65 - leg_swing)
        skeleton['left_knee'] = (-0.10, leg_swing_forward * 0.5, 0.88 + leg_swing * 0.5)
        skeleton['right_knee'] = (0.10, -leg_swing_forward * 0.5, 0.88 - leg_swing * 0.5)
        skeleton['left_hip'] = (-0.12, leg_swing_forward * 0.3, 1.25)
        skeleton['right_hip'] = (0.12, -leg_swing_forward * 0.3, 1.25)

        # 身体上下起伏（更明显）
        body_bob = np.sin(self.walk_step) * 0.04  # 增大到0.04
        skeleton['head'] = (0.0, 0.0, 1.88 + body_bob)
        skeleton['neck'] = (0.0, 0.0, 1.78 + body_bob * 0.8)
        skeleton['chest'] = (0.0, 0.0, 1.65 + body_bob * 0.6)
        skeleton['waist'] = (0.0, 0.0, 1.45 + body_bob * 0.4)

        # 肩膀转动（伴随手臂摆动）
        shoulder_rotation = np.sin(self.walk_step) * 0.03
        skeleton['left_shoulder'] = (-0.18, shoulder_rotation, 1.68)
        skeleton['right_shoulder'] = (0.18, -shoulder_rotation, 1.68)

        # 头部轻微晃动
        head_sway_x = np.cos(self.walk_step * 2) * 0.02
        head_sway_y = np.sin(self.walk_step * 3) * 0.015
        skeleton['head'] = (head_sway_x, head_sway_y, skeleton['head'][2])

        all_points = []
        for part_name in skeleton.keys():
            # 使用骨架的绝对坐标，加上水平偏移
            skeleton_pos = np.array(skeleton[part_name])
            base_pos = np.array([
                skeleton_pos[0] + self.position[0],
                skeleton_pos[1] + self.position[1],
                skeleton_pos[2]
            ])

            # 行走时各部位有不同的速度特征
            # 脚部速度最大（着地和抬起），头部速度最小
            if 'foot' in part_name:
                # 脚部：着地时向下，抬起时向上
                velocity_z = np.sin(self.walk_step) * 0.08
            elif 'hand' in part_name:
                # 手部：随手臂摆动有轻微垂直运动
                velocity_z = np.cos(self.walk_step) * 0.04
            elif 'knee' in part_name:
                # 膝盖：随腿部摆动
                velocity_z = np.sin(self.walk_step + np.pi) * 0.05
            else:
                # 躯干和头部：稳定的上下起伏
                velocity_z = np.sin(self.walk_step) * 0.02

            part_points = self._generate_points_for_body_part(base_pos, part_name, noise, velocity_z)
            all_points.append(part_points)

        return np.vstack(all_points) if all_points else np.array([])

    def _generate_falling_points(self) -> np.ndarray:
        """生成摔倒过程的点云（骨架逐渐变化）"""
        noise = self._get_noise_scale()
        # 更慢的摔倒进度（约15帧完成，10fps下约1.5秒）
        self.fall_progress += 0.065  # 从0.08降低到0.065

        # 获取插值后的骨架
        skeleton = self._get_skeleton_for_state()

        # 摔倒速度逐渐增加 - 更真实的加速度
        # 真实摔倒时，开始慢，然后加速，最后受地面阻力减速
        if self.fall_progress < 0.5:
            # 初期：较慢的速度
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * (0.3 + self.fall_progress * 1.4)
        elif self.fall_progress < 0.8:
            # 中期：最快速度
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * 1.2
        else:
            # 末期：接近地面时减速
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * (1.2 - (self.fall_progress - 0.8) * 2)

        all_points = []
        for part_name in skeleton.keys():
            # 使用骨架的绝对坐标，加上水平偏移
            skeleton_pos = np.array(skeleton[part_name])
            base_pos = np.array([
                skeleton_pos[0] + self.position[0],
                skeleton_pos[1] + self.position[1],
                skeleton_pos[2]
            ])

            # 所有点都带有向下的速度
            part_points = self._generate_points_for_body_part(base_pos, part_name, noise, fall_speed)
            all_points.append(part_points)

        # 检查是否已倒地
        if self.fall_progress >= 1.0:
            self.fall_progress = 1.0
            self.state = "fallen"
            self.position[2] = 0.08  # 倒地时的高度
            logger.info("人体已倒地")

        return np.vstack(all_points) if all_points else np.array([])

    def _generate_fallen_points(self) -> np.ndarray:
        """生成倒地状态的点云（平躺姿势）"""
        noise = self._get_noise_scale()
        skeleton = self.skeleton_fallen
        all_points = []

        for part_name in skeleton.keys():
            # 使用骨架的绝对坐标，加上水平偏移
            skeleton_pos = np.array(skeleton[part_name])
            base_pos = np.array([
                skeleton_pos[0] + self.position[0],
                skeleton_pos[1] + self.position[1],
                skeleton_pos[2]
            ])

            part_points = self._generate_points_for_body_part(base_pos, part_name, noise)
            all_points.append(part_points)

        return np.vstack(all_points) if all_points else np.array([])

    def get_next_frame(self) -> np.ndarray:
        """
        获取下一帧点云数据
        :return: 点云数组 (N, 4)，列顺序: x, y, z, velocity_z
        """
        try:
            if self.state == "standing":
                points = self._generate_standing_points()
            elif self.state == "walking":
                points = self._generate_walking_points()
            elif self.state == "falling":
                points = self._generate_falling_points()
            elif self.state == "fallen":
                points = self._generate_fallen_points()
            else:
                points = self._generate_standing_points()

            # 更新轨迹
            self.trajectory.append(self.position.copy())
            # 保持轨迹长度
            max_frames = 5
            if len(self.trajectory) > max_frames:
                self.trajectory = self.trajectory[-max_frames:]

            return points
        except Exception as e:
            logger.error(f"生成点云失败: {e}")
            return np.array([])

    def get_trajectory(self) -> list:
        """获取最近的轨迹记录"""
        return self.trajectory

    def get_skeleton_ground_truth(self) -> np.ndarray:
        """
        获取骨架的ground truth数据
        :return: 骨架关节坐标数组 (16, 3)
        """
        skeleton = self._get_skeleton_for_state()
        skeleton_array = np.zeros((16, 3))
        joint_names = [
            'head', 'neck', 'chest', 'waist',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_hand', 'right_hand',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_foot', 'right_foot'
        ]
        for i, name in enumerate(joint_names):
            if name in skeleton:
                skeleton_array[i] = skeleton[name]
        return skeleton_array

    def get_center_position(self) -> Tuple[float, float, float]:
        """获取当前人体中心位置"""
        return tuple(self.position)