#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达点云模拟器模块
生成模拟的4D毫米波雷达点云数据（x, y, z, velocity_z）
物理版：基于几何模型和雷达物理特性生成点云
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from config import SIMULATOR_CONFIG, RADAR_PHYSICS_CONFIG, HumanState, get_noise_level
from utils import logger


class GeometricModel:
    """
    人体几何模型
    使用椭球体和圆柱体组合表示人体
    """
    
    def __init__(self):
        """初始化几何模型"""
        self.config = RADAR_PHYSICS_CONFIG["body_geometry"]
        self._init_body_parts()
    
    def _init_body_parts(self):
        """初始化身体部位几何参数"""
        # 躯干：椭球体
        self.torso = {
            "type": "ellipsoid",
            "semi_axes": self.config["torso"]["semi_axes"],  # (a, b, c)
            "center": np.array([0.0, 0.0, 1.35]),  # 站立时中心高度
        }
        
        # 头部：球体
        self.head = {
            "type": "sphere",
            "radius": self.config["head"]["radius"],
            "center": np.array([0.0, 0.0, 1.75]),  # 站立时头部中心
        }
        
        # 四肢：圆柱体（简化表示）
        self.limbs = {
            "left_upper_arm": {
                "type": "cylinder",
                "radius": self.config["upper_arm"]["radius"],
                "length": self.config["upper_arm"]["length"],
                "start": np.array([-0.20, 0.0, 1.55]),  # 肩关节
                "end": np.array([-0.45, 0.0, 1.40]),  # 肘关节
            },
            "right_upper_arm": {
                "type": "cylinder",
                "radius": self.config["upper_arm"]["radius"],
                "length": self.config["upper_arm"]["length"],
                "start": np.array([0.20, 0.0, 1.55]),
                "end": np.array([0.45, 0.0, 1.40]),
            },
            "left_lower_arm": {
                "type": "cylinder",
                "radius": self.config["lower_arm"]["radius"],
                "length": self.config["lower_arm"]["length"],
                "start": np.array([-0.45, 0.0, 1.40]),  # 肘关节
                "end": np.array([-0.65, 0.0, 1.25]),  # 腕关节
            },
            "right_lower_arm": {
                "type": "cylinder",
                "radius": self.config["lower_arm"]["radius"],
                "length": self.config["lower_arm"]["length"],
                "start": np.array([0.45, 0.0, 1.40]),
                "end": np.array([0.65, 0.0, 1.25]),
            },
            "left_upper_leg": {
                "type": "cylinder",
                "radius": self.config["upper_leg"]["radius"],
                "length": self.config["upper_leg"]["length"],
                "start": np.array([-0.12, 0.0, 1.10]),  # 髋关节
                "end": np.array([-0.12, 0.0, 0.70]),  # 膝关节
            },
            "right_upper_leg": {
                "type": "cylinder",
                "radius": self.config["upper_leg"]["radius"],
                "length": self.config["upper_leg"]["length"],
                "start": np.array([0.12, 0.0, 1.10]),
                "end": np.array([0.12, 0.0, 0.70]),
            },
            "left_lower_leg": {
                "type": "cylinder",
                "radius": self.config["lower_leg"]["radius"],
                "length": self.config["lower_leg"]["length"],
                "start": np.array([-0.12, 0.0, 0.70]),  # 膝关节
                "end": np.array([-0.12, 0.0, 0.35]),  # 踝关节
            },
            "right_lower_leg": {
                "type": "cylinder",
                "radius": self.config["lower_leg"]["radius"],
                "length": self.config["lower_leg"]["length"],
                "start": np.array([0.12, 0.0, 0.70]),
                "end": np.array([0.12, 0.0, 0.35]),
            },
        }
    
    def update_pose(self, state: HumanState, fall_progress: float = 0.0, walk_phase: float = 0.0):
        """
        根据状态更新身体部位位置
        :param state: 人体状态
        :param fall_progress: 摔倒进度 (0-1)
        :param walk_phase: 行走相位 (0-2π)
        """
        if state == HumanState.STANDING:
            self._update_standing_pose()
        elif state == HumanState.WALKING:
            self._update_walking_pose(walk_phase)
        elif state == HumanState.FALLING:
            self._update_falling_pose(fall_progress)
        elif state == HumanState.FALLEN:
            self._update_fallen_pose()
    
    def _update_standing_pose(self):
        """更新站立姿势"""
        # 重置为标准站立姿势
        self.torso["center"] = np.array([0.0, 0.0, 1.35])
        self.head["center"] = np.array([0.0, 0.0, 1.75])
        # 四肢位置已在初始化时设置
    
    def _update_walking_pose(self, phase: float):
        """
        更新行走姿势
        :param phase: 行走相位 (0-2π)
        """
        # 身体上下起伏
        body_bob = np.sin(phase) * 0.03
        self.torso["center"][2] = 1.35 + body_bob
        self.head["center"][2] = 1.75 + body_bob
        
        # 手臂摆动
        arm_swing = np.sin(phase) * 0.15
        self.limbs["left_upper_arm"]["end"][1] = arm_swing
        self.limbs["right_upper_arm"]["end"][1] = -arm_swing
        self.limbs["left_lower_arm"]["end"][1] = arm_swing * 0.8
        self.limbs["right_lower_arm"]["end"][1] = -arm_swing * 0.8
        
        # 腿部摆动
        leg_swing = np.sin(phase + np.pi) * 0.10
        self.limbs["left_upper_leg"]["end"][1] = leg_swing
        self.limbs["right_upper_leg"]["end"][1] = -leg_swing
        self.limbs["left_lower_leg"]["end"][1] = leg_swing * 0.6
        self.limbs["right_lower_leg"]["end"][1] = -leg_swing * 0.6
    
    def _update_falling_pose(self, progress: float):
        """
        更新摔倒姿势
        :param progress: 摔倒进度 (0-1)
        """
        # 缓动函数
        t = (1 - np.cos(progress * np.pi)) / 2
        
        # 身体倾斜（绕X轴旋转）
        angle = t * np.pi / 2  # 最大倾斜90度
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 更新躯干位置
        self.torso["center"] = np.array([
            0.0,
            sin_a * 0.25,  # 向前倾斜
            1.35 * cos_a + 0.15  # 高度降低
        ])
        
        # 更新头部位置
        self.head["center"] = np.array([
            0.0,
            sin_a * 0.4,
            1.75 * cos_a + 0.15
        ])
        
        # 更新四肢位置（简化处理）
        # 这里可以根据需要添加更复杂的运动学模型
    
    def _update_fallen_pose(self):
        """更新倒地姿势（平躺）"""
        # 身体平躺在地面上
        self.torso["center"] = np.array([0.0, 0.0, 0.15])
        self.head["center"] = np.array([0.0, 0.35, 0.15])
        
        # 四肢平放
        self.limbs["left_upper_arm"]["start"] = np.array([-0.20, 0.08, 0.15])
        self.limbs["left_upper_arm"]["end"] = np.array([-0.45, 0.08, 0.15])
        self.limbs["right_upper_arm"]["start"] = np.array([0.20, 0.08, 0.15])
        self.limbs["right_upper_arm"]["end"] = np.array([0.45, 0.08, 0.15])
        # ... 其他四肢类似
    
    def get_skeleton_points(self) -> Dict[str, np.ndarray]:
        """
        获取骨架关键点坐标
        :return: 关键点字典 {名称: 坐标}
        """
        skeleton = {}
        
        # 头部关键点
        skeleton["head"] = self.head["center"]
        
        # 颈部（头部和躯干的中间点）
        skeleton["neck"] = (self.head["center"] + self.torso["center"]) / 2
        
        # 胸部（躯干中心）
        skeleton["chest"] = self.torso["center"]
        
        # 腰部（躯干下方）
        skeleton["waist"] = self.torso["center"] - np.array([0.0, 0.0, 0.2])
        
        # 肩部关键点
        skeleton["left_shoulder"] = self.limbs["left_upper_arm"]["start"]
        skeleton["right_shoulder"] = self.limbs["right_upper_arm"]["start"]
        
        # 肘部关键点
        skeleton["left_elbow"] = self.limbs["left_upper_arm"]["end"]
        skeleton["right_elbow"] = self.limbs["right_upper_arm"]["end"]
        
        # 手部关键点
        skeleton["left_hand"] = self.limbs["left_lower_arm"]["end"]
        skeleton["right_hand"] = self.limbs["right_lower_arm"]["end"]
        
        # 髋部关键点
        skeleton["left_hip"] = self.limbs["left_upper_leg"]["start"]
        skeleton["right_hip"] = self.limbs["right_upper_leg"]["start"]
        
        # 膝部关键点
        skeleton["left_knee"] = self.limbs["left_upper_leg"]["end"]
        skeleton["right_knee"] = self.limbs["right_upper_leg"]["end"]
        
        # 脚部关键点
        skeleton["left_foot"] = self.limbs["left_lower_leg"]["end"]
        skeleton["right_foot"] = self.limbs["right_lower_leg"]["end"]
        
        return skeleton
    
    def generate_surface_points(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成身体表面点云
        :param num_points: 总点数
        :return: (点云坐标, RCS值)
        """
        all_points = []
        all_rcs = []
        
        # 分配点数到各部位（基于RCS权重）
        rcs_config = RADAR_PHYSICS_CONFIG["rcs"]
        total_rcs = (rcs_config["torso"] + rcs_config["head"] + 
                    rcs_config["upper_arm"] * 2 + rcs_config["lower_arm"] * 2 +
                    rcs_config["upper_leg"] * 2 + rcs_config["lower_leg"] * 2)
        
        # 躯干点云
        torso_points = int(num_points * rcs_config["torso"] / total_rcs)
        if torso_points > 0:
            points, rcs = self._generate_ellipsoid_points(
                self.torso["center"], self.torso["semi_axes"], torso_points, rcs_config["torso"]
            )
            all_points.append(points)
            all_rcs.append(rcs)
        
        # 头部点云
        head_points = int(num_points * rcs_config["head"] / total_rcs)
        if head_points > 0:
            points, rcs = self._generate_sphere_points(
                self.head["center"], self.head["radius"], head_points, rcs_config["head"]
            )
            all_points.append(points)
            all_rcs.append(rcs)
        
        # 四肢点云
        for limb_name, limb_config in self.limbs.items():
            limb_type = "upper_arm" if "upper_arm" in limb_name else "lower_arm" if "lower_arm" in limb_name else "upper_leg" if "upper_leg" in limb_name else "lower_leg"
            limb_points = int(num_points * rcs_config[limb_type] / total_rcs)
            if limb_points > 0:
                points, rcs = self._generate_cylinder_points(
                    limb_config["start"], limb_config["end"], limb_config["radius"],
                    limb_points, rcs_config[limb_type]
                )
                all_points.append(points)
                all_rcs.append(rcs)
        
        if not all_points:
            return np.array([]).reshape(0, 3), np.array([])
        
        return np.vstack(all_points), np.concatenate(all_rcs)
    
    def _generate_ellipsoid_points(self, center: np.ndarray, semi_axes: Tuple[float, float, float],
                                  num_points: int, base_rcs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成椭球体表面点云
        :param center: 中心坐标
        :param semi_axes: 半轴长度 (a, b, c)
        :param num_points: 点数
        :param base_rcs: 基础RCS值
        :return: (点云坐标, RCS值)
        """
        a, b, c = semi_axes
        
        # 使用球面坐标生成均匀分布的点
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        cos_theta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(cos_theta)
        
        # 椭球体参数方程
        x = center[0] + a * np.sin(theta) * np.cos(phi)
        y = center[1] + b * np.sin(theta) * np.sin(phi)
        z = center[2] + c * np.cos(theta)
        
        points = np.column_stack((x, y, z))
        
        # RCS随入射角变化（简化模型）
        # 假设雷达在原点(0,0,0)，计算入射角
        incident_angles = np.arccos(np.abs(np.cos(theta)))  # 简化计算
        rcs_values = base_rcs * np.cos(incident_angles) ** 2  # 余弦平方衰减
        rcs_values = np.maximum(rcs_values, 0.01)  # 最小RCS
        
        return points, rcs_values
    
    def _generate_sphere_points(self, center: np.ndarray, radius: float,
                               num_points: int, base_rcs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成球体表面点云
        :param center: 中心坐标
        :param radius: 半径
        :param num_points: 点数
        :param base_rcs: 基础RCS值
        :return: (点云坐标, RCS值)
        """
        return self._generate_ellipsoid_points(center, (radius, radius, radius), num_points, base_rcs)
    
    def _generate_cylinder_points(self, start: np.ndarray, end: np.ndarray, radius: float,
                                 num_points: int, base_rcs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成圆柱体表面点云
        :param start: 起点坐标
        :param end: 终点坐标
        :param radius: 半径
        :param num_points: 点数
        :param base_rcs: 基础RCS值
        :return: (点云坐标, RCS值)
        """
        # 圆柱体方向向量
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        
        # 生成圆柱体表面点
        t = np.random.uniform(0, 1, num_points)  # 沿圆柱体长度
        theta = np.random.uniform(0, 2 * np.pi, num_points)  # 圆周角度
        
        # 计算垂直于方向向量的两个基向量
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # 计算点坐标
        points = []
        for i in range(num_points):
            # 沿圆柱体长度的位置
            pos_along = start + t[i] * length * direction
            
            # 圆周上的偏移
            offset = radius * (np.cos(theta[i]) * perp1 + np.sin(theta[i]) * perp2)
            
            point = pos_along + offset
            points.append(point)
        
        points = np.array(points)
        
        # RCS随入射角变化
        # 计算表面法向量与雷达方向的夹角
        radar_direction = -points / np.linalg.norm(points, axis=1, keepdims=True)
        surface_normals = points - (start + t.reshape(-1, 1) * length * direction)
        surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1, keepdims=True)
        
        cos_incident = np.abs(np.sum(radar_direction * surface_normals, axis=1))
        rcs_values = base_rcs * cos_incident ** 2
        rcs_values = np.maximum(rcs_values, 0.01)
        
        return points, rcs_values


class RadarPhysicsModel:
    """
    雷达物理模型
    模拟毫米波雷达的信号处理特性
    """
    
    def __init__(self):
        """初始化雷达物理模型"""
        self.config = RADAR_PHYSICS_CONFIG
        self.noise_config = self.config["noise"]
        self.env_config = self.config["environment"]
    
    def add_measurement_noise(self, points: np.ndarray, rcs_values: np.ndarray) -> np.ndarray:
        """
        添加测量噪声
        :param points: 真实坐标 (N, 3)
        :param rcs_values: RCS值 (N,)
        :return: 带噪声的点云 (N, 4) [x, y, z, velocity_z]
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 4)
        
        num_points = len(points)
        
        # 计算距离（用于距离相关噪声）
        distances = np.linalg.norm(points, axis=1)
        
        # 距离噪声（随距离增大）
        range_noise = np.random.normal(0, self.noise_config["range_std"], num_points)
        range_noise *= (1 + distances / self.config["max_range"])  # 距离衰减
        
        # 角度噪声（转换为坐标噪声）
        angle_noise = np.random.normal(0, self.noise_config["angle_std"], (num_points, 2))
        
        # 将角度噪声转换为坐标噪声
        # 假设雷达在原点，计算方位角和俯仰角
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        elevation = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))
        
        # 添加角度噪声
        azimuth += angle_noise[:, 0]
        elevation += angle_noise[:, 1]
        
        # 重新计算坐标（带角度噪声）
        r = distances + range_noise
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        
        # 速度噪声（垂直速度）
        velocity_noise = np.random.normal(0, self.noise_config["velocity_std"], num_points)
        
        # 基础垂直速度（可以基于RCS和距离计算）
        # 这里简化处理，假设大部分点静止，部分点有微小速度
        base_velocity = np.random.normal(0, 0.1, num_points)  # 基础速度噪声
        
        # 组合点云数据
        noisy_points = np.column_stack((x, y, z, base_velocity + velocity_noise))
        
        return noisy_points
    
    def apply_rcs_filter(self, points: np.ndarray, rcs_values: np.ndarray, 
                        snr_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于RCS的检测过滤
        :param points: 点云坐标 (N, 3)
        :param rcs_values: RCS值 (N,)
        :param snr_threshold: 信噪比阈值
        :return: (过滤后的点云, 过滤后的RCS)
        """
        if len(points) == 0:
            return points, rcs_values
        
        # 计算信噪比（简化模型）
        # SNR ∝ RCS / (距离^4 * 噪声功率)
        distances = np.linalg.norm(points, axis=1)
        noise_power = self.noise_config["thermal_noise_power"]
        
        snr = rcs_values / (distances ** 4 * noise_power + 1e-10)
        
        # 基于SNR的检测概率
        detection_prob = 1 - np.exp(-snr / snr_threshold)
        
        # 随机检测
        detected = np.random.random(len(points)) < detection_prob
        
        return points[detected], rcs_values[detected]
    
    def add_clutter(self, points: np.ndarray, room_size: Tuple[float, float, float]) -> np.ndarray:
        """
        添加环境杂波
        :param points: 原始点云 (N, 4)
        :param room_size: 房间尺寸 (长, 宽, 高)
        :return: 添加杂波后的点云
        """
        if len(points) == 0:
            return points
        
        # 计算杂波点数（基于房间面积和杂波密度）
        floor_area = room_size[0] * room_size[1]
        clutter_density = self.noise_config["clutter_density"]
        num_clutter = int(floor_area * clutter_density)
        
        if num_clutter == 0:
            return points
        
        # 生成杂波点（随机分布在房间内）
        clutter_x = np.random.uniform(-room_size[0]/2, room_size[0]/2, num_clutter)
        clutter_y = np.random.uniform(-room_size[1]/2, room_size[1]/2, num_clutter)
        clutter_z = np.random.uniform(0, room_size[2], num_clutter)
        clutter_v = np.random.normal(0, 0.05, num_clutter)  # 杂波速度接近0
        
        clutter_points = np.column_stack((clutter_x, clutter_y, clutter_z, clutter_v))
        
        # 合并点云
        combined = np.vstack((points, clutter_points))
        
        return combined
    
    def simulate_multipath(self, points: np.ndarray, room_size: Tuple[float, float, float]) -> np.ndarray:
        """
        模拟多径反射
        :param points: 原始点云 (N, 4)
        :param room_size: 房间尺寸
        :return: 添加多径反射后的点云
        """
        if len(points) == 0 or not self.env_config["multipath_enabled"]:
            return points
        
        multipath_points = []
        
        # 地面反射（最常见）
        floor_reflectivity = self.env_config["floor_reflectivity"]
        if np.random.random() < floor_reflectivity:
            # 地面反射点：z坐标变为负值（镜像）
            floor_points = points.copy()
            floor_points[:, 2] = -floor_points[:, 2]  # 镜像反射
            floor_points[:, 3] *= 0.5  # 反射后速度减半
            multipath_points.append(floor_points)
        
        # 墙壁反射（较弱）
        wall_reflectivity = self.env_config["wall_reflectivity"]
        if np.random.random() < wall_reflectivity * 0.3:  # 墙壁反射概率较低
            # 随机选择一面墙壁
            wall_side = np.random.choice(['left', 'right', 'front', 'back'])
            wall_points = points.copy()
            
            if wall_side == 'left':
                wall_points[:, 0] = -room_size[0] - wall_points[:, 0]  # 镜像
            elif wall_side == 'right':
                wall_points[:, 0] = room_size[0] - wall_points[:, 0]
            elif wall_side == 'front':
                wall_points[:, 1] = -room_size[1] - wall_points[:, 1]
            elif wall_side == 'back':
                wall_points[:, 1] = room_size[1] - wall_points[:, 1]
            
            wall_points[:, 3] *= 0.3  # 墙壁反射速度衰减更多
            multipath_points.append(wall_points)
        
        if not multipath_points:
            return points
        
        # 合并所有点云
        all_points = [points] + multipath_points
        return np.vstack(all_points)


class RadarPointCloudSimulator:
    """
    毫米波雷达点云模拟器（物理版）
    支持4种人体状态：站立(standing)、行走(walking)、摔倒(falling)、倒地(fallen)
    基于几何模型和雷达物理特性生成点云
    """

    def __init__(self):
        """初始化模拟器"""
        self.reset()
        # 初始化几何模型和物理模型
        self.geometric_model = GeometricModel()
        self.physics_model = RadarPhysicsModel()
        logger.info("物理毫米波雷达模拟器已初始化")

    def reset(self) -> None:
        """重置到初始状态"""
        self.state = HumanState.STANDING  # 当前状态
        self.position = np.array([0.0, 0.0, 1.6])  # 人体中心位置（站立时约1.6m）
        self.fall_progress = 0.0  # 摔倒进度 0-1
        self.trajectory = []  # 轨迹记录
        self.walk_direction = np.random.uniform(0, 2*np.pi)  # 行走方向
        self.walk_step = 0.0  # 行走步幅相位
        self.walk_phase = 0.0  # 行走相位
        logger.info("物理毫米波雷达模拟器已重置")

    def set_state(self, state: HumanState) -> None:
        """
        设置人体状态
        :param state: 目标状态
        """
        if state in HumanState:
            self.state = state
            if state == HumanState.FALLING:
                self.fall_progress = 0.0
            logger.info(f"人体状态已切换为: {state.value}")

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state

    def trigger_fall(self) -> None:
        """手动触发摔倒事件"""
        if self.state in (HumanState.STANDING, HumanState.WALKING):
            self.state = HumanState.FALLING
            self.fall_progress = 0.0
            logger.info("已触发摔倒事件")

    def _get_noise_scale(self) -> float:
        """获取当前噪声水平"""
        return float(get_noise_level())

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
        if self.state == HumanState.FALLEN:
            return self.skeleton_fallen
        elif self.state == HumanState.FALLING:
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
            self.state = HumanState.FALLEN
            self.position[2] = 0.08  # 倒地时的高度
            logger.info("人体已倒地")

        return np.vstack(all_points) if all_points else np.array([])

    def _generate_fallen_points(self) -> np.ndarray:
        """
        生成倒地状态的点云（平躺姿势）
        改进：倒地时点云更稀疏，高度更符合真实情况
        """
        noise = self._get_noise_scale()
        skeleton = self.skeleton_fallen
        all_points = []
        
        # 倒地时点云稀疏度系数（倒地后反射面积变化，点更稀疏）
        fallen_point_ratio = 0.6  # 只生成60%的点
        
        for part_name in skeleton.keys():
            skeleton_pos = np.array(skeleton[part_name])
            base_pos = np.array([
                skeleton_pos[0] + self.position[0],
                skeleton_pos[1] + self.position[1],
                skeleton_pos[2]  # 保持原始高度（已在skeleton_fallen中定义）
            ])
            
            # 生成该部位的点云
            part_points = self._generate_points_for_body_part(
                base_pos, part_name, noise * 1.3  # 倒地时噪声略大
            )
            
            # 倒地时减少点数（模拟反射面积变化）
            if len(part_points) > 5:
                n_keep = max(3, int(len(part_points) * fallen_point_ratio))
                indices = np.random.choice(len(part_points), n_keep, replace=False)
                part_points = part_points[indices]
            
            all_points.append(part_points)

        return np.vstack(all_points) if all_points else np.array([])

    def get_next_frame(self) -> np.ndarray:
        """
        获取下一帧点云数据
        :return: 点云数组 (N, 4)，列顺序: x, y, z, velocity_z
        """
        try:
            # 更新几何模型姿势
            self.geometric_model.update_pose(self.state, self.fall_progress, self.walk_phase)
            
            # 生成表面点云
            num_points = SIMULATOR_CONFIG["point_num_per_frame"]
            surface_points, rcs_values = self.geometric_model.generate_surface_points(num_points)
            
            if len(surface_points) == 0:
                return np.array([]).reshape(0, 4)
            
            # 添加位置偏移（人体中心位置）
            surface_points[:, 0] += self.position[0]
            surface_points[:, 1] += self.position[1]
            
            # 应用RCS检测过滤
            filtered_points, filtered_rcs = self.physics_model.apply_rcs_filter(
                surface_points, rcs_values
            )
            
            if len(filtered_points) == 0:
                return np.array([]).reshape(0, 4)
            
            # 添加测量噪声
            noisy_points = self.physics_model.add_measurement_noise(filtered_points, filtered_rcs)
            
            # 添加环境杂波
            room_size = RADAR_PHYSICS_CONFIG["environment"]["room_size"]
            noisy_points = self.physics_model.add_clutter(noisy_points, room_size)
            
            # 模拟多径反射
            noisy_points = self.physics_model.simulate_multipath(noisy_points, room_size)
            
            # 更新行走状态
            if self.state == HumanState.WALKING:
                self._update_walking_state()
            
            # 更新摔倒进度
            if self.state == HumanState.FALLING:
                self._update_falling_state()
            
            # 更新轨迹
            self.trajectory.append(self.position.copy())
            # 保持轨迹长度
            max_frames = 5
            if len(self.trajectory) > max_frames:
                self.trajectory = self.trajectory[-max_frames:]

            return noisy_points
        except Exception as e:
            logger.error(f"生成点云失败: {e}")
            return np.array([]).reshape(0, 4)

    def _update_walking_state(self):
        """更新行走状态"""
        # 更新行走相位
        self.walk_phase += 0.1  # 行走频率
        if self.walk_phase > 2 * np.pi:
            self.walk_phase -= 2 * np.pi
        
        # 计算行走位移
        step_length = 0.02  # 步长
        dx = np.cos(self.walk_direction) * step_length * np.sin(self.walk_phase)
        dy = np.sin(self.walk_direction) * step_length * np.sin(self.walk_phase)
        
        # 更新位置
        self.position[0] += dx
        self.position[1] += dy
        
        # 限制移动范围
        self.position[0] = np.clip(self.position[0], -1.8, 1.8)
        self.position[1] = np.clip(self.position[1], -1.8, 1.8)
        
        # 随机改变方向
        if np.random.random() < 0.01:
            self.walk_direction += np.random.uniform(-0.3, 0.3)
    
    def _update_falling_state(self):
        """更新摔倒状态"""
        # 摔倒进度更新（约15帧完成摔倒）
        self.fall_progress += 0.065
        
        # 摔倒时位置变化
        if self.fall_progress < 0.5:
            # 初期：较慢的速度
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * (0.3 + self.fall_progress * 1.4)
        elif self.fall_progress < 0.8:
            # 中期：最快速度
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * 1.2
        else:
            # 末期：接近地面时减速
            fall_speed = SIMULATOR_CONFIG["fall_speed"] * (1.2 - (self.fall_progress - 0.8) * 2)
        
        # 更新高度
        self.position[2] += fall_speed * 0.1  # 0.1秒帧间隔
        
        # 检查是否倒地
        if self.fall_progress >= 1.0:
            self.fall_progress = 1.0
            self.state = HumanState.FALLEN
            self.position[2] = 0.08  # 倒地时的高度
            logger.info("人体已倒地")
    
    def get_trajectory(self) -> list:
        """获取最近的轨迹记录"""
        return self.trajectory

    def get_skeleton_ground_truth(self) -> np.ndarray:
        """
        获取骨架的ground truth数据
        :return: 骨架关节坐标数组 (16, 3)
        """
        # 使用几何模型的关键点
        skeleton = self.geometric_model.get_skeleton_points()
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