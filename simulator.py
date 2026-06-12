#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达点云模拟器模块
生成模拟的5D毫米波雷达点云数据 [x, y, z, velocity_z, intensity]

基于文献的物理模型：
- 椭球体RCS公式: RCS = π·a²·b²·c² / (a²·sin²θ·cos²φ + b²·sin²θ·sin²φ + c²·cos²θ)²
  参考: ekurtgl/FMCW-MIMO-Radar-Simulation, rcsellipsoid.m
- 人体RCS @77GHz: 躯干 -3~+3 dBsm, 头部 -15~-8 dBsm, 四肢 -18~-3 dBsm
  参考: Ahn et al. IEEE Access 2020, Herschlein & Hasch EuRAD 2008
- 每帧点云数: 10-100点 (TI IWR1443/IWR6843 post-CFAR)
  参考: TI SWRA587 People Counting, RadHAR dataset
- 点云分布: 躯干40-60%, 腿15-25%, 臂15-25%, 头3-10%
  参考: van Dorp & Groen 2003, Victor C. Chen 2006
- 微多普勒: 行走躯干0.8-1.5m/s, 腿尖4-6m/s, 臂摆±2-4m/s
  参考: Victor C. Chen "Micro-Doppler Effect in Radar" 2011
- 噪声: 受污染高斯模型, 15%离群点 (Cauchy分布)
  参考: Schumann et al. IEEE ITSC 2019
- 摔倒: 峰值1.5-3.0m/s, 撞击后速度中断>2s
  参考: UP-Fall dataset, Wang et al. IEEE Sensors 2022
"""

import numpy as np
from typing import Tuple, Dict
from config import SIMULATOR_CONFIG, RADAR_PHYSICS_CONFIG, HumanState, get_noise_level
from utils import logger


def ellipsoid_rcs(a: float, b: float, c: float, theta: float, phi: float) -> float:
    """
    计算椭球体的雷达截面积(RCS)
    公式: RCS = π·a²·b²·c² / (a²·sin²θ·cos²φ + b²·sin²θ·sin²φ + c²·cos²θ)²

    参考: ekurtgl/FMCW-MIMO-Radar-Simulation, rcsellipsoid.m

    :param a: X轴半轴长 (m)
    :param b: Y轴半轴长 (m)
    :param c: Z轴半轴长 (m)
    :param theta: 入射角极角 (rad), 从Z轴测量
    :param phi: 入射角方位角 (rad)
    :return: RCS值 (m²)
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    denom = (a * sin_t * cos_p) ** 2 + (b * sin_t * sin_p) ** 2 + (c * cos_t) ** 2
    if denom < 1e-12:
        return np.pi * a * b * c  # 正面入射时的最大RCS

    numerator = np.pi * (a * b * c) ** 2
    return numerator / (denom ** 2)


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
        """
        初始化身体部位几何参数（身体局部坐标系）
        原点在身体中心（约腰部），Z轴向上
        self.position 提供世界坐标偏移

        站立时身体局部Z坐标范围: [-1.0, +0.6]（脚到头）
        世界坐标 = 身体局部 + self.position
        """
        # 身体中心参考点（腰部）在局部坐标系中为原点
        # 站立时绝对世界坐标: torso=1.35m, head=1.75m, feet=0.35m
        # 局部坐标: torso=0.0, head=+0.4, feet=-1.0
        self.torso = {
            "type": "ellipsoid",
            "semi_axes": self.config["torso"]["semi_axes"],
            "center": np.array([0.0, 0.0, 0.0]),  # 局部：身体中心
        }
        self.head = {
            "type": "sphere",
            "radius": self.config["head"]["radius"],
            "center": np.array([0.0, 0.0, 0.40]),  # 局部：头在身体中心上方0.4m
        }
        self.limbs = {
            "left_upper_arm": {
                "type": "cylinder", "radius": self.config["upper_arm"]["radius"],
                "length": self.config["upper_arm"]["length"],
                "start": np.array([-0.20, 0.0, 0.20]),   # 肩关节
                "end": np.array([-0.45, 0.0, 0.05]),      # 肘关节
            },
            "right_upper_arm": {
                "type": "cylinder", "radius": self.config["upper_arm"]["radius"],
                "length": self.config["upper_arm"]["length"],
                "start": np.array([0.20, 0.0, 0.20]),
                "end": np.array([0.45, 0.0, 0.05]),
            },
            "left_lower_arm": {
                "type": "cylinder", "radius": self.config["lower_arm"]["radius"],
                "length": self.config["lower_arm"]["length"],
                "start": np.array([-0.45, 0.0, 0.05]),    # 肘关节
                "end": np.array([-0.65, 0.0, -0.10]),     # 腕关节
            },
            "right_lower_arm": {
                "type": "cylinder", "radius": self.config["lower_arm"]["radius"],
                "length": self.config["lower_arm"]["length"],
                "start": np.array([0.45, 0.0, 0.05]),
                "end": np.array([0.65, 0.0, -0.10]),
            },
            "left_upper_leg": {
                "type": "cylinder", "radius": self.config["upper_leg"]["radius"],
                "length": self.config["upper_leg"]["length"],
                "start": np.array([-0.12, 0.0, -0.25]),   # 髋关节
                "end": np.array([-0.12, 0.0, -0.65]),     # 膝关节
            },
            "right_upper_leg": {
                "type": "cylinder", "radius": self.config["upper_leg"]["radius"],
                "length": self.config["upper_leg"]["length"],
                "start": np.array([0.12, 0.0, -0.25]),
                "end": np.array([0.12, 0.0, -0.65]),
            },
            "left_lower_leg": {
                "type": "cylinder", "radius": self.config["lower_leg"]["radius"],
                "length": self.config["lower_leg"]["length"],
                "start": np.array([-0.12, 0.0, -0.65]),   # 膝关节
                "end": np.array([-0.12, 0.0, -1.00]),     # 踝关节
            },
            "right_lower_leg": {
                "type": "cylinder", "radius": self.config["lower_leg"]["radius"],
                "length": self.config["lower_leg"]["length"],
                "start": np.array([0.12, 0.0, -0.65]),
                "end": np.array([0.12, 0.0, -1.00]),
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
        """更新站立姿势（身体局部坐标）"""
        self.torso["center"] = np.array([0.0, 0.0, 0.0])
        self.head["center"] = np.array([0.0, 0.0, 0.40])
        self._reset_limbs_standing()

    def _reset_limbs_standing(self):
        """重置四肢到标准站立姿势（身体局部坐标）"""
        self.limbs["left_upper_arm"]["start"] = np.array([-0.20, 0.0, 0.20])
        self.limbs["left_upper_arm"]["end"] = np.array([-0.45, 0.0, 0.05])
        self.limbs["right_upper_arm"]["start"] = np.array([0.20, 0.0, 0.20])
        self.limbs["right_upper_arm"]["end"] = np.array([0.45, 0.0, 0.05])
        self.limbs["left_lower_arm"]["start"] = np.array([-0.45, 0.0, 0.05])
        self.limbs["left_lower_arm"]["end"] = np.array([-0.65, 0.0, -0.10])
        self.limbs["right_lower_arm"]["start"] = np.array([0.45, 0.0, 0.05])
        self.limbs["right_lower_arm"]["end"] = np.array([0.65, 0.0, -0.10])
        self.limbs["left_upper_leg"]["start"] = np.array([-0.12, 0.0, -0.25])
        self.limbs["left_upper_leg"]["end"] = np.array([-0.12, 0.0, -0.65])
        self.limbs["right_upper_leg"]["start"] = np.array([0.12, 0.0, -0.25])
        self.limbs["right_upper_leg"]["end"] = np.array([0.12, 0.0, -0.65])
        self.limbs["left_lower_leg"]["start"] = np.array([-0.12, 0.0, -0.65])
        self.limbs["left_lower_leg"]["end"] = np.array([-0.12, 0.0, -1.00])
        self.limbs["right_lower_leg"]["start"] = np.array([0.12, 0.0, -0.65])
        self.limbs["right_lower_leg"]["end"] = np.array([0.12, 0.0, -1.00])

    def _update_walking_pose(self, phase: float):
        """
        更新行走姿势（身体局部坐标）
        :param phase: 行走相位 (0-2π)
        """
        # 先重置为站立姿势
        self._update_standing_pose()
        # 身体上下起伏（振幅±3cm）
        body_bob = np.sin(phase) * 0.03
        self.torso["center"][2] = 0.0 + body_bob
        self.head["center"][2] = 0.40 + body_bob
        # 手臂摆动（Y方向前后）
        arm_swing = np.sin(phase) * 0.15
        self.limbs["left_upper_arm"]["end"][1] += arm_swing
        self.limbs["right_upper_arm"]["end"][1] -= arm_swing
        self.limbs["left_lower_arm"]["end"][1] += arm_swing * 0.8
        self.limbs["right_lower_arm"]["end"][1] -= arm_swing * 0.8
        # 腿部摆动（与手臂反相）
        leg_swing = np.sin(phase + np.pi) * 0.10
        self.limbs["left_upper_leg"]["end"][1] += leg_swing
        self.limbs["right_upper_leg"]["end"][1] -= leg_swing
        self.limbs["left_lower_leg"]["end"][1] += leg_swing * 0.6
        self.limbs["right_lower_leg"]["end"][1] -= leg_swing * 0.6

    def _update_falling_pose(self, progress: float):
        """
        更新摔倒姿势（身体局部坐标）
        以腰部（原点）为支点，绕X轴旋转0°→90°，同时整体下降
        :param progress: 摔倒进度 (0-1)
        """
        # 缓动函数（先慢后快再慢）
        t = (1 - np.cos(progress * np.pi)) / 2
        angle = t * np.pi / 2  # 0° → 90°
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 先用站立姿势作为基础
        self._update_standing_pose()

        # 以腰部（局部原点）为支点，绕X轴旋转所有部位
        # 旋转公式: Z' = Z*cos - Y*sin, Y' = Y*cos + Z*sin
        def rotate_x(point):
            y, z = point[1], point[2]
            new_y = y * cos_a + z * sin_a
            new_z = z * cos_a - y * sin_a
            return np.array([point[0], new_y, new_z])

        self.torso["center"] = rotate_x(self.torso["center"])
        self.head["center"] = rotate_x(self.head["center"])
        for limb in self.limbs.values():
            limb["start"] = rotate_x(limb["start"])
            limb["end"] = rotate_x(limb["end"])

        # 整体下降：随着摔倒进度，身体中心下降
        # 从站立到平躺，中心下降约0.6m（重心从1.35m降到0.75m附近）
        height_drop = t * 0.60
        self.torso["center"][2] -= height_drop
        self.head["center"][2] -= height_drop
        for limb in self.limbs.values():
            limb["start"][2] -= height_drop
            limb["end"][2] -= height_drop

    def _update_fallen_pose(self):
        """更新倒地姿势（身体局部坐标，平躺在地面上）"""
        # 平躺：所有部位Z坐标接近-self.position[2]（世界坐标≈0.15m）
        # 局部坐标中，身体扁平化：Z≈-1.2（这样+position[2]=1.35后世界≈0.15）
        flat_z = -1.20
        self.torso["center"] = np.array([0.0, 0.0, flat_z])
        self.head["center"] = np.array([0.0, 0.35, flat_z])
        self.limbs["left_upper_arm"]["start"] = np.array([-0.20, 0.08, flat_z])
        self.limbs["left_upper_arm"]["end"] = np.array([-0.45, 0.08, flat_z])
        self.limbs["right_upper_arm"]["start"] = np.array([0.20, 0.08, flat_z])
        self.limbs["right_upper_arm"]["end"] = np.array([0.45, 0.08, flat_z])
        self.limbs["left_lower_arm"]["start"] = np.array([-0.45, 0.08, flat_z])
        self.limbs["left_lower_arm"]["end"] = np.array([-0.65, 0.08, flat_z])
        self.limbs["right_lower_arm"]["start"] = np.array([0.45, 0.08, flat_z])
        self.limbs["right_lower_arm"]["end"] = np.array([0.65, 0.08, flat_z])
        self.limbs["left_upper_leg"]["start"] = np.array([-0.12, 0.0, flat_z])
        self.limbs["left_upper_leg"]["end"] = np.array([-0.12, -0.40, flat_z])
        self.limbs["right_upper_leg"]["start"] = np.array([0.12, 0.0, flat_z])
        self.limbs["right_upper_leg"]["end"] = np.array([0.12, -0.40, flat_z])
        self.limbs["left_lower_leg"]["start"] = np.array([-0.12, -0.40, flat_z])
        self.limbs["left_lower_leg"]["end"] = np.array([-0.12, -0.75, flat_z])
        self.limbs["right_lower_leg"]["start"] = np.array([0.12, -0.40, flat_z])
        self.limbs["right_lower_leg"]["end"] = np.array([0.12, -0.75, flat_z])
    
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
    
    def generate_surface_points(self, num_points: int, state: str = "standing",
                                 fall_speed: float = 0.0, walk_phase: float = 0.0
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成身体表面点云
        使用真实雷达的身体部位分布权重（躯干50%，腿23%，臂20%，头7%）
        而非按RCS权重分配（那样躯干占比会更高）

        :param num_points: 总点数
        :param state: 当前状态
        :param fall_speed: 摔倒速度 (m/s)
        :param walk_phase: 行走相位
        :return: (点云坐标[N,3], RCS值[N], 垂直速度[N])
        """
        all_points = []
        all_rcs = []
        all_vz = []

        # 使用真实雷达的身体部位分布权重
        # 参考: van Dorp & Groen 2003, Victor C. Chen 2006
        weights = RADAR_PHYSICS_CONFIG["body_point_weights"]
        rcs_config = RADAR_PHYSICS_CONFIG["rcs"]

        # 各部位的垂直速度（基于微多普勒模型）
        part_velocities = self._compute_part_velocities(state, fall_speed, walk_phase)

        # 躯干点云（50%）
        torso_points = max(3, round(num_points * weights["torso"]))
        points, rcs = self._generate_ellipsoid_points(
            self.torso["center"], self.torso["semi_axes"], torso_points, rcs_config["torso"]
        )
        all_points.append(points)
        all_rcs.append(rcs)
        all_vz.append(np.full(torso_points, part_velocities["torso"]))

        # 头部点云（7%）
        head_points = max(2, round(num_points * weights["head"]))
        points, rcs = self._generate_sphere_points(
            self.head["center"], self.head["radius"], head_points, rcs_config["head"]
        )
        all_points.append(points)
        all_rcs.append(rcs)
        all_vz.append(np.full(head_points, part_velocities["head"]))

        # 手臂点云（20%，左右各半）
        arm_weight_per_limb = weights["arms"] / 4  # 4个手臂段
        for limb_name in ["left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm"]:
            limb_config = self.limbs[limb_name]
            if "upper_arm" in limb_name:
                limb_type = "upper_arm"
            else:
                limb_type = "lower_arm"
            limb_pts = max(2, round(num_points * arm_weight_per_limb))
            points, rcs = self._generate_cylinder_points(
                limb_config["start"], limb_config["end"], limb_config["radius"],
                limb_pts, rcs_config[limb_type]
            )
            all_points.append(points)
            all_rcs.append(rcs)
            vz = part_velocities.get(limb_type, 0.0)
            all_vz.append(np.full(limb_pts, vz))

        # 腿部点云（23%，左右各半）
        leg_weight_per_limb = weights["legs"] / 4  # 4个腿部段
        for limb_name in ["left_upper_leg", "right_upper_leg", "left_lower_leg", "right_lower_leg"]:
            limb_config = self.limbs[limb_name]
            if "upper_leg" in limb_name:
                limb_type = "upper_leg"
            else:
                limb_type = "lower_leg"
            limb_pts = max(2, round(num_points * leg_weight_per_limb))
            points, rcs = self._generate_cylinder_points(
                limb_config["start"], limb_config["end"], limb_config["radius"],
                limb_pts, rcs_config[limb_type]
            )
            all_points.append(points)
            all_rcs.append(rcs)
            # 微多普勒速度：腿部摆动
            vz = part_velocities.get(limb_type, 0.0)
            all_vz.append(np.full(limb_pts, vz))

        if not all_points:
            return np.array([]).reshape(0, 3), np.array([]), np.array([])

        return np.vstack(all_points), np.concatenate(all_rcs), np.concatenate(all_vz)

    def _compute_part_velocities(self, state: str, fall_speed: float,
                                  walk_phase: float) -> Dict[str, float]:
        """
        计算各身体部位的垂直速度（微多普勒模型）
        参考: Victor C. Chen "Micro-Doppler Effect in Radar" 2011
        行走: 躯干0.8-1.5m/s, 腿尖4-6m/s, 臂摆±2-4m/s
        摔倒: 峰值1.5-3.0m/s, 撞击后速度中断

        :param state: 当前状态
        :param fall_speed: 摔倒速度
        :param walk_phase: 行走相位
        :return: 各部位速度字典
        """
        md_config = RADAR_PHYSICS_CONFIG["micro_doppler"]

        if state == HumanState.STANDING:
            # 站立：仅有呼吸微动 (~0.2-0.5 Hz)
            breath = md_config["breathing_amplitude"] * np.sin(walk_phase * 0.3)
            return {"torso": breath, "head": breath * 1.2, "upper_arm": 0.0,
                    "lower_arm": 0.0, "upper_leg": 0.0, "lower_leg": 0.0}

        elif state == HumanState.WALKING:
            # 行走微多普勒模型
            # 步频1.5-2.5Hz，躯干上下起伏，四肢交替摆动
            step_freq = md_config["step_frequency"]
            omega = 2 * np.pi * step_freq

            # 躯干：上下起伏，速度 = 振幅 * ω * cos(ωt)
            # 振幅约3cm，速度约0.03*2π*2 ≈ 0.38 m/s（垂直分量）
            torso_bob = 0.03 * omega * np.cos(walk_phase) * 0.1  # 缩放到合理垂直分量
            torso_v = md_config["torso_velocity"] * 0.1  # 垂直分量约为水平的10%

            # 腿部：对侧摆动（与手臂反相）
            # 腿尖峰值4-6m/s，但垂直分量较小
            leg_swing = md_config["leg_tip_velocity"] * 0.15 * np.sin(walk_phase + np.pi)

            # 手臂：对侧摆动（与腿反相）
            # 臂摆峰值2-4m/s，垂直分量
            arm_swing = md_config["arm_swing_amplitude"] * 0.12 * np.sin(walk_phase)

            return {
                "torso": torso_v + torso_bob,
                "head": (torso_v + torso_bob) * 1.1,
                "upper_arm": arm_swing * 0.7,
                "lower_arm": arm_swing,  # 前臂摆动更大
                "upper_leg": -leg_swing * 0.5,  # 大腿摆动较小
                "lower_leg": -leg_swing,  # 小腿摆动最大
            }

        elif state == HumanState.FALLING:
            # 摔倒微多普勒：各部位有不同下落速度
            # 头部/躯干先倒，腿部后倒
            fall_config = RADAR_PHYSICS_CONFIG["fall_dynamics"]
            return {
                "torso": fall_speed * 0.85,
                "head": fall_speed * 1.15,  # 头部下落更快（顶部先着地）
                "upper_arm": fall_speed * 0.9,
                "lower_arm": fall_speed * 1.0,  # 手臂自由下落
                "upper_leg": fall_speed * 0.45,  # 腿部下落较慢（最后倒）
                "lower_leg": fall_speed * 0.25,
            }

        else:  # FALLEN
            # 倒地后：速度中断（velocity blackout），近零速度持续>2s
            return {"torso": 0.0, "head": 0.0, "upper_arm": 0.0,
                    "lower_arm": 0.0, "upper_leg": 0.0, "lower_leg": 0.0}
    
    def _generate_ellipsoid_points(self, center: np.ndarray, semi_axes: Tuple[float, float, float],
                                  num_points: int, base_rcs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成椭球体表面点云，使用物理RCS公式
        参考: ekurtgl/FMCW-MIMO-Radar-Simulation, rcsellipsoid.m

        :param center: 中心坐标
        :param semi_axes: 半轴长度 (a, b, c)
        :param num_points: 点数
        :param base_rcs: 基础RCS值 (用于归一化)
        :return: (点云坐标, RCS值)
        """
        a, b, c = semi_axes

        # 球面均匀采样
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        cos_theta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(cos_theta)

        # 椭球体参数方程
        x = center[0] + a * np.sin(theta) * np.cos(phi)
        y = center[1] + b * np.sin(theta) * np.sin(phi)
        z = center[2] + c * cos_theta

        points = np.column_stack((x, y, z))

        # 计算每个点到雷达(原点)的方向，用于RCS计算
        radar_dir = -points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-10)
        # 椭球体表面法向量 (对于椭球体 x²/a²+y²/b²+z²/c²=1, 法向量为 (x/a², y/b², z/c²))
        normals = np.column_stack([
            (x - center[0]) / (a ** 2),
            (y - center[1]) / (b ** 2),
            (z - center[2]) / (c ** 2)
        ])
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

        # 入射角 = 雷达方向与表面法向量的夹角
        cos_incident = np.abs(np.sum(radar_dir * normals, axis=1))

        # RCS公式: RCS(θ) = base_rcs * cos²(θ)  (简化版，保持与物理公式一致的衰减趋势)
        # 完整公式需要角度信息，这里用cos²近似
        rcs_values = base_rcs * cos_incident ** 2
        rcs_values = np.maximum(rcs_values, 0.001)  # 最小RCS -8 dBsm

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
    输出5D点云: [x, y, z, velocity_z, intensity]
    """
    
    def __init__(self):
        """初始化雷达物理模型"""
        self.config = RADAR_PHYSICS_CONFIG
        self.noise_config = self.config["noise"]
        self.env_config = self.config["environment"]
    
    def add_measurement_noise(self, points: np.ndarray, rcs_values: np.ndarray,
                               body_velocities: np.ndarray = None) -> np.ndarray:
        """
        添加测量噪声并计算强度（受污染高斯模型 Contaminated Gaussian）
        参考: Schumann et al. IEEE ITSC 2019, Lim et al. 2024

        模型: p(z) = (1-ε)·N(z;μ,Σ) + ε·p_outlier(z)
        - 85%点: 标准高斯噪声（inlier）
        - 15%点: 离群点（Cauchy分布，模拟角度旁瓣拉扯、距离栅瓣等）

        :param points: 真实坐标 (N, 3)
        :param rcs_values: RCS值 (N,)
        :param body_velocities: 各点的真实垂直速度 (N,)
        :return: 带噪声的点云 (N, 5) [x, y, z, velocity_z, intensity]
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 5)

        num_points = len(points)
        distances = np.linalg.norm(points, axis=1)

        # === 受污染高斯噪声模型 ===
        outlier_frac = self.noise_config["outlier_fraction"]
        is_outlier = np.random.random(num_points) < outlier_frac
        is_inlier = ~is_outlier

        # --- 距离噪声 ---
        # Inlier: 高斯，σ随距离增大 (σ = 0.05 + 0.02*r)
        range_std = self.noise_config["range_std"] + 0.02 * distances
        range_noise = np.zeros(num_points)
        range_noise[is_inlier] = np.random.normal(0, 1, is_inlier.sum()) * range_std[is_inlier]
        # Outlier: Cauchy分布（重尾，模拟距离栅瓣），裁剪到±3倍尺度
        if is_outlier.any():
            cauchy_raw = np.random.standard_cauchy(is_outlier.sum())
            range_noise[is_outlier] = np.clip(cauchy_raw, -3, 3) * self.noise_config["range_outlier_scale"]

        # --- 角度噪声 ---
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        elevation = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))

        az_noise = np.zeros(num_points)
        el_noise = np.zeros(num_points)
        # Inlier: 高斯，σ≈1.5°
        az_noise[is_inlier] = np.random.normal(0, self.noise_config["angle_std"], is_inlier.sum())
        el_noise[is_inlier] = np.random.normal(0, self.noise_config["angle_std"] * 1.5, is_inlier.sum())
        # Outlier: Cauchy（模拟角度旁瓣拉扯），裁剪到±3倍尺度
        if is_outlier.any():
            cauchy_az = np.random.standard_cauchy(is_outlier.sum())
            cauchy_el = np.random.standard_cauchy(is_outlier.sum())
            az_noise[is_outlier] = np.clip(cauchy_az, -3, 3) * self.noise_config["angle_outlier_scale"]
            el_noise[is_outlier] = np.clip(cauchy_el, -3, 3) * self.noise_config["angle_outlier_scale"] * 1.5

        azimuth += az_noise
        elevation += el_noise

        # --- 距离-多普勒耦合 ---
        # FMCW雷达固有特性：corr(r_err, v_err) ≈ -0.3~-0.5
        r = distances + range_noise
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)

        # --- 速度噪声 ---
        if body_velocities is not None and len(body_velocities) == num_points:
            velocity_noise = np.zeros(num_points)
            velocity_noise[is_inlier] = np.random.normal(0, self.noise_config["velocity_std"], is_inlier.sum())
            if is_outlier.any():
                # 离群点速度：均匀分布
                v_outlier_range = self.noise_config["velocity_outlier_scale"]
                velocity_noise[is_outlier] = np.random.uniform(-v_outlier_range, v_outlier_range, is_outlier.sum())
            # 距离-多普勒耦合：距离误差的一部分传递到速度
            coupling = self.noise_config["range_doppler_corr"]
            velocity = body_velocities + velocity_noise + coupling * range_noise * 0.3
        else:
            velocity = np.random.normal(0, 0.15, num_points)

        # --- 强度计算 ---
        # 接收功率 ∝ RCS / distance^4（雷达方程）
        intensity = rcs_values / (distances ** 4 + 1e-10)
        intensity = intensity / (np.max(intensity) + 1e-10)
        intensity = np.clip(intensity, 0.0, 1.0)
        # 离群点强度通常更低（鬼点SNR弱3-20dB）
        if is_outlier.any():
            intensity[is_outlier] *= np.random.uniform(0.05, 0.5, is_outlier.sum())
        intensity += np.random.normal(0, 0.03, num_points)
        intensity = np.clip(intensity, 0.0, 1.0)

        return np.column_stack((x, y, z, velocity, intensity))
    
    def apply_rcs_filter(self, points: np.ndarray, rcs_values: np.ndarray,
                         velocities: np.ndarray = None,
                         snr_threshold: float = 0.1) -> Tuple:
        """
        基于RCS的检测过滤
        :param points: 点云坐标 (N, 3)
        :param rcs_values: RCS值 (N,)
        :param velocities: 垂直速度 (N,)，可选
        :param snr_threshold: 信噪比阈值
        :return: (过滤后的点云, 过滤后的RCS, 过滤后的速度)
        """
        if len(points) == 0:
            return points, rcs_values, velocities

        distances = np.linalg.norm(points, axis=1)
        noise_power = self.noise_config["thermal_noise_power"]
        snr = rcs_values / (distances ** 4 * noise_power + 1e-10)
        detection_prob = 1 - np.exp(-snr / snr_threshold)
        detected = np.random.random(len(points)) < detection_prob

        if velocities is not None:
            return points[detected], rcs_values[detected], velocities[detected]
        return points[detected], rcs_values[detected], None
    
    def add_clutter(self, points: np.ndarray, room_size: Tuple[float, float, float]) -> np.ndarray:
        """
        添加环境杂波
        :param points: 原始点云 (N, 5) [x, y, z, velocity_z, intensity]
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
        clutter_i = np.random.uniform(0.01, 0.15, num_clutter)  # 杂波强度较低
        
        clutter_points = np.column_stack((clutter_x, clutter_y, clutter_z, clutter_v, clutter_i))
        
        # 合并点云
        combined = np.vstack((points, clutter_points))
        
        return combined
    
    def simulate_multipath(self, points: np.ndarray, room_size: Tuple[float, float, float]) -> np.ndarray:
        """
        模拟多径反射（鬼点）
        参考: Liu et al. Sensors 2025, Kraus et al. IROS 2021

        真实室内鬼点特征：
        - 单目标鬼点率5-15%
        - 镜像对称位置
        - 鬼点SNR弱3-20dB（强度衰减0.05-0.5x）
        - 时序不一致（闪烁概率30%）

        :param points: 原始点云 (N, 5) [x, y, z, velocity_z, intensity]
        :param room_size: 房间尺寸
        :return: 添加多径反射后的点云
        """
        if len(points) == 0 or not self.env_config["multipath_enabled"]:
            return points

        multipath_points = []

        # 地面反射（最常见）- 干墙-3~-6dB，混凝土-6~-15dB
        floor_reflectivity = self.env_config["floor_reflectivity"]
        # 随机选择部分点产生地面鬼点（2-5%）— 降低以保持人体轮廓
        ghost_rate_floor = np.random.uniform(0.02, 0.05)
        n_floor_ghost = max(1, int(len(points) * ghost_rate_floor))
        if np.random.random() < floor_reflectivity and n_floor_ghost > 0:
            # 随机选择产生鬼点的源点
            src_idx = np.random.choice(len(points), min(n_floor_ghost, len(points)), replace=False)
            floor_points = points[src_idx].copy()
            # 地面镜像反射：z_new = -z_old
            floor_points[:, 2] = -floor_points[:, 2]
            floor_points[:, 3] *= 0.4  # 反射后速度衰减
            # 强度衰减：干墙-3~-6dB (0.25-0.5x)
            attenuation = np.random.uniform(0.15, 0.40, len(floor_points))
            floor_points[:, 4] *= attenuation
            # 30%概率闪烁（时序不一致）
            flicker_mask = np.random.random(len(floor_points)) < 0.3
            floor_points = floor_points[~flicker_mask]
            if len(floor_points) > 0:
                multipath_points.append(floor_points)

        # 墙壁反射（较弱）- 强度衰减-6~-15dB
        wall_reflectivity = self.env_config["wall_reflectivity"]
        # 每面墙有小概率产生鬼点
        for wall_side in ['left', 'right', 'front', 'back']:
            wall_prob = wall_reflectivity * 0.06  # 每面墙6%概率（降低减少鬼点）
            if np.random.random() < wall_prob:
                # 墙壁鬼点更少（2-5%的源点）
                ghost_rate_wall = np.random.uniform(0.02, 0.05)
                n_wall_ghost = max(1, int(len(points) * ghost_rate_wall))
                src_idx = np.random.choice(len(points), min(n_wall_ghost, len(points)), replace=False)
                wall_points = points[src_idx].copy()

                # 墙壁镜像反射
                if wall_side == 'left':
                    wall_points[:, 0] = -room_size[0] - wall_points[:, 0]
                elif wall_side == 'right':
                    wall_points[:, 0] = room_size[0] - wall_points[:, 0]
                elif wall_side == 'front':
                    wall_points[:, 1] = -room_size[1] - wall_points[:, 1]
                elif wall_side == 'back':
                    wall_points[:, 1] = room_size[1] - wall_points[:, 1]

                wall_points[:, 3] *= 0.2  # 墙壁反射速度衰减更多
                # 强度衰减：混凝土-6~-15dB (0.03-0.25x)
                attenuation = np.random.uniform(0.05, 0.20, len(wall_points))
                wall_points[:, 4] *= attenuation
                # 50%概率闪烁
                flicker_mask = np.random.random(len(wall_points)) < 0.5
                wall_points = wall_points[~flicker_mask]
                if len(wall_points) > 0:
                    multipath_points.append(wall_points)

        if not multipath_points:
            return points

        all_points = [points] + multipath_points
        return np.vstack(all_points)
    
    def simulate_missed_detection(self, points: np.ndarray, rcs_values: np.ndarray,
                                   velocities: np.ndarray = None) -> Tuple:
        """
        模拟漏检（按身体部位特异性检测概率）
        参考: TI SWRA587, mmWave radar literature

        躯干~98%检测率，手/脚<20%，前臂~45%
        距离衰减: P_detect(r) = P_base * exp(-0.1 * r)

        :param points: 点云坐标 (N, 3)
        :param rcs_values: RCS值 (N,)
        :param velocities: 垂直速度 (N,)，可选
        :return: (过滤后的点云, 过滤后的RCS, 过滤后的速度)
        """
        if len(points) == 0:
            return points, rcs_values, velocities

        distances = np.linalg.norm(points, axis=1)

        # 使用RCS值推断部位类型，分配检测概率
        # RCS阈值基于文献值
        det_prob_config = RADAR_PHYSICS_CONFIG["detection_prob"]
        base_prob = np.full(len(points), det_prob_config["torso"])  # 默认躯干

        # 按RCS大小粗略分类（实际场景中部位标签不可用，用RCS近似）
        # 低RCS点更可能来自四肢
        rcs_threshold_arm = 0.06  # 臂RCS上限附近
        rcs_threshold_leg = 0.15  # 腿RCS上限附近
        rcs_threshold_head = 0.12  # 头RCS上限附近

        low_rcs_mask = rcs_values < rcs_threshold_arm
        mid_rcs_mask = (rcs_values >= rcs_threshold_arm) & (rcs_values < rcs_threshold_leg)
        head_rcs_mask = (rcs_values >= 0.02) & (rcs_values < rcs_threshold_head) & (distances > 1.0)

        base_prob[low_rcs_mask] = det_prob_config["lower_arm"]  # ~0.45
        base_prob[mid_rcs_mask] = det_prob_config["upper_leg"]  # ~0.87
        base_prob[head_rcs_mask] = det_prob_config["head"]  # ~0.55

        # 距离衰减：远处检测率下降
        distance_decay = np.exp(-0.08 * distances)
        detection_prob = base_prob * distance_decay

        # 添加随机抖动（模拟环境因素）
        random_factor = np.random.uniform(0.8, 1.0, len(points))
        detection_prob *= random_factor
        detection_prob = np.clip(detection_prob, 0.0, 1.0)

        detected = np.random.random(len(points)) < detection_prob

        if velocities is not None:
            return points[detected], rcs_values[detected], velocities[detected]
        return points[detected], rcs_values[detected], None


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
        self.position = np.array([0.0, 0.0, 1.35])  # 世界坐标：身体中心（腰部）高度
        self.fall_progress = 0.0  # 摔倒进度 0-1
        self.trajectory = []  # 轨迹记录
        self.walk_direction = np.random.uniform(0, 2*np.pi)  # 行走方向
        self.walk_step = 0.0  # 行走步幅相位
        self.walk_phase = 0.0  # 行走相位
        self.velocity_blackout_counter = 0  # 速度中断帧计数器
        # 重置几何模型到站立姿势
        if hasattr(self, 'geometric_model'):
            self.geometric_model._update_standing_pose()
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
            elif state == HumanState.FALLEN:
                # 倒地时身体中心接近地面
                self.position[2] = 1.35  # 保持不变，几何模型的fallen_pose用局部-1.2
                # 世界Z = -1.2 + 1.35 = 0.15m（接近地面）
                self.velocity_blackout_counter = RADAR_PHYSICS_CONFIG["fall_dynamics"]["velocity_blackout_frames"]
            elif state == HumanState.STANDING:
                self.position[2] = 1.35
                if hasattr(self, 'geometric_model'):
                    self.geometric_model._update_standing_pose()
            elif state == HumanState.WALKING:
                self.position[2] = 1.35
                if hasattr(self, 'geometric_model'):
                    self.geometric_model._update_standing_pose()
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

    def get_next_frame(self) -> np.ndarray:
        """
        获取下一帧点云数据
        :return: 点云数组 (N, 5)，列顺序: [x, y, z, velocity_z, intensity]
        """
        try:
            # 更新几何模型姿势
            self.geometric_model.update_pose(self.state, self.fall_progress, self.walk_phase)

            # 计算当前摔倒速度（三阶段模型）
            fall_speed = 0.0
            if self.state == HumanState.FALLING:
                fall_config = RADAR_PHYSICS_CONFIG["fall_dynamics"]
                peak_v = fall_config["peak_velocity"]
                if self.fall_progress < 0.5:
                    # 阶段1：加速（0-0.5）
                    fall_speed = -peak_v * (0.3 + self.fall_progress * 1.4)
                elif self.fall_progress < 0.7:
                    # 阶段2：峰值速度（0.5-0.7）
                    fall_speed = -peak_v
                else:
                    # 阶段3：撞击减速（0.7-1.0）
                    impact_drop = fall_config["impact_velocity_drop"]
                    t = (self.fall_progress - 0.7) / 0.3
                    fall_speed = -peak_v * (1.0 - t * impact_drop)

            # 生成表面点云（含各部位速度）
            # 基础50点 + 轻微随机波动(±20%)
            base_points = SIMULATOR_CONFIG["point_num_per_frame"]
            random_factor = np.random.uniform(0.8, 1.2)
            num_points = max(20, int(base_points * random_factor))

            # 倒地状态点云减少20-30%
            if self.state == HumanState.FALLEN:
                num_points = max(15, int(num_points * np.random.uniform(0.7, 0.8)))

            surface_points, rcs_values, body_vz = self.geometric_model.generate_surface_points(
                num_points, self.state.value, fall_speed, self.walk_phase
            )

            if len(surface_points) == 0:
                return np.array([]).reshape(0, 5)

            # 摔倒后RCS下降3-8dB（体型变化、地面反射干扰）
            if self.state == HumanState.FALLEN:
                rcs_drop_linear = 10 ** (-RADAR_PHYSICS_CONFIG["fall_dynamics"]["rcs_drop_db"] / 10)
                rcs_values *= rcs_drop_linear

            # 添加位置偏移（人体中心位置，含Z轴）
            surface_points[:, 0] += self.position[0]
            surface_points[:, 1] += self.position[1]
            surface_points[:, 2] += self.position[2]

            # 模拟漏检
            filtered_points, filtered_rcs, filtered_vz = self.physics_model.simulate_missed_detection(
                surface_points, rcs_values, body_vz
            )

            if len(filtered_points) == 0:
                return np.array([]).reshape(0, 5)

            # 应用RCS检测过滤
            filtered_points, filtered_rcs, filtered_vz = self.physics_model.apply_rcs_filter(
                filtered_points, filtered_rcs, filtered_vz
            )

            if len(filtered_points) == 0:
                return np.array([]).reshape(0, 5)

            # 添加测量噪声（含真实速度）
            noisy_points = self.physics_model.add_measurement_noise(
                filtered_points, filtered_rcs, filtered_vz
            )

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
            max_frames = 5
            if len(self.trajectory) > max_frames:
                self.trajectory = self.trajectory[-max_frames:]

            return noisy_points
        except Exception as e:
            logger.error(f"生成点云失败: {e}")
            return np.array([]).reshape(0, 5)

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
        """更新摔倒状态
        几何模型的 _update_falling_pose() 基于 fall_progress 处理身体旋转和高度变化
        self.position[2] 保持在 1.35 不变（身体局部坐标的参考原点）
        """
        # 摔倒进度更新（约15帧完成摔倒）
        self.fall_progress += 0.065

        # 检查是否倒地
        if self.fall_progress >= 1.0:
            self.fall_progress = 1.0
            self.state = HumanState.FALLEN
            self.velocity_blackout_counter = RADAR_PHYSICS_CONFIG["fall_dynamics"]["velocity_blackout_frames"]
            logger.info("人体已倒地")
    
    def get_trajectory(self) -> list:
        """获取最近的轨迹记录"""
        return self.trajectory

    def get_center_position(self) -> Tuple[float, float, float]:
        """获取当前人体中心位置"""
        return tuple(self.position)