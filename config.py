#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 集中管理所有系统参数
"""

from enum import Enum


class HumanState(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    FALLING = "falling"
    FALLEN = "fallen"

# 项目信息
PROJECT_INFO = {
    "name": "毫米波雷达摔倒检测系统",
    "version": "3.2.0",
    "author": "Radar Fall Detection Team",
    "description": "基于毫米波雷达点云的人体摔倒检测原型系统",
}

# 雷达模拟器配置
SIMULATOR_CONFIG = {
    "frame_rate": 10,  # 雷达帧率（Hz）
    "point_num_per_frame": 50,  # 每帧模拟点云数量（真实雷达20-64点）
    "noise_level": 0.1,  # 噪声水平（0-1）
    "fall_speed": -1.2,  # 摔倒垂直速度（m/s）
}

# 物理毫米波雷达参数
RADAR_PHYSICS_CONFIG = {
    # FMCW雷达参数
    "frequency": 77e9,  # 77GHz毫米波
    "bandwidth": 4e9,  # 4GHz带宽
    "chirp_duration": 50e-6,  # 50μs扫频周期
    "num_rx": 4,  # 4个接收天线
    "num_tx": 3,  # 3个发射天线
    "max_range": 5.0,  # 最大探测距离5m
    "range_resolution": 0.0375,  # 距离分辨率3.75cm
    "angle_resolution": 0.26,  # 角度分辨率~15° (TI IWR6843实测值)
    
    # 人体几何模型参数（米）
    "body_geometry": {
        "torso": {"semi_axes": (0.15, 0.10, 0.25)},  # 躯干椭球体半轴
        "head": {"radius": 0.10},  # 头部球体半径
        "upper_arm": {"radius": 0.035, "length": 0.30},  # 上臂圆柱体
        "lower_arm": {"radius": 0.030, "length": 0.25},  # 前臂圆柱体
        "upper_leg": {"radius": 0.06, "length": 0.40},  # 大腿圆柱体
        "lower_leg": {"radius": 0.045, "length": 0.35},  # 小腿圆柱体
    },
    
    # 雷达截面积(RCS)参数（平方米）
    # 参考: Ahn et al. IEEE Access 2020, Herschlein & Hasch EuRAD 2008, Tracy & Young 2006
    "rcs": {
        "torso": 1.0,  # 躯干: -3~+3 dBsm, 中值0 dBsm (1.0 m²)
        "head": 0.08,  # 头部: -15~-8 dBsm, 中值-11 dBsm (0.08 m²)
        "upper_arm": 0.05,  # 上臂: -18~-5 dBsm, 中值-13 dBsm (0.05 m²)
        "lower_arm": 0.03,  # 前臂: -18~-5 dBsm, 中值-15 dBsm (0.03 m²)
        "upper_leg": 0.12,  # 大腿: -15~-3 dBsm, 中值-9 dBsm (0.12 m²)
        "lower_leg": 0.06,  # 小腿: -15~-3 dBsm, 中值-12 dBsm (0.06 m²)
        "wall": 10.0,  # 墙壁强反射
        "floor": 5.0,  # 地面强反射
    },

    # 身体部位点云分布权重（真实雷达躯干占40-60%，非按RCS均匀分配）
    # 参考: van Dorp & Groen 2003, Victor C. Chen 2006
    "body_point_weights": {
        "torso": 0.50,  # 躯干: 40-60% (取中值50%)
        "head": 0.07,  # 头部: 3-10% (取中值7%)
        "arms": 0.20,  # 双臂合计: 15-25% (取中值20%)
        "legs": 0.23,  # 双腿合计: 15-25% (取中值23%)
    },

    # 身体部位检测概率（5m内，post-CFAR）
    # 参考: TI SWRA587, mmWave radar literature review
    "detection_prob": {
        "torso": 0.98,  # 躯干几乎总能检测到
        "head": 0.55,  # 头部：可变，受头发/帽子影响
        "upper_arm": 0.65,  # 上臂：8-10m外常丢失
        "lower_arm": 0.45,  # 前臂：更细，检测率更低
        "upper_leg": 0.87,  # 大腿：两条大腿常融合为单回波
        "lower_leg": 0.75,  # 小腿：被近侧腿遮挡
        "hands": 0.15,  # 手部：几乎检测不到
        "feet": 0.15,  # 脚部：3-5m外消失在噪声中
    },
    
    # 噪声模型参数（受污染高斯模型 Contaminated Gaussian）
    # 参考: Schumann et al. IEEE ITSC 2019, Lim et al. 2024
    "noise": {
        # 离群点（inlier）高斯噪声参数
        "range_std": 0.03,  # 距离噪声标准差3cm (降低以保持点云紧凑)
        "angle_std": 0.018,  # 角度噪声标准差~1.0° (降低减少横向散射)
        "velocity_std": 0.08,  # 速度噪声标准差0.08m/s
        "thermal_noise_power": 1e-10,  # 热噪声功率
        "clutter_density": 0.04,  # 杂波密度（点/m²）— 降低以减少随机散点
        # 受污染高斯模型：epsilon比例的离群点
        "outlier_fraction": 0.05,  # 5%离群点比例（降低以保持人体轮廓清晰）
        "range_outlier_scale": 0.06,  # 距离离群点Cauchy尺度(m)
        "angle_outlier_scale": 0.02,  # 角度离群点Cauchy尺度(rad, ~1.1°)
        "velocity_outlier_scale": 0.2,  # 速度离群点均匀分布半宽(m/s)
        # 距离-多普勒耦合系数（FMCW雷达固有特性）
        "range_doppler_corr": -0.4,  # corr(r_err, v_err) ≈ -0.3~-0.5
    },
    
    # 环境模型参数
    "environment": {
        "room_size": (4.0, 4.0, 2.5),  # 房间尺寸（长宽高）
        "wall_reflectivity": 0.8,  # 墙壁反射率
        "floor_reflectivity": 0.6,  # 地面反射率
        "multipath_enabled": True,  # 是否启用多径效应
    },

    # 微多普勒参数（行走）
    # 参考: Victor C. Chen "Micro-Doppler Effect in Radar" 2011
    "micro_doppler": {
        "step_frequency": 2.0,  # 步频 (Hz), 典型1.5-2.5
        "torso_velocity": 1.2,  # 躯干行走速度 (m/s), 典型0.8-1.5
        "arm_swing_amplitude": 3.0,  # 臂摆峰值速度 (m/s相对身体), 典型2-4
        "leg_tip_velocity": 5.0,  # 腿尖峰值速度 (m/s), 典型4-6
        "breathing_amplitude": 0.02,  # 呼吸微动幅度 (m/s)
    },

    # 摔倒过程参数
    # 参考: UP-Fall dataset, Wang et al. IEEE Sensors 2022
    "fall_dynamics": {
        "peak_velocity": 2.5,  # 摔倒峰值速度 (m/s), 典型1.5-3.0
        "impact_velocity_drop": 0.85,  # 撞击后速度衰减比例
        "velocity_blackout_frames": 20,  # 速度中断持续帧数 (>2s @10fps)
        "rcs_drop_db": 5.0,  # 摔倒后RCS下降 (dB), 典型3-8
    },
}

# 预处理配置
PROCESSOR_CONFIG = {
    "human_height_range": (0.05, 1.8),  # 人体高度范围（m）— 下限0.05m以允许倒地状态点云通过
    "statistical_k": 3,  # 统计滤波邻域点数（从5减到3，适应稀疏点云）
    "statistical_std_ratio": 1.5,  # 统计滤波标准差倍数（从1.0增到1.5，保留更多点）
}

# 摔倒检测配置
DETECTOR_CONFIG = {
    "fall_height_threshold": 0.4,  # 摔倒高度阈值（m）- 与倒地高度0.15m匹配
    "fall_speed_threshold": -0.6,  # 摔倒垂直速度阈值（m/s）- 更敏感
    "fall_frames_threshold": 2,  # 连续多少帧满足条件判定为摔倒 - 更快响应
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "point_size": 30,  # 点云尺寸（统一设置为30）
    "trajectory_frames": 5,  # 轨迹线帧数
    "history_seconds": 60,  # 历史数据显示时长（秒）
    "view_elevation": 45,  # 默认视角仰角（度）
    "view_azimuth": -45,  # 默认视角方位角（度）
}

# 颜色配置
COLOR_CONFIG = {
    "background": "#2b2b2b",  # 背景色（深灰色）
    "grid": "#aaaaaa",  # 网格线颜色（浅灰色）
    "axis": "#aaaaaa",  # 坐标轴颜色（浅灰色）
    "text": "#ffffff",  # 文本颜色
    
    # 点云状态颜色（更直观）
    "standing": "#4A90D9",      # 站立：蓝色（静止/冷静）
    "walking": "#50C878",       # 行走：绿色（运动/正常）
    "falling": "#FF8C00",       # 摔倒中：橙色（警告）
    "fallen": "#DC143C",        # 倒地：深红色（危险）
    
    # 保留旧配置以兼容
    "normal_point": "#4A90D9",  # 正常状态点云颜色（蓝色）
    "falling_point": "#FF8C00", # 摔倒过程点云颜色（橙色）
    "fallen_point": "#DC143C",  # 倒地状态点云颜色（深红色）
    
    # 轨迹线
    "trajectory": "#ffffff",    # 运动轨迹线颜色（白色）
    "center_trajectory": "#ffffff",  # 人体中心轨迹线颜色（白色）
    
    # 其他元素
    "orientation_arrow": "#00ff00",  # 朝向箭头颜色（绿色）
    "status_standby": "#888888",  # 待机状态颜色
    "status_running": "#00cc00",  # 运行状态颜色
    "status_alarm": "#ff4444",  # 报警状态颜色
    "status_stopped": "#ffaa00",  # 停止状态颜色
}

# 界面配置
UI_CONFIG = {
    "window_title": "毫米波雷达摔倒检测系统",
    "window_size": (1200, 700),
    "left_panel_ratio": 0.6,  # 左侧面板占比
    "right_panel_ratio": 0.4,  # 右侧面板占比
}

# 报警配置
ALARM_CONFIG = {
    "enable_sound": True,  # 是否启用蜂鸣报警
    "enable_voice": False,  # 是否启用语音报警
    "beep_frequency": 800,  # 蜂鸣频率（Hz）
    "beep_duration": 500,  # 蜂鸣持续时间（ms）
    "screenshot_enabled": True,  # 是否自动截图
    "screenshot_dir": "alarms",  # 截图保存目录
    "log_max_entries": 100,  # 日志最大条目数
}


def get_fall_height_threshold() -> float:
    """获取摔倒高度阈值"""
    return DETECTOR_CONFIG["fall_height_threshold"]


def set_fall_height_threshold(value: float) -> None:
    """设置摔倒高度阈值"""
    DETECTOR_CONFIG["fall_height_threshold"] = value


def get_fall_speed_threshold() -> float:
    """获取摔倒速度阈值"""
    return DETECTOR_CONFIG["fall_speed_threshold"]


def set_fall_speed_threshold(value: float) -> None:
    """设置摔倒速度阈值"""
    DETECTOR_CONFIG["fall_speed_threshold"] = value


def get_fall_frames_threshold() -> int:
    """获取连续帧数阈值"""
    return DETECTOR_CONFIG["fall_frames_threshold"]


def set_fall_frames_threshold(value: int) -> None:
    """设置连续帧数阈值"""
    DETECTOR_CONFIG["fall_frames_threshold"] = value


def get_noise_level() -> float:
    """获取噪声水平"""
    return SIMULATOR_CONFIG["noise_level"]


def set_noise_level(value: float) -> None:
    """设置噪声水平"""
    SIMULATOR_CONFIG["noise_level"] = value


def get_sim_speed() -> float:
    """获取模拟速度"""
    return SIMULATOR_CONFIG.get("sim_speed", 1.0)


def set_sim_speed(value: float) -> None:
    """设置模拟速度"""
    SIMULATOR_CONFIG["sim_speed"] = value
