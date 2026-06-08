#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 集中管理所有系统参数
"""

from typing import Tuple
from enum import Enum


class HumanState(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    FALLING = "falling"
    FALLEN = "fallen"

# 项目信息
PROJECT_INFO = {
    "name": "毫米波雷达摔倒检测系统",
    "version": "2.0.0",
    "author": "Radar Fall Detection Team",
    "description": "基于毫米波雷达点云的人体摔倒检测原型系统",
}

# 雷达模拟器配置
SIMULATOR_CONFIG = {
    "frame_rate": 10,  # 雷达帧率（Hz）
    "point_num_per_frame": 30,  # 每帧模拟点云数量
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
    "angle_resolution": 0.1,  # 角度分辨率~6°
    
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
    "rcs": {
        "torso": 0.5,  # 躯干强反射
        "head": 0.1,  # 头部中等反射
        "upper_arm": 0.05,  # 上臂弱反射
        "lower_arm": 0.03,  # 前臂更弱
        "upper_leg": 0.08,  # 大腿中等反射
        "lower_leg": 0.04,  # 小腿弱反射
        "wall": 10.0,  # 墙壁强反射
        "floor": 5.0,  # 地面强反射
    },
    
    # 噪声模型参数
    "noise": {
        "range_std": 0.02,  # 距离噪声标准差2cm
        "angle_std": 0.05,  # 角度噪声标准差~3°
        "velocity_std": 0.05,  # 速度噪声标准差0.05m/s
        "thermal_noise_power": 1e-10,  # 热噪声功率
        "clutter_density": 0.1,  # 杂波密度（点/m²）
    },
    
    # 环境模型参数
    "environment": {
        "room_size": (4.0, 4.0, 2.5),  # 房间尺寸（长宽高）
        "wall_reflectivity": 0.8,  # 墙壁反射率
        "floor_reflectivity": 0.6,  # 地面反射率
        "multipath_enabled": True,  # 是否启用多径效应
    },
}

# 预处理配置
PROCESSOR_CONFIG = {
    "human_height_range": (0.05, 1.8),  # 人体高度范围（m）— 下限0.05m以允许倒地状态点云通过
    "statistical_k": 5,  # 统计滤波邻域点数
    "statistical_std_ratio": 1.0,  # 统计滤波标准差倍数
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
    "history_seconds": 30,  # 历史数据显示时长（秒）
    "view_elevation": 45,  # 默认视角仰角（度）
    "view_azimuth": -45,  # 默认视角方位角（度）
}

# 颜色配置
COLOR_CONFIG = {
    "background": "#2b2b2b",  # 背景色（深灰色）
    "grid": "#aaaaaa",  # 网格线颜色（浅灰色）
    "axis": "#aaaaaa",  # 坐标轴颜色（浅灰色）
    "text": "#ffffff",  # 文本颜色
    "normal_point": "#00aaff",  # 正常状态点云颜色
    "falling_point": "#ff8800",  # 摔倒过程点云颜色
    "fallen_point": "#ff0000",  # 倒地状态点云颜色（纯红色）
    "trajectory": "#00ff88",  # 轨迹线颜色
    "center_trajectory": "#ffffff",  # 人体中心轨迹线颜色（白色）
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
