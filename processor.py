#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云预处理模块
对原始点云进行滤波去噪
物理版：优化直通滤波和统计算法，适应真实毫米波雷达点云特性
"""

import numpy as np
from typing import Tuple, Dict, Any
from config import PROCESSOR_CONFIG, RADAR_PHYSICS_CONFIG
from utils import logger


class PointCloudPreprocessor:
    """
    点云预处理模块
    提供直通滤波、统计滤波等预处理功能
    """

    def __init__(self):
        """初始化预处理模块"""
        self.height_min, self.height_max = PROCESSOR_CONFIG["human_height_range"]
        self.statistical_k = PROCESSOR_CONFIG["statistical_k"]
        self.statistical_std_ratio = PROCESSOR_CONFIG["statistical_std_ratio"]
        
        # 物理雷达参数
        self.radar_config = RADAR_PHYSICS_CONFIG
        self.max_range = self.radar_config["max_range"]
        self.min_range = 0.1  # 最小探测距离10cm
        
        # 速度过滤参数
        self.velocity_threshold = 0.1  # 速度阈值，用于过滤静态杂波
        
        logger.info("物理毫米波雷达点云预处理模块已初始化")

    def _pass_through_filter(self, points: np.ndarray) -> np.ndarray:
        """
        直通滤波：只保留指定高度范围内的点，并添加距离过滤
        :param points: 原始点云 (N, 4)
        :return: 滤波后的点云
        """
        if points.size == 0:
            return points
        
        # 高度过滤
        height_mask = (points[:, 2] >= self.height_min) & (points[:, 2] <= self.height_max)
        
        # 距离过滤（去除太远或太近的点）
        distances = np.linalg.norm(points[:, :3], axis=1)
        range_mask = (distances >= self.min_range) & (distances <= self.max_range)
        
        # 组合掩码
        mask = height_mask & range_mask
        filtered = points[mask]
        
        logger.debug(f"直通滤波: 输入 {len(points)} 点, 输出 {len(filtered)} 点")
        return filtered

    def _statistical_filter(self, points: np.ndarray) -> np.ndarray:
        """
        统计滤波：去除离群噪声点（优化版，适应稀疏点云）
        :param points: 输入点云 (N, 4)
        :return: 滤波后的点云
        """
        n = len(points)
        if n == 0:
            return points
        
        # 对于非常稀疏的点云，使用更简单的方法
        if n <= 3:
            return points
        
        # 动态调整K值（适应点云数量变化）
        k = min(self.statistical_k, n - 1)
        
        # 向量化计算每对点之间的距离 (N, N)
        coords = points[:, :3]
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        dist.sort(axis=1)
        
        # 计算K近邻平均距离
        avg_dist = np.mean(dist[:, 1:k + 1], axis=1)
        
        # 使用更鲁棒的统计方法（中位数和MAD）
        median_dist = np.median(avg_dist)
        mad_dist = np.median(np.abs(avg_dist - median_dist))
        
        # 使用MAD（中位数绝对偏差）代替标准差，更鲁棒
        threshold = median_dist + self.statistical_std_ratio * 1.4826 * mad_dist  # 1.4826是正态分布的MAD到标准差的转换系数
        
        mask = avg_dist < threshold
        filtered = points[mask]

        logger.debug(f"统计滤波: 输入 {n} 点, 输出 {len(filtered)} 点")
        return filtered
    
    def _velocity_filter(self, points: np.ndarray) -> np.ndarray:
        """
        速度滤波：去除静态杂波（速度接近0的点）
        改进：对站立状态更友好，保留低速点
        
        :param points: 输入点云 (N, 4) 或 (N, 5)
        :return: 滤波后的点云
        """
        if points.size == 0:
            return points
        
        # 计算速度绝对值
        velocity_magnitude = np.abs(points[:, 3])
        
        # 大幅放宽速度阈值：保留绝大部分点
        # 只过滤掉速度极低的点（可能是静态杂波）
        velocity_mask = velocity_magnitude > self.velocity_threshold * 0.05  # 从0.2降到0.05
        
        # 如果过滤后点数太少，放宽条件
        filtered = points[velocity_mask]
        if len(filtered) < 8 and len(points) > 8:
            # 保留速度最大的前90%点
            velocity_indices = np.argsort(velocity_magnitude)[::-1]
            top_90 = velocity_indices[:int(len(velocity_indices) * 0.9)]
            filtered = points[top_90]
        
        logger.debug(f"速度滤波: 输入 {len(points)} 点, 输出 {len(filtered)} 点")
        return filtered
    
    def _density_filter(self, points: np.ndarray, min_points: int = 1) -> np.ndarray:
        """
        密度过滤：去除过于稀疏的区域（可能是噪声）
        :param points: 输入点云 (N, 4)
        :param min_points: 最小点数阈值（设为1，保留所有有效点）
        :return: 滤波后的点云
        """
        if points.size == 0 or len(points) < min_points:
            return points

        # 使用简单的网格密度过滤
        # 将空间划分为网格，只保留包含足够点的网格
        grid_size = 0.15  # 15cm网格，更精细
        
        # 计算网格索引
        grid_x = np.floor(points[:, 0] / grid_size).astype(int)
        grid_y = np.floor(points[:, 1] / grid_size).astype(int)
        grid_z = np.floor(points[:, 2] / grid_size).astype(int)
        
        # 统计每个网格的点数
        grid_indices = np.column_stack((grid_x, grid_y, grid_z))
        unique_grids, counts = np.unique(grid_indices, axis=0, return_counts=True)
        
        # 找出包含足够点的网格
        valid_grids = unique_grids[counts >= min_points]
        
        if len(valid_grids) == 0:
            # 如果没有足够密集的网格，返回原始点云
            return points
        
        # 创建掩码，只保留在有效网格中的点
        mask = np.zeros(len(points), dtype=bool)
        for grid in valid_grids:
            grid_mask = (grid_indices == grid).all(axis=1)
            mask |= grid_mask
        
        filtered = points[mask]
        
        logger.debug(f"密度过滤: 输入 {len(points)} 点, 输出 {len(filtered)} 点")
        return filtered

    def process(self, raw_points: np.ndarray) -> np.ndarray:
        """
        完整预处理流程（物理雷达优化版）
        :param raw_points: 原始点云
        :return: 预处理后的点云
        """
        if raw_points.size == 0:
            return raw_points
        
        try:
            # 1. 直通滤波（高度+距离）
            filtered = self._pass_through_filter(raw_points)
            
            # 2. 统计滤波（去除离群点）
            filtered = self._statistical_filter(filtered)
            
            # 3. 速度滤波（去除静态杂波）
            filtered = self._velocity_filter(filtered)
            
            # 4. 密度过滤（去除稀疏噪声）
            filtered = self._density_filter(filtered)
            
            return filtered
        except Exception as e:
            logger.error(f"点云预处理失败: {e}")
            return raw_points

    def get_frame_features(self, points: np.ndarray) -> Tuple[float, float]:
        """
        获取单帧点云的特征
        :param points: 点云数据 (N, 4) 或 (N, 5)
        :return: (平均高度, 平均垂直速度)
        """
        if points.size == 0:
            return (0.0, 0.0)
        
        avg_height = np.mean(points[:, 2])
        avg_vz = np.mean(points[:, 3])
        
        return (avg_height, avg_vz)
    
    def get_intensity_features(self, points: np.ndarray) -> Dict[str, float]:
        """
        获取强度相关特征（5D点云）
        :param points: 点云数据 (N, 5) [x, y, z, velocity_z, intensity]
        :return: 强度特征字典
        """
        if points.size == 0 or points.shape[1] < 5:
            return {
                'avg_intensity': 0.0,
                'max_intensity': 0.0,
                'min_intensity': 0.0,
                'intensity_std': 0.0,
            }
        
        intensity = points[:, 4]
        return {
            'avg_intensity': float(np.mean(intensity)),
            'max_intensity': float(np.max(intensity)),
            'min_intensity': float(np.min(intensity)),
            'intensity_std': float(np.std(intensity)),
        }
    
    def get_advanced_features(self, points: np.ndarray) -> Dict[str, float]:
        """
        获取高级特征（用于改进的摔倒检测）
        :param points: 点云数据
        :return: 特征字典
        """
        if points.size == 0:
            return {
                'avg_height': 0.0,
                'avg_vz': 0.0,
                'height_std': 0.0,
                'velocity_std': 0.0,
                'point_density': 0.0,
                'height_range': 0.0,
                'velocity_range': 0.0,
            }
        
        # 基本特征
        avg_height = np.mean(points[:, 2])
        avg_vz = np.mean(points[:, 3])
        
        # 高度分布特征
        height_std = np.std(points[:, 2])
        height_range = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # 速度分布特征
        velocity_std = np.std(points[:, 3])
        velocity_range = np.max(points[:, 3]) - np.min(points[:, 3])
        
        # 点云密度（点数/体积）
        if len(points) > 1:
            # 计算点云的包围盒体积
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            z_range = np.max(points[:, 2]) - np.min(points[:, 2])
            volume = max(x_range * y_range * z_range, 0.001)  # 避免除零
            point_density = len(points) / volume
        else:
            point_density = 0.0
        
        return {
            'avg_height': avg_height,
            'avg_vz': avg_vz,
            'height_std': height_std,
            'velocity_std': velocity_std,
            'point_density': point_density,
            'height_range': height_range,
            'velocity_range': velocity_range,
        }

    def get_point_cloud_stats(self, points: np.ndarray) -> Dict[str, Any]:
        """
        获取点云统计信息
        :param points: 点云数据
        :return: 统计信息字典
        """
        if points.size == 0:
            return {
                'count': 0,
                'min_height': 0.0,
                'max_height': 0.0,
                'avg_height': 0.0,
                'min_vz': 0.0,
                'max_vz': 0.0,
                'avg_vz': 0.0,
                'std_vz': 0.0,
                'avg_intensity': 0.0,
                'centroid_x': 0.0,
                'centroid_y': 0.0,
                'centroid_z': 0.0,
                'density': 0.0,
            }

        count = len(points)
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        area = max(x_range * y_range, 0.01)  # 避免除零

        avg_intensity = float(np.mean(points[:, 4])) if points.shape[1] >= 5 else 0.0

        return {
            'count': count,
            'min_height': float(np.min(points[:, 2])),
            'max_height': float(np.max(points[:, 2])),
            'avg_height': float(np.mean(points[:, 2])),
            'min_vz': float(np.min(points[:, 3])),
            'max_vz': float(np.max(points[:, 3])),
            'avg_vz': float(np.mean(points[:, 3])),
            'std_vz': float(np.std(points[:, 3])),
            'avg_intensity': avg_intensity,
            'centroid_x': float(np.mean(points[:, 0])),
            'centroid_y': float(np.mean(points[:, 1])),
            'centroid_z': float(np.mean(points[:, 2])),
            'density': float(count / area),
        }

    def set_height_range(self, min_h: float, max_h: float) -> None:
        """
        设置高度范围
        :param min_h: 最小高度
        :param max_h: 最大高度
        """
        self.height_min = min_h
        self.height_max = max_h
        logger.info(f"高度范围已设置: {min_h}m - {max_h}m")
