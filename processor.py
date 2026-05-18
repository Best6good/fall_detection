#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云预处理模块
对原始点云进行滤波去噪
"""

import numpy as np
from typing import Tuple, Dict, Any
from config import PROCESSOR_CONFIG
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
        logger.info("点云预处理模块已初始化")

    def _pass_through_filter(self, points: np.ndarray) -> np.ndarray:
        """
        直通滤波：只保留指定高度范围内的点
        :param points: 原始点云 (N, 4)
        :return: 滤波后的点云
        """
        if points.size == 0:
            return points
        
        mask = (points[:, 2] >= self.height_min) & (points[:, 2] <= self.height_max)
        filtered = points[mask]
        
        logger.debug(f"直通滤波: 输入 {len(points)} 点, 输出 {len(filtered)} 点")
        return filtered

    def _statistical_filter(self, points: np.ndarray) -> np.ndarray:
        """
        统计滤波：去除离群噪声点
        :param points: 输入点云 (N, 4)
        :return: 滤波后的点云
        """
        if points.size == 0 or len(points) <= self.statistical_k:
            return points
        
        # 计算每个点到其他点的平均距离
        distances = []
        for i in range(len(points)):
            diff = points[:, :3] - points[i, :3]
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            dist.sort()
            avg_dist = np.mean(dist[1:self.statistical_k+1])  # 排除自身
            distances.append(avg_dist)
        
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # 保留在标准差范围内的点
        mask = distances < (mean_dist + self.statistical_std_ratio * std_dist)
        filtered = points[mask]
        
        logger.debug(f"统计滤波: 输入 {len(points)} 点, 输出 {len(filtered)} 点")
        return filtered

    def process(self, raw_points: np.ndarray) -> np.ndarray:
        """
        完整预处理流程
        :param raw_points: 原始点云
        :return: 预处理后的点云
        """
        if raw_points.size == 0:
            return raw_points
        
        try:
            # 直通滤波
            filtered = self._pass_through_filter(raw_points)
            # 统计滤波
            filtered = self._statistical_filter(filtered)
            
            return filtered
        except Exception as e:
            logger.error(f"点云预处理失败: {e}")
            return raw_points

    def get_frame_features(self, points: np.ndarray) -> Tuple[float, float]:
        """
        获取单帧点云的特征
        :param points: 点云数据
        :return: (平均高度, 平均垂直速度)
        """
        if points.size == 0:
            return (0.0, 0.0)
        
        avg_height = np.mean(points[:, 2])
        avg_vz = np.mean(points[:, 3])
        
        return (avg_height, avg_vz)

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
            }
        
        return {
            'count': len(points),
            'min_height': float(np.min(points[:, 2])),
            'max_height': float(np.max(points[:, 2])),
            'avg_height': float(np.mean(points[:, 2])),
            'min_vz': float(np.min(points[:, 3])),
            'max_vz': float(np.max(points[:, 3])),
            'avg_vz': float(np.mean(points[:, 3])),
            'std_vz': float(np.std(points[:, 3])),
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
