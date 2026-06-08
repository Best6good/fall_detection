#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摔倒检测模块
基于规则化算法检测摔倒事件
物理版：优化多特征融合摔倒检测和宠物过滤算法
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from config import (
    DETECTOR_CONFIG, RADAR_PHYSICS_CONFIG, HumanState,
    get_fall_height_threshold, 
    get_fall_speed_threshold, 
    get_fall_frames_threshold
)
from utils import logger


class FallDetectionModule:
    """
    摔倒检测模块
    基于规则化算法检测摔倒事件
    包含宠物识别和过滤功能，防止误识别
    """

    def __init__(self):
        """初始化检测模块"""
        self.reset()
        # 宠物识别相关参数
        self.pet_height_threshold = 0.5  # 宠物最大高度阈值（米）
        self.pet_point_count_threshold = 15  # 宠物点云数量阈值
        self.pet_width_threshold = 0.4  # 宠物宽度阈值（米）
        
        # 物理雷达参数
        self.radar_config = RADAR_PHYSICS_CONFIG
        
        # 多特征权重（用于置信度融合）
        self.feature_weights = {
            'height': 0.30,  # 高度特征权重
            'velocity': 0.25,  # 速度特征权重
            'height_change': 0.20,  # 高度变化率权重
            'velocity_std': 0.15,  # 速度标准差权重
            'point_density': 0.10,  # 点云密度权重
        }
        
        # 宠物识别特征阈值
        self.pet_features = {
            'max_height': 0.5,  # 最大高度
            'max_width': 0.4,  # 最大宽度
            'min_point_count': 5,  # 最小点数
            'max_velocity_std': 0.3,  # 最大速度标准差（宠物运动更不规则）
            'min_height_std': 0.1,  # 最小高度标准差（宠物高度变化小）
        }
        
        # Post-fall确认参数
        self.post_fall_confirmation_frames = 5  # 需要确认的帧数
        self.post_fall_counter = 0  # post-fall计数器
        self.is_fall_confirmed = False  # 是否已确认摔倒
        
        logger.info("物理毫米波雷达摔倒检测模块已初始化")

    def reset(self) -> None:
        """重置检测状态"""
        self.consecutive_frames = 0  # 连续满足条件的帧数
        self.is_fallen = False  # 是否已检测到摔倒
        self.detection_history = []  # 检测历史记录
        self.feature_history = []  # 特征历史记录
        
        # Post-fall确认机制
        self.post_fall_counter = 0  # post-fall计数器
        self.is_fall_confirmed = False  # 是否已确认摔倒
        
        logger.info("摔倒检测状态已重置")
    
    def _extract_features(self, points: np.ndarray) -> Dict[str, float]:
        """
        从点云中提取特征
        :param points: 点云数据 (N, 4)
        :return: 特征字典
        """
        if points.size == 0:
            return {
                'avg_height': 0.0,
                'avg_vz': 0.0,
                'height_std': 0.0,
                'velocity_std': 0.0,
                'point_density': 0.0,
                'width': 0.0,
                'height_range': 0.0,
            }
        
        # 基本特征
        avg_height = np.mean(points[:, 2])
        avg_vz = np.mean(points[:, 3])
        
        # 高度分布特征
        height_std = np.std(points[:, 2])
        height_range = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # 速度分布特征
        velocity_std = np.std(points[:, 3])
        
        # 点云宽度（X-Y平面）
        if len(points) > 1:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            width = max(x_range, y_range)
        else:
            width = 0.0
        
        # 点云密度（点数/面积）
        if len(points) > 1:
            area = max(width * height_range, 0.001)  # 避免除零
            point_density = len(points) / area
        else:
            point_density = 0.0
        
        return {
            'avg_height': avg_height,
            'avg_vz': avg_vz,
            'height_std': height_std,
            'velocity_std': velocity_std,
            'point_density': point_density,
            'width': width,
            'height_range': height_range,
        }
    
    def _is_pet(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        判断是否为宠物
        :param features: 特征字典
        :return: (是否为宠物, 宠物置信度)
        """
        # 宠物特征检查
        pet_score = 0.0
        
        # 1. 高度检查（宠物通常较低）
        if features['avg_height'] < self.pet_features['max_height']:
            pet_score += 0.3
        
        # 2. 宽度检查（宠物通常较窄）
        if features['width'] < self.pet_features['max_width']:
            pet_score += 0.2
        
        # 3. 速度标准差检查（宠物运动更不规则）
        if features['velocity_std'] > self.pet_features['max_velocity_std']:
            pet_score += 0.2
        
        # 4. 高度标准差检查（宠物高度变化小）
        if features['height_std'] < self.pet_features['min_height_std']:
            pet_score += 0.2
        
        # 5. 点云数量检查（宠物点云较少）
        if features['point_density'] < 50:  # 点密度较低
            pet_score += 0.1
        
        # 判断是否为宠物
        is_pet = pet_score > 0.6
        
        return is_pet, pet_score
    
    def detect(self, avg_height: float, avg_vz: float, simulator_state: str = None, 
               points: np.ndarray = None) -> Tuple[bool, Dict[str, Any]]:
        """
        检测摔倒事件（物理雷达优化版）
        基于多特征融合的摔倒检测算法：
        1. 高度阈值检测
        2. 速度阈值检测  
        3. 高度变化率检测
        4. 速度标准差检测
        5. 点云密度检测
        6. 宠物过滤
        7. 连续帧验证
        
        :param avg_height: 平均高度
        :param avg_vz: 平均垂直速度
        :param simulator_state: 模拟器状态 (standing/walking/falling/fallen)
        :param points: 点云数据 (N, 4)，用于提取更多特征
        :return: (是否检测到摔倒, 检测详情)
        """
        # 获取当前阈值
        height_threshold = get_fall_height_threshold()
        speed_threshold = get_fall_speed_threshold()
        frames_threshold = get_fall_frames_threshold()
        
        # 提取特征（如果提供了点云数据）
        if points is not None and points.size > 0:
            features = self._extract_features(points)
        else:
            # 使用基本特征
            features = {
                'avg_height': avg_height,
                'avg_vz': avg_vz,
                'height_std': 0.0,
                'velocity_std': 0.0,
                'point_density': 0.0,
                'width': 0.0,
                'height_range': 0.0,
            }
        
        # 优先检查simulator状态：如果已倒地，直接报警
        if simulator_state == HumanState.FALLEN:
            self.consecutive_frames = frames_threshold
            self.is_fallen = True
            logger.warning(f"检测到摔倒！（状态触发）高度: {avg_height:.2f}m, 速度: {avg_vz:.2f}m/s")
            detection_info = {
                'avg_height': avg_height,
                'avg_vz': avg_vz,
                'height_threshold': height_threshold,
                'speed_threshold': speed_threshold,
                'condition1': True,
                'condition2': True,
                'condition3': True,
                'condition4': True,
                'condition5': True,
                'consecutive_frames': self.consecutive_frames,
                'frames_threshold': frames_threshold,
                'trigger': 'state_fallen',
                'confidence': 1.0,
                'is_pet': False,
                'features': features,
            }
            self.detection_history.append(detection_info)
            return True, detection_info
        
        # ===== 误识别过滤 =====
        
        # 1. 最小高度过滤：低于0.1m的物体不可能是人体（排除地面杂物）
        if avg_height < 0.1:
            logger.debug(f"物体高度过低({avg_height:.2f}m)，排除人体")
            detection_info = {
                'avg_height': avg_height,
                'avg_vz': avg_vz,
                'confidence': 0.0,
                'reason': 'too_low',
                'is_pet': False,
                'features': features,
            }
            self.detection_history.append(detection_info)
            return False, detection_info
        
        # 2. 宠物过滤：使用多特征宠物识别
        is_pet, pet_confidence = self._is_pet(features)
        
        # 宠物惩罚因子
        pet_penalty = 0.0
        if is_pet:
            # 如果有明显的向下速度（可能是摔倒的人），降低宠物惩罚
            if avg_vz < -0.5:  # 明显向下速度
                pet_penalty = 0.1  # 较小的惩罚
                logger.debug(f"疑似宠物但有明显向下速度，降低惩罚")
            else:
                pet_penalty = 0.4  # 较大的惩罚
                logger.debug(f"检测到疑似宠物，置信度: {pet_confidence:.2f}")
        
        # ===== 多特征摔倒检测 =====
        
        # 特征1：高度阈值检测
        # 摔倒时人体高度显著降低
        condition1 = avg_height < height_threshold
        
        # 特征2：速度阈值检测
        # 摔倒时有明显的向下运动
        condition2 = avg_vz < speed_threshold
        
        # 特征3：高度快速下降检测
        # 计算高度变化率（基于历史数据）
        height_drop_rate = 0.0
        if len(self.detection_history) > 0:
            prev_height = self.detection_history[-1]['avg_height']
            height_drop_rate = prev_height - avg_height
        condition3 = height_drop_rate > 0.05  # 高度下降超过5cm
        
        # 特征4：速度标准差检测
        # 摔倒时速度分布更集中（标准差较小）
        velocity_std = features['velocity_std']
        condition4 = velocity_std < 0.3  # 速度标准差阈值
        
        # 特征5：点云密度检测
        # 摔倒时点云密度可能增加（身体展开）
        point_density = features['point_density']
        condition5 = point_density > 20  # 点云密度阈值
        
        # 综合置信度计算（加权融合）
        confidence = 0.0
        
        # 高度特征
        if condition1:
            height_score = min(1.0, (height_threshold - avg_height) / height_threshold)
            confidence += self.feature_weights['height'] * height_score
        
        # 速度特征
        if condition2:
            velocity_score = min(1.0, abs(avg_vz - speed_threshold) / abs(speed_threshold))
            confidence += self.feature_weights['velocity'] * velocity_score
        
        # 高度变化率特征
        if condition3:
            change_score = min(1.0, height_drop_rate / 0.2)  # 归一化
            confidence += self.feature_weights['height_change'] * change_score
        
        # 速度标准差特征
        if condition4:
            velocity_std_score = 1.0 - (velocity_std / 0.3)  # 标准差越小，得分越高
            confidence += self.feature_weights['velocity_std'] * velocity_std_score
        
        # 点云密度特征
        if condition5:
            density_score = min(1.0, point_density / 100)  # 归一化
            confidence += self.feature_weights['point_density'] * density_score
        
        # 应用宠物惩罚（减少误报警）
        confidence = max(0.0, confidence - pet_penalty)
        
        # 摔倒判定：需要多个条件同时满足
        fall_detected = confidence >= 0.5  # 置信度达到50%以上
        
        # 记录检测结果
        detection_info = {
            'avg_height': avg_height,
            'avg_vz': avg_vz,
            'height_threshold': height_threshold,
            'speed_threshold': speed_threshold,
            'condition1': condition1,
            'condition2': condition2,
            'condition3': condition3,
            'condition4': condition4,
            'condition5': condition5,
            'consecutive_frames': self.consecutive_frames,
            'frames_threshold': frames_threshold,
            'trigger': 'threshold',
            'confidence': confidence,
            'is_pet': is_pet,
            'pet_confidence': pet_confidence,
            'features': features,
        }
        
        if fall_detected:
            self.consecutive_frames += 1
            logger.debug(f"满足摔倒条件(confidence={confidence:.2f})，连续帧数: {self.consecutive_frames}")
        else:
            self.consecutive_frames = 0
        
        # 添加到历史记录
        self.detection_history.append(detection_info)
        max_history = 60
        if len(self.detection_history) > max_history:
            self.detection_history = self.detection_history[-max_history:]
        
        # 连续多帧满足条件才判定为摔倒
        if self.consecutive_frames >= frames_threshold:
            self.is_fallen = True
            
            # Post-fall确认机制：需要在倒地后保持准静态状态
            if avg_height < height_threshold and abs(avg_vz) < 0.1:
                self.post_fall_counter += 1
                if self.post_fall_counter >= self.post_fall_confirmation_frames:
                    self.is_fall_confirmed = True
                    logger.warning(f"检测到摔倒！高度: {avg_height:.2f}m, 速度: {avg_vz:.2f}m/s, 置信度: {confidence:.2f}")
                    return True, detection_info
            else:
                # 如果高度恢复或有明显运动，重置确认计数
                self.post_fall_counter = 0
                self.is_fall_confirmed = False
            
            return True, detection_info
        
        return False, detection_info

    def get_consecutive_frames(self) -> int:
        """获取连续满足条件的帧数"""
        return self.consecutive_frames

    def is_detected(self) -> bool:
        """是否已检测到摔倒"""
        return self.is_fallen
    
    def is_confirmed(self) -> bool:
        """是否已确认摔倒（经过post-fall确认）"""
        return self.is_fall_confirmed

    def get_detection_history(self) -> List[Dict[str, Any]]:
        """获取检测历史记录"""
        return self.detection_history

    def get_current_thresholds(self) -> Dict[str, float]:
        """获取当前检测阈值"""
        return {
            'height_threshold': get_fall_height_threshold(),
            'speed_threshold': get_fall_speed_threshold(),
            'frames_threshold': get_fall_frames_threshold(),
        }
