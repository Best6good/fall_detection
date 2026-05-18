#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摔倒检测模块
基于规则化算法检测摔倒事件
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from config import (
    DETECTOR_CONFIG, HumanState,
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
        logger.info("摔倒检测模块已初始化")

    def reset(self) -> None:
        """重置检测状态"""
        self.consecutive_frames = 0  # 连续满足条件的帧数
        self.is_fallen = False  # 是否已检测到摔倒
        self.detection_history = []  # 检测历史记录
        logger.info("摔倒检测状态已重置")
    
    def detect(self, avg_height: float, avg_vz: float, simulator_state: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        检测摔倒事件（优化版）
        基于多特征融合的摔倒检测算法：
        1. 高度阈值检测
        2. 速度阈值检测  
        3. 高度变化率检测
        4. 连续帧验证
        
        :param avg_height: 平均高度
        :param avg_vz: 平均垂直速度
        :param simulator_state: 模拟器状态 (standing/walking/falling/fallen)
        :return: (是否检测到摔倒, 检测详情)
        """
        # 获取当前阈值
        height_threshold = get_fall_height_threshold()
        speed_threshold = get_fall_speed_threshold()
        frames_threshold = get_fall_frames_threshold()
        
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
                'condition3': True,  # 高度变化率条件
                'consecutive_frames': self.consecutive_frames,
                'frames_threshold': frames_threshold,
                'trigger': 'state_fallen',
                'confidence': 1.0,
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
                'is_pet': False
            }
            self.detection_history.append(detection_info)
            return False, detection_info
        
        # 2. 宠物过滤：在进行摔倒检测之前，先判断是否为宠物（猫/狗等小动物）
        # 宠物特征：高度低（0.1-0.5m），但有明显向下速度的可能是摔倒的人
        is_pet = avg_height < 0.5 and avg_height > 0.1  # 宠物高度范围
        
        # 如果检测到疑似宠物，降低置信度权重（减少误报警）
        # 但如果有明显的向下速度（可能是摔倒），不应用宠物惩罚
        pet_penalty = 0.0
        if is_pet and avg_vz >= -0.3:  # 只有当垂直速度较小时才应用宠物惩罚
            pet_penalty = 0.3  # 宠物惩罚因子
            logger.debug(f"检测到疑似宠物，高度: {avg_height:.2f}m")
        
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
        
        # 综合置信度计算
        confidence = 0.0
        if condition1:
            confidence += 0.4
        if condition2:
            confidence += 0.3
        if condition3:
            confidence += 0.3
        
        # 应用宠物惩罚（减少误报警）
        confidence = max(0.0, confidence - pet_penalty)
        
        # 摔倒判定：需要多个条件同时满足
        fall_detected = confidence >= 0.6  # 置信度达到60%以上
        
        # 记录检测结果
        detection_info = {
            'avg_height': avg_height,
            'avg_vz': avg_vz,
            'height_threshold': height_threshold,
            'speed_threshold': speed_threshold,
            'condition1': condition1,
            'condition2': condition2,
            'condition3': condition3,
            'consecutive_frames': self.consecutive_frames,
            'frames_threshold': frames_threshold,
            'trigger': 'threshold',
            'confidence': confidence,
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
            logger.warning(f"检测到摔倒！高度: {avg_height:.2f}m, 速度: {avg_vz:.2f}m/s, 置信度: {confidence:.2f}")
            return True, detection_info
        
        return False, detection_info

    def get_consecutive_frames(self) -> int:
        """获取连续满足条件的帧数"""
        return self.consecutive_frames

    def is_detected(self) -> bool:
        """是否已检测到摔倒"""
        return self.is_fallen

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
