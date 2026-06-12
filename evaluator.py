#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估框架模块
提供摔倒检测系统的量化评估指标

支持：
- 混淆矩阵（TP/FP/FN/TN）
- 精确率、召回率、F1分数
- 检测延迟
- 误报率、漏报率
- 批量评估和报告生成
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from config import HumanState


@dataclass
class DetectionEvent:
    """单次检测事件记录"""
    frame_idx: int
    true_state: str
    predicted_fall: bool
    confidence: float
    timestamp: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """评估结果"""
    total_frames: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_alarm_rate: float
    miss_rate: float
    avg_detection_delay: float
    events: List[DetectionEvent] = field(default_factory=list)


class Evaluator:
    """
    摔倒检测评估器
    
    使用方法：
        evaluator = Evaluator()
        for frame in test_sequence:
            is_fall, info = detector.detect(...)
            evaluator.add_frame(frame.true_state, is_fall, info['confidence'])
        result = evaluator.evaluate()
    """

    def __init__(self):
        self.events: List[DetectionEvent] = []
        self._fall_start_idx: Optional[int] = None

    def reset(self):
        self.events.clear()
        self._fall_start_idx = None

    def add_frame(self, true_state: str, predicted_fall: bool,
                  confidence: float = 0.0, features: Optional[Dict] = None):
        """
        记录一帧的检测结果
        
        :param true_state: 真实状态（HumanState 值）
        :param predicted_fall: 检测器是否判定为摔倒
        :param confidence: 检测置信度
        :param features: 可选的特征字典
        """
        idx = len(self.events)
        event = DetectionEvent(
            frame_idx=idx,
            true_state=true_state,
            predicted_fall=predicted_fall,
            confidence=confidence,
            timestamp=time.time(),
            features=features or {},
        )
        self.events.append(event)

        # 跟踪摔倒起始帧
        if true_state == HumanState.FALLING and self._fall_start_idx is None:
            self._fall_start_idx = idx

    def evaluate(self) -> EvaluationResult:
        """
        计算评估指标
        :return: EvaluationResult
        """
        if not self.events:
            return EvaluationResult(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 定义"真实摔倒"：FALLING 或 FALLEN 状态
        fall_states = {HumanState.FALLING, HumanState.FALLEN}

        tp = fp = tn = fn = 0
        delays: List[int] = []
        fall_detected_for_current = False
        fall_start: Optional[int] = None

        for i, evt in enumerate(self.events):
            is_real_fall = evt.true_state in fall_states

            # 跟踪摔倒区间
            if evt.true_state == HumanState.FALLING and fall_start is None:
                fall_start = i
                fall_detected_for_current = False

            if is_real_fall and evt.predicted_fall:
                tp += 1
                if fall_start is not None and not fall_detected_for_current:
                    delays.append(i - fall_start)
                    fall_detected_for_current = True
            elif is_real_fall and not evt.predicted_fall:
                fn += 1
            elif not is_real_fall and evt.predicted_fall:
                fp += 1
            else:
                tn += 1

            # 摔倒结束
            if evt.true_state not in fall_states and fall_start is not None:
                fall_start = None
                fall_detected_for_current = False

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        miss = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        avg_delay = float(np.mean(delays)) if delays else 0.0

        return EvaluationResult(
            total_frames=total,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_alarm_rate=far,
            miss_rate=miss,
            avg_detection_delay=avg_delay,
            events=self.events,
        )

    @staticmethod
    def format_report(result: EvaluationResult) -> str:
        """格式化评估报告"""
        lines = [
            "=" * 50,
            "摔倒检测评估报告",
            "=" * 50,
            f"总帧数:         {result.total_frames}",
            f"真正例 (TP):    {result.true_positives}",
            f"假正例 (FP):    {result.false_positives}",
            f"真负例 (TN):    {result.true_negatives}",
            f"假负例 (FN):    {result.false_negatives}",
            "-" * 50,
            f"精确率 (Precision):  {result.precision:.4f}",
            f"召回率 (Recall):     {result.recall:.4f}",
            f"F1 分数:             {result.f1_score:.4f}",
            f"误报率 (FAR):        {result.false_alarm_rate:.4f}",
            f"漏报率 (Miss Rate):  {result.miss_rate:.4f}",
            f"平均检测延迟:        {result.avg_detection_delay:.1f} 帧",
            "=" * 50,
        ]
        return "\n".join(lines)
