#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# 配置Matplotlib中文支持
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 关闭不必要的Matplotlib功能以提高性能
plt.rcParams['toolbar'] = 'None'
plt.rcParams['interactive'] = False

# 初始化日志系统
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_timestamp_filename() -> str:
    """获取用于文件名的时间戳"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir_exists(dir_path: str) -> None:
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"创建目录: {dir_path}")


def export_to_csv(data: List[Dict[str, Any]], filepath: str) -> bool:
    """
    将数据导出为CSV文件
    :param data: 数据列表，每个元素是字典
    :param filepath: 输出文件路径
    :return: 是否成功
    """
    try:
        if not data:
            logger.warning("没有数据可导出")
            return False

        ensure_dir_exists(os.path.dirname(filepath))

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"数据已导出到: {filepath}")
        return True
    except Exception as e:
        logger.error(f"导出CSV失败: {e}")
        return False


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    计算两点之间的距离
    :param point1: 点1坐标 [x, y, z]
    :param point2: 点2坐标 [x, y, z]
    :return: 距离
    """
    import numpy as np
    return np.linalg.norm(np.array(point1) - np.array(point2))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内
    :param value: 输入值
    :param min_val: 最小值
    :param max_val: 最大值
    :return: 限制后的值
    """
    return max(min_val, min(value, max_val))


def interpolate_color(value: float, min_val: float, max_val: float, 
                     color_low: str, color_high: str) -> str:
    """
    根据值插值计算颜色
    :param value: 当前值
    :param min_val: 最小值
    :param max_val: 最大值
    :param color_low: 低值颜色 (hex)
    :param color_high: 高值颜色 (hex)
    :return: 插值后的颜色 (hex)
    """
    # 将hex颜色转换为RGB
    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # 将RGB转换为hex颜色
    def rgb_to_hex(rgb: tuple) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    # 计算归一化值
    normalized = (value - min_val) / (max_val - min_val)
    normalized = clamp(normalized, 0.0, 1.0)
    
    # 获取RGB值
    r1, g1, b1 = hex_to_rgb(color_low)
    r2, g2, b2 = hex_to_rgb(color_high)
    
    # 线性插值
    r = int(r1 + (r2 - r1) * normalized)
    g = int(g1 + (g2 - g1) * normalized)
    b = int(b1 + (b2 - b1) * normalized)
    
    return rgb_to_hex((r, g, b))


def play_beep(frequency: int = 800, duration: int = 500, repeat: int = 3) -> None:
    """
    播放蜂鸣报警声
    :param frequency: 频率（Hz）
    :param duration: 持续时间（ms）
    :param repeat: 重复次数
    """
    try:
        import winsound
        for _ in range(repeat):
            winsound.Beep(frequency, duration)
        logger.info("蜂鸣报警已播放")
    except Exception as e:
        logger.error(f"播放蜂鸣失败: {e}")


def speak_text(text: str) -> None:
    """
    使用语音合成播放文本（需要pyttsx3库）
    :param text: 要播放的文本
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        logger.info(f"语音报警已播放: {text}")
    except ImportError:
        logger.warning("pyttsx3库未安装，无法播放语音报警")
    except Exception as e:
        logger.error(f"播放语音失败: {e}")


def take_screenshot(widget, filepath: str) -> bool:
    """
    截取控件的屏幕截图
    :param widget: PyQt控件
    :param filepath: 保存路径
    :return: 是否成功
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        pixmap = widget.grab()
        pixmap.save(filepath)
        logger.info(f"截图已保存到: {filepath}")
        return True
    except Exception as e:
        logger.error(f"截图失败: {e}")
        return False


def format_number(value: float, decimals: int = 2) -> str:
    """
    格式化数字为字符串
    :param value: 数值
    :param decimals: 小数位数
    :return: 格式化后的字符串
    """
    return f"{value:.{decimals}f}"


def format_duration(seconds: float) -> str:
    """
    格式化时长为可读字符串
    :param seconds: 秒数
    :return: 格式化后的字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}小时{minutes}分{secs:.1f}秒"
    elif minutes > 0:
        return f"{minutes}分{secs:.1f}秒"
    else:
        return f"{secs:.1f}秒"
