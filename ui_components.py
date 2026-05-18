#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义UI组件模块
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSlider, 
    QPushButton, QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QFormLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon

from config import COLOR_CONFIG, VISUALIZATION_CONFIG, PROJECT_INFO
from utils import logger


class SkeletonVisualizer(FigureCanvas):
    """
    3D人体骨架可视化组件
    显示从点云推断出的人体骨架结构
    """

    def __init__(self, parent=None):
        """初始化骨架可视化器"""
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=COLOR_CONFIG["background"])
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.skeleton_joints = np.array([])
        self.skeleton_connections = []
        self.current_state = "standing"
        self.fall_progress = 0.0  # 摔倒进度
        
        # 骨架连接定义
        self._init_skeleton_connections()
        
        # 关节和骨骼的绘图对象
        self.joints_scatter = None
        self.bones_plot = []
        
        self._init_plot()

    def _init_skeleton_connections(self):
        """定义人体骨架的连接关系"""
        # 关节索引: 0-head, 1-neck, 2-chest, 3-waist, 
        #          4-left_shoulder, 5-right_shoulder,
        #          6-left_elbow, 7-right_elbow,
        #          8-left_hand, 9-right_hand,
        #          10-left_hip, 11-right_hip,
        #          12-left_knee, 13-right_knee,
        #          14-left_foot, 15-right_foot
        self.skeleton_connections = [
            (0, 1),   # head -> neck
            (1, 2),   # neck -> chest
            (2, 3),   # chest -> waist
            (2, 4),   # chest -> left_shoulder
            (2, 5),   # chest -> right_shoulder
            (4, 6),   # left_shoulder -> left_elbow
            (5, 7),   # right_shoulder -> right_elbow
            (6, 8),   # left_elbow -> left_hand
            (7, 9),   # right_elbow -> right_hand
            (3, 10),  # waist -> left_hip
            (3, 11),  # waist -> right_hip
            (10, 12), # left_hip -> left_knee
            (11, 13), # right_hip -> right_knee
            (12, 14), # left_knee -> left_foot
            (13, 15), # right_knee -> right_foot
        ]
        
        # 关节名称
        self.joint_names = [
            'head', 'neck', 'chest', 'waist',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_hand', 'right_hand',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_foot', 'right_foot'
        ]

    def _init_plot(self):
        """初始化3D绘图区域"""
        self.ax.clear()
        self.ax.set_xlabel('X (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_ylabel('Y (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_zlabel('Z (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2.0)
        
        # 设置背景色和网格颜色
        self.ax.set_facecolor(COLOR_CONFIG["background"])
        self.ax.grid(color=COLOR_CONFIG["grid"], alpha=0.3)
        self.ax.xaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.yaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.zaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.tick_params(axis='x', colors=COLOR_CONFIG["axis"], labelsize=7)
        self.ax.tick_params(axis='y', colors=COLOR_CONFIG["axis"], labelsize=7)
        self.ax.tick_params(axis='z', colors=COLOR_CONFIG["axis"], labelsize=7)
        
        # 设置视角（与点云视图一致）
        self.ax.view_init(elev=VISUALIZATION_CONFIG["view_elevation"], 
                         azim=VISUALIZATION_CONFIG["view_azimuth"])
        
        # 禁用自动缩放
        self.ax.autoscale(False)
        
        # 添加标题
        self.ax.set_title('人体骨架', color=COLOR_CONFIG["text"], fontsize=11, pad=10)
        
        self.fig.subplots_adjust(left=0.12, right=0.9, bottom=0.12, top=0.88)
    
    def clear(self):
        """清空骨架显示"""
        self.skeleton_joints = np.array([])
        self.current_state = "standing"
        self.fall_progress = 0.0
        
        # 清除绘图对象
        if self.joints_scatter:
            self.joints_scatter.remove()
            self.joints_scatter = None
        
        for bone in self.bones_plot:
            bone.remove()
        self.bones_plot = []
        
        # 重新初始化绘图区域
        self._init_plot()
        self.draw()

    def _estimate_skeleton_from_pointcloud(self, points: np.ndarray, state: str) -> np.ndarray:
        """
        从点云数据推断人体骨架（模拟Point Cloud Transformer处理）
        基于点云的空间分布特征和聚类分析推断骨架结构
        
        处理流程：
        1. 点云预处理和分割
        2. 高度分层分析（识别人体部位）
        3. 空间聚类（找到身体各部分的中心）
        4. 骨架结构推理
        
        :param points: 点云数据
        :param state: 当前状态
        :return: 骨架关节坐标数组 (16, 3)
        """
        if points.size == 0:
            return self._get_default_skeleton(state)
        
        # 获取点云统计信息
        x_mean, y_mean, z_mean = np.mean(points[:, :3], axis=0)
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_std = np.std(points[:, 2])
        
        # 初始化骨架（站立姿态）
        skeleton = self._get_default_skeleton(state)
        
        # ===== 基于点云分析的骨架推断 =====
        
        # 1. 高度分层分析：识别人体各部位的大致高度
        # 使用聚类思想：将点云按高度分层
        z_height = points[:, 2]
        
        # 计算身体各部位的特征高度（基于点云分布）
        # 头部特征：z方向最高点集中区域
        # 躯干特征：z方向中间区域，密度最高
        # 腿部特征：z方向最低区域
        
        # 高度分层聚类（模拟Point Cloud Transformer的特征提取）
        if state in ["standing", "walking"]:
            # 站立状态：人体直立，各部位高度分明
            
            # 识别人体中心线（基于密度估计）
            body_center_z = z_mean
            
            # 根据点云高度分布推断各部位
            # 头部：最高点附近
            head_height = z_max - 0.05
            # 颈部：头部下方
            neck_height = head_height - 0.12
            # 胸部：身体中心偏上
            chest_height = head_height - 0.20
            # 腰部：胸部下方
            waist_height = chest_height - 0.18
            
            # 下肢
            hip_height = waist_height - 0.15
            knee_height = hip_height - 0.32
            foot_height = knee_height - 0.28
            
            # 上肢（相对于躯干）
            shoulder_offset = 0.18
            elbow_offset = 0.30
            hand_offset = 0.38
            
            # 更新骨架各关节高度
            skeleton[0, 2] = head_height       # head
            skeleton[1, 2] = neck_height       # neck
            skeleton[2, 2] = chest_height      # chest
            skeleton[3, 2] = waist_height      # waist
            skeleton[4, 2] = chest_height + 0.03  # left_shoulder
            skeleton[5, 2] = chest_height + 0.03  # right_shoulder
            skeleton[6, 2] = chest_height - 0.08  # left_elbow
            skeleton[7, 2] = chest_height - 0.08  # right_elbow
            skeleton[8, 2] = chest_height - 0.18  # left_hand
            skeleton[9, 2] = chest_height - 0.18  # right_hand
            skeleton[10, 2] = hip_height         # left_hip
            skeleton[11, 2] = hip_height         # right_hip
            skeleton[12, 2] = knee_height        # left_knee
            skeleton[13, 2] = knee_height        # right_knee
            skeleton[14, 2] = foot_height        # left_foot
            skeleton[15, 2] = foot_height        # right_foot
            
        elif state == "falling":
            # 摔倒过程：根据点云高度变化推断身体倾斜
            progress = getattr(self, 'fall_progress', 0.0)
            
            # 基于点云高度范围判断摔倒程度
            height_range = z_max - z_min
            
            # 如果点云变矮，说明正在倒下
            if height_range < 1.2:  # 人体高度范围缩小
                # 身体正在倾斜/倒下
                tilt_angle = (1.2 - height_range) * 0.8
                
                # 计算倾斜后的各部位高度
                base_height = z_min + 0.15  # 身体底部高度
                
                # 头部最低（撞击地面）
                head_height = base_height + 0.02
                # 躯干逐渐增高
                neck_height = base_height + 0.05
                chest_height = base_height + 0.10
                waist_height = base_height + 0.12
                
                # 下肢最高（抬起）
                hip_height = base_height + 0.15
                knee_height = base_height + 0.20
                foot_height = base_height + 0.25
                
                skeleton[0, 2] = head_height
                skeleton[1, 2] = neck_height
                skeleton[2, 2] = chest_height
                skeleton[3, 2] = waist_height
                skeleton[10, 2] = hip_height
                skeleton[11, 2] = hip_height
                skeleton[12, 2] = knee_height
                skeleton[13, 2] = knee_height
                skeleton[14, 2] = foot_height
                skeleton[15, 2] = foot_height
                
                # 上肢高度
                skeleton[4, 2] = chest_height + 0.02
                skeleton[5, 2] = chest_height + 0.02
                skeleton[6, 2] = chest_height - 0.05
                skeleton[7, 2] = chest_height - 0.05
                skeleton[8, 2] = chest_height - 0.10
                skeleton[9, 2] = chest_height - 0.10
            else:
                # 还在站立
                skeleton = self._get_default_skeleton("standing")
                
        else:  # fallen
            # 倒地状态：所有关节接近地面
            base_height = z_min + 0.10
            
            # 平躺姿势：所有关节高度相近
            skeleton[:, 2] = base_height
            skeleton[0, 2] = base_height + 0.02  # 头部稍高
        
        # 2. 水平位置调整（基于点云中心）
        skeleton[:, 0] += x_mean - np.mean(skeleton[2:4, 0])  # 以躯干为中心
        skeleton[:, 1] += y_mean - np.mean(skeleton[2:4, 1])
        
        # 3. 添加处理误差（模拟真实点云处理的误差）
        # 高度推断误差较大（±0.05m），水平位置误差较小（±0.02m）
        skeleton[:, 0] += np.random.normal(0, 0.02, 16)
        skeleton[:, 1] += np.random.normal(0, 0.02, 16)
        skeleton[:, 2] += np.random.normal(0, 0.05, 16)
        
        return skeleton

    def _get_default_skeleton(self, state: str) -> np.ndarray:
        """获取指定状态的默认骨架"""
        # 站立状态骨架
        standing_skeleton = np.array([
            [0.0, 0.0, 1.95],   # head
            [0.0, 0.0, 1.82],   # neck
            [0.0, 0.0, 1.70],   # chest
            [0.0, 0.0, 1.50],   # waist
            [-0.18, 0.0, 1.75], # left_shoulder
            [0.18, 0.0, 1.75],  # right_shoulder
            [-0.30, 0.0, 1.60], # left_elbow
            [0.30, 0.0, 1.60],  # right_elbow
            [-0.40, 0.0, 1.45], # left_hand
            [0.40, 0.0, 1.45],  # right_hand
            [-0.12, 0.0, 1.30], # left_hip
            [0.12, 0.0, 1.30],  # right_hip
            [-0.10, 0.0, 1.00], # left_knee
            [0.10, 0.0, 1.00],  # right_knee
            [-0.10, 0.0, 0.75], # left_foot
            [0.10, 0.0, 0.75],  # right_foot
        ])
        
        if state == "fallen":
            # 倒地状态骨架（平躺）
            fallen_skeleton = standing_skeleton.copy()
            # 旋转到水平位置
            fallen_skeleton[:, [0, 2]] = standing_skeleton[:, [2, 0]] * [1, -1]
            fallen_skeleton[:, 2] = 0.08 + fallen_skeleton[:, 2] * 0.1
            return fallen_skeleton
        
        elif state == "falling":
            # 摔倒过程中，插值过渡
            # 这里简化处理，使用站立骨架
            return standing_skeleton
        
        else:
            # 站立或行走
            return standing_skeleton

    def update_skeleton(self, points: np.ndarray, state: str = "standing", fall_progress: float = 0.0):
        """
        更新骨架显示
        :param points: 点云数据
        :param state: 当前状态
        :param fall_progress: 摔倒进度（0-1）
        """
        self.current_state = state
        self.fall_progress = fall_progress
        
        # 从点云推断骨架
        self.skeleton_joints = self._estimate_skeleton_from_pointcloud(points, state)
        
        # 清除旧的绘图
        if self.joints_scatter:
            self.joints_scatter.remove()
            self.joints_scatter = None
        
        for bone in self.bones_plot:
            bone.remove()
        self.bones_plot = []
        
        # 绘制骨骼（连接线）- 骨骼颜色与状态对应
        if state == "standing" or state == "walking":
            bone_color = '#00cc00'  # 绿色
        elif state == "falling":
            bone_color = '#ff8800'  # 橙色
        else:  # fallen
            bone_color = '#ff0000'  # 红色
        
        for (start_idx, end_idx) in self.skeleton_connections:
            start = self.skeleton_joints[start_idx]
            end = self.skeleton_joints[end_idx]
            
            bone, = self.ax.plot3D(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=bone_color,
                linewidth=3,
                alpha=0.9
            )
            self.bones_plot.append(bone)
        
        # 计算关节点的颜色（与点云颜色对应）
        # 使用与点云相同的归一化范围
        z_min, z_max = 0.2, 1.8
        z_values = self.skeleton_joints[:, 2]
        normalized_z = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0, 1)
        
        # 使用viridis颜色映射
        cmap = cm.get_cmap('viridis')
        joint_colors = cmap(normalized_z)
        
        # 绘制关节点（颜色与点云对应）
        self.joints_scatter = self.ax.scatter(
            self.skeleton_joints[:, 0],
            self.skeleton_joints[:, 1],
            self.skeleton_joints[:, 2],
            c=joint_colors,
            marker='o',
            s=60,
            alpha=1.0
        )
        
        self.draw()

    def clear(self):
        """清空显示"""
        self.skeleton_joints = np.array([])
        self._init_plot()
        self.draw()


class PointCloudVisualizer(FigureCanvas):
    """
    3D点云可视化组件
    """

    def __init__(self, parent=None):
        """初始化可视化器"""
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COLOR_CONFIG["background"])
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self.points = np.array([])
        self.trajectory = []
        self.center_trajectory = []
        self.alarm_position = None
        self.scatter_plot = None
        self.trajectory_plot = None
        self.trajectory_scatter = None
        self.center_trajectory_plot = None
        self.alarm_marker = None
        
        # 报警状态
        self.is_alarming = False
        self.is_flashing = False
        self.flash_timer = QTimer()
        self.flash_timer.setInterval(300)
        self.flash_timer.timeout.connect(self._flash_points)
        
        # 初始化颜色映射
        self.viridis_cmap = cm.get_cmap('viridis')
        self.OrRd_cmap = cm.get_cmap('OrRd')
        
        self._init_plot()

    def _init_plot(self):
        """初始化3D绘图区域"""
        self.ax.clear()
        self.ax.set_xlabel('X (m)', color=COLOR_CONFIG["text"], fontsize=10)
        self.ax.set_ylabel('Y (m)', color=COLOR_CONFIG["text"], fontsize=10)
        self.ax.set_zlabel('Z (m)', color=COLOR_CONFIG["text"], fontsize=10)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2.0)
        
        # 设置背景色和网格颜色
        self.ax.set_facecolor(COLOR_CONFIG["background"])
        self.ax.grid(color=COLOR_CONFIG["grid"], alpha=0.3)
        self.ax.xaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.yaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.zaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.tick_params(axis='x', colors=COLOR_CONFIG["axis"], labelsize=8)
        self.ax.tick_params(axis='y', colors=COLOR_CONFIG["axis"], labelsize=8)
        self.ax.tick_params(axis='z', colors=COLOR_CONFIG["axis"], labelsize=8)
        
        # 设置视角
        self.ax.view_init(elev=VISUALIZATION_CONFIG["view_elevation"], 
                         azim=VISUALIZATION_CONFIG["view_azimuth"])
        
        # 禁用自动缩放
        self.ax.autoscale(False)
        
        # 添加标题
        self.ax.set_title('模拟点云', color=COLOR_CONFIG["text"], fontsize=12, pad=10)
        
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    def update_points(self, points: np.ndarray, state: str = "standing", is_alarming: bool = False) -> None:
        """
        更新点云数据（增量更新）
        :param points: 点云数据
        :param state: 当前状态
        :param is_alarming: 是否处于报警状态
        """
        self.points = points
        
        # 更新报警状态
        if is_alarming and state == "fallen":
            self.is_alarming = True
            if not self.flash_timer.isActive():
                self.flash_timer.start()
        else:
            self.is_alarming = False
            self.is_flashing = False
            if self.flash_timer.isActive():
                self.flash_timer.stop()
        
        # 根据状态选择颜色映射
        if state == "falling":
            # 摔倒过程：橙色→红色渐变，点大小增大到50
            cmap = self.OrRd_cmap
            point_size = 50
        elif state == "fallen":
            # 倒地状态：纯红色，可能闪烁
            self._update_points_fallen(points)
            return
        else:
            # 正常状态：viridis渐变，点大小30
            cmap = self.viridis_cmap
            point_size = VISUALIZATION_CONFIG["point_size"]
        
        # 清除旧的点云
        if self.scatter_plot:
            self.scatter_plot.remove()
            self.scatter_plot = None
        
        if self.points.size > 0:
            z_values = self.points[:, 2]
            z_min, z_max = 0.2, 1.8
            normalized_z = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0, 1)
            colors = cmap(normalized_z)
            
            self.scatter_plot = self.ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
                c=colors,
                marker='o',
                s=point_size,
                alpha=0.8
            )
        
        self.draw()

    def _update_points_fallen(self, points: np.ndarray) -> None:
        """
        倒地状态点云更新（纯红色，可能闪烁）
        """
        if self.scatter_plot:
            self.scatter_plot.remove()
            self.scatter_plot = None
        
        if points.size > 0:
            alpha = 0.8 if not self.is_flashing else 0.3
            colors = np.array([[1.0, 0.0, 0.0, alpha]] * len(points))
            point_size = 50
            
            self.scatter_plot = self.ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                marker='o',
                s=point_size,
                alpha=alpha
            )
        
        if self.alarm_marker:
            self.alarm_marker.remove()
            self.alarm_marker = None
        
        if self.alarm_position:
            alarm_alpha = 0.6 if not self.is_flashing else 0.2
            self.alarm_marker = self.ax.scatter(
                self.alarm_position[0],
                self.alarm_position[1],
                0.05,
                c='red',
                marker='o',
                s=400,
                alpha=alarm_alpha
            )
        
        self.draw()

    def _flash_points(self):
        """点云和标记闪烁效果"""
        self.is_flashing = not self.is_flashing
        if self.points.size > 0 and self.alarm_position:
            self._update_points_fallen(self.points)

    def update_trajectory(self, trajectory: list, is_alarming: bool = False) -> None:
        """
        更新轨迹显示
        :param trajectory: 轨迹点列表
        :param is_alarming: 是否报警状态（红色加粗）
        """
        self.trajectory = trajectory

        if self.trajectory_plot:
            self.trajectory_plot.remove()
            self.trajectory_plot = None
        if self.trajectory_scatter:
            self.trajectory_scatter.remove()
            self.trajectory_scatter = None

        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            # 报警时轨迹线变为红色加粗
            color = '#ff0000' if is_alarming else COLOR_CONFIG["trajectory"]
            linewidth = 4 if is_alarming else 2
            
            self.trajectory_plot, = self.ax.plot(
                traj_array[:, 0],
                traj_array[:, 1],
                traj_array[:, 2],
                color=color,
                linewidth=linewidth,
                linestyle='--',
                alpha=0.7
            )
            self.trajectory_scatter = self.ax.scatter(
                traj_array[:, 0],
                traj_array[:, 1],
                traj_array[:, 2],
                c=color,
                marker='s',
                s=30,
                alpha=0.8
            )

        self.draw()

    def update_center_trajectory(self, center_position: tuple) -> None:
        """
        更新人体中心轨迹线（白色）
        """
        self.center_trajectory.append(center_position)

        max_frames = 10
        if len(self.center_trajectory) > max_frames:
            self.center_trajectory = self.center_trajectory[-max_frames:]

        if self.center_trajectory_plot:
            self.center_trajectory_plot.remove()
            self.center_trajectory_plot = None

        if len(self.center_trajectory) > 1:
            traj_array = np.array(self.center_trajectory)
            self.center_trajectory_plot, = self.ax.plot3D(
                traj_array[:, 0],
                traj_array[:, 1],
                traj_array[:, 2],
                color=COLOR_CONFIG["center_trajectory"],
                linewidth=2
            )

        self.draw()

    def show_alarm_marker(self, position: tuple) -> None:
        """
        显示报警标记（半透明红色圆形，半径0.5m）
        """
        self.alarm_position = position
        
        if self.alarm_marker:
            self.alarm_marker.remove()
            self.alarm_marker = None
        
        self.alarm_marker = self.ax.scatter(
            position[0],
            position[1],
            0.05,
            c='red',
            marker='o',
            s=400,
            alpha=0.6
        )
        
        self.draw()

    def clear_alarm_marker(self) -> None:
        """清除报警标记"""
        if self.alarm_marker:
            self.alarm_marker.remove()
            self.alarm_marker = None
        self.alarm_position = None
        self.is_alarming = False
        self.is_flashing = False
        if self.flash_timer.isActive():
            self.flash_timer.stop()
        self.draw()

    def clear(self) -> None:
        """清空显示"""
        self.points = np.array([])
        self.trajectory = []
        self.center_trajectory = []
        self.alarm_position = None
        self.scatter_plot = None
        self.trajectory_plot = None
        self.trajectory_scatter = None
        self.center_trajectory_plot = None
        self.alarm_marker = None
        self.is_alarming = False
        self.is_flashing = False
        if self.flash_timer.isActive():
            self.flash_timer.stop()
        self._init_plot()
        self.draw()


class RealTimePlot(FigureCanvas):
    """
    实时曲线图组件
    """

    def __init__(self, title: str, y_label: str, parent=None):
        """初始化实时曲线图"""
        self.fig = Figure(figsize=(4, 2), dpi=100, facecolor=COLOR_CONFIG["background"])
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.title = title
        self.y_label = y_label
        self.data = []
        self.max_points = VISUALIZATION_CONFIG["history_seconds"] * 10
        self.line, = self.ax.plot([], [], color='#00aaff', linewidth=2)
        self.alarm_lines = []
        self.alarm_fill = None
        self.is_paused = False
        
        self._init_plot()

    def _init_plot(self):
        """初始化绘图区域"""
        self.ax.set_title(self.title, color=COLOR_CONFIG["text"], fontsize=12)
        self.ax.set_xlabel('时间 (帧)', color=COLOR_CONFIG["text"], fontsize=10)
        self.ax.set_ylabel(self.y_label, color=COLOR_CONFIG["text"], fontsize=10)
        self.ax.set_facecolor(COLOR_CONFIG["background"])
        self.ax.grid(color=COLOR_CONFIG["grid"], alpha=0.3)
        self.ax.tick_params(axis='x', colors=COLOR_CONFIG["axis"], labelsize=8)
        self.ax.tick_params(axis='y', colors=COLOR_CONFIG["axis"], labelsize=8)
        self.ax.autoscale(False)
        self.fig.tight_layout(pad=0.3)

    def add_data(self, value: float) -> None:
        """
        添加新数据点
        """
        if self.is_paused:
            return
            
        self.data.append(value)
        
        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]
        
        self.line.set_xdata(range(len(self.data)))
        self.line.set_ydata(self.data)
        
        if len(self.data) > 1:
            y_min = min(self.data) - 0.1 * abs(min(self.data)) if min(self.data) != 0 else -0.1
            y_max = max(self.data) + 0.1 * abs(max(self.data)) if max(self.data) != 0 else 0.1
            self.ax.set_ylim(y_min, y_max)
        
        self.ax.set_xlim(0, len(self.data))
        self.draw()

    def add_alarm_marker(self, position: int, pause: bool = True) -> None:
        """
        在指定位置添加报警标记
        :param position: 帧位置
        :param pause: 是否暂停更新
        """
        line = self.ax.axvline(x=position, color='#dc2626', linestyle='-', linewidth=3, alpha=0.9)
        self.alarm_lines.append(line)
        
        if len(self.data) > 0:
            start_frame = max(0, position - 10)
            end_frame = min(len(self.data), position + 10)
            
            x_fill = range(start_frame, end_frame)
            y_fill = [self.data[i] if i < len(self.data) else 0 for i in x_fill]
            
            y_lim = self.ax.get_ylim()
            self.alarm_fill = self.ax.fill_between(
                x_fill, 
                y_lim[0], 
                y_lim[1], 
                where=[True]*len(x_fill),
                color='#dc2626',
                alpha=0.2
            )
        
        if pause:
            self.is_paused = True
        
        self.draw()

    def resume(self) -> None:
        """恢复曲线更新"""
        self.is_paused = False
        self.draw()

    def clear_alarm_markers(self) -> None:
        """清除所有报警标记"""
        for line in self.alarm_lines:
            line.remove()
        self.alarm_lines = []
        
        if self.alarm_fill:
            self.alarm_fill.remove()
            self.alarm_fill = None
        
        self.is_paused = False
        self.draw()

    def clear(self) -> None:
        """清空数据"""
        self.data = []
        self.clear_alarm_markers()
        self.line.set_xdata([])
        self.line.set_ydata([])
        self.draw()


class StatusCard(QGroupBox):
    """
    状态卡片组件
    """

    def __init__(self, title: str, parent=None):
        """初始化状态卡片"""
        super().__init__(title, parent)
        self.layout = QVBoxLayout(self)
        self.status_label = QLabel('待机')
        self.status_label.setObjectName('status_label')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 24px; font-weight: bold;')
        self.layout.addWidget(self.status_label)
        
        # 闪烁定时器
        self.flash_timer = QTimer()
        self.flash_timer.setInterval(500)
        self.flash_timer.timeout.connect(self._flash_status)
        self.is_flashing = False
        
    def set_status(self, status: str, is_alarming: bool = False) -> None:
        """
        设置状态
        :param status: 状态名称
        :param is_alarming: 是否报警状态
        """
        self.status_label.setText(status)
        
        if is_alarming:
            # 报警状态：红色背景，白色文字闪烁
            self.setStyleSheet('background-color: #dc2626; border: 2px solid #ffffff;')
            self.status_label.setStyleSheet('color: #ffffff; font-size: 24px; font-weight: bold;')
            self.is_flashing = True
            self.flash_timer.start()
        else:
            # 正常状态
            self.flash_timer.stop()
            self.is_flashing = False
            self.setStyleSheet('background-color: transparent;')
            
            if status == '待机':
                self.status_label.setStyleSheet('color: #888888; font-size: 24px; font-weight: bold;')
            elif status == '运行中':
                self.status_label.setStyleSheet('color: #00cc00; font-size: 24px; font-weight: bold;')
            elif status == '报警':
                self.status_label.setStyleSheet('color: #ff4444; font-size: 24px; font-weight: bold;')
            elif status == '已停止':
                self.status_label.setStyleSheet('color: #ffaa00; font-size: 24px; font-weight: bold;')
    
    def _flash_status(self):
        """状态文字闪烁效果"""
        if self.is_flashing:
            current_color = self.status_label.styleSheet()
            if 'opacity: 1.0' in current_color or 'opacity' not in current_color:
                self.status_label.setStyleSheet('color: #ffffff; font-size: 24px; font-weight: bold; opacity: 0.5;')
            else:
                self.status_label.setStyleSheet('color: #ffffff; font-size: 24px; font-weight: bold; opacity: 1.0;')

    def clear_alarm(self):
        """清除报警状态"""
        self.flash_timer.stop()
        self.is_flashing = False
        self.setStyleSheet('background-color: transparent;')


class DataLabel(QLabel):
    """
    数据显示标签组件
    """

    def __init__(self, label: str, value: str = '--', parent=None):
        """初始化数据标签"""
        super().__init__(parent)
        self.label = label
        self.value = value
        self.update_text()
    
    def update_text(self) -> None:
        """更新显示文本"""
        self.setText(f'{self.label}: <span style="color: #00ff88; font-family: Consolas;">{self.value}</span>')
    
    def set_value(self, value: str) -> None:
        """
        设置值
        """
        self.value = value
        self.update_text()


class AlarmBar(QWidget):
    """
    报警闪烁条组件
    """
    
    # 定义信号
    alarm_confirmed = pyqtSignal()

    def __init__(self, parent=None):
        """初始化报警条"""
        super().__init__(parent)
        self.setFixedHeight(30)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        self.layout.setSpacing(20)
        
        # 报警文本标签
        self.label = QLabel('⚠️ 检测到摔倒！请立即救援！')
        self.label.setStyleSheet('color: white; font-weight: bold; font-size: 14px;')
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        
        # 弹簧
        self.layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # 确认按钮
        self.confirm_btn = QPushButton('确认报警')
        self.confirm_btn.setStyleSheet('''
            QPushButton {
                background-color: #ffffff;
                color: #dc2626;
                font-weight: bold;
                border: none;
                padding: 4px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        ''')
        self.confirm_btn.clicked.connect(self._confirm_alarm)
        self.layout.addWidget(self.confirm_btn)
        
        # 闪烁定时器
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._flash)
        self.is_flashing = True
        self.is_confirmed = False
        
        self.hide()

    def show(self) -> None:
        """显示并开始闪烁"""
        super().show()
        self.is_flashing = True
        self.is_confirmed = False
        self.setStyleSheet('background-color: #dc2626;')
        self.confirm_btn.setEnabled(True)
        self.timer.start()

    def hide(self) -> None:
        """隐藏并停止闪烁"""
        super().hide()
        self.timer.stop()
        self.is_flashing = False
        self.is_confirmed = False

    def _flash(self) -> None:
        """闪烁效果：在#dc2626和#991b1b之间交替"""
        if not self.is_confirmed:
            self.is_flashing = not self.is_flashing
            color = '#dc2626' if self.is_flashing else '#991b1b'
            self.setStyleSheet(f'background-color: {color};')

    def _confirm_alarm(self):
        """确认报警：停止闪烁，变为灰色常亮"""
        self.is_confirmed = True
        self.timer.stop()
        self.setStyleSheet('background-color: #6b7280;')
        self.label.setText('✅ 报警已确认')
        self.confirm_btn.setEnabled(False)
        self.alarm_confirmed.emit()

    def reset(self):
        """重置报警条状态"""
        self.hide()
        self.label.setText('⚠️ 检测到摔倒！请立即救援！')
        self.is_confirmed = False
        self.is_flashing = True


class AlarmLogTable(QTableWidget):
    """
    报警日志表格组件
    """

    def __init__(self, parent=None):
        """初始化日志表格"""
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['序号', '报警时间', '报警高度 (m)', '报警速度 (m/s)'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setStyleSheet('font-size: 12px;')
        
    def add_log(self, timestamp: str, height: float, speed: float) -> None:
        """
        添加报警日志
        """
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.setItem(row, 1, QTableWidgetItem(timestamp))
        self.setItem(row, 2, QTableWidgetItem(f'{height:.2f}'))
        self.setItem(row, 3, QTableWidgetItem(f'{speed:.2f}'))
        
        self.scrollToBottom()
    
    def clear_logs(self) -> None:
        """清除所有日志"""
        self.setRowCount(0)
    
    def get_logs(self) -> list:
        """获取所有日志数据"""
        logs = []
        for row in range(self.rowCount()):
            logs.append({
                '序号': self.item(row, 0).text(),
                '报警时间': self.item(row, 1).text(),
                '报警高度 (m)': self.item(row, 2).text(),
                '报警速度 (m/s)': self.item(row, 3).text(),
            })
        return logs


class AboutDialog(QDialog):
    """
    关于对话框
    """

    def __init__(self, parent=None):
        """初始化关于对话框"""
        super().__init__(parent)
        self.setWindowTitle('关于')
        self.setFixedSize(350, 250)
        
        layout = QVBoxLayout()
        
        title_label = QLabel(PROJECT_INFO["name"])
        title_label.setStyleSheet('font-size: 18px; font-weight: bold; color: #00aaff;')
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        version_label = QLabel(f'版本: {PROJECT_INFO["version"]}')
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        desc_label = QLabel(PROJECT_INFO["description"])
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        author_label = QLabel(f'作者: {PROJECT_INFO["author"]}')
        author_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(author_label)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        shortcut_group = QGroupBox('快捷键')
        shortcut_layout = QFormLayout(shortcut_group)
        shortcuts = [
            ('空格键', '开始/停止模拟'),
            ('F', '触发模拟摔倒'),
            ('R', '重置系统'),
            ('Esc', '退出程序'),
        ]
        for key, desc in shortcuts:
            shortcut_layout.addRow(QLabel(f'<b>{key}</b>'), QLabel(desc))
        layout.addWidget(shortcut_group)
        
        ok_btn = QPushButton('确定')
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
        
        self.setLayout(layout)
