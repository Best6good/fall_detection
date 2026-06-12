#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义UI组件模块
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSlider, 
    QPushButton, QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QFormLayout, QSpacerItem, QSizePolicy,
    QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from config import COLOR_CONFIG, VISUALIZATION_CONFIG, PROJECT_INFO, HumanState
from skeleton import (
    SkeletonEstimator, SKELETON_CONNECTIONS, JOINT_NAMES,
)
from utils import logger


class SkeletonVisualizer(FigureCanvas):
    """
    3D人体骨架可视化组件
    显示从点云推断出的人体骨架结构
    """

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=COLOR_CONFIG["background"])
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)

        self.skeleton_joints = np.array([])
        self.current_state = "standing"
        self.fall_progress = 0.0
        self.fused_centroid = None

        self.estimator = SkeletonEstimator(buffer_size=5, bone_elasticity=0.05)

        self.joints_scatter = None
        self.bones_plot = []
        self._viridis_cmap = cm.get_cmap('viridis')

        self._init_plot()
        self._init_artists()

    def _init_artists(self):
        """初始化持久化的绘图元素（避免每帧删除重建导致闪烁）"""
        # 骨架线条：每个连接一条 Line3D
        self.bones_plot = []
        for _ in SKELETON_CONNECTIONS:
            bone, = self.ax.plot3D([0, 0], [0, 0], [0, 0], color='#00cc00', linewidth=3, alpha=0.9)
            bone.set_visible(False)
            self.bones_plot.append(bone)

        # 关节散点：用一个不可见的占位点初始化
        self.joints_scatter = self.ax.scatter(
            [0], [0], [0], c='white', marker='o', s=0, alpha=0.0
        )
        # 初始化 3D 偏移（必须在首次 _offsets3d 赋值前调用一次）
        self.joints_scatter._offsets3d = (
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        self.joints_scatter.set_visible(False)

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('X (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_ylabel('Y (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_zlabel('Z (m)', color=COLOR_CONFIG["text"], fontsize=9)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2.0)
        self.ax.set_facecolor(COLOR_CONFIG["background"])
        self.ax.grid(color=COLOR_CONFIG["grid"], alpha=0.3)
        self.ax.xaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.yaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.zaxis.label.set_color(COLOR_CONFIG["text"])
        self.ax.tick_params(axis='x', colors=COLOR_CONFIG["axis"], labelsize=7)
        self.ax.tick_params(axis='y', colors=COLOR_CONFIG["axis"], labelsize=7)
        self.ax.tick_params(axis='z', colors=COLOR_CONFIG["axis"], labelsize=7)
        self.ax.view_init(elev=VISUALIZATION_CONFIG["view_elevation"],
                          azim=VISUALIZATION_CONFIG["view_azimuth"])
        self.ax.autoscale(False)
        self.ax.set_title('人体骨架', color=COLOR_CONFIG["text"], fontsize=11, pad=10)
        self.fig.subplots_adjust(left=0.12, right=0.9, bottom=0.12, top=0.88)

    def clear(self):
        self.skeleton_joints = np.array([])
        self.current_state = "standing"
        self.fall_progress = 0.0
        self.fused_centroid = None
        self.estimator.reset()

        # 重新初始化坐标轴和持久化元素
        self._init_plot()
        self._init_artists()
        self.draw()

    def update_skeleton(self, points: np.ndarray, state: str = "standing",
                        fall_progress: float = 0.0, walk_phase: float = 0.0):
        self.current_state = state
        self.fall_progress = fall_progress

        self.skeleton_joints = self.estimator.estimate(points, state, fall_progress, walk_phase)
        self.fused_centroid = self.estimator.fused_centroid

        # 选择骨骼颜色
        if state in (HumanState.STANDING, HumanState.WALKING):
            bone_color = '#00cc00'
        elif state == HumanState.FALLING:
            bone_color = '#ff8800'
        else:
            bone_color = '#ff0000'

        # 原地更新骨骼线条（不删除重建，消除闪烁）
        for idx, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
            start = self.skeleton_joints[start_idx]
            end = self.skeleton_joints[end_idx]
            bone = self.bones_plot[idx]
            bone.set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
            bone.set_color(bone_color)
            bone.set_visible(True)

        # 原地更新关节散点（不删除重建）
        z_min, z_max = 0.2, 1.8
        z_values = self.skeleton_joints[:, 2]
        normalized_z = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0, 1)
        joint_colors = self._viridis_cmap(normalized_z)

        self.joints_scatter._offsets3d = (
            self.skeleton_joints[:, 0].copy(),
            self.skeleton_joints[:, 1].copy(),
            self.skeleton_joints[:, 2].copy()
        )
        self.joints_scatter.set_facecolors(joint_colors)
        self.joints_scatter.set_sizes(np.full(16, 60))
        self.joints_scatter.set_alpha(1.0)
        self.joints_scatter.set_visible(True)

        self.draw()


class PointCloudVisualizer(FigureCanvas):
    """
    3D点云可视化组件
    
    改进特性：
    - 支持5D点云数据 [x, y, z, velocity_z, intensity]
    - 多种颜色映射模式（高度、速度、强度）
    - 更好的视觉效果
    """

    def __init__(self, parent=None):
        """初始化可视化器"""
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COLOR_CONFIG["background"])
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self.points = np.array([])
        self.center_trajectory = []
        self.alarm_position = None
        self.scatter_plot = None
        self.center_trajectory_plot = None
        self.alarm_marker = None
        
        # 报警状态
        self.is_alarming = False
        self.is_flashing = False
        self.flash_timer = QTimer()
        self.flash_timer.setInterval(300)
        self.flash_timer.timeout.connect(self._flash_points)
        
        # 可视化增强：质心标记和速度箭头
        self.center_marker = None  # 质心标记
        self.velocity_arrow = None  # 速度方向箭头
        self.last_center_position = None  # 上次的融合质心位置
        
        # 初始化颜色映射
        self.viridis_cmap = cm.get_cmap('viridis')
        self.OrRd_cmap = cm.get_cmap('OrRd')
        self.RdYlBu_cmap = cm.get_cmap('RdYlBu')  # 用于速度映射
        
        # 颜色映射模式: 'height', 'velocity', 'intensity'
        self.color_mode = 'height'

        self._init_plot()
        self._init_artists()

    def _init_artists(self):
        """初始化持久化的绘图元素（避免每帧删除重建导致闪烁）"""
        # 点云散点（用占位点初始化，首次更新时替换数据）
        self.scatter_plot = self.ax.scatter(
            [0], [0], [0], c='white', marker='o', s=0, alpha=0.0
        )
        self.scatter_plot._offsets3d = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.scatter_plot.set_visible(False)

        # 质心标记（金色菱形）
        self.center_marker = self.ax.scatter(
            [0], [0], [0], c='#FFD700', marker='D', s=200,
            alpha=1.0, edgecolors='white', linewidths=2, zorder=10
        )
        self.center_marker._offsets3d = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.center_marker.set_visible(False)

        # 速度箭头（初始不可见）
        self.velocity_arrow = None

        # 轨迹线
        self.center_trajectory_plot, = self.ax.plot3D(
            [0], [0], [0], color=COLOR_CONFIG["center_trajectory"],
            linewidth=3, alpha=0.8
        )
        self.center_trajectory_plot.set_visible(False)

        # 报警标记（三层同心圆）
        self.alarm_marker = None

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

    def set_color_mode(self, mode: str):
        """
        设置颜色映射模式
        :param mode: 'height', 'velocity', 'intensity'
        """
        if mode in ('height', 'velocity', 'intensity'):
            self.color_mode = mode

    def update_points(self, points: np.ndarray, state: str = "standing", is_alarming: bool = False, center_position: tuple = None) -> None:
        """
        更新点云数据（原地更新，不删除重建）
        :param points: 点云数据 (N, 4) 或 (N, 5) [x, y, z, velocity_z, intensity]
        :param state: 当前状态
        :param is_alarming: 是否处于报警状态
        :param center_position: 人体中心位置 (x, y, z)，如果为 None 则从点云计算
        """
        self.points = points

        # 更新报警状态
        if is_alarming and state == HumanState.FALLEN:
            self.is_alarming = True
            if not self.flash_timer.isActive():
                self.flash_timer.start()
        else:
            self.is_alarming = False
            self.is_flashing = False
            if self.flash_timer.isActive():
                self.flash_timer.stop()

        # 根据状态选择颜色和大小
        if state == HumanState.FALLING:
            point_color = COLOR_CONFIG["falling"]
            point_size = 50
        elif state == HumanState.FALLEN:
            self._update_points_fallen(points, center_position)
            return
        elif state == HumanState.WALKING:
            point_color = COLOR_CONFIG["walking"]
            point_size = VISUALIZATION_CONFIG["point_size"]
        else:
            point_color = COLOR_CONFIG["standing"]
            point_size = VISUALIZATION_CONFIG["point_size"]

        # 原地更新点云散点（不删除重建，消除闪烁）
        if self.points.size > 0:
            n = len(self.points)
            # 根据颜色模式选择映射值
            if self.color_mode == 'velocity' and self.points.shape[1] >= 4:
                v_values = self.points[:, 3]
                v_min, v_max = -1.5, 1.5
                normalized = np.clip((v_values - v_min) / (v_max - v_min + 1e-6), 0, 1)
                colors = self.RdYlBu_cmap(normalized)
            elif self.color_mode == 'intensity' and self.points.shape[1] >= 5:
                i_values = self.points[:, 4]
                normalized = np.clip(i_values, 0, 1)
                colors = self.viridis_cmap(normalized)
            else:
                colors = point_color

            self.scatter_plot._offsets3d = (
                self.points[:, 0].copy(),
                self.points[:, 1].copy(),
                self.points[:, 2].copy()
            )
            self.scatter_plot.set_facecolors(colors)
            self.scatter_plot.set_sizes(np.full(n, point_size))
            self.scatter_plot.set_alpha(0.8)
            self.scatter_plot.set_visible(True)
        else:
            self.scatter_plot.set_visible(False)

        # 更新可视化增强元素（质心和速度箭头）
        self.update_visual_enhancements(points, state, center_position)

        self.draw()

    def update_visual_enhancements(self, points: np.ndarray, state: str, center_position: tuple = None) -> None:
        """
        更新可视化增强元素：质心标记和速度方向箭头（原地更新）
        :param points: 点云数据 (N, 4) 或 (N, 5)
        :param state: 当前状态
        :param center_position: 人体中心位置 (x, y, z)，如果为 None 则从点云计算
        """
        if points.size == 0:
            self.center_marker.set_visible(False)
            if self.velocity_arrow:
                self.velocity_arrow.remove()
                self.velocity_arrow = None
            return

        # 使用传入的中心位置或从点云计算
        if center_position is not None:
            cx, cy, cz = center_position
            self.last_center_position = center_position
        else:
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])
            cz = np.mean(points[:, 2])
            self.last_center_position = (cx, cy, cz)

        # 原地更新质心标记位置
        self.center_marker._offsets3d = (
            np.array([cx]), np.array([cy]), np.array([cz])
        )
        self.center_marker.set_visible(True)

        # 速度箭头（quiver 不支持原地更新，保留删除重建）
        if self.velocity_arrow:
            self.velocity_arrow.remove()
            self.velocity_arrow = None

        # 计算平均垂直速度
        if points.shape[1] >= 4:
            avg_vz = np.mean(points[:, 3])
            # 当向下速度超过阈值时，显示速度箭头
            if avg_vz < -0.1:
                arrow_length = abs(avg_vz) * 1.0
                self.velocity_arrow = self.ax.quiver(
                    cx, cy, cz, 0, 0, -arrow_length,
                    color='#FF4444', alpha=0.9,
                    arrow_length_ratio=0.4, linewidth=3, zorder=9
                )

    def _update_points_fallen(self, points: np.ndarray, center_position: tuple = None) -> None:
        """
        倒地状态点云更新（深红色，脉冲闪烁效果，原地更新）
        """
        if points.size > 0:
            # 使用深红色，闪烁时降低透明度
            alpha = 0.9 if not self.is_flashing else 0.4
            n = len(points)
            colors = np.array([[0.86, 0.08, 0.24, alpha]] * n)

            self.scatter_plot._offsets3d = (
                points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()
            )
            self.scatter_plot.set_facecolors(colors)
            self.scatter_plot.set_sizes(np.full(n, 60))
            self.scatter_plot.set_alpha(alpha)
            self.scatter_plot.set_visible(True)
        else:
            self.scatter_plot.set_visible(False)
        
        # 清除旧的报警标记
        if self.alarm_marker:
            if isinstance(self.alarm_marker, list):
                for marker in self.alarm_marker:
                    marker.remove()
            else:
                self.alarm_marker.remove()
            self.alarm_marker = None
        
        # 显示脉冲圆环报警标记
        if self.alarm_position:
            self.alarm_marker = []
            
            # 根据闪烁状态调整透明度
            if self.is_flashing:
                alphas = [0.4, 0.2, 0.1]  # 闪烁时变暗
            else:
                alphas = [0.9, 0.5, 0.2]  # 正常时明亮
            
            # 内层
            inner = self.ax.scatter(
                self.alarm_position[0], self.alarm_position[1], 0.05,
                c='#DC143C', marker='o', s=200, alpha=alphas[0]
            )
            self.alarm_marker.append(inner)
            
            # 中层
            middle = self.ax.scatter(
                self.alarm_position[0], self.alarm_position[1], 0.05,
                c='#FF4444', marker='o', s=400, alpha=alphas[1]
            )
            self.alarm_marker.append(middle)
            
            # 外层
            outer = self.ax.scatter(
                self.alarm_position[0], self.alarm_position[1], 0.05,
                c='#FF8888', marker='o', s=600, alpha=alphas[2]
            )
            self.alarm_marker.append(outer)
        
        # 更新可视化增强元素（质心和速度箭头）
        self.update_visual_enhancements(points, HumanState.FALLEN, center_position)
        
        self.draw()

    def _flash_points(self):
        """点云和标记闪烁效果"""
        self.is_flashing = not self.is_flashing
        if self.points.size > 0 and self.alarm_position:
            # 使用上次的融合质心位置，保持一致性
            center_position = self.last_center_position if self.last_center_position else (
                np.mean(self.points[:, 0]),
                np.mean(self.points[:, 1]),
                np.mean(self.points[:, 2])
            )
            self._update_points_fallen(self.points, center_position)

    def update_center_trajectory(self, center_position: tuple) -> None:
        """
        更新人体中心轨迹线（白色，原地更新）
        """
        # 应用指数平滑（alpha越小越平滑）
        alpha = 0.3
        if len(self.center_trajectory) > 0:
            last_pos = self.center_trajectory[-1]
            smoothed = (
                alpha * center_position[0] + (1 - alpha) * last_pos[0],
                alpha * center_position[1] + (1 - alpha) * last_pos[1],
                alpha * center_position[2] + (1 - alpha) * last_pos[2],
            )
            self.center_trajectory.append(smoothed)
        else:
            self.center_trajectory.append(center_position)

        max_frames = 30  # 增加轨迹长度到30帧
        if len(self.center_trajectory) > max_frames:
            self.center_trajectory = self.center_trajectory[-max_frames:]

        # 原地更新轨迹线数据（不删除重建）
        if len(self.center_trajectory) > 1:
            traj_array = np.array(self.center_trajectory)
            self.center_trajectory_plot.set_data_3d(
                traj_array[:, 0], traj_array[:, 1], traj_array[:, 2]
            )
            self.center_trajectory_plot.set_visible(True)
        else:
            self.center_trajectory_plot.set_visible(False)

    def show_alarm_marker(self, position: tuple) -> None:
        """
        显示报警标记（脉冲圆环效果）
        多层同心圆，由内向外扩散，颜色从深红渐变到浅红
        """
        self.alarm_position = position
        
        # 清除旧的报警标记
        if self.alarm_marker:
            if isinstance(self.alarm_marker, list):
                for marker in self.alarm_marker:
                    marker.remove()
            else:
                self.alarm_marker.remove()
            self.alarm_marker = None
        
        # 创建脉冲圆环效果（3层同心圆）
        self.alarm_marker = []
        
        # 内层：深红色，高不透明度
        inner = self.ax.scatter(
            position[0], position[1], 0.05,
            c='#DC143C', marker='o', s=200, alpha=0.9
        )
        self.alarm_marker.append(inner)
        
        # 中层：红色，中等不透明度
        middle = self.ax.scatter(
            position[0], position[1], 0.05,
            c='#FF4444', marker='o', s=400, alpha=0.5
        )
        self.alarm_marker.append(middle)
        
        # 外层：浅红色，低不透明度
        outer = self.ax.scatter(
            position[0], position[1], 0.05,
            c='#FF8888', marker='o', s=600, alpha=0.2
        )
        self.alarm_marker.append(outer)
        
        self.draw()

    def clear_alarm_marker(self) -> None:
        """清除报警标记"""
        if self.alarm_marker:
            if isinstance(self.alarm_marker, list):
                for marker in self.alarm_marker:
                    marker.remove()
            else:
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
        self.center_trajectory = []
        self.alarm_position = None
        self.is_alarming = False
        self.is_flashing = False

        if self.flash_timer.isActive():
            self.flash_timer.stop()

        # 清理报警标记（非持久化元素）
        if self.alarm_marker:
            if isinstance(self.alarm_marker, list):
                for marker in self.alarm_marker:
                    marker.remove()
            else:
                self.alarm_marker.remove()
            self.alarm_marker = None
        if self.velocity_arrow:
            self.velocity_arrow.remove()
            self.velocity_arrow = None

        # 重新初始化坐标轴和持久化元素
        self._init_plot()
        self._init_artists()
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
        self.line, = self.ax.plot([], [], color='#00aaff', linewidth=2.5)
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
        添加新数据点（Y轴自适应范围）
        """
        if self.is_paused:
            return

        self.data.append(value)

        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]

        self.line.set_xdata(range(len(self.data)))
        self.line.set_ydata(self.data)

        # Y轴自适应数据范围（15% padding，最小范围0.2）
        if len(self.data) > 1:
            d_min = min(self.data)
            d_max = max(self.data)
            d_range = d_max - d_min
            padding = max(d_range * 0.15, 0.05)
            self.ax.set_ylim(d_min - padding, d_max + padding)

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
            if '#888888' in current_color:
                self.status_label.setStyleSheet('color: #ffffff; font-size: 24px; font-weight: bold;')
            else:
                self.status_label.setStyleSheet('color: #888888; font-size: 24px; font-weight: bold;')

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
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(['序号', '报警时间', '报警高度 (m)', '报警速度 (m/s)', '连续帧数'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setStyleSheet('font-size: 12px;')

    def add_log(self, timestamp: str, height: float, speed: float, consecutive: int = 0) -> None:
        """
        添加报警日志
        """
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.setItem(row, 1, QTableWidgetItem(timestamp))
        self.setItem(row, 2, QTableWidgetItem(f'{height:.2f}'))
        self.setItem(row, 3, QTableWidgetItem(f'{speed:.2f}'))
        self.setItem(row, 4, QTableWidgetItem(str(consecutive)))

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
                '连续帧数': self.item(row, 4).text() if self.item(row, 4) else '0',
            })
        return logs


class VisualLegend(QWidget):
    """
    综合视觉图例组件
    显示所有视觉元素的说明：点云颜色、线条、图形等
    """

    def __init__(self, parent=None):
        """初始化图例组件"""
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 标题
        title = QLabel("📊 视觉元素说明")
        title.setStyleSheet("font-weight: bold; font-size: 18px; color: #ffffff; padding: 8px;")
        layout.addWidget(title)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #555555;")
        layout.addWidget(line)
        
        # 点云颜色说明
        color_group = QGroupBox("点云颜色")
        color_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(8)
        
        colors = [
            ("standing", "🔵 蓝色", "站立 / 静止"),
            ("walking", "🟢 绿色", "行走 / 运动中"),
            ("falling", "🟠 橙色", "摔倒中 / 警告"),
            ("fallen", "🔴 深红色", "倒地 / 报警"),
        ]
        
        for color_key, icon, desc in colors:
            row = QHBoxLayout()
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {COLOR_CONFIG[color_key]}; border: 2px solid #666666; border-radius: 3px;")
            desc_label = QLabel(f"{icon} {desc}")
            desc_label.setStyleSheet("color: #ffffff; font-size: 15px;")
            row.addWidget(color_box)
            row.addWidget(desc_label)
            row.addStretch()
            color_layout.addLayout(row)
        
        layout.addWidget(color_group)
        
        # 线条说明
        line_group = QGroupBox("线条")
        line_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        line_layout = QVBoxLayout(line_group)
        line_layout.setSpacing(8)
        
        lines = [
            ("━━━", "#ffffff", "运动轨迹", "人体中心移动路径"),
            ("━━━", "#4A90D9", "骨骼连接", "正常状态"),
            ("━━━", "#FF8C00", "骨骼连接", "摔倒中"),
            ("━━━", "#DC143C", "骨骼连接", "倒地状态"),
        ]
        
        for line_icon, color, name, desc in lines:
            row = QHBoxLayout()
            line_label = QLabel(line_icon)
            line_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
            line_label.setFixedWidth(50)
            desc_label = QLabel(f"{name} - {desc}")
            desc_label.setStyleSheet("color: #ffffff; font-size: 15px;")
            row.addWidget(line_label)
            row.addWidget(desc_label)
            row.addStretch()
            line_layout.addLayout(row)
        
        layout.addWidget(line_group)
        
        # 图形说明
        shape_group = QGroupBox("图形标记")
        shape_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        shape_layout = QVBoxLayout(shape_group)
        shape_layout.setSpacing(8)
        
        shapes = [
            ("●", "圆点", "关节节点（16个）"),
            ("◆", "金色菱形", "质心标记（点云中心位置）"),
            ("↓", "红色箭头", "速度方向（向下运动时显示）"),
            ("◉", "大圆环", "报警标记（倒地时显示）"),
            ("▦", "网格", "空间参考（每格0.5m）"),
        ]
        
        for icon, name, desc in shapes:
            row = QHBoxLayout()
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 20px; color: #aaaaaa;")
            icon_label.setFixedWidth(30)
            desc_label = QLabel(f"{name} - {desc}")
            desc_label.setStyleSheet("color: #ffffff; font-size: 15px;")
            row.addWidget(icon_label)
            row.addWidget(desc_label)
            row.addStretch()
            shape_layout.addLayout(row)
        
        layout.addWidget(shape_group)
        
        # 骨架说明
        skeleton_group = QGroupBox("骨架关节")
        skeleton_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        skeleton_layout = QVBoxLayout(skeleton_group)
        skeleton_layout.setSpacing(8)

        joints = [
            "头部 → 颈部 → 胸部 → 腰部",
            "肩部 → 肘部 → 手部",
            "髋部 → 膝盖 → 脚部",
        ]

        for joint_chain in joints:
            label = QLabel(f"  {joint_chain}")
            label.setStyleSheet("color: #cccccc; font-size: 15px;")
            skeleton_layout.addWidget(label)

        layout.addWidget(skeleton_group)

        # 人体状态说明
        state_group = QGroupBox("人体状态（中文标签）")
        state_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        state_layout_v = QVBoxLayout(state_group)
        state_layout_v.setSpacing(8)

        states = [
            ("站立", "🔵", "静止站立，点云集中在躯干"),
            ("行走", "🟢", "运动中，微多普勒效应明显"),
            ("摔倒中", "🟠", "正在摔倒，向下加速"),
            ("倒地", "🔴", "已倒地，速度中断，触发报警"),
        ]

        for state_name, icon, desc in states:
            row = QHBoxLayout()
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 16px;")
            icon_label.setFixedWidth(30)
            desc_label = QLabel(f"{state_name} — {desc}")
            desc_label.setStyleSheet("color: #ffffff; font-size: 14px;")
            row.addWidget(icon_label)
            row.addWidget(desc_label)
            row.addStretch()
            state_layout_v.addLayout(row)

        layout.addWidget(state_group)

        # 快捷键说明
        shortcut_group = QGroupBox("⌨️ 快捷键")
        shortcut_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        shortcut_layout = QVBoxLayout(shortcut_group)
        shortcut_layout.setSpacing(8)

        shortcuts = [
            ("Space", "开始 / 停止模拟"),
            ("S", "停止模拟"),
            ("F", "触发摔倒（运行中）"),
            ("R", "重置系统"),
            ("Esc", "退出程序"),
        ]

        for key, desc in shortcuts:
            row = QHBoxLayout()
            key_label = QLabel(key)
            key_label.setStyleSheet("color: #00aaff; font-size: 14px; font-weight: bold; font-family: Consolas;")
            key_label.setFixedWidth(60)
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #ffffff; font-size: 14px;")
            row.addWidget(key_label)
            row.addWidget(desc_label)
            row.addStretch()
            shortcut_layout.addLayout(row)

        layout.addWidget(shortcut_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 设置整体样式
        self.setStyleSheet("""
            VisualLegend {
                background-color: #2b2b2b;
            }
            QGroupBox {
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)


class CollapsibleSection(QWidget):
    """可折叠的分组组件"""
    
    def __init__(self, title, icon="", parent=None):
        super().__init__(parent)
        self.is_expanded = True
        self._init_ui(title, icon)
    
    def _init_ui(self, title, icon):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题按钮
        self.header_btn = QPushButton(f"  {icon} {title}  ▼")
        self.header_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #454545;
            }
        """)
        self.header_btn.clicked.connect(self._toggle)
        layout.addWidget(self.header_btn)
        
        # 内容区域
        self.content = QWidget()
        self.content.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border: 1px solid #555555;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }
        """)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(15, 15, 15, 15)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.content)
    
    def add_widget(self, widget):
        """添加内容组件"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """添加布局"""
        self.content_layout.addLayout(layout)
    
    def _toggle(self):
        """折叠/展开"""
        self.is_expanded = not self.is_expanded
        self.content.setVisible(self.is_expanded)
        icon = self.header_btn.text().split(" ")[1] if " " in self.header_btn.text() else ""
        title = self.header_btn.text().replace(" ▼", "").replace(" ▶", "").strip()
        if self.is_expanded:
            self.header_btn.setText(f"  {icon} {title}  ▼")
        else:
            self.header_btn.setText(f"  {icon} {title}  ▶")


class ProgramGuide(QWidget):
    """
    程序说明组件
    包含项目概述、技术架构、工作流程、功能说明、使用步骤、参考资料等
    """

    def __init__(self, parent=None):
        """初始化程序说明组件"""
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # 标题
        title = QLabel("📖 程序说明")
        title.setStyleSheet("font-weight: bold; font-size: 18px; color: #ffffff; padding: 10px;")
        main_layout.addWidget(title)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
        """)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        # 添加各个可折叠部分
        content_layout.addWidget(self._create_overview_section())
        content_layout.addWidget(self._create_architecture_section())
        content_layout.addWidget(self._create_workflow_section())
        content_layout.addWidget(self._create_features_section())
        content_layout.addWidget(self._create_usage_section())
        content_layout.addWidget(self._create_shortcuts_section())
        content_layout.addWidget(self._create_data_format_section())
        content_layout.addWidget(self._create_detection_section())
        content_layout.addWidget(self._create_references_section())
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _create_label(self, text, font_size=13, color="#ffffff", bold=False):
        """创建标签"""
        label = QLabel(text)
        style = f"color: {color}; font-size: {font_size}px;"
        if bold:
            style += " font-weight: bold;"
        label.setStyleSheet(style)
        label.setWordWrap(True)
        return label
    
    def _create_separator(self):
        """创建分隔线"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #555555; max-height: 1px;")
        return line
    
    def _create_overview_section(self):
        """项目概述"""
        section = CollapsibleSection("项目概述", "🏠")
        
        section.add_widget(self._create_label(
            "项目名称：毫米波雷达摔倒检测系统",
            font_size=14, bold=True
        ))
        section.add_widget(self._create_label(
            f"版本：{PROJECT_INFO['version']}",
            font_size=13
        ))
        section.add_widget(self._create_separator())
        section.add_widget(self._create_label(
            "简介：基于物理毫米波雷达点云模拟的摔倒检测原型系统，"
            "展示从点云生成 → 预处理 → 特征提取 → 摔倒检测 → 可视化报警的完整技术链路。",
            font_size=13
        ))
        section.add_widget(self._create_separator())
        section.add_widget(self._create_label("核心特性：", font_size=13, bold=True))
        
        features = [
            "• 纯规则算法，不含深度学习",
            "• 几何模型 + 雷达物理特性模拟",
            "• 5D点云输出（含反射强度）",
            "• 多帧融合骨架估计",
            "• 5特征加权融合检测",
            "• 多特征宠物过滤",
            "• Post-fall确认机制（倒地准静态5帧确认）",
        ]
        for f in features:
            section.add_widget(self._create_label(f"  {f}", font_size=13))
        
        return section
    
    def _create_architecture_section(self):
        """技术架构"""
        section = CollapsibleSection("技术架构", "🔧")
        
        arch_text = """
┌─────────────────────────────────────────────────────────────┐
│                    GeometricModel                           │
│              (椭球体 + 圆柱体人体模型)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   RadarPhysicsModel                         │
│     (RCS反射 + 漏检模拟 + 噪声模型 + 环境建模)                │
└─────────────────────────┬───────────────────────────────────┘
                          │ (N, 5) 点云数据
┌─────────────────────────▼───────────────────────────────────┐
│                  PointCloudPreprocessor                     │
│     (直通滤波 + 统计滤波 + 速度滤波 + 密度过滤)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   FallDetectionModule                        │
│     (5特征加权融合 + 宠物过滤 + Post-fall确认)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      MainWindow                             │
│     骨架: SkeletonEstimator (多帧融合 + 1€滤波 + 骨骼约束)   │
│     滤波: OneEuroFilter (filters.py)                        │
│     评估: Evaluator (evaluator.py)                          │
└─────────────────────────────────────────────────────────────┘
        """
        
        section.add_widget(self._create_label(arch_text, font_size=12))
        
        section.add_widget(self._create_separator())
        section.add_widget(self._create_label("模块职责：", font_size=13, bold=True))
        
        modules = [
            ("simulator.py", "点云模拟器 - 生成5D点云数据"),
            ("processor.py", "预处理器 - 滤波去噪 + 特征提取"),
            ("detector.py", "检测器 - 摔倒检测 + 宠物过滤"),
            ("ui_components.py", "可视化 - 点云/骨架/曲线显示"),
            ("skeleton.py", "骨架估计 - 多帧融合 + 骨骼约束"),
            ("filters.py", "滤波器 - 1€自适应低通滤波"),
            ("evaluator.py", "评估框架 - 精确率/召回率/F1"),
            ("config.py", "配置 - 参数集中管理"),
            ("main.py", "主程序 - 界面布局 + 事件循环"),
        ]
        
        for name, desc in modules:
            section.add_widget(self._create_label(f"  • {name}: {desc}", font_size=12))
        
        return section
    
    def _create_workflow_section(self):
        """工作流程"""
        section = CollapsibleSection("工作流程", "📊")
        
        workflows = [
            ("① 点云生成", [
                "几何模型定义人体部位（椭球体/圆柱体）",
                "按真实分布权重采样：躯干50%、腿23%、臂20%、头7%",
                "椭球体RCS公式计算各点反射强度",
                "微多普勒速度模型：步频2Hz，各部位独立速度",
                "部位特异性漏检（躯干98%，手/脚<20%）",
                "受污染高斯噪声：90%高斯 + 10%Cauchy离群点",
                "环境杂波 + 鬼点生成（5-15%，SNR弱3-20dB）",
                "输出5D点云：[x, y, z, velocity_z, intensity]",
            ]),
            ("② 点云预处理", [
                "直通滤波：保留高度0.05-1.8m，距离0.1-5m",
                "统计滤波：基于MAD去除离群点",
                "速度滤波：去除静态杂波",
                "密度过滤：去除稀疏噪声区域",
            ]),
            ("③ 骨架估计", [
                "多帧融合：融合最近3帧点云",
                "高度分层分析：推算各关节高度",
                "水平位置校正：以点云质心对齐",
                "1€滤波：自适应时序平滑",
                "骨骼约束：±10%弹性范围",
                "输出16关节骨架",
            ]),
            ("④ 摔倒检测", [
                "特征提取：高度、速度、变化率、标准差、密度",
                "5特征加权融合计算置信度",
                "宠物过滤：多特征识别排除宠物",
                "连续帧验证：默认2帧满足条件",
                "Post-fall确认：倒地后保持静止5帧",
                "触发报警",
            ]),
        ]
        
        for title, steps in workflows:
            section.add_widget(self._create_label(title, font_size=13, bold=True))
            for step in steps:
                section.add_widget(self._create_label(f"    → {step}", font_size=12))
            section.add_widget(self._create_separator())
        
        return section
    
    def _create_features_section(self):
        """功能说明"""
        section = CollapsibleSection("功能说明", "🎯")
        
        # 点云模拟器
        section.add_widget(self._create_label("【点云模拟器】", font_size=14, bold=True))
        simulator_features = [
            ("5D点云输出", "输出[x, y, z, velocity_z, intensity]，强度基于雷达方程RCS/d⁴"),
            ("4种人体状态", "站立、行走、摔倒、倒地，支持手动切换和自动触发"),
            ("真实RCS值", "躯干1.0m²(0dBsm)、头0.08m²、臂0.03-0.05m²、腿0.06-0.12m²"),
            ("部位分布权重", "躯干50%、腿23%、臂20%、头7%（基于文献实测）"),
            ("受污染高斯噪声", "95%高斯inlier + 5%Cauchy离群点，距离-多普勒耦合-0.4"),
            ("部位特异性检测", "躯干~98%、腿75-87%、臂45-65%、手/脚<20%"),
            ("微多普勒模型", "步频2Hz，躯干1.2m/s，腿尖5m/s，臂摆3m/s"),
            ("三阶段摔倒", "加速→峰值2.5m/s→撞击减速，倒地后速度中断>2s"),
            ("真实鬼点", "2-5%鬼点率，SNR弱3-20dB，镜像对称，30%闪烁"),
            ("范围依赖密度", "近处密集，远处稀疏，每帧±30%随机波动"),
        ]
        for name, desc in simulator_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 骨架估计
        section.add_widget(self._create_label("【骨架估计】", font_size=14, bold=True))
        skeleton_features = [
            ("多帧融合", "融合5帧点云，降低34%关节定位误差（参考FUSE论文）"),
            ("鲁棒质心", "使用中位数代替均值，抗离群点（Cauchy噪声、鬼点）"),
            ("行走摆动", "手臂/腿交替摆动（对侧模式），步频2Hz，身体上下微动"),
            ("摔倒倾斜", "基于fall_progress的旋转倾斜（0°→90°），以腰部为中心"),
            ("倒地平躺", "所有关节接近同一高度，模拟平躺在地面"),
            ("1€滤波", "自适应低通滤波器，参数：min_cutoff=0.3, beta=0.1（强平滑）"),
            ("骨骼约束", "15条骨骼连接，±5%弹性范围，符合人体解剖学"),
            ("16关节", "头、颈、胸、腰、肩×2、肘×2、手×2、髋×2、膝×2、脚×2"),
        ]
        for name, desc in skeleton_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 摔倒检测
        section.add_widget(self._create_label("【摔倒检测】", font_size=14, bold=True))
        detection_features = [
            ("5特征融合", "高度30% + 速度25% + 变化率20% + 速度标准差15% + 密度10%"),
            ("宠物过滤", "高度、宽度、速度标准差、高度标准差多特征识别"),
            ("连续帧验证", "默认2帧满足条件才判定摔倒，减少误报"),
            ("Post-fall确认", "倒地后保持准静态5帧确认，排除快速蹲下等动作"),
        ]
        for name, desc in detection_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 可视化
        section.add_widget(self._create_label("【可视化】", font_size=14, bold=True))
        visual_features = [
            ("状态颜色", "蓝(站立)、绿(行走)、橙(摔倒)、红(倒地)"),
            ("颜色模式", "高度模式、速度模式、强度模式可切换"),
            ("质心标记", "金色菱形标记，始终显示点云中心位置"),
            ("速度箭头", "红色向下箭头，仅在速度<-0.1m/s时显示"),
            ("报警效果", "3层脉冲圆环 + 闪烁效果"),
            ("实时曲线", "高度和速度实时曲线图"),
            ("骨架显示", "骨骼连接线 + 关节节点，颜色随状态变化"),
            ("运动轨迹", "白色轨迹线，指数平滑(alpha=0.3)，30帧长度"),
            ("持久化渲染", "绘图元素原地更新，消除闪烁，帧率15FPS+"),
            ("质心平滑", "5帧滑动平均，质心/骨架/轨迹严格对齐"),
        ]
        for name, desc in visual_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 报警系统
        section.add_widget(self._create_label("【报警系统】", font_size=14, bold=True))
        alarm_features = [
            ("报警触发", "摔倒检测后自动触发，点云变红闪烁"),
            ("报警日志", "记录报警时间、高度、速度、连续帧数，支持导出CSV"),
            ("报警截图", "自动保存报警时的3D视图截图到alarms/目录"),
            ("蜂鸣报警", "Windows系统下播放蜂鸣声（800Hz, 500ms）"),
            ("报警确认", "点击确认按钮解除报警，恢复待机状态"),
            ("报警弹窗", "显示完整检测参数：时间、高度、速度、连续帧数"),
        ]
        for name, desc in alarm_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 参数调节
        section.add_widget(self._create_label("【参数调节】", font_size=14, bold=True))
        param_features = [
            ("高度阈值", "0.3-1.0m，默认0.4m，低于此值触发条件1"),
            ("速度阈值", "-1.5-0 m/s，默认-0.6 m/s，低于此值触发条件2"),
            ("帧数阈值", "1-10帧，默认2帧，连续满足帧数要求"),
            ("噪声水平", "0-1，默认0.1，影响点云散布程度"),
            ("模拟速度", "0.5-2.0x，默认1.0x，调节定时器间隔"),
            ("人体状态", "中文标签：站立/行走/摔倒中/倒地，支持手动切换"),
        ]
        for name, desc in param_features:
            section.add_widget(self._create_label(f"  • {name}：{desc}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        # 功能关联
        section.add_widget(self._create_label("【功能关联】", font_size=14, bold=True))
        relations = [
            "• 点云模拟器 → 预处理器：原始点云经过滤波后用于检测和骨架估计",
            "• 预处理器 → 检测器：提取的特征用于摔倒判定",
            "• 预处理器 → 骨架估计：滤波后的点云用于骨架推断",
            "• 检测器 → 报警系统：检测结果触发报警流程",
            "• 参数调节 → 检测器：实时调节检测灵敏度",
            "• 模拟控制 → 点云模拟器：控制模拟运行和状态切换",
        ]
        for r in relations:
            section.add_widget(self._create_label(f"  {r}", font_size=12))
        
        return section
    
    def _create_usage_section(self):
        """使用步骤"""
        section = CollapsibleSection("使用步骤", "📋")
        
        steps = [
            ("1. 启动程序", "双击 run.bat 或执行 python main.py"),
            ("2. 开始模拟", "点击[开始模拟]按钮或按空格键"),
            ("3. 观察点云", "左侧3D视图显示点云和骨架，右侧显示实时曲线和参数"),
            ("4. 触发摔倒", "点击[触发摔倒]按钮或按F键，观察点云颜色变化和骨架姿态变化"),
            ("5. 调节参数", "右侧面板可调节检测阈值、噪声水平等"),
            ("6. 查看报警", "倒地后自动触发报警，点击[确认报警]解除"),
            ("7. 导出日志", "在[报警日志]标签页可导出CSV日志文件"),
            ("8. 重置系统", "点击[重置系统]按钮或按R键恢复初始状态"),
        ]
        
        for title, desc in steps:
            section.add_widget(self._create_label(title, font_size=13, bold=True))
            section.add_widget(self._create_label(f"    {desc}", font_size=12))
            section.add_widget(self._create_separator())
        
        return section
    
    def _create_shortcuts_section(self):
        """快捷键"""
        section = CollapsibleSection("快捷键", "⌨️")
        
        shortcuts = [
            ("Space", "开始/停止模拟"),
            ("S", "停止模拟"),
            ("F", "触发摔倒"),
            ("R", "重置系统"),
            ("Esc", "退出程序"),
        ]
        
        for key, desc in shortcuts:
            row = QHBoxLayout()
            key_label = QLabel(f"  {key}")
            key_label.setStyleSheet("color: #00aaff; font-size: 14px; font-weight: bold; font-family: Consolas;")
            key_label.setFixedWidth(120)
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #ffffff; font-size: 13px;")
            row.addWidget(key_label)
            row.addWidget(desc_label)
            row.addStretch()
            section.add_layout(row)
        
        return section
    
    def _create_data_format_section(self):
        """数据格式"""
        section = CollapsibleSection("数据格式", "📁")
        
        section.add_widget(self._create_label("点云数据 (N, 5)：", font_size=13, bold=True))
        section.add_widget(self._create_label("  [x, y, z, velocity_z, intensity]", font_size=12, color="#00ff88"))
        section.add_widget(self._create_label("  x, y, z: 3D坐标 (米)", font_size=12))
        section.add_widget(self._create_label("  velocity_z: 垂直速度 (m/s)", font_size=12))
        section.add_widget(self._create_label("  intensity: 反射强度 (0-1)", font_size=12))
        
        section.add_widget(self._create_separator())
        
        section.add_widget(self._create_label("骨架数据 (16, 3)：", font_size=13, bold=True))
        joints = [
            "head, neck, chest, waist",
            "left_shoulder, right_shoulder",
            "left_elbow, right_elbow",
            "left_hand, right_hand",
            "left_hip, right_hip",
            "left_knee, right_knee",
            "left_foot, right_foot",
        ]
        for j in joints:
            section.add_widget(self._create_label(f"  • {j}", font_size=12))
        
        section.add_widget(self._create_separator())
        
        section.add_widget(self._create_label("强度计算公式：", font_size=13, bold=True))
        section.add_widget(self._create_label("  intensity ∝ RCS / distance⁴", font_size=12, color="#00ff88"))
        section.add_widget(self._create_label("  杂波点强度：0.01-0.15", font_size=12))
        section.add_widget(self._create_label("  人体目标强度：0.3-1.0", font_size=12))
        
        return section
    
    def _create_detection_section(self):
        """检测特征"""
        section = CollapsibleSection("检测特征", "📊")
        
        section.add_widget(self._create_label("5特征加权融合：", font_size=13, bold=True))
        
        # 表格式展示
        features = [
            ("特征名称", "权重", "阈值", "说明"),
            ("人体高度", "30%", "< 0.4m", "平均高度低于阈值"),
            ("垂直速度", "25%", "< -0.6 m/s", "向下运动速度"),
            ("帧间高度变化率", "20%", "> 5 cm/帧", "高度快速下降"),
            ("速度标准差", "15%", "< 0.3", "速度分布集中"),
            ("点云密度", "10%", "> 20 点/m²", "身体展开密度增加"),
        ]
        
        for i, (name, weight, threshold, desc) in enumerate(features):
            row = QHBoxLayout()
            
            if i == 0:
                style = "color: #00aaff; font-size: 12px; font-weight: bold;"
            else:
                style = "color: #ffffff; font-size: 12px;"
            
            name_label = QLabel(f"  {name}")
            name_label.setStyleSheet(style)
            name_label.setFixedWidth(140)
            
            weight_label = QLabel(weight)
            weight_label.setStyleSheet(style)
            weight_label.setFixedWidth(60)
            
            threshold_label = QLabel(threshold)
            threshold_label.setStyleSheet(style)
            threshold_label.setFixedWidth(100)
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(style)
            
            row.addWidget(name_label)
            row.addWidget(weight_label)
            row.addWidget(threshold_label)
            row.addWidget(desc_label)
            row.addStretch()
            section.add_layout(row)
            
            if i == 0:
                section.add_widget(self._create_separator())
        
        section.add_widget(self._create_separator())
        
        section.add_widget(self._create_label("宠物过滤特征：", font_size=13, bold=True))
        pet_features = [
            "• 高度检查：宠物通常 < 0.5m",
            "• 宽度检查：宠物通常 < 0.4m",
            "• 速度标准差：宠物运动更不规则（> 0.3）",
            "• 高度标准差：宠物高度变化小（< 0.1）",
        ]
        for f in pet_features:
            section.add_widget(self._create_label(f"  {f}", font_size=12))
        
        return section
    
    def _create_references_section(self):
        """参考资料"""
        section = CollapsibleSection("参考资料", "📚")
        
        categories = [
            ("点云仿真", [
                ("mmSim", "mmWave雷达模拟器", "https://github.com/yizzfz/mmSim"),
                ("RadHARSimulator", "FMCW雷达HAR模拟器", "https://github.com/JoeyBGOfficial/RadHARSimulatorV1"),
                ("Human Walking Radar Simulator", "行走人体雷达仿真", "https://github.com/cidcom/human-walking-radar-simulator"),
            ]),
            ("骨架估计", [
                ("FUSE", "多帧融合骨架估计 (2022)", "https://arxiv.org/pdf/2205.00097"),
                ("mmChainPose", "几何感知时序链式估计 (2026)", "Neurocomputing"),
                ("mmDiff", "扩散模型姿态估计 (ECCV 2024)", "https://arxiv.org/html/2403.16198"),
            ]),
            ("摔倒检测", [
                ("Advanced mmWave Radar Fall Detection", "摔倒检测 (2024)", "Sensors 24(11), 3660"),
                ("Post-fall Detection", "倒地检测 (2025)", "arXiv:2601.17710"),
            ]),
            ("滤波算法", [
                ("1€ Filter", "自适应低通滤波器 (CHI 2012)", "https://hal.inria.fr/hal-00670496/document"),
            ]),
        ]
        
        for cat_name, refs in categories:
            section.add_widget(self._create_label(f"【{cat_name}】", font_size=13, bold=True))
            for name, desc, link in refs:
                section.add_widget(self._create_label(f"  • {name} - {desc}", font_size=12))
                section.add_widget(self._create_label(f"    {link}", font_size=11, color="#888888"))
            section.add_widget(self._create_separator())
        
        return section


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
