#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import traceback
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSlider, QTabWidget, QGroupBox,
    QComboBox, QStatusBar, QAction, QMenuBar, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt, QTime
from PyQt5.QtGui import QIcon

# 导入自定义模块
from config import (
    UI_CONFIG, SIMULATOR_CONFIG, ALARM_CONFIG, COLOR_CONFIG, HumanState,
    set_fall_height_threshold, set_fall_speed_threshold, 
    set_fall_frames_threshold, set_noise_level, set_sim_speed,
    get_fall_height_threshold, get_fall_speed_threshold,
    get_fall_frames_threshold, get_noise_level, get_sim_speed
)
from simulator import RadarPointCloudSimulator
from processor import PointCloudPreprocessor
from detector import FallDetectionModule
from ui_components import (
    PointCloudVisualizer, SkeletonVisualizer, RealTimePlot, StatusCard, DataLabel,
    AlarmBar, AlarmLogTable, AboutDialog
)
from utils import (
    logger, get_timestamp, get_timestamp_filename, 
    export_to_csv, play_beep, take_screenshot, format_duration,
    ensure_dir_exists
)


class MainWindow(QMainWindow):
    """
    主窗口类
    包含完整的图形界面和系统控制逻辑
    """

    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle(UI_CONFIG["window_title"])
        self.setGeometry(100, 100, *UI_CONFIG["window_size"])
        
        # 初始化核心模块
        self.simulator = RadarPointCloudSimulator()
        self.preprocessor = PointCloudPreprocessor()
        self.detector = FallDetectionModule()
        
        # 状态标志
        self.is_running = False
        self.has_alarmed = False
        self.start_time = None
        self.frame_count = 0
        
        # 定时器
        self.timer = QTimer()
        self.timer.setInterval(int(1000 / SIMULATOR_CONFIG["frame_rate"]))
        self.timer.timeout.connect(self._update_frame)
        
        # 创建UI
        self._create_ui()
        
        # 加载样式表
        self._load_stylesheet()
        
        logger.info("主窗口初始化完成")

    def _create_ui(self):
        """创建用户界面"""
        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 报警闪烁条
        self.alarm_bar = AlarmBar()
        self.alarm_bar.alarm_confirmed.connect(self._confirm_alarm)
        main_layout.addWidget(self.alarm_bar)
        
        # 主内容区域
        content_layout = QHBoxLayout()
        
        # 左侧区域：点云和骨架可视化
        left_container = QFrame()
        left_container_layout = QHBoxLayout(left_container)
        
        # 左侧-左：点云可视化
        point_cloud_frame = QFrame()
        point_cloud_layout = QVBoxLayout(point_cloud_frame)
        self.visualizer = PointCloudVisualizer()
        point_cloud_layout.addWidget(self.visualizer)
        left_container_layout.addWidget(point_cloud_frame, stretch=1)
        
        # 左侧-右：骨架可视化
        skeleton_frame = QFrame()
        skeleton_layout = QVBoxLayout(skeleton_frame)
        self.skeleton_visualizer = SkeletonVisualizer()
        skeleton_layout.addWidget(self.skeleton_visualizer)
        left_container_layout.addWidget(skeleton_frame, stretch=1)
        
        content_layout.addWidget(left_container, stretch=5)
        
        # 右侧：控制和状态区域
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        
        # 顶部控制按钮
        top_btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton('开始模拟')
        self.start_btn.setObjectName('start_btn')
        self.start_btn.setIcon(QIcon.fromTheme('media-playback-start'))
        self.start_btn.clicked.connect(self._start_simulation)
        top_btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton('停止模拟')
        self.stop_btn.setObjectName('stop_btn')
        self.stop_btn.setIcon(QIcon.fromTheme('media-playback-stop'))
        self.stop_btn.clicked.connect(self._stop_simulation)
        self.stop_btn.setEnabled(False)
        top_btn_layout.addWidget(self.stop_btn)
        
        self.fall_btn = QPushButton('触发摔倒')
        self.fall_btn.setObjectName('alarm_btn')
        self.fall_btn.setIcon(QIcon.fromTheme('alert-warning'))
        self.fall_btn.clicked.connect(self._trigger_fall)
        self.fall_btn.setEnabled(False)
        top_btn_layout.addWidget(self.fall_btn)
        
        self.reset_btn = QPushButton('重置系统')
        self.reset_btn.setObjectName('reset_btn')
        self.reset_btn.setIcon(QIcon.fromTheme('edit-clear'))
        self.reset_btn.clicked.connect(self._reset_system)
        top_btn_layout.addWidget(self.reset_btn)
        
        right_layout.addLayout(top_btn_layout)
        
        # 多标签页
        self.tab_widget = QTabWidget()
        
        # 实时监控标签
        self.monitor_tab = QWidget()
        self._create_monitor_tab()
        self.tab_widget.addTab(self.monitor_tab, '实时监控')
        
        # 点云详情标签
        self.details_tab = QWidget()
        self._create_details_tab()
        self.tab_widget.addTab(self.details_tab, '点云详情')
        
        # 报警日志标签
        self.log_tab = QWidget()
        self._create_log_tab()
        self.tab_widget.addTab(self.log_tab, '报警日志')
        
        right_layout.addWidget(self.tab_widget)
        content_layout.addWidget(right_frame, stretch=4)
        
        main_layout.addLayout(content_layout)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('系统就绪')
        
        # 菜单栏
        self._create_menu_bar()

    def _create_monitor_tab(self):
        """创建实时监控标签页"""
        layout = QVBoxLayout(self.monitor_tab)
        
        # 状态卡片
        self.status_card = StatusCard('系统状态')
        layout.addWidget(self.status_card)
        
        # 实时曲线图
        graphs_group = QGroupBox('实时曲线')
        graphs_layout = QVBoxLayout(graphs_group)
        
        self.height_plot = RealTimePlot('人体平均高度', '高度 (m)')
        graphs_layout.addWidget(self.height_plot)
        
        self.speed_plot = RealTimePlot('垂直速度', '速度 (m/s)')
        graphs_layout.addWidget(self.speed_plot)
        
        layout.addWidget(graphs_group)
        
        # 关键参数显示
        params_group = QGroupBox('关键参数')
        params_layout = QVBoxLayout(params_group)
        
        self.height_label = DataLabel('当前高度', '-- m')
        params_layout.addWidget(self.height_label)
        
        self.speed_label = DataLabel('当前速度', '-- m/s')
        params_layout.addWidget(self.speed_label)
        
        self.consecutive_label = DataLabel('连续满足帧数', '--')
        params_layout.addWidget(self.consecutive_label)
        
        layout.addWidget(params_group)
        
        # 参数调节
        adjust_group = QGroupBox('检测参数调节')
        adjust_layout = QVBoxLayout(adjust_group)
        
        # 高度阈值
        ht_layout = QHBoxLayout()
        ht_layout.addWidget(QLabel('高度阈值:'))
        self.height_threshold_slider = QSlider(Qt.Horizontal)
        self.height_threshold_slider.setRange(30, 100)  # 0.3m - 1.0m
        self.height_threshold_slider.setValue(int(get_fall_height_threshold() * 100))
        self.height_threshold_slider.valueChanged.connect(self._update_height_threshold)
        ht_layout.addWidget(self.height_threshold_slider)
        self.height_threshold_label = QLabel(f'{get_fall_height_threshold():.2f}m')
        ht_layout.addWidget(self.height_threshold_label)
        adjust_layout.addLayout(ht_layout)
        
        # 速度阈值
        st_layout = QHBoxLayout()
        st_layout.addWidget(QLabel('速度阈值:'))
        self.speed_threshold_slider = QSlider(Qt.Horizontal)
        self.speed_threshold_slider.setRange(-150, 0)  # -1.5m/s - 0m/s
        self.speed_threshold_slider.setValue(int(get_fall_speed_threshold() * 100))
        self.speed_threshold_slider.valueChanged.connect(self._update_speed_threshold)
        st_layout.addWidget(self.speed_threshold_slider)
        self.speed_threshold_label = QLabel(f'{get_fall_speed_threshold():.2f}m/s')
        st_layout.addWidget(self.speed_threshold_label)
        adjust_layout.addLayout(st_layout)
        
        # 帧数阈值
        ft_layout = QHBoxLayout()
        ft_layout.addWidget(QLabel('帧数阈值:'))
        self.frames_threshold_slider = QSlider(Qt.Horizontal)
        self.frames_threshold_slider.setRange(1, 10)
        self.frames_threshold_slider.setValue(get_fall_frames_threshold())
        self.frames_threshold_slider.valueChanged.connect(self._update_frames_threshold)
        ft_layout.addWidget(self.frames_threshold_slider)
        self.frames_threshold_label = QLabel(f'{get_fall_frames_threshold()}帧')
        ft_layout.addWidget(self.frames_threshold_label)
        adjust_layout.addLayout(ft_layout)
        
        # 恢复默认按钮
        reset_params_btn = QPushButton('恢复默认参数')
        reset_params_btn.clicked.connect(self._reset_parameters)
        adjust_layout.addWidget(reset_params_btn)
        
        layout.addWidget(adjust_group)
        
        # 模拟控制
        sim_group = QGroupBox('模拟控制')
        sim_layout = QVBoxLayout(sim_group)
        
        # 模拟速度
        ss_layout = QHBoxLayout()
        ss_layout.addWidget(QLabel('模拟速度:'))
        self.sim_speed_slider = QSlider(Qt.Horizontal)
        self.sim_speed_slider.setRange(50, 200)  # 0.5x - 2.0x
        self.sim_speed_slider.setValue(100)
        self.sim_speed_slider.valueChanged.connect(self._update_sim_speed)
        ss_layout.addWidget(self.sim_speed_slider)
        self.sim_speed_label = QLabel('1.0x')
        ss_layout.addWidget(self.sim_speed_label)
        sim_layout.addLayout(ss_layout)
        
        # 噪声水平
        nl_layout = QHBoxLayout()
        nl_layout.addWidget(QLabel('噪声水平:'))
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(int(get_noise_level() * 100))
        self.noise_slider.valueChanged.connect(self._update_noise_level)
        nl_layout.addWidget(self.noise_slider)
        self.noise_label = QLabel(f'{get_noise_level():.2f}')
        nl_layout.addWidget(self.noise_label)
        sim_layout.addLayout(nl_layout)
        
        # 状态切换
        state_layout = QHBoxLayout()
        state_layout.addWidget(QLabel('人体状态:'))
        self.state_combo = QComboBox()
        self.state_combo.addItems([s.value for s in HumanState])
        self.state_combo.setCurrentText(HumanState.STANDING.value)
        self.state_combo.currentTextChanged.connect(self._change_state)
        state_layout.addWidget(self.state_combo)
        sim_layout.addLayout(state_layout)
        
        layout.addWidget(sim_group)
        
        layout.addStretch()

    def _create_details_tab(self):
        """创建点云详情标签页"""
        layout = QVBoxLayout(self.details_tab)
        
        # 点云统计
        stats_group = QGroupBox('点云统计')
        stats_layout = QVBoxLayout(stats_group)
        
        self.raw_count_label = DataLabel('原始点云数量', '--')
        stats_layout.addWidget(self.raw_count_label)
        
        self.filtered_count_label = DataLabel('滤波后点云数量', '--')
        stats_layout.addWidget(self.filtered_count_label)
        
        self.min_height_label = DataLabel('最小高度', '-- m')
        stats_layout.addWidget(self.min_height_label)
        
        self.max_height_label = DataLabel('最大高度', '-- m')
        stats_layout.addWidget(self.max_height_label)
        
        self.avg_height_label = DataLabel('平均高度', '-- m')
        stats_layout.addWidget(self.avg_height_label)
        
        self.min_vz_label = DataLabel('最小速度', '-- m/s')
        stats_layout.addWidget(self.min_vz_label)
        
        self.max_vz_label = DataLabel('最大速度', '-- m/s')
        stats_layout.addWidget(self.max_vz_label)
        
        self.std_vz_label = DataLabel('速度标准差', '--')
        stats_layout.addWidget(self.std_vz_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()

    def _create_log_tab(self):
        """创建报警日志标签页"""
        layout = QVBoxLayout(self.log_tab)
        
        # 日志表格
        self.log_table = AlarmLogTable()
        layout.addWidget(self.log_table)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton('导出日志')
        export_btn.clicked.connect(self._export_logs)
        btn_layout.addWidget(export_btn)
        
        clear_btn = QPushButton('清除日志')
        clear_btn.clicked.connect(self._clear_logs)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)

    def _create_menu_bar(self):
        """创建菜单栏"""
        menu_bar = QMenuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu('文件')
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Esc')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 控制菜单
        control_menu = menu_bar.addMenu('控制')
        
        start_action = QAction('开始模拟', self)
        start_action.setShortcut(' ')
        start_action.triggered.connect(self._start_simulation)
        control_menu.addAction(start_action)
        
        stop_action = QAction('停止模拟', self)
        stop_action.setShortcut('Ctrl+Space')
        stop_action.triggered.connect(self._stop_simulation)
        control_menu.addAction(stop_action)
        
        fall_action = QAction('触发摔倒', self)
        fall_action.setShortcut('F')
        fall_action.triggered.connect(self._trigger_fall)
        control_menu.addAction(fall_action)
        
        reset_action = QAction('重置系统', self)
        reset_action.setShortcut('R')
        reset_action.triggered.connect(self._reset_system)
        control_menu.addAction(reset_action)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        self.setMenuBar(menu_bar)

    def _load_stylesheet(self):
        """加载样式表"""
        style_path = os.path.join(os.path.dirname(__file__), 'styles.qss')
        if os.path.exists(style_path):
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
            logger.info("样式表加载成功")
        else:
            logger.warning("样式表文件不存在")

    def _update_height_threshold(self, value):
        """更新高度阈值"""
        threshold = value / 100.0
        set_fall_height_threshold(threshold)
        self.height_threshold_label.setText(f'{threshold:.2f}m')
        logger.info(f"高度阈值已更新: {threshold:.2f}m")

    def _update_speed_threshold(self, value):
        """更新速度阈值"""
        threshold = value / 100.0
        set_fall_speed_threshold(threshold)
        self.speed_threshold_label.setText(f'{threshold:.2f}m/s')
        logger.info(f"速度阈值已更新: {threshold:.2f}m/s")

    def _update_frames_threshold(self, value):
        """更新帧数阈值"""
        set_fall_frames_threshold(value)
        self.frames_threshold_label.setText(f'{value}帧')
        logger.info(f"帧数阈值已更新: {value}帧")

    def _update_sim_speed(self, value):
        """更新模拟速度"""
        speed = value / 100.0
        set_sim_speed(speed)
        self.sim_speed_label.setText(f'{speed:.1f}x')
        # 更新定时器间隔
        interval = int(1000 / (SIMULATOR_CONFIG["frame_rate"] * speed))
        self.timer.setInterval(interval)

    def _update_noise_level(self, value):
        """更新噪声水平"""
        noise = value / 100.0
        set_noise_level(noise)
        self.noise_label.setText(f'{noise:.2f}')
        logger.info(f"噪声水平已更新: {noise:.2f}")

    def _change_state(self, state):
        """切换人体状态"""
        if self.is_running:
            self.simulator.set_state(state)
            logger.info(f"人体状态已切换为: {state}")

    def _reset_parameters(self):
        """恢复默认参数"""
        self.height_threshold_slider.setValue(50)  # 0.5m
        self.speed_threshold_slider.setValue(-80)  # -0.8m/s
        self.frames_threshold_slider.setValue(3)  # 3帧
        self.sim_speed_slider.setValue(100)  # 1.0x
        self.noise_slider.setValue(10)  # 0.1
        logger.info("参数已恢复默认值")

    def _start_simulation(self):
        """开始模拟"""
        self.is_running = True
        self.has_alarmed = False
        self.frame_count = 0
        self.start_time = QTime.currentTime()
        self.detector.reset()

        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fall_btn.setEnabled(True)

        # 更新状态显示
        self.status_card.set_status('运行中')
        self.status_bar.showMessage('模拟运行中...')
        self._set_status_bar_color('green')

        # 启动定时器
        self.timer.start()

        # 立即触发一次更新，确保UI立即显示
        self._update_frame()
        QApplication.processEvents()

        logger.info("模拟已启动")

    def _stop_simulation(self):
        """停止模拟"""
        self.is_running = False
        self.timer.stop()
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fall_btn.setEnabled(False)
        
        # 更新状态显示
        self.status_card.set_status('已停止')
        self.status_bar.showMessage('模拟已停止')
        self._set_status_bar_color('orange')
        
        logger.info("模拟已停止")

    def _trigger_fall(self):
        """触发摔倒事件"""
        self.simulator.trigger_fall()
        # 阻止信号触发 _change_state，避免将 enum 状态覆盖为 string
        self.state_combo.blockSignals(True)
        self.state_combo.setCurrentText(HumanState.FALLING.value)
        self.state_combo.blockSignals(False)
        self.status_bar.showMessage('已触发摔倒模拟...')

    def _reset_system(self):
        """重置系统"""
        self.timer.stop()
        self.is_running = False
        self.has_alarmed = False
        self.frame_count = 0
        self.start_time = None
        
        # 重置所有模块
        self.simulator.reset()
        self.detector.reset()
        
        # 清空显示
        self.visualizer.clear()
        self.skeleton_visualizer.clear()
        self.height_plot.clear()
        self.speed_plot.clear()
        self.alarm_bar.reset()
        self.status_card.clear_alarm()
        
        # 重置参数显示
        self.height_label.set_value('-- m')
        self.speed_label.set_value('-- m/s')
        self.consecutive_label.set_value('--')
        self.raw_count_label.set_value('--')
        self.filtered_count_label.set_value('--')
        self.min_height_label.set_value('-- m')
        self.max_height_label.set_value('-- m')
        self.avg_height_label.set_value('-- m')
        self.min_vz_label.set_value('-- m/s')
        self.max_vz_label.set_value('-- m/s')
        self.std_vz_label.set_value('--')
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fall_btn.setEnabled(False)
        
        # 更新状态显示
        self.status_card.set_status('待机')
        self.state_combo.setCurrentText(HumanState.STANDING.value)
        self.status_bar.showMessage('系统已重置')
        self._set_status_bar_color('default')
        
        logger.info("系统已重置")

    def _update_frame(self):
        """更新一帧数据"""
        if not self.is_running:
            return

        self.frame_count += 1

        # --- 数据生成与预处理 (独立 try) ---
        try:
            raw_points = self.simulator.get_next_frame()
            processed_points = self.preprocessor.process(raw_points)
            avg_height, avg_vz = self.preprocessor.get_frame_features(processed_points)
            stats = self.preprocessor.get_point_cloud_stats(processed_points)
        except Exception as e:
            logger.error(f"数据生成/预处理失败: {e}\n{traceback.format_exc()}")
            self.status_bar.showMessage(f'数据错误: {str(e)}')
            return

        # --- UI 更新 (独立 try，失败不影响检测) ---
        try:
            self.height_label.set_value(f'{avg_height:.2f} m')
            self.speed_label.set_value(f'{avg_vz:.2f} m/s')
            self.consecutive_label.set_value(str(self.detector.get_consecutive_frames()))
            self.raw_count_label.set_value(str(len(raw_points)))
            self.filtered_count_label.set_value(str(stats['count']))
            self.min_height_label.set_value(f'{stats["min_height"]:.2f} m')
            self.max_height_label.set_value(f'{stats["max_height"]:.2f} m')
            self.avg_height_label.set_value(f'{stats["avg_height"]:.2f} m')
            self.min_vz_label.set_value(f'{stats["min_vz"]:.2f} m/s')
            self.max_vz_label.set_value(f'{stats["max_vz"]:.2f} m/s')
            self.std_vz_label.set_value(f'{stats["std_vz"]:.2f}')
            self.height_plot.add_data(avg_height)
            self.speed_plot.add_data(avg_vz)

            if processed_points.size > 0:
                center_x = np.mean(processed_points[:, 0])
                center_y = np.mean(processed_points[:, 1])
                center_z = np.mean(processed_points[:, 2])
                center_position = (center_x, center_y, center_z)
            else:
                center_position = (0.0, 0.0, 1.6)

            current_state = self.simulator.get_state()
            is_alarming = self.has_alarmed or current_state == HumanState.FALLEN
            self.visualizer.update_points(processed_points, current_state, is_alarming)
            self.visualizer.update_trajectory(self.simulator.get_trajectory(), is_alarming)
            self.visualizer.update_center_trajectory(center_position)
            self.skeleton_visualizer.update_skeleton(processed_points, current_state, self.simulator.fall_progress)

            elapsed_time = self.start_time.elapsed() / 1000.0
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.status_bar.showMessage(
                f'运行时间: {format_duration(elapsed_time)} | 帧率: {fps:.1f} FPS | 点云数: {len(processed_points)}'
            )
        except Exception as e:
            logger.error(f"UI更新失败: {e}\n{traceback.format_exc()}")

        # --- 摔倒检测 (独立 try，与 UI 隔离) ---
        try:
            if not self.has_alarmed:
                current_state = self.simulator.get_state()
                is_fallen, _ = self.detector.detect(avg_height, avg_vz, current_state, processed_points)
                if is_fallen:
                    self._trigger_alarm(avg_height, avg_vz)
        except Exception as e:
            logger.error(f"摔倒检测失败: {e}\n{traceback.format_exc()}")

    def _trigger_alarm(self, height: float, speed: float):
        """触发报警"""
        self.has_alarmed = True
        
        # 停止模拟
        self.timer.stop()
        self.is_running = False
        
        # 更新按钮状态 - 禁用所有按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.fall_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        # 更新状态显示 - 报警状态带闪烁效果
        self.status_card.set_status('报警', is_alarming=True)
        self._set_status_bar_color('red')
        self.status_bar.showMessage(f'⚠️ 报警！时间: {get_timestamp()}')
        
        # 显示报警条
        self.alarm_bar.show()
        
        # 在曲线图中标记报警点
        self.height_plot.add_alarm_marker(len(self.height_plot.data))
        self.speed_plot.add_alarm_marker(len(self.speed_plot.data))
        
        # 在3D视图中显示报警标记和更新点云/轨迹为报警状态
        alarm_pos = self.simulator.get_center_position()
        self.visualizer.show_alarm_marker(alarm_pos)
        self.visualizer.update_trajectory(self.simulator.get_trajectory(), is_alarming=True)
        
        # 记录报警日志
        timestamp = get_timestamp()
        self.log_table.add_log(timestamp, height, speed)
        
        # 播放蜂鸣报警声
        if ALARM_CONFIG["enable_sound"]:
            play_beep(ALARM_CONFIG["beep_frequency"], ALARM_CONFIG["beep_duration"])
        
        # 保存截图
        if ALARM_CONFIG["screenshot_enabled"]:
            ensure_dir_exists(ALARM_CONFIG["screenshot_dir"])
            screenshot_path = os.path.join(ALARM_CONFIG["screenshot_dir"], 
                                        f'alarm_{get_timestamp_filename()}.png')
            take_screenshot(self.visualizer, screenshot_path)
        
        # 弹出报警弹窗
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle('紧急报警')
        msg_box.setText('检测到人员摔倒！')
        msg_box.setInformativeText(f'报警时间: {timestamp}\n'
                                f'检测高度: {height:.2f}m\n'
                                f'检测速度: {speed:.2f}m/s\n\n'
                                '请立即前往现场查看！')
        msg_box.setStyleSheet("""
            QMessageBox { background-color: #353535; }
            QMessageBox QLabel { color: #ffffff; }
            QPushButton { 
                background-color: #ff4444; 
                color: white; 
                padding: 8px 20px;
                border-radius: 6px;
            }
        """)
        msg_box.exec_()
        
        logger.warning(f"报警触发: 高度={height:.2f}m, 速度={speed:.2f}m/s")
    
    def _confirm_alarm(self):
        """确认报警"""
        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fall_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        
        # 更新状态显示
        self.status_card.set_status('待机')
        self._set_status_bar_color('gray')
        self.status_bar.showMessage('报警已确认，系统就绪')
        
        # 清除报警标记
        self.visualizer.clear_alarm_marker()
        self.height_plot.resume()
        self.speed_plot.resume()
        
        logger.info("报警已确认")

    def _set_status_bar_color(self, color):
        """设置状态栏颜色"""
        bg_map = {
            'red': '#ffc8c8',
            'green': '#c8ffc8',
            'orange': '#ffffc8',
        }
        bg = bg_map.get(color)
        if bg:
            self.status_bar.setStyleSheet(f'QStatusBar {{ background-color: {bg}; color: #000000; }}')
        else:
            self.status_bar.setStyleSheet('')

    def _export_logs(self):
        """导出报警日志"""
        logs = self.log_table.get_logs()
        if not logs:
            QMessageBox.information(self, '提示', '没有日志可导出')
            return
        
        ensure_dir_exists('logs')
        filepath = os.path.join('logs', f'alarm_logs_{get_timestamp_filename()}.csv')
        if export_to_csv(logs, filepath):
            QMessageBox.information(self, '成功', f'日志已导出到:\n{filepath}')
        else:
            QMessageBox.warning(self, '失败', '导出日志失败')

    def _clear_logs(self):
        """清除报警日志"""
        reply = QMessageBox.question(self, '确认', '确定要清除所有报警日志吗？',
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.log_table.clear_logs()
            logger.info("报警日志已清除")

    def _show_about(self):
        """显示关于对话框"""
        dialog = AboutDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        """关闭窗口事件"""
        reply = QMessageBox.question(self, '确认退出', '确定要退出程序吗？',
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timer.stop()
            logger.info("程序已退出")
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序信息
    app.setApplicationName('毫米波雷达摔倒检测系统')
    app.setApplicationVersion('2.0.0')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
