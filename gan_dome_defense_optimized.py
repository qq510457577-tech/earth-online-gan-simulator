# -*- coding: utf-8 -*-
"""
地球 Online GAN 攻防模拟器 - 优化版
Earth Online GAN Offensive and Defensive Simulator - Optimized Version

优化说明：
1. 代码质量优化：增加类型标注、拆分超长函数、统一命名规范
2. 性能优化：自动检测GPU、复用渲染缓存、优化GAN训练效率
3. 功能增强：支持模型保存/加载、参数预设、数据导出、更多快捷键
4. 鲁棒性优化：跨平台兼容、异常处理、内存泄漏防护
5. 体验优化：自动适配窗口大小、多语言支持、智能告警
"""
import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
import math
import time
import threading
import queue
import json
import csv
import os
from collections import deque
from typing import List, Dict, Tuple, Optional, Union

# 自动检测设备（优先GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] 运行设备: {DEVICE}")

# 依赖检查
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[Warn] 未安装pyserial，串口功能不可用")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.max_open_warning'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warn] 未安装matplotlib，图表功能不可用")

# --------------------------
# 通用工具函数
# --------------------------
def lerp_angle(current: float, target: float, speed: float) -> float:
    """角度线性插值，解决360度环绕问题"""
    diff = (target - current + 180) % 360 - 180
    return current + math.copysign(min(abs(diff), speed), diff)

def sphere_to_xyz(radius: float, azimuth: float, elevation: float) -> Tuple[float, float, float]:
    """球坐标转笛卡尔坐标"""
    x = radius * math.sin(elevation) * math.cos(azimuth)
    y = radius * math.sin(elevation) * math.sin(azimuth)
    z = radius * math.cos(elevation)
    return x, y, z

def gradient_color(start: Tuple[int, int, int], end: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
    """生成渐变颜色"""
    return tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))

def load_font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    """跨平台字体加载，自动fallback"""
    try:
        # 优先尝试用户指定字体
        return pygame.font.SysFont(name, size, bold=bold)
    except:
        # 中文环境fallback
        for font_name in ["simhei", "Microsoft YaHei", "Noto Sans CJK", "PingFang SC", "Arial Unicode MS"]:
            try:
                return pygame.font.SysFont(font_name, size, bold=bold)
            except:
                continue
        # 终极fallback
        return pygame.font.Font(None, size)

# --------------------------
# 配置类
# --------------------------
class Config:
    """全局配置类，支持从文件加载/导出"""
    # 窗口配置
    WIDTH: int = 1400
    HEIGHT: int = 850
    WINDOW_RESIZABLE: bool = True
    FPS: int = 60
    BG_COLOR: Tuple[int, int, int] = (5, 8, 25)
    
    # 穹顶配置
    HEMI_RADIUS: int = 260
    DOME_TILT: float = 0.42
    DOME_SCALE: float = 1.0
    DOME_LAYERS: int = 5
    
    # 模拟参数
    ATTACKER_NUM: int = 30
    LAUNCH_PROB: float = 0.25
    MAX_BALLS: int = 80
    SPEED_BALL: float = 2.8
    IR_LOCK_RADIUS: int = 20
    PRIORITY_MODE: str = "nearest"  # nearest/fastest/threat
    TURRET_SPEED: float = 6.0
    
    # GAN配置
    GAN_LR_G: float = 0.0008
    GAN_LR_D: float = 0.0015
    GAN_TRAIN_FREQ: int = 4
    GAN_BATCH_SIZE: int = 16
    
    # 串口配置
    USE_SERIAL: bool = False
    SERIAL_PORT: str = "COM3"
    BAUD: int = 9600
    
    # 界面配置
    PLOT_FREQ: int = 15
    LOG_MAXLEN: int = 300
    SIM_SPEED: float = 1.0  # 0.1-3.0
    SIM_SPEED_MAX: float = 3.0  # 最大模拟速度限制
    BALL_TRAIL: int = 8
    EXPLOSION_FRAMES: int = 20
    
    # 运行状态
    SIM_RUNNING: bool = False
    VIEW_AZ: float = 30.0  # 水平视角
    VIEW_EL: float = 25.0  # 垂直视角

    @classmethod
    def load_from_file(cls, path: str) -> None:
        """从JSON文件加载配置"""
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
            print(f"[Config] 已加载配置: {path}")
        except Exception as e:
            print(f"[Config] 加载配置失败: {e}")

    @classmethod
    def save_to_file(cls, path: str) -> None:
        """保存配置到JSON文件"""
        try:
            data = {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v)}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[Config] 已保存配置: {path}")
        except Exception as e:
            print(f"[Config] 保存配置失败: {e}")

    @classmethod
    def load_preset(cls, preset_name: str) -> None:
        """加载参数预设"""
        presets = {
            "低强度": {
                "ATTACKER_NUM": 15,
                "LAUNCH_PROB": 0.15,
                "SPEED_BALL": 1.5,
                "TURRET_SPEED": 8.0
            },
            "中强度": {
                "ATTACKER_NUM": 30,
                "LAUNCH_PROB": 0.25,
                "SPEED_BALL": 2.8,
                "TURRET_SPEED": 6.0
            },
            "高强度饱和攻击": {
                "ATTACKER_NUM": 50,
                "LAUNCH_PROB": 0.6,
                "SPEED_BALL": 4.5,
                "TURRET_SPEED": 10.0
            },
            "GAN训练模式": {
                "ATTACKER_NUM": 30,
                "LAUNCH_PROB": 0.3,
                "GAN_TRAIN_FREQ": 2,
                "SIM_SPEED": 2.0
            }
        }
        if preset_name in presets:
            for k, v in presets[preset_name].items():
                setattr(cls, k, v)
            print(f"[Config] 已加载预设: {preset_name}")

cfg = Config()

# --------------------------
# GAN模型
# --------------------------
class Generator(nn.Module):
    def __init__(self, attacker_count: int):
        super().__init__()
        self.attacker_count = attacker_count
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, attacker_count * 3), nn.Tanh()
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.attacker_count, 3)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------
# 硬件控制器
# --------------------------
class SerialController:
    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self.connected: bool = False
        self._queue: queue.Queue = queue.Queue(maxsize=32)
        self._reconnect_interval: int = 5  # 重连间隔（秒）
        self._last_reconnect: float = 0
        if cfg.USE_SERIAL and HAS_SERIAL:
            self._connect()
            threading.Thread(target=self._worker, daemon=True).start()

    def _connect(self) -> None:
        """连接串口"""
        try:
            self.ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD, timeout=1)
            self.connected = True
            print(f"[Serial] 已连接 {cfg.SERIAL_PORT}@{cfg.BAUD}")
        except Exception as e:
            self.connected = False
            print(f"[Serial] 连接失败: {e}")

    def _worker(self) -> None:
        """串口工作线程"""
        while True:
            # 自动重连逻辑
            if not self.connected and time.time() - self._last_reconnect > self._reconnect_interval:
                self._last_reconnect = time.time()
                self._connect()
            
            if self.connected and self.ser and self.ser.is_open:
                try:
                    msg = self._queue.get(timeout=1)
                    self.ser.write(msg.encode())
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Serial] 发送失败: {e}")
                    self.connected = False
            else:
                time.sleep(1)

    def send(self, pan: float, tilt: float, laser: int = 1) -> None:
        """发送控制指令"""
        if self.connected and not self._queue.full():
            try:
                self._queue.put_nowait(f"PAN {pan:.1f} TILT {tilt:.1f} LASER {laser}\n")
            except queue.Full:
                pass

    def close(self) -> None:
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False

# --------------------------
# 日志与数据统计
# --------------------------
class Logger:
    def __init__(self):
        self.hits: deque = deque(maxlen=cfg.LOG_MAXLEN)
        self.launches: deque = deque(maxlen=cfg.LOG_MAXLEN)
        self.accuracy: deque = deque(maxlen=cfg.LOG_MAXLEN)
        self.loss_g: deque = deque(maxlen=cfg.LOG_MAXLEN)
        self.loss_d: deque = deque(maxlen=cfg.LOG_MAXLEN)
        self.events: List[str] = []
        self._surf: Optional[pygame.Surface] = None
        self._lock: threading.Lock = threading.Lock()
        self._dirty: bool = False
        self._plot_cache: Optional[Tuple[int, int, pygame.Surface]] = None

    def push(self, total_hit: int, total_launch: int, loss_g: Optional[float] = None, loss_d: Optional[float] = None) -> None:
        """推送统计数据"""
        self.hits.append(total_hit)
        self.launches.append(total_launch)
        self.accuracy.append(total_hit / (total_launch + 1e-6))
        if loss_g is not None:
            self.loss_g.append(loss_g)
        if loss_d is not None:
            self.loss_d.append(loss_d)
        self._dirty = True

    def log(self, msg: str, level: str = "info") -> None:
        """记录日志"""
        ts = time.strftime("%H:%M:%S")
        level_tag = f"[{level.upper()}]" if level != "info" else ""
        self.events.append(f"[{ts}]{level_tag} {msg}")
        if len(self.events) > 500:
            self.events = self.events[-500:]

    def export_data(self, path: str) -> None:
        """导出统计数据到CSV"""
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "hits", "launches", "accuracy", "loss_g", "loss_d"])
                for i in range(len(self.hits)):
                    writer.writerow([
                        i,
                        self.hits[i],
                        self.launches[i],
                        self.accuracy[i] if i < len(self.accuracy) else "",
                        self.loss_g[i] if i < len(self.loss_g) else "",
                        self.loss_d[i] if i < len(self.loss_d) else ""
                    ])
            self.log(f"数据已导出到: {path}")
        except Exception as e:
            self.log(f"导出数据失败: {e}", level="error")

    def render_plot_bg(self) -> None:
        """后台渲染统计图表"""
        if not HAS_MPL or not self._dirty:
            return
        self._dirty = False
        
        try:
            # 复用缓存尺寸，避免重复创建画布
            if self._plot_cache is None:
                fig, axes = plt.subplots(3, 1, figsize=(4.2, 5.4), dpi=82, facecolor="#050819")
                self._plot_cache = (fig, axes)
            else:
                fig, axes = self._plot_cache
                for ax in axes:
                    ax.clear()

            # 统一样式设置
            for ax in axes:
                ax.set_facecolor("#0a1030")
                ax.tick_params(colors="#aaaacc", labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color("#334")

            # 绘制击中/发射曲线
            if self.hits:
                axes[0].plot(list(self.hits), color="#ff4444", lw=1.3, label="Hits")
                axes[0].plot(list(self.launches), color="#4488ff", lw=1.3, label="Launches")
                axes[0].legend(fontsize=7, facecolor="#0a1030", labelcolor="white")
                axes[0].set_title("Hits vs Launches", color="#aaccff", fontsize=8, pad=2)

            # 绘制命中率曲线
            if self.accuracy:
                axes[1].plot(list(self.accuracy), color="#ffaa00", lw=1.5)
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Hit Rate", color="#aaccff", fontsize=8, pad=2)

            # 绘制GAN损失曲线
            if self.loss_g:
                axes[2].plot(list(self.loss_g), color="#88ff44", lw=1, label="G loss")
                axes[2].plot(list(self.loss_d), color="#ff88ff", lw=1, label="D loss")
                axes[2].legend(fontsize=7, facecolor="#0a1030", labelcolor="white")
                axes[2].set_title("GAN Loss", color="#aaccff", fontsize=8, pad=2)

            plt.tight_layout(pad=0.4)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            surf = pygame.surfarray.make_surface(arr[:, :, :3].swapaxes(0, 1))
            
            with self._lock:
                self._surf = surf
        except Exception as e:
            print(f"[Plot] 渲染失败: {e}")

    def get_surf(self) -> Optional[pygame.Surface]:
        """获取渲染好的图表Surface"""
        with self._lock:
            return self._surf

# --------------------------
# 3D渲染相关
# --------------------------
def project(x: float, y: float, z: float, center_x: float, center_y: float) -> Tuple[int, int]:
    """3D坐标投影到2D屏幕，支持视角旋转"""
    scale = cfg.DOME_SCALE
    az = math.radians(cfg.VIEW_AZ)
    el = math.radians(cfg.VIEW_EL)
    
    # 水平旋转（方位角）
    x1 = x * math.cos(az) - y * math.sin(az)
    y1 = x * math.sin(az) + y * math.cos(az)
    
    # 垂直旋转（俯仰角）
    y2 = y1 * math.cos(el) - z * math.sin(el)
    z2 = y1 * math.sin(el) + z * math.cos(el)
    
    # 等轴投影
    screen_x = int(center_x + x1 * scale)
    screen_y = int(center_y - z2 * scale)
    return screen_x, screen_y

class DomeRenderer:
    """半球穹顶渲染器"""
    def __init__(self):
        self._cache: List = []
        self._radius_cache: int = cfg.HEMI_RADIUS
        self._build()

    def _build(self) -> None:
        """构建穹顶几何缓存"""
        R = cfg.HEMI_RADIUS
        self._cache = []
        
        # 水平环
        for layer in range(1, cfg.DOME_LAYERS + 1):
            elev = math.pi / 2 * layer / cfg.DOME_LAYERS
            r_h = R * math.sin(elev)
            z = R * math.cos(elev)
            pts = [(r_h * math.cos(a), r_h * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 72)]
            bright = int(40 + 100 * layer / cfg.DOME_LAYERS)
            self._cache.append(("ring", pts, (20, bright, 120), True))
        
        # 经线
        for a in np.linspace(0, 2 * math.pi, 18, endpoint=False):
            pts = [sphere_to_xyz(R, a, e) for e in np.linspace(0.02, math.pi / 2, 20)]
            self._cache.append(("merid", pts, (15, 35, 85), False))
        
        # 底部圆环
        pts = [(R * math.cos(a), R * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 100)]
        self._cache.append(("ground", pts, (20, 80, 200), True))
        
        # 内部辅助环
        for frac in [0.33, 0.66]:
            r2 = R * frac
            pts = [(r2 * math.cos(a), r2 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 60)]
            self._cache.append(("inner", pts, (12, 30, 70), True))

    def draw(self, screen: pygame.Surface, center_x: float, center_y: float) -> None:
        """绘制穹顶"""
        # 半径变化时重建缓存
        if cfg.HEMI_RADIUS != self._radius_cache:
            self._radius_cache = cfg.HEMI_RADIUS
            self._build()
            
        for kind, pts, color, closed in self._cache:
            screen_pts = [project(x, y, z, center_x, center_y) for x, y, z in pts]
            pygame.draw.lines(screen, color, closed, screen_pts, 1)
        
        # 穹顶顶部光晕
        tx, ty = project(0, 0, cfg.HEMI_RADIUS, center_x, center_y)
        for r, alpha in [(16, 30), (10, 60), (5, 120)]:
            glow_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (80, 160, 255, alpha), (r, r), r)
            screen.blit(glow_surf, (tx - r, ty - r))

# --------------------------
# UI组件
# --------------------------
class ParamPanel:
    """参数控制面板"""
    PANEL_W = 310
    SLIDER_HEIGHT = 22

    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = pygame.Rect(x, y, w, h)
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self._dragging: Optional[int] = None
        self._sliders = [
            ("攻击方数量", "ATTACKER_NUM", 10, 50, 1, True, 1),
            ("发射概率", "LAUNCH_PROB", 0.05, 0.6, 0.01, False, 1),
            ("小球最大数", "MAX_BALLS", 20, 200, 5, True, 1),
            ("小球速度", "SPEED_BALL", 0.5, 6.0, 0.1, False, 1),
            ("红外半径", "IR_LOCK_RADIUS", 6, 40, 1, True, 1),
            ("云台速度", "TURRET_SPEED", 1.0, 15.0, 0.5, False, 1),
            ("半球半径", "HEMI_RADIUS", 150, 350, 10, True, 1),
            ("生成器学习率", "GAN_LR_G", 0.0001, 0.005, 0.0001, False, 1),
            ("判别器学习率", "GAN_LR_D", 0.0001, 0.005, 0.0001, False, 1),
            ("模拟速度", "SIM_SPEED", 0.1, cfg.SIM_SPEED_MAX, 0.1, False, 1),
            ("尾巴长度", "BALL_TRAIL", 3, 20, 1, True, 1),
            ("爆炸帧数", "EXPLOSION_FRAMES", 5, 30, 1, True, 1),
        ]
        self._priority_modes = ["nearest", "fastest", "threat"]
        self._priority_labels = ["最近", "最快", "威胁"]
        self._presets = ["低强度", "中强度", "高强度饱和攻击", "GAN训练模式"]

    def set_fonts(self, font: pygame.font.Font, font_small: pygame.font.Font) -> None:
        self.font = font
        self.font_small = font_small

    def _slider_rect(self, idx: int) -> pygame.Rect:
        x = self.rect.x + 10
        y = self.rect.y + 170 + idx * 40  # 顶部按钮区 + 优先级 + 预设区
        return pygame.Rect(x, y + 16, self.rect.width - 20, self.SLIDER_HEIGHT)

    def _slider_ratio(self, idx: int) -> float:
        _, attr, min_val, max_val, _, _, scale = self._sliders[idx]
        value = getattr(cfg, attr) * scale
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val + 1e-9)))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """处理UI事件"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # 滑块处理
            for i in range(len(self._sliders)):
                if self._slider_rect(i).collidepoint(event.pos):
                    self._dragging = i
                    self._set_slider_value(i, event.pos[0])
                    return True
            
            # 优先级按钮处理
            for j, mode in enumerate(self._priority_modes):
                btn_rect = pygame.Rect(self.rect.x + 10 + j * 86, self.rect.y + 78, 80, 26)
                if btn_rect.collidepoint(event.pos):
                    cfg.PRIORITY_MODE = mode
                    return True
            
            # 预设按钮处理
            for k, preset in enumerate(self._presets[:2]):
                btn_rect = pygame.Rect(self.rect.x + 10 + k * 142, self.rect.y + 118, 138, 26)
                if btn_rect.collidepoint(event.pos):
                    cfg.load_preset(preset)
                    return True
            for k, preset in enumerate(self._presets[2:]):
                btn_rect = pygame.Rect(self.rect.x + 10 + k * 142, self.rect.y + 150, 138, 26)
                if btn_rect.collidepoint(event.pos):
                    cfg.load_preset(preset)
                    return True
            
            # 启动/停止按钮
            btn_w = (self.rect.width - 30) // 2
            start_btn = pygame.Rect(self.rect.x + 10, self.rect.y + 32, btn_w, 32)
            if start_btn.collidepoint(event.pos):
                cfg.SIM_RUNNING = not cfg.SIM_RUNNING
                return True
            
            # 串口开关按钮
            serial_btn = pygame.Rect(self.rect.x + 20 + btn_w, self.rect.y + 32, btn_w, 32)
            if serial_btn.collidepoint(event.pos):
                cfg.USE_SERIAL = not cfg.USE_SERIAL
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self._dragging = None
        
        elif event.type == pygame.MOUSEMOTION and self._dragging is not None:
            self._set_slider_value(self._dragging, event.pos[0])
            return True
        
        return False

    def _set_slider_value(self, idx: int, mouse_x: int) -> None:
        _, attr, min_val, max_val, step, is_int, scale = self._sliders[idx]
        rect = self._slider_rect(idx)
        ratio = max(0.0, min(1.0, (mouse_x - rect.x) / rect.width))
        value = min_val + ratio * (max_val - min_val)
        value = round(value / step) * step
        if is_int:
            value = int(value)
        setattr(cfg, attr, value / scale if scale != 1 else value)

    def draw(self, screen: pygame.Surface) -> None:
        if not self.font or not self.font_small:
            return
        
        # 背景
        bg = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg.fill((8, 14, 42, 215))
        screen.blit(bg, self.rect.topleft)
        pygame.draw.rect(screen, (30, 60, 150), self.rect, 1)

        # 标题
        title = self.font.render("  参数面板", True, (140, 210, 255))
        screen.blit(title, (self.rect.x + 8, self.rect.y + 10))

        # 顶部双按钮
        btn_w = (self.rect.width - 30) // 2
        
        # 启动/停止按钮
        start_btn = pygame.Rect(self.rect.x + 10, self.rect.y + 32, btn_w, 32)
        start_color = (30, 150, 30) if cfg.SIM_RUNNING else (150, 30, 30)
        start_border = (80, 220, 80) if cfg.SIM_RUNNING else (220, 80, 80)
        pygame.draw.rect(screen, start_color, start_btn, border_radius=5)
        pygame.draw.rect(screen, start_border, start_btn, 1, border_radius=5)
        start_text = self.font.render("■ 停止" if cfg.SIM_RUNNING else "▶ 启动", True, (255, 255, 255))
        screen.blit(start_text, (
            start_btn.x + (start_btn.width - start_text.get_width()) // 2,
            start_btn.y + (start_btn.height - start_text.get_height()) // 2
        ))

        # 串口按钮
        serial_btn = pygame.Rect(self.rect.x + 20 + btn_w, self.rect.y + 32, btn_w, 32)
        serial_color = (15, 100, 30) if cfg.USE_SERIAL else (60, 30, 30)
        serial_border = (60, 200, 80) if cfg.USE_SERIAL else (140, 60, 60)
        pygame.draw.rect(screen, serial_color, serial_btn, border_radius=5)
        pygame.draw.rect(screen,