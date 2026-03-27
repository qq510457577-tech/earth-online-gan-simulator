# -*- coding: utf-8 -*-
"""Dome GAN Defense System - Ultimate Edition"""
import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
import math
import time
import threading
import queue
from collections import deque

try:
    import serial
    HAS_SERIAL = True
except:
    HAS_SERIAL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.max_open_warning'] = False  # 禁用警告
    HAS_MPL = True
except:
    HAS_MPL = False


class Config:
    WIDTH = 1400
    HEIGHT = 850
    WINDOW_RESIZABLE = True
    FPS = 60
    BG_COLOR = (5, 8, 25)
    HEMI_RADIUS = 260
    DOME_TILT = 0.42
    DOME_SCALE = 1.0
    DOME_LAYERS = 5
    ATTACKER_NUM = 30
    LAUNCH_PROB = 0.25
    MAX_BALLS = 80
    SPEED_BALL = 2.8
    IR_LOCK_RADIUS = 20
    PRIORITY_MODE = "nearest"  # 默认最近优先
    TURRET_SPEED = 6.0
    GAN_LR_G = 0.0008
    GAN_LR_D = 0.0015
    GAN_TRAIN_FREQ = 4
    USE_SERIAL = False
    SERIAL_PORT = "COM3"
    BAUD = 9600
    PLOT_FREQ = 15
    LOG_MAXLEN = 300
    # 新增参数
    SIM_SPEED = 1.0  # 模拟速度倍率 0.1-3.0
    BALL_TRAIL = 8   # 小球尾巴长度
    EXPLOSION_FRAMES = 20  # 爆炸效果持续帧数
    SIM_RUNNING = False  # 模拟运行状态
    # 视角参数
    VIEW_AZ = 30.0   # 水平旋转角（方位角），度
    VIEW_EL = 25.0   # 俯仰角，度

cfg = Config()

class Generator(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, n * 3), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).view(-1, self.n, 3)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class SerialController:
    def __init__(self):
        self.ser = None
        self.connected = False
        self._q = queue.Queue(maxsize=16)
        if cfg.USE_SERIAL and HAS_SERIAL:
            try:
                self.ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD, timeout=1)
                self.connected = True
                print(f"[Serial] Connected {cfg.SERIAL_PORT}@{cfg.BAUD}")
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception as e:
                print(f"[Serial] Failed: {e}")

    def _worker(self):
        while True:
            msg = self._q.get()
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(msg.encode())
                except:
                    pass

    def send(self, pan, tilt, laser=1):
        if self.connected and not self._q.full():
            self._q.put_nowait(f"PAN {pan:.1f} TILT {tilt:.1f} LASER {laser}\n")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

class Logger:
    def __init__(self):
        n = cfg.LOG_MAXLEN
        self.hits = deque(maxlen=n)
        self.launches = deque(maxlen=n)
        self.acc = deque(maxlen=n)
        self.loss_g = deque(maxlen=n)
        self.loss_d = deque(maxlen=n)
        self.events = []
        self._surf = None
        self._lock = threading.Lock()
        self._dirty = False

    def push(self, total_hit, total_launch, lg=None, ld=None):
        self.hits.append(total_hit)
        self.launches.append(total_launch)
        self.acc.append(total_hit / (total_launch + 1e-6))
        if lg is not None:
            self.loss_g.append(lg)
        if ld is not None:
            self.loss_d.append(ld)
        self._dirty = True

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {msg}")
        if len(self.events) > 300:
            self.events = self.events[-300:]

    def render_plot_bg(self):
        if not HAS_MPL or not self._dirty:
            return
        self._dirty = False
        try:
            fig, axes = plt.subplots(3, 1, figsize=(4.2, 5.4), dpi=82, facecolor="#050819")
            for ax in axes:
                ax.set_facecolor("#0a1030")
                ax.tick_params(colors="#aaaacc", labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color("#334")
            if self.hits:
                axes[0].plot(list(self.hits), color="#ff4444", lw=1.3, label="Hits")
                axes[0].plot(list(self.launches), color="#4488ff", lw=1.3, label="Launches")
                axes[0].legend(fontsize=7, facecolor="#0a1030", labelcolor="white")
                axes[0].set_title("Hits vs Launches", color="#aaccff", fontsize=8, pad=2)
            if self.acc:
                axes[1].plot(list(self.acc), color="#ffaa00", lw=1.5)
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Hit Rate", color="#aaccff", fontsize=8, pad=2)
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
            plt.close(fig)
            with self._lock:
                self._surf = surf
        except Exception as e:
            print(f"[Plot] {e}")

    def get_surf(self):
        with self._lock:
            return self._surf

def project(x, y, z, cx, cy):
    """3D旋转矩阵投影：支持方位角(VIEW_AZ)和俯仰角(VIEW_EL)视角旋转"""
    s = cfg.DOME_SCALE
    az = math.radians(cfg.VIEW_AZ)
    el = math.radians(cfg.VIEW_EL)
    # 绕Z轴旋转（水平方位角）
    x1 = x * math.cos(az) - y * math.sin(az)
    y1 = x * math.sin(az) + y * math.cos(az)
    z1 = z
    # 绕X轴旋转（俯仰角）
    y2 = y1 * math.cos(el) - z1 * math.sin(el)
    z2 = y1 * math.sin(el) + z1 * math.cos(el)
    # 等轴投影到屏幕
    sx = cx + x1 * s
    sy = cy - z2 * s
    return int(sx), int(sy)


def sphere_to_xyz(r, ang, elev):
    x = r * math.sin(elev) * math.cos(ang)
    y = r * math.sin(elev) * math.sin(ang)
    z = r * math.cos(elev)
    return x, y, z

class DomeRenderer:
    def __init__(self):
        self._cache = []
        self._build()

    def _build(self):
        R = cfg.HEMI_RADIUS
        self._cache = []
        for layer in range(1, cfg.DOME_LAYERS + 1):
            elev = math.pi / 2 * layer / cfg.DOME_LAYERS
            r_h = R * math.sin(elev)
            z = R * math.cos(elev)
            pts = [(r_h * math.cos(a), r_h * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 72)]
            bright = int(40 + 100 * layer / cfg.DOME_LAYERS)
            self._cache.append(("ring", pts, (20, bright, 120), True))
        for a in np.linspace(0, 2 * math.pi, 18, endpoint=False):
            pts = [sphere_to_xyz(R, a, e) for e in np.linspace(0.02, math.pi / 2, 20)]
            self._cache.append(("merid", pts, (15, 35, 85), False))
        pts = [(R * math.cos(a), R * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 100)]
        self._cache.append(("ground", pts, (20, 80, 200), True))
        for frac in [0.33, 0.66]:
            r2 = R * frac
            pts = [(r2 * math.cos(a), r2 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 60)]
            self._cache.append(("inner", pts, (12, 30, 70), True))

    def draw(self, screen, cx, cy):
        for kind, pts, color, closed in self._cache:
            spx = [project(x, y, z, cx, cy) for x, y, z in pts]
            pygame.draw.lines(screen, color, closed, spx, 1)
        tx, ty = project(0, 0, cfg.HEMI_RADIUS, cx, cy)
        for r, al in [(16, 30), (10, 60), (5, 120)]:
            gs = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (80, 160, 255, al), (r, r), r)
            screen.blit(gs, (tx - r, ty - r))

    def rebuild(self):
        self._build()

def compute_threat(ball):
    dist = ball["dist"]
    speed = ball["speed"]
    if cfg.PRIORITY_MODE == "nearest":
        return dist
    if cfg.PRIORITY_MODE == "fastest":
        return -speed
    return dist * 0.6 - speed * 0.4

class ParamPanel:
    PANEL_W = 310
    SH = 22

    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = None
        self.font2 = None
        self._dragging = None
        self._sliders = [
            ("攻击方数量", "ATTACKER_NUM", 10, 50, 1, True, 1),
            ("发射概率", "LAUNCH_PROB", 0.05, 0.6, 0.01, False, 1),
            ("小球最大数", "MAX_BALLS", 20, 200, 5, True, 1),
            ("小球速度", "SPEED_BALL", 0.5, 6.0, 0.1, False, 1),
            ("红外半径", "IR_LOCK_RADIUS", 6, 40, 1, True, 1),
            ("穹顶倾斜", "DOME_TILT", 0.1, 0.9, 0.02, False, 1),
            ("云台速度", "TURRET_SPEED", 1.0, 15.0, 0.5, False, 1),
            # 新增滑块
            ("半球半径", "HEMI_RADIUS", 150, 350, 10, True, 1),
            ("生成器学习率", "GAN_LR_G", 0.0001, 0.005, 0.0001, False, 1),
            ("判别器学习率", "GAN_LR_D", 0.0001, 0.005, 0.0001, False, 1),
            ("模拟速度", "SIM_SPEED", 0.1, 3.0, 0.1, False, 1),
            ("尾巴长度", "BALL_TRAIL", 3, 20, 1, True, 1),
            ("爆炸帧数", "EXPLOSION_FRAMES", 5, 30, 1, True, 1),
        ]
        self._priority_modes = ["nearest", "fastest", "threat"]
        self._priority_labels = ["最近", "最快", "威胁"]

    def set_fonts(self, f, f2):
        self.font = f
        self.font2 = f2

    def _slider_rect(self, idx):
        x = self.rect.x + 10
        y = self.rect.y + 120 + idx * 40  # 顶部按钮区 + 优先级区 = 约120px
        return pygame.Rect(x, y + 16, self.rect.width - 20, self.SH)

    def _ratio(self, idx):
        _, attr, mn, mx, _, _, scale = self._sliders[idx]
        v = getattr(cfg, attr) * scale
        return max(0.0, min(1.0, (v - mn) / (mx - mn + 1e-9)))

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i in range(len(self._sliders)):
                if self._slider_rect(i).collidepoint(e.pos):
                    self._dragging = i
                    self._set_val(i, e.pos[0])
                    return True
            for j, mode in enumerate(self._priority_modes):
                br = pygame.Rect(self.rect.x + 10 + j * 86, self.rect.y + 78, 80, 26)
                if br.collidepoint(e.pos):
                    cfg.PRIORITY_MODE = mode
                    return True
            # 启动/停止按钮（顶部中间左）
            btn_w = (self.rect.width - 30) // 2
            start_btn = pygame.Rect(self.rect.x + 10, self.rect.y + 32, btn_w, 32)
            if start_btn.collidepoint(e.pos):
                cfg.SIM_RUNNING = not cfg.SIM_RUNNING
                return True
            # 串口按钮（顶部中间右）
            sr = pygame.Rect(self.rect.x + 20 + btn_w, self.rect.y + 32, btn_w, 32)
            if sr.collidepoint(e.pos):
                cfg.USE_SERIAL = not cfg.USE_SERIAL
                return True
        elif e.type == pygame.MOUSEBUTTONUP:
            self._dragging = None
        elif e.type == pygame.MOUSEMOTION and self._dragging is not None:
            self._set_val(self._dragging, e.pos[0])
            return True
        return False

    def _set_val(self, idx, mx_pos):
        _, attr, mn, mx, step, is_int, scale = self._sliders[idx]
        r = self._slider_rect(idx)
        ratio = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
        val = mn + ratio * (mx - mn)
        val = round(val / step) * step
        if is_int:
            val = int(val)
        setattr(cfg, attr, val / scale if scale != 1 else val)

    def draw(self, screen):
        if not self.font:
            return
        bg = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg.fill((8, 14, 42, 215))
        screen.blit(bg, self.rect.topleft)
        pygame.draw.rect(screen, (30, 60, 150), self.rect, 1)

        # ── 标题 ──
        t = self.font.render("  参数面板", True, (140, 210, 255))
        screen.blit(t, (self.rect.x + 8, self.rect.y + 10))

        # ── 顶部双按钮（启动/串口） ──
        btn_w = (self.rect.width - 30) // 2
        # 启动/停止
        start_btn = pygame.Rect(self.rect.x + 10, self.rect.y + 32, btn_w, 32)
        pygame.draw.rect(screen, (30, 150, 30) if cfg.SIM_RUNNING else (150, 30, 30), start_btn, border_radius=5)
        pygame.draw.rect(screen, (80, 220, 80) if cfg.SIM_RUNNING else (220, 80, 80), start_btn, 1, border_radius=5)
        start_txt = self.font.render("■ 停止" if cfg.SIM_RUNNING else "▶ 启动", True, (255, 255, 255))
        screen.blit(start_txt, (start_btn.x + (start_btn.width - start_txt.get_width()) // 2,
                                start_btn.y + (start_btn.height - start_txt.get_height()) // 2))
        # 串口
        sr = pygame.Rect(self.rect.x + 20 + btn_w, self.rect.y + 32, btn_w, 32)
        pygame.draw.rect(screen, (15, 100, 30) if cfg.USE_SERIAL else (60, 30, 30), sr, border_radius=5)
        pygame.draw.rect(screen, (60, 200, 80) if cfg.USE_SERIAL else (140, 60, 60), sr, 1, border_radius=5)
        st_txt = self.font2.render("串口: 开" if cfg.USE_SERIAL else "串口: 关", True, (255, 255, 255))
        screen.blit(st_txt, (sr.x + (sr.width - st_txt.get_width()) // 2,
                             sr.y + (sr.height - st_txt.get_height()) // 2))

        # ── 优先级按钮 ──
        lbl = self.font2.render("锁定优先级:", True, (180, 180, 230))
        screen.blit(lbl, (self.rect.x + 10, self.rect.y + 72))
        for j, (mode, name) in enumerate(zip(self._priority_modes, self._priority_labels)):
            br = pygame.Rect(self.rect.x + 10 + j * 86, self.rect.y + 78, 80, 26)
            active = (cfg.PRIORITY_MODE == mode)
            pygame.draw.rect(screen, (35, 90, 200) if active else (18, 38, 80), br, border_radius=4)
            pygame.draw.rect(screen, (60, 130, 255) if active else (40, 60, 130), br, 1, border_radius=4)
            ct = self.font2.render(name, True, (255, 255, 255) if active else (150, 160, 200))
            screen.blit(ct, (br.x + (br.width - ct.get_width()) // 2, br.y + (br.height - ct.get_height()) // 2))

        # ── 滑块列表 ──
        for i, (label, attr, mn, mx, step, is_int, scale) in enumerate(self._sliders):
            y = self.rect.y + 120 + i * 40
            # 标签
            lt = self.font2.render(label, True, (160, 195, 245))
            screen.blit(lt, (self.rect.x + 10, y))
            # 数值（右对齐到标签行）
            raw = getattr(cfg, attr)
            if attr in ("GAN_LR_G", "GAN_LR_D"):
                vstr = f"{raw:.4f}"
            elif is_int:
                vstr = str(int(raw))
            else:
                vstr = f"{raw:.2f}"
            vt = self.font2.render(vstr, True, (255, 225, 80))
            screen.blit(vt, (self.rect.right - vt.get_width() - 12, y))
            # 滑块轨道
            r = self._slider_rect(i)
            pygame.draw.rect(screen, (18, 38, 100), r, border_radius=4)
            ratio = self._ratio(i)
            fr = pygame.Rect(r.x, r.y, max(4, int(r.width * ratio)), r.height)
            pygame.draw.rect(screen, (35, 100, 230), fr, border_radius=4)
            # 滑块把手
            hx = r.x + int(r.width * ratio)
            pygame.draw.circle(screen, (120, 185, 255), (hx, r.y + r.height // 2), 8)

class LogPanel:
    MAX_LINES = 5

    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = None

    def set_font(self, f):
        self.font = f

    def draw(self, screen, logger):
        if not self.font:
            return
        bg = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        bg.fill((8, 14, 38, 200))
        screen.blit(bg, self.rect.topleft)
        pygame.draw.rect(screen, (30, 55, 130), self.rect, 1)
        t = self.font.render(" Combat Log", True, (140, 210, 255))
        screen.blit(t, (self.rect.x + 6, self.rect.y + 5))
        for i, line in enumerate(logger.events[-self.MAX_LINES:]):
            if "HIT" in line.upper() or "命中" in line:
                col = (100, 255, 130)
            elif "LAUNCH" in line.upper() or "FIRE" in line.upper() or "发射" in line:
                col = (255, 180, 80)
            else:
                col = (160, 160, 215)
            lt = self.font.render(line, True, col)
            screen.blit(lt, (self.rect.x + 6, self.rect.y + 24 + i * 16))

class GameController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Dome GAN Defense System - Ultimate Edition")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("simhei", 15)
            self.font2 = pygame.font.SysFont("simhei", 13)
            self.font_b = pygame.font.SysFont("simhei", 17, bold=True)
            self.font_s = pygame.font.SysFont("simhei", 12)
        except:
            self.font = pygame.font.SysFont(None, 16)
            self.font2 = pygame.font.SysFont(None, 14)
            self.font_b = pygame.font.SysFont(None, 18)
            self.font_s = pygame.font.SysFont(None, 13)
        self.dome = DomeRenderer()
        self.serial = SerialController()
        self.logger = Logger()
        PW = ParamPanel.PANEL_W
        px = cfg.WIDTH - PW - 4
        # 调整布局：参数面板(右上) + 日志(右下) + 图表(底部居中)
        self.param_panel = ParamPanel(px, 4, PW, 590)  # 参数面板（增大高度容纳13个滑块）
        self.param_panel.set_fonts(self.font_b, self.font2)
        # 日志面板放在右侧底部
        self.log_panel = LogPanel(px, 600, PW, 100)  # 稍大的日志
        self.log_panel.set_font(self.font_s)
        self.G = Generator(cfg.ATTACKER_NUM)
        self.D = Discriminator()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()
        self.attackers = []
        self.balls = []
        self.ball_trails = {}  # 小球尾巴 {ball_id: [(x,y,z), ...]}
        self.laser_trails = []  # 激光轨迹列表
        self.hit_effects = []  # 击中特效列表
        self.total_hit = 0
        self.total_launch = 0
        self.frame = 0
        self.loss_g_val = 0.0
        self.loss_d_val = 0.0
        self.turret_pan = 0.0
        self.turret_tilt = 45.0
        self._prev_attacker_num = cfg.ATTACKER_NUM
        self._prev_lr_g = cfg.GAN_LR_G
        self._prev_lr_d = cfg.GAN_LR_D
        self._init_attackers()
        self.logger.log("System start - ready")

    def _init_attackers(self):
        self.attackers = []
        for i in range(cfg.ATTACKER_NUM):
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2)
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x": x, "y": y, "z": z, "ang": ang, "elev": elev})

    def _train_gan(self):
        if len(self.balls) < 4:
            return
        batch, labels = [], []
        for b in self.balls:
            d, sp = b["dist"], b["speed"]
            batch.append([b["x"], b["y"], b["z"], sp, d, 1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
            labels.append([1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
        x = torch.tensor(batch, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        self.optD.zero_grad()
        ld = self.bce(self.D(x), y)
        ld.backward()
        self.optD.step()
        self.optG.zero_grad()
        lg = -torch.mean(torch.log(self.D(x.detach()) + 1e-7))
        lg.backward()
        self.optG.step()
        self.loss_d_val = float(ld)
        self.loss_g_val = float(lg)

    def _update_attackers(self):
        n = cfg.ATTACKER_NUM
        while len(self.attackers) < n:
            ang = np.random.uniform(0, 2 * math.pi)
            elev = np.random.uniform(0.1, math.pi / 2)
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x": x, "y": y, "z": z, "ang": ang, "elev": elev})
        self.attackers = self.attackers[:n]
        if n != self._prev_attacker_num:
            self.G = Generator(n)
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
            self._prev_attacker_num = n
        # 检测学习率变化并更新优化器
        if hasattr(self, '_prev_lr_g') and (self._prev_lr_g != cfg.GAN_LR_G or self._prev_lr_d != cfg.GAN_LR_D):
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self._prev_lr_g = cfg.GAN_LR_G
        self._prev_lr_d = cfg.GAN_LR_D
        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        noise = torch.randn(1, 10)
        state = torch.tensor([[hit_rate, len(self.balls) / (cfg.MAX_BALLS + 1e-6),
                              (self.frame * 0.001) % 1.0, self.loss_g_val, self.loss_d_val, np.random.rand()]], dtype=torch.float32)
        with torch.no_grad():
            g_out = self.G(torch.cat([noise, state], dim=1)).squeeze(0).numpy()
        launched_now = 0
        speed = cfg.SIM_SPEED
        for i, a in enumerate(self.attackers):
            if i >= g_out.shape[0]:
                break
            a["ang"] = (a["ang"] + float(g_out[i, 0]) * 0.06 * speed) % (2 * math.pi)
            a["elev"] = max(0.05, min(math.pi / 2, a["elev"] + float(g_out[i, 1]) * 0.04 * speed))
            a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            # 发射逻辑：GAN输出 + 备用随机发射（确保未经训练时也能发射）
            gan_launch = float(g_out[i, 2])  # GAN输出范围 [-1, 1]
            launch_threshold = cfg.LAUNCH_PROB * 2 - 1  # 将LAUNCH_PROB映射到 [-1,1] 范围
            # 主要用GAN输出判断，GAN输出<0.3时启用备用随机发射
            gan_decide = gan_launch > launch_threshold
            random_decide = np.random.rand() < cfg.LAUNCH_PROB * 1.5  # 备用随机发射
            can_launch = (gan_decide or (gan_launch < 0.3 and random_decide))
            if len(self.balls) < cfg.MAX_BALLS and can_launch and np.random.rand() < 0.35 * speed:
                speed_val = cfg.SPEED_BALL * (0.8 + np.random.rand() * 0.4)
                R = cfg.HEMI_RADIUS
                # 朝球心方向为基础，加入随机偏角（±25°）让弹道有散布
                # 基础方向：从攻击点指向球心
                base_vx = -a["x"] / R
                base_vy = -a["y"] / R
                base_vz = -a["z"] / R
                # 随机偏转角度（弧度），模拟发射散布
                scatter = math.radians(np.random.uniform(0, 25))
                scatter_az = np.random.uniform(0, 2 * math.pi)  # 随机偏转方向
                # 构造垂直于飞行方向的偏转
                perp1 = np.array([-base_vy, base_vx, 0.0])
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.array([1.0, 0.0, 0.0])
                else:
                    perp1 = perp1 / np.linalg.norm(perp1)
                base_dir = np.array([base_vx, base_vy, base_vz])
                perp2 = np.cross(base_dir, perp1)
                # 偏转后的方向
                offset = math.sin(scatter) * (math.cos(scatter_az) * perp1 + math.sin(scatter_az) * perp2)
                final_dir = base_dir * math.cos(scatter) + offset
                norm = np.linalg.norm(final_dir)
                if norm > 1e-6:
                    final_dir /= norm
                vx = float(final_dir[0]) * speed_val
                vy = float(final_dir[1]) * speed_val
                vz = float(final_dir[2]) * speed_val
                self.balls.append({"x": a["x"], "y": a["y"], "z": a["z"],
                                   "vx": vx, "vy": vy, "vz": vz,
                                   "dist": R, "speed": speed_val})
                self.total_launch += 1
                launched_now += 1
        if launched_now > 0 and self.frame % 12 == 0:
            self.logger.log(f"Fire {launched_now} rounds, total {self.total_launch}")

    def _update_balls(self):
        new_balls = []
        new_trails = {}
        hits_now = 0
        speed = cfg.SIM_SPEED
        # 守方激光当前指向的单位向量（从turret_pan/tilt计算）
        pan_r = math.radians(self.turret_pan)
        tilt_r = math.radians(self.turret_tilt)
        laser_dx = math.cos(tilt_r) * math.cos(pan_r)
        laser_dy = math.cos(tilt_r) * math.sin(pan_r)
        laser_dz = math.sin(tilt_r)
        # IR锁定角（半径映射为度：锁定半径越大，锁定角越宽）
        lock_angle_deg = max(2.0, cfg.IR_LOCK_RADIUS * 0.15)
        lock_cos = math.cos(math.radians(lock_angle_deg))
        for b in self.balls:
            # 记录尾巴
            bid = id(b)
            if bid not in self.ball_trails:
                self.ball_trails[bid] = []
            trail = self.ball_trails[bid]
            trail.append((b["x"], b["y"], b["z"]))
            if len(trail) > cfg.BALL_TRAIL:
                trail.pop(0)
            new_trails[bid] = trail
            
            # 移动
            b["x"] += b["vx"] * speed
            b["y"] += b["vy"] * speed
            b["z"] += b["vz"] * speed
            d = math.sqrt(b["x"] ** 2 + b["y"] ** 2 + b["z"] ** 2)
            sp = math.sqrt(b["vx"] ** 2 + b["vy"] ** 2 + b["vz"] ** 2)
            b["dist"] = d
            b["speed"] = sp
            
            # 命中判定：守方激光方向与小球方向夹角 < 锁定角，且小球在有效射程内
            if d > 1e-3:
                bx_n = b["x"] / d
                by_n = b["y"] / d
                bz_n = b["z"] / d
                dot = laser_dx * bx_n + laser_dy * by_n + laser_dz * bz_n
                in_beam = dot > lock_cos
                in_range = d < cfg.HEMI_RADIUS * 0.85  # 小球进入穹顶内85%深度才可被拦截
                if in_beam and in_range:
                    self.total_hit += 1
                    hits_now += 1
                    self.hit_effects.append({
                        "x": b["x"], "y": b["y"], "z": b["z"],
                        "life": cfg.EXPLOSION_FRAMES, "max_life": cfg.EXPLOSION_FRAMES
                    })
                    continue
            
            # 飞出穹顶外丢弃
            if d > cfg.HEMI_RADIUS + 150:
                continue
            new_balls.append(b)
        self.balls = new_balls
        self.ball_trails = new_trails
        if hits_now > 0:
            rate = self.total_hit / (self.total_launch + 1e-6)
            self.logger.log(f"HIT {hits_now}! Total hits {self.total_hit} Rate {rate:.1%}")

    def _update_turret(self):
        if not self.balls:
            # 淡出激光轨迹
            self.laser_trails = [t for t in self.laser_trails if t["life"] > 0]
            for t in self.laser_trails:
                t["life"] -= 1
            return None
        self.balls.sort(key=compute_threat)
        t = self.balls[0]
        pan = math.degrees(math.atan2(t["y"], t["x"]))
        tilt = math.degrees(math.atan2(t["z"], math.sqrt(t["x"] ** 2 + t["y"] ** 2) + 1e-6))

        def lerp_ang(cur, tgt, spd):
            d = (tgt - cur + 180) % 360 - 180
            return cur + math.copysign(min(abs(d), spd), d)
        speed = cfg.SIM_SPEED
        self.turret_pan = lerp_ang(self.turret_pan, pan, cfg.TURRET_SPEED * speed)
        self.turret_tilt = lerp_ang(self.turret_tilt, tilt, cfg.TURRET_SPEED * speed)
        self.serial.send(self.turret_pan, self.turret_tilt, laser=1)
        
        # 记录激光轨迹
        scx, scy, scz = 0, 0, 0
        self.laser_trails.append({
            "sx": scx, "sy": scy, "sz": scz,
            "ex": t["x"], "ey": t["y"], "ez": t["z"],
            "life": 15, "max_life": 15
        })
        # 限制轨迹数量
        if len(self.laser_trails) > 30:
            self.laser_trails = self.laser_trails[-20:]
        
        # 淡出旧轨迹
        for lt in self.laser_trails:
            lt["life"] -= 1
        self.laser_trails = [lt for lt in self.laser_trails if lt["life"] > 0]
        
        return t

    def run(self):
        running = True
        mouse_dragging = False   # 鼠标左键拖拽旋转
        drag_last = (0, 0)       # 上一帧鼠标位置
        while running:
            cx = cfg.WIDTH // 2 - ParamPanel.PANEL_W // 2
            cy = cfg.HEIGHT // 2
            self.screen.fill(cfg.BG_COLOR)
            # 记录上一帧的运行状态用于检测变化
            prev_running = getattr(self, '_last_sim_running', None)
            self._last_sim_running = cfg.SIM_RUNNING
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                if e.type == pygame.VIDEORESIZE:
                    cfg.WIDTH, cfg.HEIGHT = e.w, e.h
                    self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT), pygame.RESIZABLE)
                    PW = ParamPanel.PANEL_W
                    px = cfg.WIDTH - PW - 4
                    self.param_panel.rect.x = px
                    self.log_panel.rect.x = px
                # 鼠标左键拖拽旋转视角
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    # 若点击落在参数面板内，交给面板处理
                    if not self.param_panel.rect.collidepoint(e.pos):
                        mouse_dragging = True
                        drag_last = e.pos
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    mouse_dragging = False
                if e.type == pygame.MOUSEMOTION and mouse_dragging:
                    dx = e.pos[0] - drag_last[0]
                    dy = e.pos[1] - drag_last[1]
                    cfg.VIEW_AZ = (cfg.VIEW_AZ - dx * 0.4) % 360
                    cfg.VIEW_EL = max(-10.0, min(80.0, cfg.VIEW_EL + dy * 0.3))
                    drag_last = e.pos
                # 滚轮缩放
                if e.type == pygame.MOUSEWHEEL:
                    cfg.DOME_SCALE = max(0.3, min(3.0, cfg.DOME_SCALE + e.y * 0.05))
                self.param_panel.handle_event(e)
            
            # 键盘左右键长按旋转视角
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                cfg.VIEW_AZ = (cfg.VIEW_AZ + 1.2) % 360
            if keys[pygame.K_RIGHT]:
                cfg.VIEW_AZ = (cfg.VIEW_AZ - 1.2) % 360
            if keys[pygame.K_UP]:
                cfg.VIEW_EL = min(80.0, cfg.VIEW_EL + 0.8)
            if keys[pygame.K_DOWN]:
                cfg.VIEW_EL = max(-10.0, cfg.VIEW_EL - 0.8)
            
            # 检测模拟状态变化并记录日志
            if prev_running is not None and prev_running != cfg.SIM_RUNNING:
                if cfg.SIM_RUNNING:
                    self.logger.log("=== 模拟启动 ===")
                else:
                    self.logger.log("=== 模拟暂停 ===")
            
            # 仅在模拟运行时更新
            if cfg.SIM_RUNNING:
                self.frame += 1
                self._update_attackers()
                self._update_balls()
                target = self._update_turret()
                if self.frame % cfg.GAN_TRAIN_FREQ == 0:
                    self._train_gan()
                self.logger.push(self.total_hit, self.total_launch, self.loss_g_val, self.loss_d_val)
                if self.frame % cfg.PLOT_FREQ == 0:
                    threading.Thread(target=self.logger.render_plot_bg, daemon=True).start()
            else:
                target = None
            self.dome.draw(self.screen, cx, cy)
            
            # 绘制激光轨迹（更明显）
            for lt in self.laser_trails:
                ratio = lt["life"] / lt["max_life"]
                sx, sy = project(lt["sx"], lt["sy"], lt["sz"], cx, cy)
                ex, ey = project(lt["ex"], lt["ey"], lt["ez"], cx, cy)
                # 黄色激光轨迹 - 不使用透明
                color = (min(255, int(255 * ratio)), min(255, int(255 * ratio)), 0)
                pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 4)
                # 激光头部亮点
                pygame.draw.circle(self.screen, (255, 255, 200), (ex, ey), 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 2)
            
            # 绘制小球（带青色彗星尾巴）
            for b in self.balls:
                bx, by, bz = b["x"], b["y"], b["z"]
                sx, sy = project(bx, by, bz, cx, cy)
                bid = id(b)
                if bid in self.ball_trails:
                    trail = self.ball_trails[bid]
                    trail_len = len(trail)
                    # 先画尾巴 - 青色渐变，越靠近尾部越透明
                    for i, (tx, ty, tz) in enumerate(trail):
                        px, py = project(tx, ty, tz, cx, cy)
                        # 渐变：头部最亮（青色），尾部最暗（深蓝）
                        ratio = (i + 1) / trail_len  # 0到1，越靠后越透明
                        alpha = int(200 * ratio)  # 透明度渐变
                        size = max(1, int(6 * ratio))  # 大小渐变
                        # 青色渐变：从亮青(0,255,255)到尾部的深蓝(0,100,180)
                        r = 0
                        g = int(100 + 155 * ratio)
                        b_col = int(180 + 75 * ratio)
                        # 使用带透明度的圆绘制
                        trail_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
                        pygame.draw.circle(trail_surf, (r, g, b_col, alpha), (size * 2, size * 2), size)
                        self.screen.blit(trail_surf, (px - size * 2, py - size * 2))
                # 小球本体 - 青色攻击球
                pygame.draw.circle(self.screen, (0, 220, 255), (sx, sy), 8)  # 外圈
                pygame.draw.circle(self.screen, (200, 255, 255), (sx, sy), 4)  # 内核
            
            # 绘制击中特效（红色爆炸 + 橙色波纹）
            for he in self.hit_effects:
                ratio = he["life"] / he["max_life"]
                hx, hy = project(he["x"], he["y"], he["z"], cx, cy)
                
                # 1. 红色爆炸核心 - 从中心向外扩散
                core_radius = int(25 * (1 - ratio) + 3)
                for r in range(core_radius, 0, -2):
                    alpha = int(255 * ratio * (1 - r / (core_radius + 1)))
                    exp_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(exp_surf, (255, 50, 30, alpha), (r + 2, r + 2), r)
                    self.screen.blit(exp_surf, (hx - r - 2, hy - r - 2))
                
                # 2. 橙色波纹粒子 - 向外扩散的圆环
                ripple_radius = int(35 * (1 - ratio) + 8)
                ripple_alpha = int(180 * ratio)
                ripple_surf = pygame.Surface((ripple_radius * 2 + 6, ripple_radius * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(ripple_surf, (255, 150, 0, ripple_alpha), (ripple_radius + 3, ripple_radius + 3), ripple_radius, 2)
                self.screen.blit(ripple_surf, (hx - ripple_radius - 3, hy - ripple_radius - 3))
                
                # 3. 第二层橙色波纹（更外层）
                ripple2_radius = int(50 * (1 - ratio) + 12)
                ripple2_alpha = int(120 * ratio)
                ripple2_surf = pygame.Surface((ripple2_radius * 2 + 6, ripple2_radius * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(ripple2_surf, (255, 100, 0, ripple2_alpha), (ripple2_radius + 3, ripple2_radius + 3), ripple2_radius, 1)
                self.screen.blit(ripple2_surf, (hx - ripple2_radius - 3, hy - ripple2_radius - 3))
                
                # 4. 中心白色亮点
                pygame.draw.circle(self.screen, (255, 255, 220), (hx, hy), int(5 * ratio) + 1)
            
            # 更新击中特效生命周期
            self.hit_effects = [he for he in self.hit_effects if he["life"] > 0]
            for he in self.hit_effects:
                he["life"] -= 1
            
            # 绘制防御塔（中心点）
            scx, scy = project(0, 0, 0, cx, cy)
            # 外圈
            pygame.draw.circle(self.screen, (255, 140, 0), (scx, scy), 12)
            # 内圈
            pygame.draw.circle(self.screen, (255, 200, 50), (scx, scy), 7)
            # 中心亮点
            pygame.draw.circle(self.screen, (255, 255, 200), (scx, scy), 3)
            
            # 实时激光（当前锁定目标 - 更明显）
            if target:
                tx, ty = project(target["x"], target["y"], target["z"], cx, cy)
                # 外层粗线
                pygame.draw.line(self.screen, (255, 200, 0), (scx, scy), (tx, ty), 5)
                # 内层亮线
                pygame.draw.line(self.screen, (255, 255, 150), (scx, scy), (tx, ty), 2)
                # 目标点亮点
                pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), 4)
            
            info = [f"模拟状态: {'运行中' if cfg.SIM_RUNNING else '已暂停'}",
                    f"攻击方: {cfg.ATTACKER_NUM}  小球: {len(self.balls)}",
                    f"发射: {self.total_launch}  命中: {self.total_hit}",
                    f"命中率: {self.total_hit / (self.total_launch + 1e-6):.1%}",
                    f"优先级: {cfg.PRIORITY_MODE}",
                    f"模拟速度: {cfg.SIM_SPEED:.1f}x",
                    f"串口: {'开启' if cfg.USE_SERIAL else '关闭'}",
                    f"视角: 方位{cfg.VIEW_AZ:.0f}° 俯仰{cfg.VIEW_EL:.0f}°",
                    f"[拖拽/←→↑↓旋转  滚轮缩放]"]
            for i, t in enumerate(info):
                self.screen.blit(self.font.render(t, True, (200, 220, 255)), (20, 20 + i * 22))
            self.param_panel.draw(self.screen)
            
            # 图表显示在左下角
            ps = self.logger.get_surf()
            if ps:
                # 左下角显示
                plot_w, plot_h = ps.get_width(), ps.get_height()
                plot_x = 10  # 左侧边距
                plot_y = cfg.HEIGHT - plot_h - 10  # 底部边距
                # 绘制背景框
                bg_rect = pygame.Rect(plot_x - 5, plot_y - 5, plot_w + 10, plot_h + 10)
                pygame.draw.rect(self.screen, (10, 15, 35), bg_rect)
                pygame.draw.rect(self.screen, (40, 60, 120), bg_rect, 1)
                self.screen.blit(ps, (plot_x, plot_y))
            
            # 日志显示在图表上方（右侧）
            self.log_panel.draw(self.screen, self.logger)
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
        pygame.quit()


if __name__ == "__main__":
    GameController().run()
