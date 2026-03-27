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
    HAS_MPL = True
except:
    HAS_MPL = False


class Config:
    WIDTH = 1200
    HEIGHT = 780
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
    PRIORITY_MODE = "threat"
    TURRET_SPEED = 6.0
    GAN_LR_G = 0.0008
    GAN_LR_D = 0.0015
    GAN_TRAIN_FREQ = 4
    USE_SERIAL = False
    SERIAL_PORT = "COM3"
    BAUD = 9600
    PLOT_FREQ = 15
    LOG_MAXLEN = 300

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
    s = cfg.DOME_SCALE
    t = cfg.DOME_TILT
    sx = cx + x * s
    sy = cy - z * s * (1.0 - t * 0.5) + y * s * t
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
    PANEL_W = 228
    SH = 22

    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = None
        self.font2 = None
        self._dragging = None
        self._sliders = [
            ("AttackerN", "ATTACKER_NUM", 10, 50, 1, True, 1),
            ("LaunchProb", "LAUNCH_PROB", 0.05, 0.6, 0.01, False, 1),
            ("MaxBalls", "MAX_BALLS", 20, 200, 5, True, 1),
            ("BallSpeed", "SPEED_BALL", 0.5, 6.0, 0.1, False, 1),
            ("IRRadius", "IR_LOCK_RADIUS", 6, 40, 1, True, 1),
            ("DomeTilt", "DOME_TILT", 0.1, 0.9, 0.02, False, 1),
            ("TurretSpd", "TURRET_SPEED", 1.0, 15.0, 0.5, False, 1),
        ]
        self._priority_modes = ["nearest", "fastest", "threat"]
        self._priority_labels = ["Nearest", "Fastest", "Threat"]

    def set_fonts(self, f, f2):
        self.font = f
        self.font2 = f2

    def _slider_rect(self, idx):
        x = self.rect.x + 10
        y = self.rect.y + 90 + idx * 42
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
                br = pygame.Rect(self.rect.x + 10 + j * 68, self.rect.y + 38, 62, 26)
                if br.collidepoint(e.pos):
                    cfg.PRIORITY_MODE = mode
                    return True
            sr = pygame.Rect(self.rect.x + 10, self.rect.y + self.rect.height - 44, 120, 28)
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
        t = self.font.render("  Param Panel", True, (140, 210, 255))
        screen.blit(t, (self.rect.x + 8, self.rect.y + 10))
        lbl = self.font2.render("Priority:", True, (180, 180, 230))
        screen.blit(lbl, (self.rect.x + 10, self.rect.y + 38))
        for j, (mode, name) in enumerate(zip(self._priority_modes, self._priority_labels)):
            br = pygame.Rect(self.rect.x + 10 + j * 68, self.rect.y + 38, 62, 26)
            active = (cfg.PRIORITY_MODE == mode)
            pygame.draw.rect(screen, (35, 90, 200) if active else (18, 38, 80), br, border_radius=4)
            pygame.draw.rect(screen, (60, 130, 255) if active else (40, 60, 130), br, 1, border_radius=4)
            ct = self.font2.render(name, True, (255, 255, 255) if active else (150, 160, 200))
            screen.blit(ct, (br.x + (br.width - ct.get_width()) // 2, br.y + (br.height - ct.get_height()) // 2))
        for i, (label, attr, mn, mx, step, is_int, scale) in enumerate(self._sliders):
            y = self.rect.y + 80 + i * 42
            lt = self.font2.render(label, True, (160, 195, 245))
            screen.blit(lt, (self.rect.x + 10, y))
            r = self._slider_rect(i)
            pygame.draw.rect(screen, (18, 38, 100), r, border_radius=4)
            ratio = self._ratio(i)
            fr = pygame.Rect(r.x, r.y, max(4, int(r.width * ratio)), r.height)
            pygame.draw.rect(screen, (35, 100, 230), fr, border_radius=4)
            hx = r.x + int(r.width * ratio)
            pygame.draw.circle(screen, (120, 185, 255), (hx, r.y + r.height // 2), 8)
            raw = getattr(cfg, attr)
            vstr = str(int(raw)) if is_int else f"{raw:.2f}"
            vt = self.font2.render(vstr, True, (255, 225, 80))
            screen.blit(vt, (r.right - vt.get_width() - 2, y))
        sr = pygame.Rect(self.rect.x + 10, self.rect.y + self.rect.height - 44, 130, 28)
        pygame.draw.rect(screen, (15, 100, 30) if cfg.USE_SERIAL else (90, 15, 15), sr, border_radius=4)
        pygame.draw.rect(screen, (40, 180, 60) if cfg.USE_SERIAL else (160, 40, 40), sr, 1, border_radius=4)
        st_txt = "Serial: ON" if cfg.USE_SERIAL else "Serial: OFF"
        st = self.font2.render(st_txt, True, (255, 255, 255))
        screen.blit(st, (sr.x + (sr.width - st.get_width()) // 2, sr.y + (sr.height - st.get_height()) // 2))

class LogPanel:
    MAX_LINES = 11

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
        self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
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
        self.param_panel = ParamPanel(px, 4, PW, 520)
        self.param_panel.set_fonts(self.font_b, self.font2)
        self.log_panel = LogPanel(px, 528, PW, cfg.HEIGHT - 532)
        self.log_panel.set_font(self.font_s)
        self.G = Generator(cfg.ATTACKER_NUM)
        self.D = Discriminator()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()
        self.attackers = []
        self.balls = []
        self.total_hit = 0
        self.total_launch = 0
        self.frame = 0
        self.loss_g_val = 0.0
        self.loss_d_val = 0.0
        self.turret_pan = 0.0
        self.turret_tilt = 45.0
        self._prev_attacker_num = cfg.ATTACKER_NUM
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
            self._prev_attacker_num = n
        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        noise = torch.randn(1, 10)
        state = torch.tensor([[hit_rate, len(self.balls) / (cfg.MAX_BALLS + 1e-6),
                              (self.frame * 0.001) % 1.0, self.loss_g_val, self.loss_d_val, np.random.rand()]], dtype=torch.float32)
        with torch.no_grad():
            g_out = self.G(torch.cat([noise, state], dim=1)).squeeze(0).numpy()
        launched_now = 0
        for i, a in enumerate(self.attackers):
            if i >= g_out.shape[0]:
                break
            a["ang"] = (a["ang"] + float(g_out[i, 0]) * 0.06) % (2 * math.pi)
            a["elev"] = max(0.05, min(math.pi / 2, a["elev"] + float(g_out[i, 1]) * 0.04))
            a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])
            if len(self.balls) < cfg.MAX_BALLS and float(g_out[i, 2]) > cfg.LAUNCH_PROB and np.random.rand() < 0.35:
                speed = cfg.SPEED_BALL * (0.8 + np.random.rand() * 0.4)
                R = cfg.HEMI_RADIUS
                vx = -a["x"] / R * speed * 10
                vy = -a["y"] / R * speed * 10
                vz = -a["z"] / R * speed * 10
                self.balls.append({"x": a["x"], "y": a["y"], "z": a["z"], "vx": vx, "vy": vy, "vz": vz, "dist": R, "speed": speed * 10})
                self.total_launch += 1
                launched_now += 1
        if launched_now > 0 and self.frame % 12 == 0:
            self.logger.log(f"Fire {launched_now} rounds, total {self.total_launch}")

    def _update_balls(self):
        new_balls = []
        hits_now = 0
        for b in self.balls:
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["z"] += b["vz"]
            d = math.sqrt(b["x"] ** 2 + b["y"] ** 2 + b["z"] ** 2)
            sp = math.sqrt(b["vx"] ** 2 + b["vy"] ** 2 + b["vz"] ** 2)
            b["dist"] = d
            b["speed"] = sp
            if d < cfg.IR_LOCK_RADIUS:
                self.total_hit += 1
                hits_now += 1
                continue
            if d > cfg.HEMI_RADIUS + 150:
                continue
            new_balls.append(b)
        self.balls = new_balls
        if hits_now > 0:
            rate = self.total_hit / (self.total_launch + 1e-6)
            self.logger.log(f"HIT {hits_now}! Total hits {self.total_hit} Rate {rate:.1%}")

    def _update_turret(self):
        if not self.balls:
            return None
        self.balls.sort(key=compute_threat)
        t = self.balls[0]
        pan = math.degrees(math.atan2(t["y"], t["x"]))
        tilt = math.degrees(math.atan2(t["z"], math.sqrt(t["x"] ** 2 + t["y"] ** 2) + 1e-6))

        def lerp_ang(cur, tgt, spd):
            d = (tgt - cur + 180) % 360 - 180
            return cur + math.copysign(min(abs(d), spd), d)
        self.turret_pan = lerp_ang(self.turret_pan, pan, cfg.TURRET_SPEED)
        self.turret_tilt = lerp_ang(self.turret_tilt, tilt, cfg.TURRET_SPEED)
        self.serial.send(self.turret_pan, self.turret_tilt, laser=1)
        return t

    def run(self):
        cx, cy = cfg.WIDTH // 2 - 50, cfg.HEIGHT // 2
        running = True
        while running:
            self.screen.fill(cfg.BG_COLOR)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                self.param_panel.handle_event(e)
            self.frame += 1
            self._update_attackers()
            self._update_balls()
            target = self._update_turret()
            if self.frame % cfg.GAN_TRAIN_FREQ == 0:
                self._train_gan()
            self.logger.push(self.total_hit, self.total_launch, self.loss_g_val, self.loss_d_val)
            if self.frame % cfg.PLOT_FREQ == 0:
                threading.Thread(target=self.logger.render_plot_bg, daemon=True).start()
            self.dome.draw(self.screen, cx, cy)
            for a in self.attackers:
                sx, sy = project(a["x"], a["y"], a["z"], cx, cy)
                pygame.draw.circle(self.screen, (255, 60, 60), (sx, sy), 6)
            for b in self.balls:
                sx, sy = project(b["x"], b["y"], b["z"], cx, cy)
                pygame.draw.circle(self.screen, (100, 180, 255), (sx, sy), 4)
            scx, scy = project(0, 0, 0, cx, cy)
            pygame.draw.circle(self.screen, (255, 160, 0), (scx, scy), 10)
            if target:
                tx, ty = project(target["x"], target["y"], target["z"], cx, cy)
                pygame.draw.line(self.screen, (255, 255, 0), (scx, scy), (tx, ty), 2)
            info = [f"Attackers: {cfg.ATTACKER_NUM}  Balls: {len(self.balls)}",
                    f"Launched: {self.total_launch}  Hits: {self.total_hit}",
                    f"Hit Rate: {self.total_hit / (self.total_launch + 1e-6):.1%}",
                    f"Priority: {cfg.PRIORITY_MODE}",
                    f"Serial: {'ON' if cfg.USE_SERIAL else 'OFF'}"]
            for i, t in enumerate(info):
                self.screen.blit(self.font.render(t, True, (255, 255, 255)), (20, 20 + i * 20))
            self.param_panel.draw(self.screen)
            self.log_panel.draw(self.screen, self.logger)
            ps = self.logger.get_surf()
            if ps:
                self.screen.blit(ps, (cfg.WIDTH - ps.get_width() - 8, cfg.HEIGHT - ps.get_height() - 8))
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
        pygame.quit()


if __name__ == "__main__":
    GameController().run()
