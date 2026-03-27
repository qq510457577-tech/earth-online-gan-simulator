# -*- coding: utf-8 -*-
"""
==============================================================
  半球穹顶 GAN 对抗系统 ── 终极升级版
  功能: 3D伪立体穹顶 | 多目标优先级 | 串口云台 | 实时曲线 | GUI参数面板
  依赖: pip install pygame torch numpy matplotlib pyserial
==============================================================
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
from collections import deque

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ===================== 全局配置 =====================
class Config:
    WIDTH          = 1200
    HEIGHT         = 780
    FPS            = 60
    BG_COLOR       = (5, 8, 25)
    HEMI_RADIUS    = 260
    DOME_TILT      = 0.42
    DOME_SCALE     = 1.0
    DOME_LAYERS    = 5
    ATTACKER_NUM   = 30
    LAUNCH_PROB    = 0.25
    MAX_BALLS      = 80
    SPEED_BALL     = 2.8
    IR_LOCK_RADIUS = 20
    PRIORITY_MODE  = "threat"
    TURRET_SPEED   = 6.0
    GAN_LR_G       = 0.0008
    GAN_LR_D       = 0.0015
    GAN_TRAIN_FREQ = 4
    USE_SERIAL     = False
    SERIAL_PORT    = "COM3"
    BAUD           = 9600
    PLOT_FREQ      = 15
    LOG_MAXLEN     = 300

cfg = Config()

# ===================== GAN 缃戠粶 =====================
class Generator(nn.Module):
    """杈撳叆: noise(10)+鎴樺満鐘舵€?6)=16缁?-> 姣忔敾鏂筟dAng,dElev,launch]*N"""
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
    """杈撳叆: [x,y,z,speed,dist,hit_flag]=6缁?-> 绌块槻鎴愬姛姒傜巼"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1),  nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ===================== 涓插彛浜戝彴鎺у埗鍣?=====================
class SerialController:
    def __init__(self):
        self.ser = None
        self.connected = False
        self._q = queue.Queue(maxsize=16)
        if cfg.USE_SERIAL and HAS_SERIAL:
            try:
                self.ser = serial.Serial(cfg.SERIAL_PORT, cfg.BAUD, timeout=1)
                self.connected = True
                print(f"[涓插彛] 宸茶繛鎺?{cfg.SERIAL_PORT}@{cfg.BAUD}")
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception as e:
                print(f"[涓插彛] 鎵撳紑澶辫触: {e}")

    def _worker(self):
        while True:
            msg = self._q.get()
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(msg.encode())
                except:
                    pass

    def send(self, pan: float, tilt: float, laser: int = 1):
        """杈撳嚭鏍煎紡: PAN <pan> TILT <tilt> LASER <laser>\n"""
        if self.connected and not self._q.full():
            self._q.put_nowait(f"PAN {pan:.1f} TILT {tilt:.1f} LASER {laser}\n")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

# ===================== 鏃ュ織 & 鏇茬嚎鍥?=====================
class Logger:
    def __init__(self):
        n = cfg.LOG_MAXLEN
        self.hits     = deque(maxlen=n)
        self.launches = deque(maxlen=n)
        self.acc      = deque(maxlen=n)
        self.loss_g   = deque(maxlen=n)
        self.loss_d   = deque(maxlen=n)
        self.events   = []
        self._surf    = None
        self._lock    = threading.Lock()
        self._dirty   = False

    def push(self, total_hit, total_launch, lg=None, ld=None):
        self.hits.append(total_hit)
        self.launches.append(total_launch)
        self.acc.append(total_hit / (total_launch + 1e-6))
        if lg is not None: self.loss_g.append(lg)
        if ld is not None: self.loss_d.append(ld)
        self._dirty = True

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {msg}")
        if len(self.events) > 300:
            self.events = self.events[-300:]

    def render_plot_bg(self):
        """鍚庡彴绾跨▼: matplotlib -> pygame Surface"""
        if not HAS_MPL or not self._dirty:
            return
        self._dirty = False
        try:
            fig, axes = plt.subplots(3, 1, figsize=(4.2, 5.4), dpi=82,
                                     facecolor="#050819")
            for ax in axes:
                ax.set_facecolor("#0a1030")
                ax.tick_params(colors="#aaaacc", labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color("#334")

            if self.hits:
                axes[0].plot(list(self.hits),    color="#ff4444", lw=1.3, label="鍛戒腑")
                axes[0].plot(list(self.launches), color="#4488ff", lw=1.3, label="鍙戝皠")
                axes[0].legend(fontsize=7, facecolor="#0a1030", labelcolor="white")
                axes[0].set_title("鍛戒腑 vs 鍙戝皠", color="#aaccff", fontsize=8, pad=2)
            if self.acc:
                axes[1].plot(list(self.acc), color="#ffaa00", lw=1.5)
                axes[1].set_ylim(0, 1)
                axes[1].set_title("鍛戒腑鐜?, color="#aaccff", fontsize=8, pad=2)
            if self.loss_g:
                axes[2].plot(list(self.loss_g), color="#88ff44", lw=1, label="G鎹熷け")
                axes[2].plot(list(self.loss_d), color="#ff88ff", lw=1, label="D鎹熷け")
                axes[2].legend(fontsize=7, facecolor="#0a1030", labelcolor="white")
                axes[2].set_title("GAN鎹熷け", color="#aaccff", fontsize=8, pad=2)

            plt.tight_layout(pad=0.4)
            fig.canvas.draw()
            buf  = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            arr  = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            surf = pygame.surfarray.make_surface(arr[:, :, :3].swapaxes(0, 1))
            plt.close(fig)
            with self._lock:
                self._surf = surf
        except Exception as e:
            print(f"[Plot] {e}")

    def get_surf(self):
        with self._lock:
            return self._surf

# ===================== 3D 伪立体投影 =====================
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

# ===================== 穹顶渲染器 =====================
class DomeRenderer:
    def __init__(self):
        self._cache = []
        self._build()

    def _build(self):
        R = cfg.HEMI_RADIUS
        self._cache = []
        # 水平等高线
        for layer in range(1, cfg.DOME_LAYERS + 1):
            elev = math.pi / 2 * layer / cfg.DOME_LAYERS
            r_h  = R * math.sin(elev)
            z    = R * math.cos(elev)
            pts  = [(r_h * math.cos(a), r_h * math.sin(a), z)
                    for a in np.linspace(0, 2*math.pi, 72)]
            bright = int(40 + 100 * layer / cfg.DOME_LAYERS)
            self._cache.append(("ring", pts, (20, bright, 120), True))
        # 经线
        for a in np.linspace(0, 2*math.pi, 18, endpoint=False):
            pts = [sphere_to_xyz(R, a, e)
                   for e in np.linspace(0.02, math.pi/2, 20)]
            self._cache.append(("merid", pts, (15, 35, 85), False))
        # 地平线
        pts = [(R*math.cos(a), R*math.sin(a), 0)
               for a in np.linspace(0, 2*math.pi, 100)]
        self._cache.append(("ground", pts, (20, 80, 200), True))
        # 内层参考圈
        for frac in [0.33, 0.66]:
            r2 = R * frac
            pts = [(r2*math.cos(a), r2*math.sin(a), 0)
                   for a in np.linspace(0, 2*math.pi, 60)]
            self._cache.append(("inner", pts, (12, 30, 70), True))

    def draw(self, screen, cx, cy):
        for kind, pts, color, closed in self._cache:
            spx = [project(x, y, z, cx, cy) for x, y, z in pts]
            pygame.draw.lines(screen, color, closed, spx, 1)
        # 顶点光晕
        tx, ty = project(0, 0, cfg.HEMI_RADIUS, cx, cy)
        for r, al in [(16, 30), (10, 60), (5, 120)]:
            gs = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (80, 160, 255, al), (r, r), r)
            screen.blit(gs, (tx-r, ty-r))

    def rebuild(self):
        self._build()

# ===================== 目标优先级 =====================
def compute_threat(ball):
    dist  = ball["dist"]
    speed = ball["speed"]
    if cfg.PRIORITY_MODE == "nearest":  return dist
    if cfg.PRIORITY_MODE == "fastest":  return -speed
    return dist * 0.6 - speed * 0.4

# ===================== GUI 鍙傛暟璋冭妭闈㈡澘 =====================
class ParamPanel:
    PANEL_W = 228
    SH      = 22

    def __init__(self, x, y, w, h):
        self.rect      = pygame.Rect(x, y, w, h)
        self.font      = None
        self.font2     = None
        self._dragging = None
        self._sliders  = [
            # (label, attr, min, max, step, is_int, lr_scale)
            ("鏀绘柟鏁伴噺",     "ATTACKER_NUM",   10,  50,   1,    True,  1),
            ("鍙戝皠姒傜巼",     "LAUNCH_PROB",     0.05,0.6,  0.01, False, 1),
            ("鏈€澶у皬鐞?,     "MAX_BALLS",       20,  200,  5,    True,  1),
            ("灏忕悆閫熷害",     "SPEED_BALL",      0.5, 6.0,  0.1,  False, 1),
            ("鍛戒腑鍗婂緞",     "IR_LOCK_RADIUS",  6,   40,   1,    True,  1),
            ("绌归《鍘嬬缉",     "DOME_TILT",       0.1, 0.9,  0.02, False, 1),
            ("G瀛︿範鐜噚1000","GAN_LR_G",         0.1, 5.0,  0.05, False, 1000),
            ("D瀛︿範鐜噚1000","GAN_LR_D",         0.1, 5.0,  0.05, False, 1000),
            ("浜戝彴閫熷害",     "TURRET_SPEED",    1.0, 15.0, 0.5,  False, 1),
        ]
        self._priority_modes  = ["nearest", "fastest", "threat"]
        self._priority_labels = ["鏈€杩?,    "鏈€蹇?,    "濞佽儊"]

    def set_fonts(self, f, f2):
        self.font  = f
        self.font2 = f2

    def _slider_rect(self, idx):
        x = self.rect.x + 10
        y = self.rect.y + 110 + idx * 50
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
            # 浼樺厛绾ф寜閽?            for j, mode in enumerate(self._priority_modes):
                br = pygame.Rect(self.rect.x + 10 + j * 68, self.rect.y + 58, 62, 26)
                if br.collidepoint(e.pos):
                    cfg.PRIORITY_MODE = mode
                    return True
            # 涓插彛寮€鍏?            sr = pygame.Rect(self.rect.x + 10, self.rect.y + self.rect.height - 44, 120, 28)
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
        val   = mn + ratio * (mx - mn)
        val   = round(val / step) * step
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

        # 鏍囬
        t = self.font.render("  鍙傛暟璋冭妭闈㈡澘", True, (140, 210, 255))
        screen.blit(t, (self.rect.x + 8, self.rect.y + 10))

        # 浼樺厛绾?        lbl = self.font2.render("鐩爣浼樺厛绾?", True, (180, 180, 230))
        screen.blit(lbl, (self.rect.x + 10, self.rect.y + 38))
        for j, (mode, name) in enumerate(zip(self._priority_modes, self._priority_labels)):
            br = pygame.Rect(self.rect.x + 10 + j*68, self.rect.y + 58, 62, 26)
            active = (cfg.PRIORITY_MODE == mode)
            pygame.draw.rect(screen, (35, 90, 200) if active else (18, 38, 80), br, border_radius=4)
            pygame.draw.rect(screen, (60, 130, 255) if active else (40, 60, 130), br, 1, border_radius=4)
            ct = self.font2.render(name, True, (255,255,255) if active else (150,160,200))
            screen.blit(ct, (br.x + (br.width - ct.get_width())//2,
                              br.y + (br.height - ct.get_height())//2))

        # 婊戝潡
        for i, (label, attr, mn, mx, step, is_int, scale) in enumerate(self._sliders):
            y  = self.rect.y + 103 + i * 50
            lt = self.font2.render(label, True, (160, 195, 245))
            screen.blit(lt, (self.rect.x + 10, y))
            r  = self._slider_rect(i)
            pygame.draw.rect(screen, (18, 38, 100), r, border_radius=4)
            ratio = self._ratio(i)
            fr = pygame.Rect(r.x, r.y, max(4, int(r.width * ratio)), r.height)
            pygame.draw.rect(screen, (35, 100, 230), fr, border_radius=4)
            hx = r.x + int(r.width * ratio)
            pygame.draw.circle(screen, (120, 185, 255), (hx, r.y + r.height//2), 8)
            # 鏁板€?            raw = getattr(cfg, attr)
            vstr = str(int(raw)) if is_int else f"{raw:.3f}" if scale == 1000 else f"{raw:.2f}"
            vt = self.font2.render(vstr, True, (255, 225, 80))
            screen.blit(vt, (r.right - vt.get_width() - 2, y))

        # 涓插彛鎸夐挳
        sr = pygame.Rect(self.rect.x + 10, self.rect.y + self.rect.height - 44, 130, 28)
        pygame.draw.rect(screen, (15, 100, 30) if cfg.USE_SERIAL else (90, 15, 15), sr, border_radius=4)
        pygame.draw.rect(screen, (40, 180, 60) if cfg.USE_SERIAL else (160, 40, 40), sr, 1, border_radius=4)
        st = self.font2.render(f"涓插彛: {'寮€鍚?(COM'+cfg.SERIAL_PORT[-1]+')' if cfg.USE_SERIAL else '鍏抽棴'}", True, (255,255,255))
        screen.blit(st, (sr.x + (sr.width - st.get_width())//2,
                          sr.y + (sr.height - st.get_height())//2))

# ===================== 鏃ュ織鏂囧瓧闈㈡澘 =====================
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
        t = self.font.render(" 瀵规姉鏃ュ織", True, (140, 210, 255))
        screen.blit(t, (self.rect.x + 6, self.rect.y + 5))
        for i, line in enumerate(logger.events[-self.MAX_LINES:]):
            if "鍛戒腑" in line:
                col = (100, 255, 130)
            elif "鍙戝皠" in line:
                col = (255, 180, 80)
            else:
                col = (160, 160, 215)
            lt = self.font.render(line, True, col)
            screen.blit(lt, (self.rect.x + 6, self.rect.y + 24 + i * 16))

# ===================== 涓绘帶鍒跺櫒 =====================
class GameController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
        pygame.display.set_caption("鍗婄悆绌归《 GAN 瀵规姉绯荤粺 鈥?缁堟瀬鍗囩骇鐗?)
        self.clock  = pygame.time.Clock()
        try:
            self.font   = pygame.font.SysFont("simhei", 15)
            self.font2  = pygame.font.SysFont("simhei", 13)
            self.font_b = pygame.font.SysFont("simhei", 17, bold=True)
            self.font_s = pygame.font.SysFont("simhei", 12)
        except:
            self.font   = pygame.font.SysFont(None, 16)
            self.font2  = pygame.font.SysFont(None, 14)
            self.font_b = pygame.font.SysFont(None, 18)
            self.font_s = pygame.font.SysFont(None, 13)

        self.dome   = DomeRenderer()
        self.serial = SerialController()
        self.logger = Logger()

        PW = ParamPanel.PANEL_W
        px = cfg.WIDTH - PW - 4
        self.param_panel = ParamPanel(px, 4, PW, 570)
        self.param_panel.set_fonts(self.font_b, self.font2)
        self.log_panel = LogPanel(px, 578, PW, cfg.HEIGHT - 582)
        self.log_panel.set_font(self.font_s)

        self.G    = Generator(cfg.ATTACKER_NUM)
        self.D    = Discriminator()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.GAN_LR_D, betas=(0.5, 0.999))
        self.bce  = nn.BCELoss()

        self.attackers    = []
        self.balls        = []
        self.total_hit    = 0
        self.total_launch = 0
        self.frame        = 0
        self.loss_g_val   = 0.0
        self.loss_d_val   = 0.0
        self.turret_pan   = 0.0
        self.turret_tilt  = 45.0
        self._prev_n      = cfg.ATTACKER_NUM
        self._plot_thread = None

        self._init_attackers()
        self.logger.log("绯荤粺鍚姩锛屾敾鏂归泦缇ゅ氨浣?)

    def _init_attackers(self):
        self.attackers = []
        for i in range(cfg.ATTACKER_NUM):
            ang  = np.random.uniform(0, 2*math.pi)
            elev = np.random.uniform(0.1, math.pi/2)
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x":x,"y":y,"z":z,"ang":ang,"elev":elev})

    def _train_gan(self):
        if len(self.balls) < 4:
            return
        batch, labels = [], []
        for b in self.balls:
            d, sp = b["dist"], b["speed"]
            batch.append([b["x"], b["y"], b["z"], sp, d,
                          1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
            labels.append([1.0 if d < cfg.IR_LOCK_RADIUS else 0.0])
        x = torch.tensor(batch,  dtype=torch.float32)
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
            ang  = np.random.uniform(0, 2*math.pi)
            elev = np.random.uniform(0.1, math.pi/2)
            x, y, z = sphere_to_xyz(cfg.HEMI_RADIUS, ang, elev)
            self.attackers.append({"x":x,"y":y,"z":z,"ang":ang,"elev":elev})
        self.attackers = self.attackers[:n]
        if n != self._prev_n:
            self.G    = Generator(n)
            self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.GAN_LR_G, betas=(0.5, 0.999))
            self._prev_n = n

        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        noise    = torch.randn(1, 10)
        state    = torch.tensor([[
            hit_rate,
            len(self.balls) / (cfg.MAX_BALLS + 1e-6),
            (self.frame * 0.001) % 1.0,
            self.loss_g_val,
            self.loss_d_val,
            np.random.rand()
        ]], dtype=torch.float32)
        with torch.no_grad():
            g_out = self.G(torch.cat([noise, state], dim=1)).squeeze(0).numpy()

        launched_now = 0
        for i, a in enumerate(self.attackers):
            if i >= g_out.shape[0]:
                break
            a["ang"]  = (a["ang"]  + float(g_out[i,0]) * 0.06) % (2*math.pi)
            a["elev"] = max(0.05, min(math.pi/2, a["elev"] + float(g_out[i,1]) * 0.04))
            a["x"], a["y"], a["z"] = sphere_to_xyz(cfg.HEMI_RADIUS, a["ang"], a["elev"])

            if (len(self.balls) < cfg.MAX_BALLS
                    and float(g_out[i,2]) > cfg.LAUNCH_PROB
                    and np.random.rand() < 0.35):
                speed = cfg.SPEED_BALL * (0.8 + np.random.rand() * 0.4)
                R = cfg.HEMI_RADIUS
                vx = -a["x"] / R * speed * 10
                vy = -a["y"] / R * speed * 10
                vz = -a["z"] / R * speed * 10
                self.balls.append({
                    "x":a["x"],"y":a["y"],"z":a["z"],
                    "vx":vx,"vy":vy,"vz":vz,
                    "dist": R, "speed": speed * 10
                })
                self.total_launch += 1
                launched_now += 1
        if launched_now > 0 and self.frame % 12 == 0:
            self.logger.log(f"鍙戝皠 {launched_now}鏋氾紝鎬?{self.total_launch}")

    def _update_balls(self):
        new_balls = []
        hits_now  = 0
        for b in self.balls:
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["z"] += b["vz"]
            d  = math.sqrt(b["x"]**2 + b["y"]**2 + b["z"]**2)
            sp = math.sqrt(b["vx"]**2 + b["vy"]**2 + b["vz"]**2)
            b["dist"]  = d
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
            self.logger.log(f"鍛戒腑{hits_now}! 鎬粄self.total_hit} 鐜噞rate:.1%}")

    def _update_turret(self):
        if not self.balls:
            return None
        self.balls.sort(key=compute_threat)
        t    = self.balls[0]
        pan  = math.degrees(math.atan2(t["y"], t["x"]))
        tilt = math.degrees(math.atan2(t["z"],
               math.sqrt(t["x"]**2 + t["y"]**2) + 1e-6))
        def lerp_ang(cur, tgt, spd):
            d = (tgt - cur + 180) % 360 - 180
            return cur + math.copysign(min(abs(d), spd), d)
        self.turret_pan  = lerp_ang(self.turret_pan,  pan,  cfg.TURRET_SPEED)
        self.turret_tilt = lerp_ang(self.turret_tilt, tilt, cfg.TURRET_SPEED)
        self.serial.send(self.turret_pan, self.turret_tilt, laser=1)
        return t

    # -------- 缁樺埗涓荤敾闈?--------
    def _draw(self, target):
        self.screen.fill(cfg.BG_COLOR)
        # 缁樺浘鍖轰腑蹇冿紙宸︿晶涓诲尯鍩燂級
        area_w = cfg.WIDTH - ParamPanel.PANEL_W - 10
        cx     = area_w // 2
        cy     = cfg.HEIGHT // 2 + 15

        # 绌归《
        self.dome.draw(self.screen, cx, cy)

        # 鏀绘柟鑺傜偣
        for a in self.attackers:
            sx, sy = project(a["x"], a["y"], a["z"], cx, cy)
            gs = pygame.Surface((22, 22), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 60, 60, 55), (11, 11), 11)
            self.screen.blit(gs, (sx-11, sy-11))
            pygame.draw.circle(self.screen, (255, 80, 55), (sx, sy), 5)
            pygame.draw.circle(self.screen, (255, 210, 190), (sx, sy), 2)

        # 椋炶灏忕悆
        for b in self.balls:
            sx, sy = project(b["x"], b["y"], b["z"], cx, cy)
            frac = max(0, 1.0 - b["dist"] / cfg.HEMI_RADIUS)
            r    = max(2, int(2 + 4 * frac))
            al   = int(180 * frac + 60)
            gs   = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
            pygame.draw.circle(gs, (80, 160, 255, al//3), (r*2, r*2), r*2)
            self.screen.blit(gs, (sx-r*2, sy-r*2))
            pygame.draw.circle(self.screen, (100, 185, 255), (sx, sy), r)

        # 瀹堟柟锛堜腑蹇冨師鐐癸級
        scx, scy = project(0, 0, 0, cx, cy)
        for r, al in [(20,18),(13,40),(8,80),(5,180)]:
            gs = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 160, 0, al), (r,r), r)
            self.screen.blit(gs, (scx-r, scy-r))
        pygame.draw.circle(self.screen, (255, 185, 55), (scx, scy), 7)

        # 婵€鍏夐攣瀹氱嚎 + 鐩爣鍦?
        if target:
            tx, ty = project(target["x"], target["y"], target["z"], cx, cy)
            pygame.draw.line(self.screen, (255, 255,   0), (scx, scy), (tx, ty), 3)
            pygame.draw.line(self.screen, (255, 200,  50), (scx, scy), (tx, ty), 1)
            pygame.draw.circle(self.screen, (255,  80,  80), (tx, ty), 14, 2)
            pygame.draw.circle(self.screen, (255, 220,  80), (tx, ty),  8, 1)
            # 娆¤鐩爣鏍囪锛堟渶澶?涓級
            for k, b in enumerate(self.balls[1:4], 2):
                bx, by = project(b["x"], b["y"], b["z"], cx, cy)
                pygame.draw.circle(self.screen, (200, 80, 220), (bx, by), 10, 1)
                kt = self.font_s.render(str(k), True, (200, 140, 220))
                self.screen.blit(kt, (bx+8, by-8))

        # 宸︿笂淇℃伅闈㈡澘
        hit_rate = self.total_hit / (self.total_launch + 1e-6)
        infos = [
            ("  鍗婄悆绌归《 GAN 瀵规姉绯荤粺", (140, 210, 255), self.font_b),
            (f"  鏀绘柟: {len(self.attackers)}  |  灏忕悆: {len(self.balls)}/{cfg.MAX_BALLS}", (200,210,255), self.font),
            (f"  鍙戝皠: {self.total_launch}  |  鍛戒腑: {self.total_hit}", (200,210,255), self.font),
            (f"  鍛戒腑鐜? {hit_rate:.1%}  |  甯? {self.frame}", (255,225,90), self.font),
            (f"  浼樺厛绾? {cfg.PRIORITY_MODE}  |  浜戝彴: {self.turret_pan:.1f}/{self.turret_tilt:.1f}", (170,255,170), self.font),
            (f"  G鎹熷け: {self.loss_g_val:.3f}  D鎹熷け: {self.loss_d_val:.3f}", (170,255,170), self.font),
            (f"  涓插彛: {'宸茶繛鎺?'+cfg.SERIAL_PORT if self.serial.connected else '妯℃嫙妯″紡'}", (160,160,210), self.font),
        ]
        bg = pygame.Surface((295, len(infos)*21 + 12), pygame.SRCALPHA)
        bg.fill((0, 10, 32, 185))
        self.screen.blit(bg, (4, 4))
        for i, (text, col, fnt) in enumerate(infos):
            self.screen.blit(fnt.render(text, True, col), (6, 6 + i*21))

        # 鏇茬嚎鍥撅紙鍙充笅鏂瑰祵鍏ワ級
        surf = self.logger.get_surf()
        if surf:
            sx2 = cfg.WIDTH - ParamPanel.PANEL_W - surf.get_width() - 8
            sy2 = cfg.HEIGHT - surf.get_height() - 6
            self.screen.blit(surf, (sx2, sy2))

        # 鍙充晶鍙傛暟闈㈡澘
        self.param_panel.draw(self.screen)
        # 鏃ュ織闈㈡澘
        self.log_panel.draw(self.screen, self.logger)

        # 搴曢儴鎻愮ず
        hint = self.font_s.render("榧犳爣鎷栧姩婊戝潡鍙疄鏃惰皟鍙? |  ESC閫€鍑?, True, (80, 80, 130))
        self.screen.blit(hint, (6, cfg.HEIGHT - 18))

        pygame.display.flip()

    # -------- 涓诲惊鐜?--------
    def run(self):
        while True:
            self.frame += 1

            # 浜嬩欢
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self._cleanup()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self._cleanup()
                    return
                self.param_panel.handle_event(e)

            # 鏇存柊閫昏緫
            self._update_attackers()
            self._update_balls()
            target = self._update_turret()

            # GAN 璁粌
            if self.frame % cfg.GAN_TRAIN_FREQ == 0:
                self._train_gan()

            # 鏃ュ織鎺ㄩ€?
            self.logger.push(self.total_hit, self.total_launch,
                             self.loss_g_val, self.loss_d_val)

            # 绌归《閲嶅缓锛堝綋鍗婂緞鍙傛暟鍙樺寲鏃讹級
            if self.frame % 120 == 0:
                self.dome.rebuild()

            # 鍚庡彴鏇茬嚎娓叉煋
            if self.frame % cfg.PLOT_FREQ == 0:
                if self._plot_thread is None or not self._plot_thread.is_alive():
                    self._plot_thread = threading.Thread(
                        target=self.logger.render_plot_bg, daemon=True)
                    self._plot_thread.start()

            self._draw(target)
            self.clock.tick(cfg.FPS)

    def _cleanup(self):
        self.serial.close()
        pygame.quit()
        sys.exit()

# ===================== 鍏ュ彛 =====================
if __name__ == "__main__":
    ctrl = GameController()
    ctrl.run()
