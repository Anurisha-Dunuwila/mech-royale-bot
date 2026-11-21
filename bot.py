import array
import heapq
import json
import math
import os
import random
import sys
import time
from collections import deque

import mechfight as mf

TAU = 2 * math.pi
G = 2.0  # grid cell size in meters
KITE = False  # tried it, practice score dropped 3.85 -> 3.38
# careful fight-or-flight on top of the 3.85 bot, False gives the 3.85 bot back
TACTICS = False  # scored 3.46
DANGER_SCALE = 0.75  # we were overrating enemy damage and running too much
# positioning, each one on its own so we can tell what helps
COVER_PLUS = False    # if no full cover, take the spot hidden from the most enemies
SAFE_RETREAT = False  # cover we can reach without walking past enemies
PREDICT = False       # also hide from where enemies will be in 2 s
DYN_STRAFE = True     # less predictable strafing
SURROUND = False      # move toward the gap when enemies are on several sides (like KITE)
HIGH_GROUND = False   # prefer higher ground when picking a zone spot
BOTTLENECK = False    # don't park in narrow spots
ZONE_SPOT = False     # wait in the next zone next to some walls, not in the open
EARLY_ROTATE = False  # leave for the next zone sooner
HIDE_MODE = False     # no enemy in sight: go next to a building and wait there
HP_BANDS = True       # >70% fight, 40-70% back off only vs stronger or outnumbered, <40% escape (3.96)
ZONE_FIRST = False    # when hurt and the zone is closing, go to the zone instead of cover
BOOST_BANDS = False   # <60% back off with boost while shooting, <40% boost out, no boost when alone
ATTACK_UPGRADES = False  # take damage and fire rate upgrades before armor and regen (3.83)
# farm crates and drones for cubes: each cube is +4% hp and damage, upgrades at 1/3/6/10/15/21
FARM = False  # scored 3.77
# rough dps and range per weapon
WDPS = {"gun": (187, 200), "laser": (210, 280), "lightning": (150, 45), "mortar": (60, 220)}

WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
# kills win points, so finish fights faster
UPGRADE_SCORE_ATK = {
    "gun_damage": 10, "gun_rate": 10, "laser_damage": 9, "laser_charge": 9, "armor": 8,
    "plating": 7, "nanites": 6, "lightning_efficiency": 6, "lightning_range": 5, "gun_mag": 6,
    "gun_velocity": 6, "gyros": 5, "servos": 4, "sensors": 4, "reactor": 3, "lightning_chain": 4,
    "capacitor": 3, "jets": 2, "mortar_reload": 3, "mortar_splash": 3, "mortar_velocity": 2,
    "laser_pierce": 2,
}
UPGRADE_SCORE = {
    # we hide and regen a lot, so faster regen and more hp come first
    "nanites": 10, "plating": 10, "armor": 9, "gun_rate": 7, "gun_damage": 7,
    "laser_damage": 6, "laser_charge": 6, "lightning_efficiency": 5, "lightning_range": 5,
    "gyros": 5, "servos": 5, "gun_mag": 5, "gun_velocity": 4, "sensors": 4, "reactor": 3,
    "lightning_chain": 3, "capacitor": 3, "jets": 3, "mortar_reload": 3, "mortar_splash": 3,
    "mortar_velocity": 2, "laser_pierce": 1,
}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    f = 0.0 if L < 1e-9 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(ax + dx * f - px, ay + dy * f - py)


def norm2(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


class Map:
    def __init__(self, mapdir):
        w = json.load(open(os.path.join(mapdir, "v2", "world.json")))
        b = w["bounds"]
        self.bx0, self.bx1, self.by0, self.by1 = b["minX"], b["maxX"], b["minY"], b["maxY"]
        tr = w["terrain"]
        self.tnx, self.tny = tr["nx"], tr["ny"]
        self.tx0, self.ty0, self.tstep = tr["minX"], tr["minY"], tr["step"]
        self.terr = array.array("h")
        with open(os.path.join(mapdir, "v2", tr.get("file", "terrain.i16")), "rb") as f:
            self.terr.frombytes(f.read())
        if sys.byteorder != "little":
            self.terr.byteswap()

        self.gx0, self.gy0 = self.bx0, self.by0
        self.nx = int(math.ceil((self.bx1 - self.bx0) / G)) + 1
        self.ny = int(math.ceil((self.by1 - self.by0) / G)) + 1
        n = self.nx * self.ny
        self.top = array.array("f", [-1e9]) * n
        self.base = array.array("f", [1e9]) * n
        for p in w["prisms"]:
            cells = self._fill(p["ring"])
            for h in p.get("holes") or []:
                cells -= self._fill(h)
            t, bs = p["top"], p["base"]
            for c in cells:
                if t > self.top[c]:
                    self.top[c] = t
                if bs < self.base[c]:
                    self.base[c] = bs

        # coarse 4m grid for pathfinding
        self.nx4, self.ny4 = self.nx // 2, self.ny // 2
        self.blk = bytearray(self.nx4 * self.ny4)
        for j in range(self.ny4):
            for i in range(self.nx4):
                cnt = 0
                for dj in (0, 1):
                    for di in (0, 1):
                        c = (2 * j + dj) * self.nx + 2 * i + di
                        if self.top[c] > -1e8:
                            gz = self.ground(self.gx0 + (2 * i + di + 0.5) * G, self.gy0 + (2 * j + dj + 0.5) * G)
                            if self.top[c] > gz + 1.2 and self.base[c] < gz + 9:
                                cnt += 1
                if cnt >= 2:
                    self.blk[j * self.nx4 + i] = 1
        # small extra cost next to walls so paths don't hug them
        self.pen = bytearray(self.nx4 * self.ny4)
        for j in range(1, self.ny4 - 1):
            for i in range(1, self.nx4 - 1):
                k = j * self.nx4 + i
                if not self.blk[k]:
                    self.pen[k] = self.blk[k - 1] + self.blk[k + 1] + self.blk[k - self.nx4] + self.blk[k + self.nx4]

    def _fill(self, ring):
        xs_ = ring[0::2]
        ys_ = ring[1::2]
        m = len(xs_)
        out = set()
        if m < 3:
            return out
        j0 = max(0, int(math.ceil((min(ys_) - self.gy0) / G - 0.5)))
        j1 = min(self.ny - 1, int(math.floor((max(ys_) - self.gy0) / G - 0.5)))
        edges = [(xs_[k], ys_[k], xs_[(k + 1) % m], ys_[(k + 1) % m]) for k in range(m)]
        for j in range(j0, j1 + 1):
            yc = self.gy0 + (j + 0.5) * G
            xs = []
            for x1, y1, x2, y2 in edges:
                if (y1 <= yc < y2) or (y2 <= yc < y1):
                    xs.append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[k] - self.gx0) / G - 0.5)))
                i1 = min(self.nx - 1, int(math.floor((xs[k + 1] - self.gx0) / G - 0.5)))
                row = j * self.nx
                for i in range(i0, i1 + 1):
                    out.add(row + i)
        return out

    def ground(self, x, y):
        i = int(round((x - self.tx0) / self.tstep))
        j = int(round((y - self.ty0) / self.tstep))
        if 0 <= i < self.tnx and 0 <= j < self.tny:
            return self.terr[j * self.tnx + i] * 0.01
        return 0.0

    def cell(self, x, y):
        i = int((x - self.gx0) / G)
        j = int((y - self.gy0) / G)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return -1

    def los(self, a, b):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(d / 1.5)
        for s in range(1, n):
            f = s / n
            if f * d < 3.0 or (1 - f) * d < 3.0:
                continue
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            c = self.cell(x, y)
            if c < 0:
                return False
            if self.base[c] <= z <= self.top[c]:
                return False
            if z < self.ground(x, y) - 0.3:
                return False
        return True

    def wall_at(self, x, y, fz):
        if not (self.bx0 + 3 < x < self.bx1 - 3 and self.by0 + 3 < y < self.by1 - 3):
            return True
        c = self.cell(x, y)
        if c < 0:
            return True
        return self.top[c] > fz + 1.2 and self.base[c] < fz + 9

    def path(self, sx, sy, gx, gy, budget=0.008):
        C = 2 * G
        nx4, ny4 = self.nx4, self.ny4

        def idx(x, y):
            i = min(nx4 - 1, max(0, int((x - self.gx0) / C)))
            j = min(ny4 - 1, max(0, int((y - self.gy0) / C)))
            return j * nx4 + i

        s, g = idx(sx, sy), idx(gx, gy)
        if self.blk[s]:
            return None
        if self.blk[g]:
            g = self._nearest_free(g)
            if g is None:
                return None
        gi, gj = g % nx4, g // nx4

        def h(k):
            dx, dy = abs(k % nx4 - gi), abs(k // nx4 - gj)
            return (dx + dy) + (1.4142 - 2) * min(dx, dy)

        t0 = time.perf_counter()
        openq = [(h(s), 0.0, s)]
        came = {s: -1}
        cost = {s: 0.0}
        best, besth = s, h(s)
        nb = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        it = 0
        while openq:
            f, gc, k = heapq.heappop(openq)
            if k == g:
                best = g
                break
            if gc > cost.get(k, 1e18):
                continue
            if f - gc < besth:
                best, besth = k, f - gc
            it += 1
            if it % 256 == 0 and time.perf_counter() - t0 > budget:
                break  # out of time, go toward the closest node so far
            ki, kj = k % nx4, k // nx4
            for di, dj, w in nb:
                ii, jj = ki + di, kj + dj
                if not (0 < ii < nx4 - 1 and 0 < jj < ny4 - 1):
                    continue
                q = jj * nx4 + ii
                if self.blk[q]:
                    continue
                if di and dj and (self.blk[kj * nx4 + ii] or self.blk[jj * nx4 + ki]):
                    continue
                nc = gc + w + 0.6 * self.pen[q]
                if nc < cost.get(q, 1e18):
                    cost[q] = nc
                    came[q] = k
                    heapq.heappush(openq, (nc + h(q), nc, q))
        pts = []
        k = best
        while k != -1:
            pts.append((self.gx0 + (k % nx4 + 0.5) * C, self.gy0 + (k // nx4 + 0.5) * C))
            k = came[k]
        pts.reverse()
        return pts

    def _nearest_free(self, g):
        nx4, ny4 = self.nx4, self.ny4
        gi, gj = g % nx4, g // nx4
        for r in range(1, 12):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    ii, jj = gi + di, gj + dj
                    if 0 <= ii < nx4 and 0 <= jj < ny4 and not self.blk[jj * nx4 + ii]:
                        return jj * nx4 + ii
        return None


class Bot:
    name = "survivor"

    def __init__(self, hello):
        random.seed(hello.get("seed", 0))
        self.cfg = hello["config"]
        self.reach = self.cfg["loot"]["pickup_radius"]
        self.regen_delay = self.cfg["mech"].get("regen_delay", 5)
        try:
            self.map = Map(hello["map"]["dir"])
        except Exception as e:
            mf.log("map load failed:", e)
            self.map = None
        self.hist = deque()
        self.stuck_until = -1.0
        self.stuck_flip = 1
        self.strafe = 1
        self.strafe_until = 0.0
        self.threat = None  # (bearing, time) of the last thing that hurt us
        self.path = []
        self.path_goal = None
        self.path_t = -99.0
        self.mortar_until = -1.0
        self.last_hurt = -99.0
        self.last_dealt = -99.0
        self.recover = False
        self.cover = None
        self.cover_t = -99.0
        # things we've seen but can't see right now: id -> (x, y, z, kind)
        self.known_pickups = {}
        self.known_crates = {}
        self.roam = None
        self.attacker = None
        self.dmg_log = deque()
        self.tgt_id = None
        self.panic_until = -99.0
        self.cover_len = None
        self.strafe_amt = 0.9
        self.zspot = None
        self.hspot = None

    def zone_at(self, z, t):
        s, e = z["shrink_start"], z["shrink_end"]
        if t <= s:
            return z["center"], z["radius"]
        f = 1.0 if t >= e or e <= s else (t - s) / (e - s)
        c = [z["center"][0] + (z["next_center"][0] - z["center"][0]) * f,
             z["center"][1] + (z["next_center"][1] - z["center"][1]) * f]
        return c, z["radius"] + (z["next_radius"] - z["radius"]) * f

    def los(self, a, b):
        return True if self.map is None else self.map.los(a, b)

    def free(self, x, y, fz):
        return True if self.map is None else not self.map.wall_at(x, y, fz)

    def steer(self, me, dx, dy):
        if self.map is None or (dx == 0 and dy == 0):
            return dx, dy
        x, y, fz = me["pos"]
        base = math.atan2(dy, dx)
        mag = min(1.0, math.hypot(dx, dy))
        for off in (0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.1, -2.1):
            a = base + off * self.stuck_flip
            ca, sa = math.cos(a), math.sin(a)
            if all(self.free(x + ca * r, y + sa * r, fz) for r in (3.0, 5.5, 8.0)):
                return ca * mag, sa * mag
        return dx, dy

    def nav_dir(self, me, t, gx, gy):
        x, y, z = me["pos"]
        if self.map is None:
            return norm2(gx - x, gy - y)
        if math.hypot(gx - x, gy - y) < 12 or self.map.los((x, y, z + 3), (gx, gy, self.map.ground(gx, gy) + 3)):
            self.path = []
            return norm2(gx - x, gy - y)
        if (not self.path or self.path_goal is None or t - self.path_t > 4.0
                or math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 15):
            p = self.map.path(x, y, gx, gy)
            self.path = p[1:] if p else []
            self.path_goal = (gx, gy)
            self.path_t = t
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 5:
            self.path.pop(0)
        # cut corners: head for the furthest waypoint we can walk straight to
        tgt = None
        for wp in self.path[:6]:
            if self.map.los((x, y, z + 3), (wp[0], wp[1], self.map.ground(wp[0], wp[1]) + 3)):
                tgt = wp
            else:
                break
        if tgt is None:
            tgt = self.path[0] if self.path else (gx, gy)
        return norm2(tgt[0] - x, tgt[1] - y)

    def find_cover(self, me, threats, in_safe):
        """Closest walkable spot nearby that none of the threats can see."""
        if self.map is None or not threats:
            return None
        x, y, z = me["pos"]
        near = sorted(threats, key=lambda m: m["_d"])[:4] if PREDICT else threats
        eyes = [(m["pos"][0], m["pos"][1], m["pos"][2] + 8) for m in near]
        if PREDICT:
            for m in near:
                v = m.get("vel") or [0, 0, 0]
                eyes.append((m["pos"][0] + v[0] * 2, m["pos"][1] + v[1] * 2, m["pos"][2] + 8))
        best, bs = None, 1e9
        part, ps = None, (99, 1e9)
        t0 = time.perf_counter()
        for r in (10, 18, 28, 40, 55, 70):
            for k in range(16):
                a = k * TAU / 16
                px, py = x + math.cos(a) * r, y + math.sin(a) * r
                gz = self.map.ground(px, py)
                if not self.free(px, py, gz) or not in_safe(px, py):
                    continue
                # mostly walking away from the enemy is better than past them
                toward = 0.0
                for e in eyes:
                    ex, ey = norm2(e[0] - x, e[1] - y)
                    toward = max(toward, math.cos(a) * ex + math.sin(a) * ey)
                score = r + 25 * max(0.0, toward)
                if SAFE_RETREAT:
                    # route passes close to someone
                    for m in near:
                        if seg_dist(m["pos"][0], m["pos"][1], x, y, px, py) < 25:
                            score += 40
                if score >= bs and not COVER_PLUS:
                    continue
                if not self.map.los((x, y, z + 3), (px, py, gz + 3)):
                    continue
                seen = sum(1 for e in eyes if self.map.los(e, (px, py, gz + 7)))
                if seen == 0 and score < bs:
                    best, bs = (px, py), score
                elif COVER_PLUS and 0 < seen < len(eyes) and (seen, score) < ps:
                    part, ps = (px, py), (seen, score)
            if best is not None or time.perf_counter() - t0 > 0.008:
                break
        return best if best is not None else part

    @staticmethod
    def lead(src, tpos, tvel, speed):
        p = tpos
        for _ in range(3):
            tt = math.dist(src, p) / speed
            p = (tpos[0] + tvel[0] * tt, tpos[1] + tvel[1] * tt, tpos[2] + tvel[2] * tt * 0.5)
        return p

    @staticmethod
    def aim_to(src, p):
        return [math.atan2(p[1] - src[1], p[0] - src[0]),
                math.atan2(p[2] - src[2], math.hypot(p[0] - src[0], p[1] - src[1]))]

    @staticmethod
    def aim_err(cur, want):
        return math.hypot(wrap(cur[0] - want[0]), cur[1] - want[1])

    @staticmethod
    def center(m):
        if m.get("kind") == "npc":
            return (m["pos"][0], m["pos"][1], m["pos"][2] + 1.75)
        s = 1 + 0.05 * (m.get("cubes") or 0)
        return (m["pos"][0], m["pos"][1], m["pos"][2] + 4.5 * s)

    def dps(self, kinds, d, cubes, los=True):
        tot = 0.0
        for k in kinds:
            if k not in WDPS or d >= WDPS[k][1]:
                continue
            if k == "gun":
                acc = max(0.25, min(0.9, 1.0 - d / 250))
            elif k == "laser":
                acc = 0.7
            elif k == "mortar":
                acc = 0.3 if los else 0.15
            else:
                acc = 1.0
            if los or k == "mortar":
                tot += WDPS[k][0] * acc
        return tot * (1 + 0.04 * (cubes or 0))

    def threat_dps(self, m, x, y):
        a = abs(wrap(m["aim"][0] - math.atan2(y - m["pos"][1], x - m["pos"][0])))
        facing = 1.0 if a < 0.35 else (0.5 if a < 1.0 else 0.2)
        return self.dps(m.get("weapons") or [], m["_d"], m.get("cubes"), m["_los"]) * facing * DANGER_SCALE

    def path_len(self, me, gx, gy):
        x, y = me["pos"][0], me["pos"][1]
        pts = self.map.path(x, y, gx, gy, budget=0.004) if self.map else None
        if not pts:
            return math.hypot(gx - x, gy - y) * 1.5
        n, px, py = 0.0, x, y
        for qx, qy in pts:
            n += math.hypot(qx - px, qy - py)
            px, py = qx, qy
        return n + math.hypot(gx - px, gy - py) * 1.3

    def hide_spot(self, x, y, in_safe):
        # closest walkable spot inside the safe area with buildings around it, but not boxed in
        if self.map is None:
            return None
        best, bs = None, 1e9
        for r in (0, 10, 20, 35, 50, 70):
            for k in range(1 if r == 0 else 16):
                a = k * TAU / 16
                px, py = x + math.cos(a) * r, y + math.sin(a) * r
                if not in_safe(px, py):
                    continue
                w = self.walls_near(px, py)
                if w is None or w < 5 or w > 14:
                    continue
                sc = r - 3 * min(w, 12)
                if sc < bs:
                    best, bs = (px, py), sc
        return best

    def walls_near(self, px, py):
        # blocked 4 m cells in the 5x5 block around a point, None if the point itself is blocked
        M = self.map
        C = 2 * G
        i, j = int((px - M.gx0) / C), int((py - M.gy0) / C)
        if not (2 < i < M.nx4 - 3 and 2 < j < M.ny4 - 3) or M.blk[j * M.nx4 + i]:
            return None
        return sum(M.blk[(j + dj) * M.nx4 + i + di] for dj in range(-2, 3) for di in range(-2, 3))

    def open_spot(self, px, py, in_safe):
        # nearest spot around (px, py) that isn't a narrow passage
        w = self.walls_near(px, py)
        if w is not None and w < 10:
            return px, py
        for r in (8, 16, 24):
            for k in range(12):
                a = k * TAU / 12
                qx, qy = px + math.cos(a) * r, py + math.sin(a) * r
                w = self.walls_near(qx, qy)
                if w is not None and w < 10 and in_safe(qx, qy):
                    return qx, qy
        return px, py

    def zone_spot(self, nc, rmax, x, y):
        # a walkable spot in the next zone with some walls around it but not boxed in
        if self.map is None or rmax < 10:
            return None
        M = self.map
        C = 2 * G
        best, bs = None, 1e9
        for ring in (0.25, 0.45, 0.65, 0.85):
            for k in range(12):
                a = k * TAU / 12 + ring
                px, py = nc[0] + math.cos(a) * rmax * ring, nc[1] + math.sin(a) * rmax * ring
                i, j = int((px - M.gx0) / C), int((py - M.gy0) / C)
                if not (2 < i < M.nx4 - 3 and 2 < j < M.ny4 - 3) or M.blk[j * M.nx4 + i]:
                    continue
                walls = sum(M.blk[(j + dj) * M.nx4 + i + di] for dj in range(-2, 3) for di in range(-2, 3))
                if BOTTLENECK and walls >= 10:
                    continue
                sc = math.hypot(px - x, py - y) * 0.5 + ring * rmax * 0.3
                if 2 <= walls <= 8:
                    sc -= 30
                if HIGH_GROUND:
                    sc -= 2 * (M.ground(px, py) - M.ground(nc[0], nc[1]))
                if sc < bs:
                    best, bs = (px, py), sc
        return best

    def act(self, obs):
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        st = me["stats"]
        muz = (x, y, z + st.get("muzzle_height", 7))
        act = {"tick": obs["tick"]}
        hpf = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        weapons = me["weapons"]
        gun = weapons[0] if weapons[0] and weapons[0]["kind"] == "gun" else None
        w1 = weapons[1] if len(weapons) > 1 else None
        kinds = {w["kind"]: i for i, w in enumerate(weapons) if w}

        for ev in obs.get("events", []):
            if ev["type"] == "damaged":
                self.dmg_log.append((t, ev.get("amount", 0)))
                if ev.get("weapon") != "zone":
                    self.last_hurt = t
                if ev.get("by") is not None and ev["by"] < 100:
                    self.attacker = (ev["by"], t)
                if ev.get("bearing") is not None and ev.get("weapon") != "zone":
                    self.threat = (ev["bearing"], t)
            elif ev["type"] == "hit":
                self.last_dealt = t
        if self.threat is None or t - self.threat[1] > 3:
            for s in obs.get("sounds", []):
                if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar") and s["dist"] != "far":
                    self.threat = (s["bearing"], t - s.get("age", 0))
                    break

        if obs.get("offer"):
            ch = obs["offer"]["choices"]
            table = UPGRADE_SCORE_ATK if ATTACK_UPGRADES else UPGRADE_SCORE
            act["pick"] = max(range(len(ch)), key=lambda i: table.get(ch[i]["id"], 1))

        charging = any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in weapons)
        cur_rank = WEAPON_RANK.get(w1["kind"], 0) if w1 else -1
        for p in obs.get("pickups", []):
            if p["kind"] in kinds or p["kind"] == "gun":
                continue
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) < self.reach - 0.3 and abs(p["pos"][2] - z) < 4:
                if WEAPON_RANK.get(p["kind"], 0) > cur_rank and not charging:
                    act["take"] = p["id"]
                    if w1:
                        act["drop"] = 1
                    break

        mechs = [m for m in obs.get("mechs", []) if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m["kind"] == "npc"]
        for m in mechs:
            m["_d"] = math.dist(m["pos"], me["pos"])
            m["_los"] = self.los(muz, self.center(m))
        threats = [m for m in mechs if m["_d"] < 220]
        enemy = None
        if mechs:
            atk = self.attacker[0] if self.attacker and t - self.attacker[1] < 3 else None

            def score(m):
                s = m["_d"] * (0.6 if m["_los"] else 1.0) * (0.7 + 0.6 * m["hp"] / max(1, m["max_hp"]))
                if TACTICS and m["hp"] < 0.5 * m["max_hp"]:
                    s *= 0.7  # already hurt, finish it
                return s * 0.6 if m["id"] == atk else s  # shoot whoever is shooting us

            enemy = min(mechs, key=score)
            if TACTICS:
                cur = next((m for m in mechs if m["id"] == self.tgt_id), None)
                # stick with the current target unless the new one is clearly better
                if cur is not None and score(enemy) > 0.7 * score(cur):
                    enemy = cur
                self.tgt_id = enemy["id"]

        # zone
        nc, nr = zone["next_center"], zone["next_radius"]
        cc, cr = self.zone_at(zone, t + 4)
        margin = min(25.0, max(4.0, nr * 0.35))
        d_next = math.hypot(x - nc[0], y - nc[1])
        d_cur = math.hypot(x - cc[0], y - cc[1])
        speed = st.get("walk_speed", 11.7)
        travel = max(0.0, d_next - (nr - margin)) / speed
        lead_t = travel * 1.3 + 35 if EARLY_ROTATE else travel + 20
        must_go = (d_cur > cr - 12) or (t > zone["shrink_start"] - lead_t and d_next > nr - margin)
        urgent = (not me["in_zone"]) or d_cur > cr - 4
        ux, uy = ((x - nc[0]) / d_next, (y - nc[1]) / d_next) if d_next > 1 else (0.0, 0.0)
        # sit a bit inside the next circle, not on its edge, so the following circle is never far
        rr = max(0.0, min(d_next, (nr - margin) * 0.65))
        safe = (nc[0] + ux * rr, nc[1] + uy * rr)
        if ZONE_SPOT:
            key = (round(nc[0]), round(nc[1]), round(nr))
            if self.zspot is None or self.zspot[0] != key:
                self.zspot = (key, self.zone_spot(nc, (nr - margin) * 0.65, x, y))
            if self.zspot[1] is not None:
                safe = self.zspot[1]

        def in_safe(px, py):
            return (math.hypot(px - nc[0], py - nc[1]) < nr - margin * 0.5
                    and math.hypot(px - cc[0], py - cc[1]) < cr - 8)

        if BOTTLENECK and not ZONE_SPOT and self.map is not None:
            safe = self.open_spot(safe[0], safe[1], in_safe)

        # recover mode: hide and let regen work (needs regen_delay s without dealing/taking damage)
        outnumbered = sum(1 for m in threats if m["_los"]) >= 2
        self.remember(obs, me)

        if HP_BANDS:
            stronger = False
            if enemy is not None and enemy["_los"]:
                mine = sum(1 for w in weapons if w)
                stronger = (enemy["hp"] > me["hp"] * 1.2
                            or (enemy.get("cubes") or 0) >= (me.get("cubes") or 0) + 3
                            or len(enemy.get("weapons") or []) > mine)
            start = hpf < 0.4 or (hpf < 0.7 and (outnumbered or stronger))
        else:
            start = hpf < 0.5 or (outnumbered and hpf < 0.7)
        if not self.recover and start:
            self.recover = True
        elif self.recover and hpf > 0.9:
            self.recover = False
        close = [m for m in threats if m["_d"] < 45 and m["_los"]]
        near = [m for m in threats if m["_d"] < 100 and m["_los"]]

        if TACTICS:
            while self.dmg_log and t - self.dmg_log[0][0] > 2.0:
                self.dmg_log.popleft()
            incoming = sum(a for _, a in self.dmg_log) / 2.0
            uniq = {m["id"]: m for m in threats}
            predicted = sum(self.threat_dps(m, x, y) for m in uniq.values())
            rate = max(incoming, predicted, 1.0)
            ttd = me["hp"] / rate
            ttk = 1e9
            if enemy is not None and enemy["_los"]:
                ttk = enemy["hp"] / max(1.0, self.dps(list(kinds), enemy["_d"], me.get("cubes")))
            if ttd < 1.5 or (hpf < 0.2 and incoming > 0):
                # about to die: get out now
                self.panic_until = t + 1.5
                self.recover = True
            elif not self.recover:
                # only leave when we'd clearly die first; healthier means braver
                losing = ttk > ttd * (1.5 + 0.5 * hpf)
                serious = incoming > 0.15 * me["max_hp"]
                if losing and serious:
                    if self.cover is None or t - self.cover_t > 1.0:
                        self.cover = self.find_cover(me, threats, in_safe)
                        self.cover_t = t
                        self.cover_len = self.path_len(me, *self.cover) if self.cover else None
                    escape = self.cover_len / speed + 0.5 if (self.cover and self.cover_len is not None) else 4.0
                    if ttd < escape + 1.5:
                        self.recover = True
        panic = t < self.panic_until

        aim = None
        fire0 = fire1 = False
        label = ""
        target = enemy
        tgt_is_mech = enemy is not None
        # don't poke npcs or crates when hurt, it resets regen
        npc_hp, npc_r = (0.5, 160) if FARM else (0.75, 100)
        if target is None and hpf > npc_hp:
            cands = [n for n in npcs if math.dist(n["pos"], me["pos"]) < npc_r and in_safe(n["pos"][0], n["pos"][1])]
            if cands:
                target = min(cands, key=lambda n: math.dist(n["pos"], me["pos"]))
                target["_d"] = math.dist(target["pos"], me["pos"])
                target["_los"] = self.los(muz, self.center(target))
        crate_hp, crate_r = (0.5, 150) if FARM else (0.9, 90)
        if target is None and hpf > crate_hp:
            cs = [c for c in obs.get("crates", []) if math.dist(c["pos"], me["pos"]) < crate_r and in_safe(c["pos"][0], c["pos"][1])]
            for c in sorted(cs, key=lambda c: math.dist(c["pos"], me["pos"])):
                cpos = (c["pos"][0], c["pos"][1], c["pos"][2] + 1.2)
                if not self.los(muz, cpos):
                    continue
                aim = self.aim_to(muz, cpos)
                d = math.dist(muz, cpos)
                if self.aim_err(me["aim"], aim) < max(0.012, math.atan(1.0 / max(d, 1))) and gun and gun.get("reload", 0) == 0 and gun.get("ammo", 1) > 0:
                    fire0 = True
                if kinds.get("lightning") == 1 and d < 40 and abs(wrap(me["aim"][0] - aim[0])) < 0.2:
                    fire1 = True
                label = "crate"
                break

        # while recovering only shoot back at someone right on top of us
        # but if we're getting hit anyway, holding fire gains nothing
        hold_fire = (self.recover and tgt_is_mech and not close and t - self.last_hurt > 1.0) or (panic and tgt_is_mech and not close)
        if BOOST_BANDS and near and 0.4 <= hpf < 0.6:
            hold_fire = False  # backing off but still shooting

        if target is not None:
            tc = self.center(target)
            tv = target.get("vel", [0, 0, 0])
            d = target["_d"]
            los_ok = target["_los"]
            tspeed = math.hypot(tv[0], tv[1])
            gspeed = gun["stats"]["speed"] if gun else 252
            gaim = self.aim_to(muz, self.lead(muz, tc, tv, gspeed))
            aim = gaim
            rad = 2.5 * (1 + 0.05 * (target.get("cubes") or 0)) if tgt_is_mech else 1.8
            tol = max(0.01, math.atan(rad * 0.75 / max(d, 1)))

            if "mortar" in kinds and not hold_fire:
                mw = weapons[kinds["mortar"]]
                ms = mw["stats"]
                # shells are slow, only worth it over cover or on someone standing still
                if mw["cooldown"] <= 0.05 and 30 < d < 220 and (not los_ok or tspeed < 3 or self.mortar_until > t):
                    tube = (x, y, z + st.get("height", 9) * (1 + 0.05 * me.get("cubes", 0)) + 0.5)
                    sol = self.mortar_solve(tube, target["pos"], tv, ms["speed"], ms.get("gravity", 22), high=not los_ok)
                    if sol:
                        aim = sol
                        self.mortar_until = t + 0.5
                        if self.aim_err(me["aim"], sol) < 0.02:
                            if kinds["mortar"] == 1:
                                fire1 = True
                            else:
                                fire0 = True

            if "laser" in kinds and self.mortar_until <= t and not hold_fire:
                lw = weapons[kinds["laser"]]
                ls = lw["stats"]
                full = ls.get("charge", 0.75)
                if los_ok and d < ls.get("range", 280) - 5:
                    laim = self.aim_to(muz, (tc[0] + tv[0] * 0.08, tc[1] + tv[1] * 0.08, tc[2]))
                    aim = laim
                    ch = lw.get("charge", 0)
                    if lw["cooldown"] <= 0.05 or ch > 0:
                        hold = ch < full or self.aim_err(me["aim"], laim) > tol
                        if ch > full + ls.get("hold_s", 2) - 0.3:
                            hold = False
                        if kinds["laser"] == 1:
                            fire1 = hold
                        else:
                            fire0 = hold

            # gun misses a lot at long range against a strafing target
            gun_range = 200 if tspeed < 2 else 120
            if (gun and los_ok and not hold_fire and d < min(gun_range, gun["stats"].get("range", 230))
                    and gun.get("reload", 0) == 0 and gun.get("ammo", 1) > 0
                    and self.aim_err(me["aim"], gaim) < tol * 1.3):
                fire0 = True

            if "lightning" in kinds and d < weapons[kinds["lightning"]]["stats"].get("range", 45) - 1:
                if abs(wrap(me["aim"][0] - gaim[0])) < math.radians(13) and los_ok and me["energy"] > 6:
                    if kinds["lightning"] == 1:
                        fire1 = True
                    else:
                        fire0 = True
            label = ("mech %d" % target["id"]) if tgt_is_mech else "npc"
            mf.draw.line(list(muz), list(tc), "#ff5040" if los_ok else "#888888")
        elif aim is None:
            if self.threat and t - self.threat[1] < 4:
                aim = [self.threat[0], 0.0]
            else:
                aim = [wrap(math.atan2(nc[1] - y, nc[0] - x) + 1.3 * math.sin(t * 0.9)), 0.0]
        act["aim"] = [aim[0], max(st.get("min_pitch", -0.7), min(st.get("max_pitch", 1.3), aim[1]))]
        if gun and gun.get("reload", 0) == 0 and (gun.get("ammo", 1) == 0 or (gun.get("ammo", 12) < 6 and not fire0)):
            act["reload"] = True

        # movement
        mx = my = 0.0
        boost = False
        if urgent or must_go:
            mx, my = self.nav_dir(me, t, safe[0], safe[1])
            boost = urgent and me["energy"] > 20
            label = "ZONE " + label

        if self.recover and threats and not (urgent or (ZONE_FIRST and must_go)):
            if self.cover is None or t - self.cover_t > 0.6:
                self.cover = self.find_cover(me, threats, in_safe)
                self.cover_t = t
            if self.cover:
                mx, my = self.nav_dir(me, t, self.cover[0], self.cover[1])
                if math.hypot(self.cover[0] - x, self.cover[1] - y) < 2.5:
                    mx, my = 0.0, 0.0
                label = "COVER " + label
            else:
                ex, ey = norm2(enemy["pos"][0] - x, enemy["pos"][1] - y)
                mx, my = -ex - ey * self.strafe * 0.8, -ey + ex * self.strafe * 0.8
                label = "RUN " + label
            boost = me["energy"] > 20 and bool(close or any(m["_los"] for m in threats))
            if panic:
                boost = me["energy"] > 10
            if t > self.strafe_until:
                self.strafe = random.choice((-1, 1))
                self.strafe_until = t + random.uniform(0.6, 1.6)
        elif tgt_is_mech:
            ex, ey = target["pos"][0] - x, target["pos"][1] - y
            d = target["_d"]
            rx, ry = norm2(ex, ey)
            tx, ty = -ry, rx
            if t > self.strafe_until:
                self.strafe = random.choice((-1, 1))
                if DYN_STRAFE:
                    self.strafe_until = t + random.uniform(0.25, 1.2)
                    self.strafe_amt = random.uniform(0.6, 1.0)
                else:
                    self.strafe_until = t + random.uniform(0.5, 1.4)
            if "laser" in kinds:
                lo, hi = 60, 130
            elif "lightning" in kinds and hpf > 0.5:
                lo, hi = 12, 32
            else:
                lo, hi = 30, 65  # gun hits much more often up close
            radial = -1.0 if d < lo else (0.8 if d > hi else 0.0)
            group = [m for m in threats if m["_d"] < 160]
            if KITE and len(group) >= 2 and not (urgent or must_go):
                # several of them: back away from the group so they reach us one at a time
                gx = sum(m["pos"][0] for m in group) / len(group) - x
                gy = sum(m["pos"][1] for m in group) / len(group) - y
                ax, ay = norm2(-gx, -gy)
                mx, my = ax + tx * self.strafe * 0.4, ay + ty * self.strafe * 0.4
                if not in_safe(x + mx * 20, y + my * 20):
                    sx, sy = norm2(nc[0] - x, nc[1] - y)
                    mx, my = mx * 0.3 + sx + tx * self.strafe * 0.6, my * 0.3 + sy + ty * self.strafe * 0.6
                label = "KITE " + label
            elif urgent or must_go:
                mx, my = mx + tx * self.strafe * 0.5, my + ty * self.strafe * 0.5
            else:
                mx = rx * radial + tx * self.strafe * self.strafe_amt
                my = ry * radial + ty * self.strafe * self.strafe_amt
                if SURROUND:
                    bs_ = sorted(math.atan2(m["pos"][1] - y, m["pos"][0] - x) for m in threats if m["_los"] and m["_d"] < 120)
                    if len(bs_) >= 2:
                        gaps = [((bs_[(i + 1) % len(bs_)] - bs_[i]) % TAU, bs_[i]) for i in range(len(bs_))]
                        gap, start = max(gaps)
                        if gap < TAU - 2.1:  # spread over more than ~120 degrees
                            ea = start + gap / 2
                            mx, my = mx + math.cos(ea) * 0.7, my + math.sin(ea) * 0.7
                if not in_safe(x + mx * 15, y + my * 15):
                    sx, sy = norm2(nc[0] - x, nc[1] - y)
                    mx, my = mx * 0.4 + sx, my * 0.4 + sy
                # cubes dropped mid-fight are worth a small detour
                near_cubes = [c for c in obs.get("cubes", []) if math.hypot(c["pos"][0] - x, c["pos"][1] - y) < 14]
                if near_cubes:
                    c = min(near_cubes, key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                    cx, cy = norm2(c["pos"][0] - x, c["pos"][1] - y)
                    mx, my = mx * 0.5 + cx, my * 0.5 + cy
            boost = boost or (me["energy"] > 45 and (d > hi + 40 or d < lo * 0.7))
        elif HIDE_MODE and not (urgent or must_go):
            # nobody in sight: wait next to a building. weapons in reach still get picked up above
            key = (round(nc[0]), round(nc[1]), round(nr))
            if self.hspot is None or self.hspot[0] != key or not in_safe(*self.hspot[1]):
                self.hspot = (key, self.hide_spot(x, y, in_safe) or safe)
            hx, hy = self.hspot[1]
            dh = math.hypot(hx - x, hy - y)
            if dh > 3:
                mx, my = self.nav_dir(me, t, hx, hy)
                if dh < 40:
                    mx, my = mx * 0.4, my * 0.4  # quiet near the spot
            label = label or "hide"
        elif not (urgent or must_go):
            goal = self.loot_goal(obs, me, in_safe, hpf)
            if FARM and target is not None and not tgt_is_mech and target.get("kind") == "npc" and target["_d"] > 50:
                # walk up to the drone, but stay outside its 45 m aggro range
                if goal is None or target["_d"] * 0.8 < math.hypot(goal[0] - x, goal[1] - y):
                    goal = (target["pos"][0], target["pos"][1], "drone")
            if goal is not None:
                mx, my = self.nav_dir(me, t, goal[0], goal[1])
                label = label or goal[2]
            elif hpf > (0.6 if FARM else 0.7) and (FARM or w1 is None or t < 150):
                # nothing known nearby: look around for loot instead of standing still
                if (self.roam is None or t > self.roam[2] or not in_safe(self.roam[0], self.roam[1])
                        or math.hypot(self.roam[0] - x, self.roam[1] - y) < 10):
                    self.roam = self.pick_roam(x, y, nc, nr - margin, t)
                mx, my = self.nav_dir(me, t, self.roam[0], self.roam[1])
                label = label or "explore"
            else:
                dsafe = math.hypot(safe[0] - x, safe[1] - y)
                if dsafe > 8:
                    mx, my = self.nav_dir(me, t, safe[0], safe[1])
                    if dsafe < 40:
                        mx, my = mx * 0.4, my * 0.4  # under 5 m/s makes no footstep sound
                label = label or "hold"

        if BOOST_BANDS:
            if near and hpf < 0.4:
                boost = me["energy"] > 10
            elif near and hpf < 0.6:
                if not self.recover and not (urgent or must_go):
                    e = min(near, key=lambda m: m["_d"])
                    ax, ay = norm2(x - e["pos"][0], y - e["pos"][1])
                    mx, my = ax - ay * self.strafe * 0.5, ay + ax * self.strafe * 0.5
                    if not in_safe(x + mx * 20, y + my * 20):
                        sx, sy = norm2(nc[0] - x, nc[1] - y)
                        mx, my = mx * 0.4 + sx, my * 0.4 + sy
                    label = "BACKOFF " + label
                boost = me["energy"] > 20
            elif not near and not urgent:
                boost = False  # nobody around, save energy

        dmx, dmy = self.dodge(obs, me)
        if dmx or dmy:
            mx, my = mx * 0.4 + dmx, my * 0.4 + dmy
            boost = boost or me["energy"] > 25

        mx, my = self.steer(me, mx, my)

        # stuck: jump/jet and turn a bit
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > 1.2:
            self.hist.popleft()
        jump = False
        if math.hypot(mx, my) > 0.5 and len(self.hist) > 8 and t - self.hist[0][0] > 1.0 and not charging:
            if math.hypot(x - self.hist[0][1], y - self.hist[0][2]) < 2.5:
                self.stuck_until = t + 1.0
                self.stuck_flip = -self.stuck_flip
                self.path = []
                self.hist.clear()
        if t < self.stuck_until:
            jump = me["grounded"]
            boost = me["energy"] > 10
            a = math.atan2(my, mx) + 0.7 * self.stuck_flip
            mx, my = math.cos(a), math.sin(a)

        n = math.hypot(mx, my)
        if n > 1:
            mx, my = mx / n, my / n
        act.update(move=[mx, my], boost=bool(boost), jump=bool(jump), fire=[bool(fire0), bool(fire1)])
        mf.draw.point([safe[0], safe[1], z], "#40c0ff", 6)
        if self.cover:
            mf.draw.point([self.cover[0], self.cover[1], z], "#40ff60", 5)
        mf.draw.label("%s hp%.0f e%.0f" % (label, me["hp"], me["energy"]))
        return act

    def mortar_solve(self, src, tpos, tvel, v, g, high):
        p = tpos
        th = 0.0
        dx = dy = 0.0
        for _ in range(3):
            dx, dy = p[0] - src[0], p[1] - src[1]
            d = math.hypot(dx, dy)
            dz = p[2] - src[2]
            disc = v ** 4 - g * (g * d * d + 2 * dz * v * v)
            if disc < 0 or d < 1:
                return None
            r = math.sqrt(disc)
            th = math.atan((v * v + r) / (g * d)) if high else math.atan((v * v - r) / (g * d))
            tt = d / (v * math.cos(th))
            p = (tpos[0] + tvel[0] * tt * 0.7, tpos[1] + tvel[1] * tt * 0.7, tpos[2])
        if th > 1.25:
            return None
        return [math.atan2(dy, dx), th]

    def loot_goal(self, obs, me, in_safe, hpf):
        x, y, z = me["pos"]
        best = None
        bs = 1e9
        weapons = me["weapons"]
        held = {w["kind"] for w in weapons if w}
        w1 = weapons[1] if len(weapons) > 1 else None
        cur_rank = WEAPON_RANK.get(w1["kind"], 0) if w1 else -1
        for c in obs.get("cubes", []):
            d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
            if d < 160 and in_safe(c["pos"][0], c["pos"][1]) and abs(c["pos"][2] - z) < 25 and d * 0.7 < bs:
                bs, best = d * 0.7, (c["pos"][0], c["pos"][1], "cube")
        for px, py, pz, kind in self.known_pickups.values():
            if kind in held or WEAPON_RANK.get(kind, 0) <= cur_rank:
                continue
            d = math.hypot(px - x, py - y)
            # an empty second slot makes any weapon worth a long walk
            w = 0.5 if w1 is None else 0.8
            if d < 250 and in_safe(px, py) and abs(pz - z) < 20 and d * w < bs:
                bs, best = d * w, (px, py, "weapon " + kind)
        if hpf > (0.5 if FARM else 0.9):
            cw, cmax = (0.8, 300) if FARM else (1.2, 200)
            for cx, cy, cz, _ in self.known_crates.values():
                d = math.hypot(cx - x, cy - y)
                if d < cmax and in_safe(cx, cy) and abs(cz - z) < 15 and d * cw < bs:
                    bs, best = d * cw, (cx, cy, "crate")
        for dr in obs.get("drops", []):
            d = math.hypot(dr["pos"][0] - x, dr["pos"][1] - y)
            if dr.get("holder") not in (None, me["id"]) and dr.get("progress", 0) > 2.5:
                continue
            # a drop is a full heal, worth more when hurt
            w = 0.3 if hpf < 0.7 else 0.5
            if d < 180 and in_safe(dr["pos"][0], dr["pos"][1]) and d / 11.7 > dr.get("lands_in", 0) - 8 and d * w < bs:
                bs, best = d * w, (dr["pos"][0], dr["pos"][1], "drop")
        return best

    def remember(self, obs, me):
        x, y, z = me["pos"]
        eye = (x, y, z + me["stats"].get("eye_height", 8))
        for store, items in ((self.known_pickups, obs.get("pickups", [])), (self.known_crates, obs.get("crates", []))):
            seen = set()
            for p in items:
                store[p["id"]] = (p["pos"][0], p["pos"][1], p["pos"][2], p.get("kind", "crate"))
                seen.add(p["id"])
            # forget things that should be in view but aren't (taken or destroyed)
            for pid, (px, py, pz, _) in list(store.items()):
                if pid not in seen and math.hypot(px - x, py - y) < 35 and self.los(eye, (px, py, pz + 1)):
                    del store[pid]
        for ev in obs.get("events", []):
            if ev["type"] == "take_failed" and ev.get("reason") == "gone":
                self.known_pickups.pop(ev.get("id"), None)
            elif ev["type"] == "took":
                self.known_pickups.pop(ev.get("id"), None)

    def pick_roam(self, x, y, nc, r, t):
        for _ in range(20):
            a = random.uniform(0, TAU)
            d = random.uniform(30, 110)
            px, py = x + math.cos(a) * d, y + math.sin(a) * d
            if math.hypot(px - nc[0], py - nc[1]) < r and (self.map is None or not self.map.wall_at(px, py, self.map.ground(px, py))):
                return (px, py, t + 20)
        return (nc[0], nc[1], t + 20)

    def dodge(self, obs, me):
        x, y, z = me["pos"]
        cz = z + 4.5
        mx = my = 0.0
        g = self.cfg["weapons"].get("mortar", {}).get("gravity", 22)
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            if p["kind"] == "mortar":
                a, b, c = -0.5 * g, vz, pz - z
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if tl <= 0 or tl > 4:
                    continue
                lx, ly = px + vx * tl, py + vy * tl
                if math.hypot(lx - x, ly - y) < 12:
                    ax, ay = norm2(x - lx, y - ly)
                    if ax == 0 and ay == 0:
                        ax, ay = norm2(-vy, vx)
                    mx += ax * 1.5
                    my += ay * 1.5
            else:
                rx, ry, rz = x - px, y - py, cz - pz
                vv = vx * vx + vy * vy + vz * vz
                if vv < 1:
                    continue
                tc = (rx * vx + ry * vy + rz * vz) / vv
                if tc <= 0 or tc > 0.8:
                    continue
                cx, cy = px + vx * tc - x, py + vy * tc - y
                if math.hypot(cx, cy) < 4.0:
                    sx, sy = norm2(-vy, vx)
                    if sx * cx + sy * cy > 0:
                        sx, sy = -sx, -sy
                    mx += sx
                    my += sy
        # someone charging a laser at us: sidestep
        for m in obs.get("mechs", []):
            if m["kind"] == "mech" and m.get("charging"):
                a = math.atan2(y - m["pos"][1], x - m["pos"][0])
                if abs(wrap(m["aim"][0] - a)) < 0.12:
                    mx += -math.sin(a) * self.strafe * 1.2
                    my += math.cos(a) * self.strafe * 1.2
        return mx, my


if __name__ == "__main__":
    mf.run(Bot)
