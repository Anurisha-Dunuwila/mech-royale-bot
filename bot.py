"""Survival-first mechfight bot.

Priorities: stay inside the zone, face and out-range attackers, dodge, collect cubes/weapons,
lead gun shots, use laser / mortar / lightning when held, pick defensive upgrades.
Pure standard library, Python 3.9+.
"""
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
G = 2.0  # fine grid (m) for line of sight / walls
WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
UPGRADE_SCORE = {
    "armor": 10, "plating": 9, "nanites": 8, "gun_damage": 7, "gun_rate": 7,
    "laser_damage": 6, "laser_charge": 6, "lightning_efficiency": 5, "lightning_range": 5,
    "gyros": 5, "servos": 5, "sensors": 4, "reactor": 4, "gun_velocity": 4, "gun_mag": 4,
    "lightning_chain": 4, "capacitor": 3, "jets": 3, "mortar_reload": 3, "mortar_splash": 3,
    "mortar_velocity": 2, "laser_pierce": 2,
}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def norm2(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


# ---------------------------------------------------------------- map
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
        # coarse nav grid (2x2 fine cells): blocked if >=2 sub-cells are walls at ground level
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
        self.pen = bytearray(self.nx4 * self.ny4)
        for j in range(1, self.ny4 - 1):
            for i in range(1, self.nx4 - 1):
                k = j * self.nx4 + i
                if not self.blk[k]:
                    s = self.blk[k - 1] + self.blk[k + 1] + self.blk[k - self.nx4] + self.blk[k + self.nx4]
                    self.pen[k] = s

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
        """True if a straight shot/sight line a->b is not blocked by buildings or terrain."""
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

    # ---- A* on coarse grid, time-capped
    def path(self, sx, sy, gx, gy, budget=0.012):
        C = 2 * G
        nx4, ny4 = self.nx4, self.ny4

        def idx(x, y):
            i = min(nx4 - 1, max(0, int((x - self.gx0) / C)))
            j = min(ny4 - 1, max(0, int((y - self.gy0) / C)))
            return j * nx4 + i

        s, g = idx(sx, sy), idx(gx, gy)
        if self.blk[s]:
            return None
        if self.blk[g]:  # nearest free cell to the goal
            gi, gj = g % nx4, g // nx4
            best = None
            for r in range(1, 12):
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        ii, jj = gi + di, gj + dj
                        if 0 <= ii < nx4 and 0 <= jj < ny4 and not self.blk[jj * nx4 + ii]:
                            best = jj * nx4 + ii
                            break
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is None:
                return None
            g = best
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
            hk = f - gc
            if hk < besth:
                best, besth = k, hk
            it += 1
            if it % 256 == 0 and time.perf_counter() - t0 > budget:
                break
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


# ---------------------------------------------------------------- bot
class Bot:
    name = "survivor"

    def __init__(self, hello):
        random.seed(hello.get("seed", 0))
        self.cfg = hello["config"]
        self.reach = self.cfg["loot"]["pickup_radius"]
        try:
            self.map = Map(hello["map"]["dir"])
        except Exception as e:  # never die because of the map
            mf.log("map load failed:", e)
            self.map = None
        self.hist = deque()
        self.stuck_until = -1.0
        self.stuck_flip = 1
        self.strafe = 1
        self.strafe_until = 0.0
        self.threat = None  # (bearing, time)
        self.path = []
        self.path_goal = None
        self.path_t = -99.0
        self.mortar_until = -1.0
        self.last_hp = None
        self.last_hit_t = -99.0
        self.scan = 0.0

    # ---------- helpers
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
        """Rotate the wanted direction away from walls just ahead."""
        if self.map is None or (dx == 0 and dy == 0):
            return dx, dy, False
        x, y, fz = me["pos"]
        base = math.atan2(dy, dx)
        mag = min(1.0, math.hypot(dx, dy))
        for off in (0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.1, -2.1):
            a = base + off * self.stuck_flip
            ca, sa = math.cos(a), math.sin(a)
            if all(self.free(x + ca * r, y + sa * r, fz) for r in (3.0, 5.5, 8.0)):
                return ca * mag, sa * mag, off != 0
        return dx, dy, True

    def nav_dir(self, me, t, gx, gy):
        """Direction toward a far goal, via cached A* path when the straight line is blocked."""
        x, y, z = me["pos"]
        if self.map is None:
            return norm2(gx - x, gy - y)
        if math.hypot(gx - x, gy - y) < 12 or self.map.los((x, y, z + 3), (gx, gy, self.map.ground(gx, gy) + 3)):
            self.path = []
            return norm2(gx - x, gy - y)
        need = (not self.path or self.path_goal is None or math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 15
                or t - self.path_t > 4.0)
        if need:
            p = self.map.path(x, y, gx, gy)
            self.path = p[1:] if p else []
            self.path_goal = (gx, gy)
            self.path_t = t
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 5:
            self.path.pop(0)
        # look ahead along the path for the farthest directly reachable waypoint
        tgt = None
        for wp in self.path[:6]:
            if self.map.los((x, y, z + 3), (wp[0], wp[1], self.map.ground(wp[0], wp[1]) + 3)):
                tgt = wp
            else:
                break
        if tgt is None:
            tgt = self.path[0] if self.path else (gx, gy)
        return norm2(tgt[0] - x, tgt[1] - y)

    def lead(self, src, tpos, tvel, speed):
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
    def center(m):
        if m.get("kind") == "npc":
            return (m["pos"][0], m["pos"][1], m["pos"][2] + 1.75)
        s = 1 + 0.05 * (m.get("cubes") or 0)
        return (m["pos"][0], m["pos"][1], m["pos"][2] + 4.5 * s)

    # ---------- main
    def act(self, obs):
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        st = me["stats"]
        muz = (x, y, z + st.get("muzzle_height", 7))
        eye = (x, y, z + st.get("eye_height", 8))
        act = {"tick": obs["tick"]}
        hpf = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        weapons = me["weapons"]
        gun = weapons[0] if weapons[0] and weapons[0]["kind"] == "gun" else None
        w1 = weapons[1] if len(weapons) > 1 else None
        kinds = {w["kind"]: i for i, w in enumerate(weapons) if w}

        # ---- events / threat memory
        for ev in obs.get("events", []):
            if ev["type"] == "damaged":
                self.last_hit_t = t
                if ev.get("bearing") is not None and ev.get("weapon") != "zone":
                    self.threat = (ev["bearing"], t)
        if self.threat is None or t - self.threat[1] > 3:
            for s in obs.get("sounds", []):
                if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar") and s["dist"] != "far":
                    self.threat = (s["bearing"], t - s.get("age", 0))
                    break

        # ---- upgrades
        if obs.get("offer"):
            ch = obs["offer"]["choices"]
            act["pick"] = max(range(len(ch)), key=lambda i: UPGRADE_SCORE.get(ch[i]["id"], 1))

        # ---- weapon pickups within reach
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

        # ---- targets
        mechs = [m for m in obs.get("mechs", []) if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m["kind"] == "npc"]
        for m in mechs:
            m["_d"] = math.dist(m["pos"], me["pos"])
            m["_los"] = self.los(muz, self.center(m))
        enemy = None
        if mechs:
            enemy = min(mechs, key=lambda m: m["_d"] * (0.6 if m["_los"] else 1.0) * (0.7 + 0.6 * m["hp"] / max(1, m["max_hp"])))

        # ---- zone planning
        nc, nr = zone["next_center"], zone["next_radius"]
        cc, cr = self.zone_at(zone, t + 4)
        margin = min(25.0, max(4.0, nr * 0.35))
        d_next = math.hypot(x - nc[0], y - nc[1])
        d_cur = math.hypot(x - cc[0], y - cc[1])
        speed = st.get("walk_speed", 11.7)
        travel = max(0.0, d_next - (nr - margin)) / speed
        must_go = (d_cur > cr - 12) or (t > zone["shrink_start"] - travel - 20 and d_next > nr - margin)
        urgent = (not me["in_zone"]) or d_cur > cr - 4
        if d_next > 1:
            ux, uy = (x - nc[0]) / d_next, (y - nc[1]) / d_next
        else:
            ux, uy = 0.0, 0.0
        rr = max(0.0, min(d_next, nr - margin))
        safe = (nc[0] + ux * rr, nc[1] + uy * rr)

        def in_safe(px, py, extra=0.0):
            return math.hypot(px - nc[0], py - nc[1]) < nr - margin * 0.5 - extra and \
                math.hypot(px - cc[0], py - cc[1]) < cr - 8

        # ---- choose aim + fire
        aim = None
        fire0 = fire1 = False
        label = ""
        target = enemy
        tgt_is_mech = enemy is not None
        if target is None:
            cands = [n for n in npcs if math.dist(n["pos"], me["pos"]) < 110 and in_safe(n["pos"][0], n["pos"][1])]
            if cands and hpf > 0.45:
                target = min(cands, key=lambda n: math.dist(n["pos"], me["pos"]))
                target["_d"] = math.dist(target["pos"], me["pos"])
                target["_los"] = self.los(muz, self.center(target))
        crate_t = None
        if target is None:
            cs = [c for c in obs.get("crates", []) if math.dist(c["pos"], me["pos"]) < 90 and in_safe(c["pos"][0], c["pos"][1])]
            for c in sorted(cs, key=lambda c: math.dist(c["pos"], me["pos"])):
                cpos = (c["pos"][0], c["pos"][1], c["pos"][2] + 1.2)
                if self.los(muz, cpos):
                    crate_t = c
                    a = self.aim_to(muz, cpos)
                    d = math.dist(muz, cpos)
                    aim = a
                    if self.aim_err(me["aim"], a) < max(0.012, math.atan(1.0 / max(d, 1))) and gun and gun.get("reload", 0) == 0 and gun.get("ammo", 1) > 0:
                        fire0 = True
                    if "lightning" in kinds and d < 40 and abs(wrap(me["aim"][0] - a[0])) < 0.2:
                        if kinds["lightning"] == 1:
                            fire1 = True
                    label = "crate"
                    break

        if target is not None:
            tc = self.center(target)
            tv = target.get("vel", [0, 0, 0])
            d = target["_d"]
            los_ok = target["_los"]
            gspeed = gun["stats"]["speed"] if gun else 252
            gaim = self.aim_to(muz, self.lead(muz, tc, tv, gspeed))
            aim = gaim
            rad = 2.5 * (1 + 0.05 * (target.get("cubes") or 0)) if tgt_is_mech else 1.8
            tol = max(0.01, math.atan(rad * 0.75 / max(d, 1)))
            # mortar
            if "mortar" in kinds:
                mw = weapons[kinds["mortar"]]
                ms = mw["stats"]
                if mw["cooldown"] <= 0.05 and 30 < d < 230 and (not los_ok or d > 90 or self.mortar_until > t):
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
            # laser
            if "laser" in kinds and self.mortar_until <= t:
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
                        fire_l = hold if ch > 0 or lw["cooldown"] <= 0.05 else False
                        if kinds["laser"] == 1:
                            fire1 = fire_l
                        else:
                            fire0 = fire_l
            # gun
            if gun and los_ok and d < gun["stats"].get("range", 230) and gun.get("reload", 0) == 0 and gun.get("ammo", 1) > 0:
                if self.aim_err(me["aim"], gaim) < tol * 1.3:
                    fire0 = True
            # lightning
            if "lightning" in kinds and d < weapons[kinds["lightning"]]["stats"].get("range", 45) - 1:
                if abs(wrap(me["aim"][0] - gaim[0])) < math.radians(13) and los_ok and me["energy"] > 6:
                    if kinds["lightning"] == 1:
                        fire1 = True
                    else:
                        fire0 = True
            label = ("mech %d" % target["id"]) if tgt_is_mech else "npc"
            if mf.draw:
                mf.draw.line(list(muz), list(tc), "#ff5040" if los_ok else "#888")
        elif aim is None:
            # nothing to shoot: face last threat, else scan around
            if self.threat and t - self.threat[1] < 4:
                aim = [self.threat[0], 0.0]
            else:
                self.scan = wrap(self.scan + 0.25)
                aim = [wrap(math.atan2(nc[1] - y, nc[0] - x) + 1.3 * math.sin(t * 0.9)), 0.0]
        act["aim"] = [aim[0], max(st.get("min_pitch", -0.7), min(st.get("max_pitch", 1.3), aim[1]))]
        if gun and (gun.get("ammo", 1) == 0 or (gun.get("ammo", 12) < 5 and target is None and crate_t is None)) and gun.get("reload", 0) == 0:
            act["reload"] = True

        # ---- movement
        mx = my = 0.0
        boost = False
        if urgent or must_go:
            mx, my = self.nav_dir(me, t, safe[0], safe[1])
            boost = urgent and me["energy"] > 20
            label = "ZONE " + label
        if target is not None and tgt_is_mech:
            ex, ey = target["pos"][0] - x, target["pos"][1] - y
            d = target["_d"]
            rx, ry = norm2(ex, ey)
            tx, ty = -ry, rx
            if t > self.strafe_until:
                self.strafe = random.choice((-1, 1))
                self.strafe_until = t + random.uniform(0.6, 1.6)
            if "laser" in kinds or "mortar" in kinds:
                lo, hi = 110, 190
            else:
                lo, hi = 70, 140
            if "lightning" in kinds and hpf > 0.5 and target["hp"] < me["hp"] * 0.8:
                lo, hi = 15, 35
            if hpf < 0.35:
                lo, hi = 170, 260
            radial = -1.0 if d < lo else (0.6 if d > hi else 0.0)
            cmx = rx * radial + tx * self.strafe * 0.9
            cmy = ry * radial + ty * self.strafe * 0.9
            if urgent or must_go:
                mx, my = mx * 1.0 + tx * self.strafe * 0.5, my * 1.0 + ty * self.strafe * 0.5
            else:
                mx, my = cmx, cmy
                # keep combat footwork inside the safe area
                if not in_safe(x + mx * 15, y + my * 15):
                    sx, sy = norm2(nc[0] - x, nc[1] - y)
                    mx, my = mx * 0.4 + sx, my * 0.4 + sy
            boost = boost or (me["energy"] > 45 and (d < lo * 0.8 or hpf < 0.35))
        elif not (urgent or must_go):
            goal = self.loot_goal(obs, me, in_safe)
            if goal is not None:
                mx, my = self.nav_dir(me, t, goal[0], goal[1])
                label = label or goal[2]
            else:
                dsafe = math.hypot(safe[0] - x, safe[1] - y)
                if dsafe > 8:
                    mx, my = self.nav_dir(me, t, safe[0], safe[1])
                    if dsafe < 40:
                        mx, my = mx * 0.4, my * 0.4  # quiet walk (no step sound under 5 m/s)
                label = label or "hold"

        # ---- dodging (adds on top)
        dmx, dmy = self.dodge(obs, me)
        if dmx or dmy:
            mx, my = mx * 0.4 + dmx, my * 0.4 + dmy
            boost = boost or me["energy"] > 25

        mx, my, blocked = self.steer(me, mx, my)

        # ---- stuck detection
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
        mf.draw.label("%s hp%.0f e%.0f" % (label, me["hp"], me["energy"]))
        return act

    def aim_err(self, cur, want):
        return math.hypot(wrap(cur[0] - want[0]), cur[1] - want[1])

    def mortar_solve(self, src, tpos, tvel, v, g, high):
        p = tpos
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

    def loot_goal(self, obs, me, in_safe):
        x, y, z = me["pos"]
        best = None
        bs = 1e9
        weapons = me["weapons"]
        held = {w["kind"] for w in weapons if w}
        w1 = weapons[1] if len(weapons) > 1 else None
        cur_rank = WEAPON_RANK.get(w1["kind"], 0) if w1 else -1
        for c in obs.get("cubes", []):
            d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
            if d < 160 and in_safe(c["pos"][0], c["pos"][1]) and abs(c["pos"][2] - z) < 25:
                if d * 0.7 < bs:
                    bs, best = d * 0.7, (c["pos"][0], c["pos"][1], "cube")
        for p in obs.get("pickups", []):
            if p["kind"] in held or WEAPON_RANK.get(p["kind"], 0) <= cur_rank:
                continue
            d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
            if d < 160 and in_safe(p["pos"][0], p["pos"][1]) and abs(p["pos"][2] - z) < 25:
                if d * 0.8 < bs:
                    bs, best = d * 0.8, (p["pos"][0], p["pos"][1], "weapon " + p["kind"])
        for dr in obs.get("drops", []):
            d = math.hypot(dr["pos"][0] - x, dr["pos"][1] - y)
            if dr.get("holder") not in (None, me["id"]) and dr.get("progress", 0) > 2.5:
                continue
            if d < 180 and in_safe(dr["pos"][0], dr["pos"][1]) and d / 11.7 > dr.get("lands_in", 0) - 8:
                if d * 0.5 < bs:
                    bs, best = d * 0.5, (dr["pos"][0], dr["pos"][1], "drop")
        return best

    def dodge(self, obs, me):
        x, y, z = me["pos"]
        cz = z + 4.5
        mx = my = 0.0
        g = self.cfg["weapons"].get("mortar", {}).get("gravity", 22)
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            if p["kind"] == "mortar":
                # predict landing point
                a, b, c = -0.5 * g, vz, pz - z
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if tl <= 0 or tl > 4:
                    continue
                lx, ly = px + vx * tl, py + vy * tl
                d = math.hypot(lx - x, ly - y)
                if d < 12:
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
        # enemy laser charging at us
        for m in obs.get("mechs", []):
            if m["kind"] == "mech" and m.get("charging"):
                a = math.atan2(y - m["pos"][1], x - m["pos"][0])
                if abs(wrap(m["aim"][0] - a)) < 0.12:
                    sx, sy = -math.sin(a), math.cos(a)
                    mx += sx * self.strafe * 1.2
                    my += sy * self.strafe * 1.2
        return mx, my


if __name__ == "__main__":
    mf.run(Bot)
