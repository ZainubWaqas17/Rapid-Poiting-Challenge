# import mujoco
# import numpy as np


# class YourCtrl:
#   def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
#     self.m = m
#     self.d = d
#     self.target_points = target_points

#     self.init_qpos = d.qpos.copy()

#     # Control gains (using similar values to CircularMotion)
#     self.kp = 150.0
#     self.kd = 10.0

#   def CtrlUpdate(self):
#     jtorque_cmd = np.zeros(6)
#     for i in range(6):
#         jtorque_cmd[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]

#     return jtorque_cmd



import mujoco
import numpy as np


class YourCtrl:
    """
    League 1 controller (torque-level):
      • Greedy nearest-unvisited targeting (with dwell).
      • Task-space impedance:  f = Kp * e - Kd * v.
      • Joint torques:         tau_task = J^T f + qfrc_bias.
      • Null-space stabilization to stop "spinning":
          - Global joint damping.
          - Posture regulation projected into null space of Jp.
      • "Approach mode": gains fade when close to target to avoid chatter.

    Returns actuator torques of length m.nu every tick.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):
        self.m, self.d = m, d
        self.points = np.asarray(target_points)                   # (3, N)
        assert self.points.ndim == 2 and self.points.shape[0] == 3
        self.N = int(self.points.shape[1])

        # EE handle (env measures from BODY "EE_Frame")
        self.ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
        if self.ee_id < 0:
            raise ValueError("Body 'EE_Frame' not found.")

        self.nv, self.nu = int(m.nv), int(m.nu)

        # --- Task-space gains (base) ---
        self.kp = 200.0
        self.kd = 10.0

        # --- Null-space stabilization ---
        self.k_damp_joint = 2.0          # global joint damping (N·m·s/rad)
        self.k_posture    = 30.0         # pull toward home in null space
        self.d_posture    = 2.0
        self.q_home = d.qpos[:self.nv].copy()   # nominal posture (start pose)

        # --- Approach / dwell (env uses 0.01 m) ---
        self.reach_thresh  = 0.010
        self.dwell_needed  = 8
        self.approach_radius = 0.06       # start fading gains inside this radius
        self.min_gain_scale = 0.25

        # --- Work buffers ---
        self.Jp = np.zeros((3, self.nv))
        self.Jr = np.zeros((3, self.nv))
        self.Iv = np.eye(self.nv)
        self.I3 = np.eye(3)

        # Visited bookkeeping
        self._dwell = np.zeros(self.N, dtype=int)
        self.visited = np.zeros(self.N, dtype=bool)

        # Clip ranges (if defined)
        if getattr(self.m, "actuator_ctrlrange", None) is not None and self.m.actuator_ctrlrange.size:
            self.ctrl_min = self.m.actuator_ctrlrange[:, 0].copy()
            self.ctrl_max = self.m.actuator_ctrlrange[:, 1].copy()
        else:
            self.ctrl_min = None
            self.ctrl_max = None

        # Debug
        self.DEBUG = True
        self.debug_every = 20
        self._tick = 0
        if self.DEBUG:
            print(f"[YourCtrl:init] nv={self.nv} nu={self.nu} kp={self.kp} kd={self.kd}")
            print(f"[YourCtrl:init] reach={self.reach_thresh} dwell={self.dwell_needed}, "
                  f"approach_r={self.approach_radius}")

    # ---------------- helpers ----------------

    def _ee_state_and_jac(self):
        x = self.d.xpos[self.ee_id].copy()
        mujoco.mj_jacBody(self.m, self.d, self.Jp, self.Jr, self.ee_id)
        v = self.Jp @ self.d.qvel
        return x, v, self.Jp

    def _mark_visited(self, x):
        diffs = (self.points.T - x)
        dists = np.linalg.norm(diffs, axis=1)
        for i in range(self.N):
            if self.visited[i]:
                continue
            if dists[i] <= self.reach_thresh:
                self._dwell[i] += 1
                if self._dwell[i] >= self.dwell_needed:
                    self.visited[i] = True
            else:
                self._dwell[i] = 0
        return dists

    def _choose_target_idx(self, x):
        d = np.linalg.norm((self.points.T - x), axis=1)
        d[self.visited] = np.inf
        idx = int(np.argmin(d))
        if not np.isfinite(d[idx]):
            return None, d
        return idx, d

    def _nullspace_projector(self, J):
        # N = I - J^T (J J^T + λI)^-1 J   (3x3 invert; robust)
        JJt = J @ J.T
        N = self.Iv - J.T @ np.linalg.inv(JJt + 1e-8 * self.I3) @ J
        return N

    # ---------------- main control ----------------

    def CtrlUpdate(self):
        self._tick += 1

        # EE state & Jacobian
        x, v, J = self._ee_state_and_jac()

        # Visit bookkeeping
        dists = self._mark_visited(x)
        tgt_idx, _ = self._choose_target_idx(x)
        if tgt_idx is None:
            if self.DEBUG and (self._tick % self.debug_every == 0):
                print("[YourCtrl] All points visited. Zero torques.")
            return np.zeros(self.nu, dtype=float)

        # Goal and distance
        x_goal = self.points[:, tgt_idx]
        e = x_goal - x
        dist = float(np.linalg.norm(e))

        # ----- Approach mode: fade gains when close -----
        if dist < self.approach_radius:
            s = max(self.min_gain_scale, dist / self.approach_radius)  # in (min_scale, 1]
        else:
            s = 1.0
        kp_use = self.kp * s
        kd_use = self.kd * s

        # ----- Task-space impedance -> joint torques -----
        f = kp_use * e - kd_use * v                     # (3,)
        tau_task = J.T @ f + self.d.qfrc_bias.copy()    # (nv,)

        # ----- Null-space stabilization to stop spinning -----
        N = self._nullspace_projector(J)                # (nv,nv)
        # Global damping in null space + posture regulation
        q = self.d.qpos[:self.nv]
        qd = self.d.qvel.copy()
        tau_null = -self.k_damp_joint * qd \
                   + N @ ( self.k_posture * (self.q_home - q) - self.d_posture * qd )

        tau = tau_task + tau_null

        # Map to actuator torques and clip
        ctrl = tau[:self.nu].astype(float)
        if self.ctrl_min is not None:
            ctrl = np.clip(ctrl, self.ctrl_min, self.ctrl_max)

        # Debug line (throttled)
        if self.DEBUG and (self._tick % self.debug_every == 0 or self._dwell[tgt_idx] == 1):
            sat = 0.0
            if self.ctrl_min is not None:
                sat = np.mean((ctrl <= self.ctrl_min + 1e-9) | (ctrl >= self.ctrl_max - 1e-9))
            n_left = int(np.sum(~self.visited))


        return ctrl
