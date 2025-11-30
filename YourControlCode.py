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
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):
        self.m, self.d = m, d
        self.points = np.asarray(target_points) # (3, N)
        assert self.points.ndim == 2 and self.points.shape[0] == 3
        self.N = int(self.points.shape[1])

        # index of ee frame
        self.ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
        if self.ee_id < 0:
            raise ValueError("Body 'EE_Frame' not found.")

        # velocity dofs and actuators
        self.nv, self.nu = int(m.nv), int(m.nu)

        # Control gains (using similar values to CircularMotion)
        self.kp = 200.0
        self.kd = 10.0

        # null-space damping
        self.k_damp_joint = 2.0 # global joint damping (N·m·s/rad)
        self.k_posture = 30.0 # pull toward home in null space
        self.d_posture = 2.0
        self.q_home = d.qpos[:self.nv].copy() # nominal posture (start pose)

        # parameters
        self.reach_thresh = 0.010
        self.dwell_needed = 8
        self.approach_radius = 0.06 # start fading gains inside this radius
        self.min_gain_scale = 0.25

        # jacobians and identity mats
        self.Jp = np.zeros((3, self.nv)) # translational jacobian
        self.Jr = np.zeros((3, self.nv)) # rotational jacobian
        self.Iv = np.eye(self.nv)
        self.I3 = np.eye(3)

        self._dwell = np.zeros(self.N, dtype=int) # counts consecutive steps within reach
        self.visited = np.zeros(self.N, dtype=bool) # which points have been visited

        # clip ranges
        if getattr(self.m, "actuator_ctrlrange", None) is not None and self.m.actuator_ctrlrange.size:
            self.ctrl_min = self.m.actuator_ctrlrange[:, 0].copy()
            self.ctrl_max = self.m.actuator_ctrlrange[:, 1].copy()
        else:
            self.ctrl_min = None
            self.ctrl_max = None

    # functions

    # get translational and rotational jacobian
    # reads ee position, computes ee translational velocity
    def _ee_state_and_jac(self):
        x = self.d.xpos[self.ee_id].copy()
        mujoco.mj_jacBody(self.m, self.d, self.Jp, self.Jr, self.ee_id)
        v = self.Jp @ self.d.qvel
        return x, v, self.Jp

    # computes euclidean distances from current ee to all points
    # for each unvisited point, if within reach_thresh, increment dwell counter
    # if dwell counter exceeds dwell_needed, mark point as visited
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

    # computes euclidean distances to unvisited points, returns index of closest unvisited point
    # if all visited, returns None, otherwise index of nearest unvisited point
    def _choose_target_idx(self, x):
        d = np.linalg.norm((self.points.T - x), axis=1)
        d[self.visited] = np.inf
        idx = int(np.argmin(d))
        if not np.isfinite(d[idx]):
            return None, d
        return idx, d

    # computes null-space projector for jacobian
    def _nullspace_projector(self, J):
        # N = I - J^T (J J^T + λI)^-1 J  
        JJt = J @ J.T
        N = self.Iv - J.T @ np.linalg.inv(JJt + 1e-8 * self.I3) @ J # nv x nv matrix, removes task space component from joint torques
        return N

    # main
    def CtrlUpdate(self):
        # curr ee, ee vel, jacobian
        x, v, J = self._ee_state_and_jac()

        # visit tracking and target selection
        dists = self._mark_visited(x)
        tgt_idx, _ = self._choose_target_idx(x)
        if tgt_idx is None:
            return np.zeros(self.nu, dtype=float)

        # computes distance and error to target
        x_goal = self.points[:, tgt_idx]
        e = x_goal - x
        dist = float(np.linalg.norm(e))

        # reduce control gains when approaching target
        # when close to target, full kp/kd causes oscillations
        # controller gains 1 -> min_gain_scale
        if dist < self.approach_radius:
            s = max(self.min_gain_scale, dist / self.approach_radius)
        else:
            s = 1.0
        kp_use = self.kp * s
        kd_use = self.kd * s

        # task-space impedance force
        # kp*e pulls towards target, -kd*v damps motion
        f = kp_use * e - kd_use * v              

        # task torques to joint torques
        tau_task = J.T @ f + self.d.qfrc_bias.copy()   

        # computes null-space projector of jacobian
        N = self._nullspace_projector(J)      

        #
        q = self.d.qpos[:self.nv] # joint pos
        qd = self.d.qvel.copy() # joint vel

        # torques that lie in null space of jacobian
        tau_null = -self.k_damp_joint * qd \
                   + N @ ( self.k_posture * (self.q_home - q) - self.d_posture * qd )

        tau = tau_task + tau_null

        # clip to actuator torque limits
        ctrl = tau[:self.nu].astype(float)
        if self.ctrl_min is not None:
            ctrl = np.clip(ctrl, self.ctrl_min, self.ctrl_max)

        return ctrl
