"""双足 RL 环境（MJX）。

本步只实现 `reset`；`step` 留到 Step 6，先把"环境能被构造、能吐出形状正确的初始状态"这件事坐实。

设计原则：
- 模型参数（kp / ctrlrange / 限位 / 质量）一律从 MjModel 读，这里不写死
- 观测契约来自 constants.py，这里只负责"按契约拼数"
- 不改 MJCF 的执行器参数：那是 BAM 辨识的结果，改了等于扔掉
"""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx

from mujoco_playground._src import mjx_env

from duck_rl import config as cfg
from duck_rl.constants import NUM_JOINTS, OBS_SIZE, sensor_slice

# 默认场景文件：用绝对路径解析，避免"必须从项目根目录运行"的隐性依赖
DEFAULT_XML = Path(__file__).resolve().parent.parent / "assets" / "scene_flat_terrain.xml"


# ---------------------------------------------------------------- 坐标变换
def _quat_to_mat(q: jax.Array) -> jax.Array:
    """四元数 (w, x, y, z) → 3x3 旋转矩阵（体坐标系 → 世界坐标系）。"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def gravity_body(q: jax.Array) -> jax.Array:
    """重力方向在体坐标系下的表达。

    世界系里重力是 (0, 0, -1)；转到体坐标系要乘 R^T：
        g_body = R^T @ (0, 0, -1) = -(R[2,0], R[2,1], R[2,2])   ← 取旋转矩阵的第三「行」
    竖直站立时结果 ≈ (0, 0, -1)；躯干前倾时 x 分量变正。

    ⚠ 注意是第三行不是第三列。取列是"体轴在世界里的方向"，取行才是"世界向量在体里的分量"，
    这两个互为转置，在机器人倾斜时会给出完全不同的数——写错了观测就废了，而且不会报错。
    """
    rot = _quat_to_mat(q)
    return -rot[2, :]



class DuckEnv(mjx_env.MjxEnv):
    """最小可用的双足环境。

    父类 MjxEnv 的 __init__ 会做三件事：锁住 config、存到 self._config、
    算出 self.n_substeps = ctrl_dt / sim_dt。所以这里 super().__init__ 之后
    只管加载模型和读初始姿态。
    """

    def __init__(self, config=None, config_overrides=None, xml_path=None):
        super().__init__(config or cfg.default_config(), config_overrides)

        self._xml = str(xml_path) if xml_path else str(DEFAULT_XML)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml)

        # 物理步长与 config 对齐（MJCF 里也是 0.002，这里做一次显式同步，
        # 免得以后改 config 忘了改 xml）
        self._mj_model.opt.timestep = self._config.sim_dt

        # 只改积分步长，**不动** actuator_gainprm / biasprm / forcerange：
        # 那些是 sts3215 辨识出来的 kp=13.37 / kv=0 / ±3.23，动它就没有 sim2real 可言
        self._mjx_model = mjx.put_model(self._mj_model)

        # 初始姿态：keyframe 在 scene 文件里，加载本体 xml 会 nkey=0
        if self._mj_model.nkey == 0:
            raise ValueError(
                f"{self._xml} 里没有 keyframe。是不是加载了本体 xml 而不是 scene？"
            )
        # qpos 布局：[x, y, z, qw, qx, qy, qz, 14 个关节角]
        self._init_qpos = jp.array(self._mj_model.key_qpos[0])
        # 默认姿态：qpos 的第 7 位往后，共 14 个
        self._default_pos = self._init_qpos[7:7 + NUM_JOINTS]

        # 一次一致性检查：constants 里的关节数必须等于模型的执行器数
        assert self._mj_model.nu == NUM_JOINTS, (
            f"constants 写了 {NUM_JOINTS} 个关节，模型有 {self._mj_model.nu} 个执行器"
        )

        # 传感器切片：按名字查一次，之后反复用（避免每次 _get_obs 都查名字）
        self._sl_gyro = sensor_slice(self._mj_model, "gyro")
        self._sl_upvec = sensor_slice(self._mj_model, "upvector")   # 调试对照用

        # qpos / qvel 里关节段的起始下标：
        # 浮基占前 7 位（3 位置 + 4 四元数）和 6 位（3 线速度 + 3 角速度），关节从后面开始
        self._q_start = 7
        self._qd_start = 6

        # 每个执行器的控制上下限。来自 MJCF 的 inheritrange="1"（等于关节限位），
        # 是 14 维数组而不是标量：左右 hip_pitch 的上下界是镜像的，用一个标量会把一条腿夹死。
        self._ctrl_lo = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_hi = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

    # ---------------------------------------------------------------- 父类要求的属性
    @property
    def xml_path(self) -> str:
        return self._xml

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def action_size(self) -> int:
        return NUM_JOINTS

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """把机器人摆到 keyframe 的蹲姿，返回初始 State。

        注意：父类的 observation_size 会用 jax.eval_shape 调一次这个函数，
        所以这里不能 print、也不能做依赖具体数值的 Python 分支。
        """
        # 拆随机数：一个给指令采样，剩下的留到 info 里给后续使用
        rng, k_cmd = jax.random.split(rng, 2)

        # 造 Data：qpos 用 keyframe（含浮基位置与四元数），ctrl 设成默认姿态。
        # ctrl 与默认姿态相同 → 位置伺服的目标就是当前位置，第一帧没有瞬态冲击。
        data = mjx_env.make_data(
            self._mjx_model,
            qpos=self._init_qpos,
            ctrl=self._default_pos,
        )
        # 必须前向一次：否则 site_xpos / sensordata 全是 0，
        # 下一步算观测时会发现重力方向恒为 (0,0,0)
        data = mjx.forward(self._mjx_model, data)

        cmd = self._sample_command(k_cmd)

        # 拼真实观测（51 维）；last_action 初始为全零，表示"上一步没有动作"
        obs = self._get_obs(data, jp.zeros(NUM_JOINTS), cmd)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics={},
            info={
                "command": cmd,                          # 本回合的速度指令（3,）
                "last_action": jp.zeros(NUM_JOINTS),     # 上一帧网络输出（14,）
                "steps": jp.zeros((), dtype=jp.int32),   # 本回合已走步数
                "rng": rng,                              # 留给后续重采样
            },
        )

    # ---------------------------------------------------------------- step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """走一个控制周期（内部 10 个物理子步）。

        数据流：action(±1) → 缩放 → 加上默认姿态 → 按 ctrlrange 剪裁 → mjx 子步进 → 新状态
        """
        c = self._config

        # 1) 动作映射：网络输出在 [-1, 1]，乘 action_scale 后叠加到默认蹲姿上
        ctrl = self._default_pos + action * c.action_scale
        # 2) 剪裁到每个关节自己的控制范围（14 维数组，不是标量）
        ctrl = jp.clip(ctrl, self._ctrl_lo, self._ctrl_hi)

        # 3) 物理子步进：n_substeps 次 mjx.step，用 jax.lax.scan 在 GPU 上跑
        #    （不要写 Python for 循环，那样无法 jit、也无法并行多个环境）
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        # 4) 新观测：last_action 传网络输出（不是 ctrl），保持与部署端一致
        obs = self._get_obs(data, action, state.info["command"])

        # 5) 终止判断：摔倒 或 回合超时
        steps = state.info["steps"] + 1
        terminated = self._get_termination(data)
        timeout = steps >= c.episode_length
        done = jp.logical_or(terminated, timeout)

        return state.replace(
            data=data,
            obs=obs,
            reward=jp.zeros(()),          # Step 7 填真实奖励
            done=done.astype(jp.float32),
            metrics={
                "height": data.qpos[2],
                "terminated": terminated.astype(jp.float32),
            },
            info={**state.info, "last_action": action, "steps": steps},
        )

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """摔倒判定。注意：不用 Python 的 if，返回的是 jax 标量布尔值。"""
        c = self._config
        too_low = data.qpos[2] < c.min_height
        # 竖直时 gravity_body[2] ≈ -1；躺平 ≈ 0。比 -0.4 更正说明倾斜过大
        tilted = gravity_body(data.qpos[3:7])[2] > -c.min_upright
        nan = jp.logical_not(jp.isfinite(data.qpos).all())
        return jp.logical_or(jp.logical_or(too_low, tilted), nan)

    # ---------------------------------------------------------------- 观测
    def _get_obs(self, data: mjx.Data, last_action: jax.Array, cmd: jax.Array) -> jax.Array:
        """按 constants.OBS_LAYOUT 的顺序拼 51 维观测。

        段的顺序必须与 OBS_LAYOUT 完全一致，否则训练和部署全乱——而且不会报错。
        """
        c = self._config
        sens = data.sensordata

        # 1) 角速度（机体系）3 维，来自 sensor "gyro"
        gyro = sens[self._sl_gyro] * c.obs_scale_gyro

        # 2) 重力方向（机体系）3 维，由浮基四元数算出（不用 upvector，符号约定自己掌控）
        grav = gravity_body(data.qpos[3:7])

        # 3) 关节位置 14 维：用「相对默认姿态」的偏移量。
        #    绝对角在 0.6~1.4 之间，相对量的量级更小、分布更接近零，网络更好学。
        q = data.qpos[self._q_start:self._q_start + NUM_JOINTS] - self._default_pos

        # 4) 关节速度 14 维
        qd = data.qvel[self._qd_start:self._qd_start + NUM_JOINTS] * c.obs_scale_qd

        # 5) 上一帧动作 14 维：让策略"记得自己刚才在干嘛"，少了它会抖
        return jp.concatenate([gyro, grav, cmd, q, qd, last_action]).astype(jp.float32)

    # ---------------------------------------------------------------- 内部工具
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """每个回合采样一次速度指令 [vx, vy, wz]，范围来自 config。"""
        c = self._config
        k1, k2, k3 = jax.random.split(rng, 3)
        return jp.array([
            jax.random.uniform(k1, (), minval=c.cmd_lin_x[0], maxval=c.cmd_lin_x[1]),
            jax.random.uniform(k2, (), minval=c.cmd_lin_y[0], maxval=c.cmd_lin_y[1]),
            jax.random.uniform(k3, (), minval=c.cmd_ang_z[0], maxval=c.cmd_ang_z[1]),
        ], dtype=jp.float32)
