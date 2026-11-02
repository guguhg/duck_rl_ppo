"""双足 RL 环境（MJX）。

整个项目**唯一实现 MDP 的地方**：观测怎么拼、奖励怎么算、什么时候算摔，全在这里。

设计原则：
- 模型参数（kp / ctrlrange / 限位 / 质量）一律从 MjModel 读，这里不写死
- 观测契约来自 constants.py，这里只负责"按契约拼数"
- 不改 MJCF 的执行器参数：那是 BAM 辨识的结果，改了等于扔掉
"""
from pathlib import Path

import jax                                       # jax.random / jax.Array
import jax.numpy as jp                           # jp.array / jp.clip / jp.logical_or
import mujoco                                    # CPU 版物理引擎，只用来加载模型
import mujoco.mjx as mjx                         # JAX 版 MuJoCo，能上 GPU 批量并行

from mujoco_playground._src import mjx_env        # 基类 + make_data + step 辅助

from duck_rl import config as cfg                 # 默认配置
from duck_rl import rewards as R                  # 七个奖励函数
from duck_rl.constants import NUM_JOINTS, OBS_SIZE, sensor_slice   # 契约

# 奖励项名字。reset 和 step 的 metrics 必须用同一批键——scan 要求结构不变。
REWARD_TERMS = (
    "tracking_lin", "tracking_ang", "upright",
    "height", "alive", "action_rate", "torque",
)

# 默认场景文件：用绝对路径解析，避免"必须从项目根目录运行"的隐性依赖
DEFAULT_XML = Path(__file__).resolve().parent.parent / "assets" / "scene_flat_terrain.xml"


# ---------------------------------------------------------------- 坐标变换
def _quat_to_mat(q: jax.Array) -> jax.Array:
    """四元数 (w, x, y, z) → 3x3 旋转矩阵（体坐标系 → 世界坐标系）。"""
    w, x, y, z = q[0], q[1], q[2], q[3]           # 拆成四个分量
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],      # 第 1 行
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],      # 第 2 行
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],      # 第 3 行
    ])                                            # 标准四元数→旋转矩阵公式，不含平移


def gravity_body(q: jax.Array) -> jax.Array:
    """重力方向在体坐标系下的表达。

    世界系里重力是 (0, 0, -1)；转到体坐标系要乘 R^T：
        g_body = R^T @ (0, 0, -1) = -(R[2,0], R[2,1], R[2,2])   ← 取旋转矩阵的第三「行」
    竖直站立时结果 ≈ (0, 0, -1)；躯干前倾时 x 分量变正。

    ⚠ 注意是第三行不是第三列。取列是"体轴在世界里的方向"，取行才是"世界向量在体里的分量"，
    这两个互为转置，在机器人倾斜时会给出完全不同的数——写错了观测就废了，而且不会报错。
    """
    rot = _quat_to_mat(q)                         # 先拿旋转矩阵
    return -rot[2, :]                             # 第三行取负 = 重力在体系的分量


def _world_to_body(q: jax.Array, v: jax.Array) -> jax.Array:
    """把世界系向量转到体坐标系（速度跟踪要用）。

    v_body = R^T · v_world。自由关节的线速度 qvel[0:3] 是世界系，
    而"往前走多快"是体坐标系的 x 分量，所以必须先转。
    """
    return _quat_to_mat(q).T @ v                  # 转置后左乘 = 世界→体


def yaw_of(q: jax.Array) -> jax.Array:
    """四元数 → 偏航角 yaw（绕世界 z 轴，范围 -π~π）。

    为什么需要它：gravity 对绕 z 轴的旋转**完全不敏感**（转多少度，重力在体系里
    的分量都一样），gyro 只给瞬时角速度。所以不加 yaw，策略就无法知道
    "我已经偏了 20 度"，也就无法纠偏——30 秒回放里掉头就是这个原因。
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class DuckEnv(mjx_env.MjxEnv):
    """最小可用的双足环境。

    父类 MjxEnv 的 __init__ 会做三件事：锁住 config、存到 self._config、
    算出 self.n_substeps = ctrl_dt / sim_dt。所以这里 super().__init__ 之后
    只管加载模型和读初始姿态。
    """

    def __init__(self, config=None, config_overrides=None, xml_path=None):
        super().__init__(config or cfg.default_config(), config_overrides)  # 父类：锁 config、算 n_substeps

        self._xml = str(xml_path) if xml_path else str(DEFAULT_XML)   # 允许换模型文件
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml)      # CPU 侧模型（读 xml + stl）

        # 物理步长与 config 对齐（MJCF 里也是 0.002，这里做一次显式同步，
        # 免得以后改 config 忘了改 xml）
        self._mj_model.opt.timestep = self._config.sim_dt             # 只改步长

        # 只改积分步长，**不动** actuator_gainprm / biasprm / forcerange：
        # 那些是 sts3215 辨识出来的 kp=13.37 / kv=0 / ±3.23，动它就没有 sim2real 可言
        self._mjx_model = mjx.put_model(self._mj_model)               # 结构搬上 GPU

        # 初始姿态：keyframe 在 scene 文件里，加载本体 xml 会 nkey=0
        if self._mj_model.nkey == 0:
            raise ValueError(
                f"{self._xml} 里没有 keyframe。是不是加载了本体 xml 而不是 scene？"
            )
        # qpos 布局：[x, y, z, qw, qx, qy, qz, 14 个关节角]
        self._init_qpos = jp.array(self._mj_model.key_qpos[0])         # 21 维完整初始位姿
        # 默认姿态：qpos 的第 7 位往后，共 14 个
        self._default_pos = self._init_qpos[7:7 + NUM_JOINTS]         # 只有关节角，蹲姿

        # 一次一致性检查：constants 里的关节数必须等于模型的执行器数
        assert self._mj_model.nu == NUM_JOINTS, (
            f"constants 写了 {NUM_JOINTS} 个关节，模型有 {self._mj_model.nu} 个执行器"
        )

        # 传感器切片：按名字查一次，之后反复用（避免每次 _get_obs 都查名字）
        self._sl_gyro = sensor_slice(self._mj_model, "gyro")          # slice(0, 3)
        self._sl_upvec = sensor_slice(self._mj_model, "upvector")     # slice(9, 12)，仅调试对照

        # qpos / qvel 里关节段的起始下标：
        # 浮基占前 7 位（3 位置 + 4 四元数）和 6 位（3 线速度 + 3 角速度），关节从后面开始
        self._q_start = 7                        # qpos 里关节角的起点
        self._qd_start = 6                       # qvel 里关节速度的起点

        # 每个执行器的控制上下限。来自 MJCF 的 inheritrange="1"（等于关节限位），
        # 是 14 维数组而不是标量：左右 hip_pitch 的上下界是镜像的，用一个标量会把一条腿夹死。
        self._ctrl_lo = jp.array(self._mj_model.actuator_ctrlrange[:, 0])   # 下界，14 个数
        self._ctrl_hi = jp.array(self._mj_model.actuator_ctrlrange[:, 1])   # 上界，14 个数

    # ---------------------------------------------------------------- 父类要求的属性
    @property
    def xml_path(self) -> str:
        return self._xml                         # 模型文件路径

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model                    # CPU 模型（画图、打印用）

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model                   # GPU 模型（仿真正在用）

    @property
    def action_size(self) -> int:
        return NUM_JOINTS                        # 动作 14 维

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """把机器人摆到 keyframe 的蹲姿，返回初始 State。
+
        注意：父类的 observation_size 会用 jax.eval_shape 调一次这个函数，
        所以这里不能 print、也不能做依赖具体数值的 Python 分支。
        """
        # 拆随机数：一个给指令采样，剩下的留到 info 里给后续使用
        rng, k_cmd = jax.random.split(rng, 2)                    # rng 复用、k_cmd 采样用

        # 造 Data：qpos 用 keyframe（含浮基位置与四元数），ctrl 设成默认姿态。
        # ctrl 与默认姿态相同 → 位置伺服的目标就是当前位置，第一帧没有瞬态冲击。
        data = mjx_env.make_data(
            self._mjx_model,                                     # GPU 上的模型
            qpos=self._init_qpos,                                # 21 维初始位姿
            ctrl=self._default_pos,                              # 伺服目标 = 当前关节角
        )
        # 必须前向一次：否则 site_xpos / sensordata 全是 0，
        # 下一步算观测时会发现重力方向恒为 (0,0,0)
        data = mjx.forward(self._mjx_model, data)                # 算一遍运动学

        cmd = self._sample_command(k_cmd)                        # 本回合指令 [vx, vy, wz]
        yaw0 = yaw_of(data.qpos[3:7])                            # 本回合的起始航向（纠偏的基准）

        # 拼真实观测（53 维）；last_action 初始为全零，表示"上一步没有动作"
        obs = self._get_obs(data, jp.zeros(NUM_JOINTS), cmd, yaw0)   # 第 2 个参数是 last_action

        return mjx_env.State(
            data=data,                                           # mjx.Data
            obs=obs,                                             # (53,)
            reward=jp.zeros(()),                                 # 标量，初始 0
            done=jp.zeros(()),                                   # 标量，初始没结束
            # metrics 的键必须与 step() 完全一致（scan 要求结构不变），
            # 所以这里统一走 _metrics()，不给自定义字典。
            metrics=self._metrics(data, jp.zeros((), dtype=jp.float32)),
            info={
                "command": cmd,                                  # 本回合的速度指令（3,）
                "last_action": jp.zeros(NUM_JOINTS),             # 上一帧网络输出（14,）
                "steps": jp.zeros((), dtype=jp.int32),           # 本回合已走步数
                "yaw0": yaw0,                                    # 起始航向，算相对偏航用
                "rng": rng,                                      # 留给后续重采样
            },
        )

    # ---------------------------------------------------------------- step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """走一个控制周期（内部 10 个物理子步）。

        数据流：action(±1) → 缩放 → 加上默认姿态 → 按 ctrlrange 剪裁 → mjx 子步进 → 新状态
        """
        c = self._config                                         # 短名字，下面反复用

        # 1) 动作映射：网络输出在 [-1, 1]，乘 action_scale 后叠加到默认蹲姿上
        ctrl = self._default_pos + action * c.action_scale        # (14,) 关节目标
        # 2) 剪裁到每个关节自己的控制范围（14 维数组，不是标量）
        ctrl = jp.clip(ctrl, self._ctrl_lo, self._ctrl_hi)        # 逐元素 clip

        # 3) 物理子步进：n_substeps 次 mjx.step，用 jax.lax.scan 在 GPU 上跑
        #    （不要写 Python for 循环，那样无法 jit、也无法并行多个环境）
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        # 4) 新观测：last_action 传网络输出（不是 ctrl），保持与部署端一致
        obs = self._get_obs(data, action, state.info["command"], state.info["yaw0"])

        # 5) 奖励：七项加权求和
        reward, terms = self._reward(data, action, state.info["last_action"], state.info["command"])

        # 6) 终止判断：摔倒 或 回合超时
        steps = state.info["steps"] + 1                           # 步数 +1
        terminated = self._get_termination(data)                  # 高度/倾角/NaN 三项
        timeout = steps >= c.episode_length                       # 满 1000 步 = 20 秒
        done = jp.logical_or(terminated, timeout)                 # 任一成立就结束

        return state.replace(                                     # State 不可变，用 replace 造新的
            data=data,                                            # 新的物理状态
            obs=obs,                                              # 新的 53 维观测
            reward=reward,                                        # 本步标量奖励
            done=done.astype(jp.float32),                         # 统一用 float32
            # 注意：必须「合并」而不是「替换」。
            # brax 的包装层会往 metrics 里塞 reward（training.py 第 201 行），
            # 如果我们直接返回新字典，那个键就丢了，scan 会因为结构变化而报错。
            metrics={**state.metrics, **self._metrics(data, terminated, terms)},
            info={**state.info, "last_action": action, "steps": steps},   # info 同样要合并
        )

    def _reward(self, data: mjx.Data, action: jax.Array, prev_action: jax.Array,
                cmd: jax.Array):
        """七项奖励加权求和。返回 (标量奖励, 各项明细字典)。"""
        c = self._config
        # 世界系线速度 → 体坐标系，才能和"往前走 m/s"的指令比较
        v_body = _world_to_body(data.qpos[3:7], data.qvel[0:3])    # qpos[3:7] 是四元数
        # 角速度取陀螺仪 z 轴（机体系），与真机可测的量一致
        gyro_z = data.sensordata[self._sl_gyro][2]               # 第 3 个分量 = z 轴

        terms = {                                                # 先算七项原始满意度
            "tracking_lin": R.tracking_lin_vel(v_body, cmd[:2], c.sigma_lin),    # 用 vx, vy
            "tracking_ang": R.tracking_ang_vel(gyro_z, cmd[2], c.sigma_ang),     # 用 wz
            "upright": R.upright(gravity_body(data.qpos[3:7]), c.sigma_upright),  # 躯干竖直
            "height": R.keep_height(data.qpos[2], c.height_ref, c.sigma_height),  # 浮基高度
            "alive": R.alive(),                                                  # 白给 1.0
            "action_rate": R.action_rate(action, prev_action),                   # 动作跳变
            "torque": R.torque_cost(data.actuator_force),                        # Στ²
        }
        reward = (c.w_tracking_lin * terms["tracking_lin"]       # 主目标，权重 2.5
                  + c.w_tracking_ang * terms["tracking_ang"]     # 0.8
                  + c.w_upright * terms["upright"]               # 0.5
                  + c.w_height * terms["height"]                 # 0.5
                  + c.w_alive * terms["alive"]                   # 0.3，白给分
                  + c.w_action_rate * terms["action_rate"]       # -0.01，惩罚
                  + c.w_torque * terms["torque"])                # -0.0002，惩罚
        return reward.astype(jp.float32), terms                  # 标量 + 明细

    def _metrics(self, data: mjx.Data, terminated: jax.Array, terms=None) -> dict:
        """metrics 的键集合必须在 reset 与 step 之间完全一致（scan 要求结构不变）。"""
        m = {
            "height": data.qpos[2],                              # 浮基高度，训练曲线要看
            "terminated": jp.asarray(terminated, dtype=jp.float32),   # 是否摔倒
        }
        for k in REWARD_TERMS:                                   # 七项明细也记下来
            # reset 时没有 terms，填 0；step 时有，填真实值
            m[f"rew/{k}"] = terms[k] if terms is not None else jp.zeros(())
        return m                                                 # 键固定 9 个

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """摔倒判定。注意：不用 Python 的 if，返回的是 jax 标量布尔值。"""
        c = self._config
        too_low = data.qpos[2] < c.min_height                    # 浮基低于 10 cm
        # 竖直时 gravity_body[2] ≈ -1；躺平 ≈ 0。比 -0.4 更正说明倾斜过大
        tilted = gravity_body(data.qpos[3:7])[2] > -c.min_upright  # 倾斜超约 66°
        nan = jp.logical_not(jp.isfinite(data.qpos).all())       # 数值炸了也算结束
        return jp.logical_or(jp.logical_or(too_low, tilted), nan)  # 三项取或

    # ---------------------------------------------------------------- 观测
    def _get_obs(self, data: mjx.Data, last_action: jax.Array, cmd: jax.Array,
                 yaw0: jax.Array) -> jax.Array:
        """按 constants.OBS_LAYOUT 的顺序拼 53 维观测。

        段的顺序必须与 OBS_LAYOUT 完全一致，否则训练和部署全乱——而且不会报错。
        """
        c = self._config
        sens = data.sensordata                                   # 传感器读数（46 维）

        # 1) 角速度（机体系）3 维，来自 sensor "gyro"
        gyro = sens[self._sl_gyro] * c.obs_scale_gyro            # 3 维，×0.25

        # 2) 重力方向（机体系）3 维，由浮基四元数算出（不用 upvector，符号约定自己掌控）
        grav = gravity_body(data.qpos[3:7])                      # 3 维，竖直时 (0,0,-1)

        # 3) 关节位置 14 维：用「相对默认姿态」的偏移量。
        #    绝对角在 0.6~1.4 之间，相对量的量级更小、分布更接近零，网络更好学。
        q = data.qpos[self._q_start:self._q_start + NUM_JOINTS] - self._default_pos  # 14 维

        # 4) 关节速度 14 维
        qd = data.qvel[self._qd_start:self._qd_start + NUM_JOINTS] * c.obs_scale_qd  # ×0.05

        # 5) 上一帧动作 14 维：让策略"记得自己刚才在干嘛"，少了它会抖
        # 6) 相对航向 2 维：sin/cos 而不是角度本身，因为角度在 ±π 处不连续，
        #    sin/cos 绕圈是连续的，网络学起来不会有跳变。
        dyaw = yaw_of(data.qpos[3:7]) - yaw0                    # 相对本回合起点的偏航
        heading = jp.array([jp.sin(dyaw), jp.cos(dyaw)])         # (2,)

        return jp.concatenate([gyro, grav, cmd, q, qd, last_action,
                               heading]).astype(jp.float32)     # 3+3+3+14+14+14+2 = 53

    # ---------------------------------------------------------------- 内部工具
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """每个回合采样一次速度指令 [vx, vy, wz]，回合内不变。

        为什么不直接三维独立均匀采样：[0, 0.3, 0]（纯侧移）这种组合的联合概率
        只有 0.09%，4 万回合里才几十个样本，策略学不会 → 实测给侧移指令它会往后退。
        两个补偿，把"单轴指令"的样本量拉上来：
          1. 30% 的回合只动一个轴，另两个强制为 0（手柄实际就是这么用的）
          2. vy 有一半概率落在两侧边界 [±0.2, ±0.3]，而不是均匀分布
        """
        c = self._config
        k_mode, k_axis, k_vx, k_vy, k_wz, k_edge = jax.random.split(rng, 6)

        vx = jax.random.uniform(k_vx, (), minval=c.cmd_lin_x[0], maxval=c.cmd_lin_x[1])
        vy = jax.random.uniform(k_vy, (), minval=c.cmd_lin_y[0], maxval=c.cmd_lin_y[1])
        wz = jax.random.uniform(k_wz, (), minval=c.cmd_ang_z[0], maxval=c.cmd_ang_z[1])

        # vy 边界加权：一半概率取 |vy| ∈ [edge_min, 上限]，符号随机
        k_e1, k_e2, k_e3 = jax.random.split(k_edge, 3)
        edge_mag = jax.random.uniform(k_e1, (), minval=c.cmd_vy_edge_min,
                                      maxval=c.cmd_lin_y[1])
        edge_sign = jnp.where(jax.random.uniform(k_e2, ()) < 0.5, -1.0, 1.0)
        vy = jnp.where(jax.random.uniform(k_e3, ()) < c.cmd_vy_edge_prob,
                       edge_mag * edge_sign, vy)

        cmd = jnp.array([vx, vy, wz], dtype=jnp.float32)

        # 单轴模式：随机挑一个轴保留，另两个置零
        axis = jax.random.randint(k_axis, (), 0, 3)              # 0 / 1 / 2
        mask = (jnp.arange(3) == axis).astype(jnp.float32)       # 选中的那位是 1
        cmd = jnp.where(jax.random.uniform(k_mode, ()) < c.cmd_single_axis_prob,
                        cmd * mask, cmd)                         # 单轴时只留一个分量
        return cmd.astype(jnp.float32)
