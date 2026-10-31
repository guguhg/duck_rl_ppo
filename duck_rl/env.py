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
from duck_rl.constants import NUM_JOINTS, OBS_SIZE

# 默认场景文件：用绝对路径解析，避免"必须从项目根目录运行"的隐性依赖
DEFAULT_XML = Path(__file__).resolve().parent.parent / "assets" / "scene_flat_terrain.xml"


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

        # 本步先返回全零观测；Step 5 换成真正的拼接
        obs = jp.zeros(OBS_SIZE, dtype=jp.float32)

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

    # ---------------------------------------------------------------- step（待实现）
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        raise NotImplementedError("Step 6 实现：action -> ctrl -> mjx_env.step -> 观测/奖励/终止")

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
