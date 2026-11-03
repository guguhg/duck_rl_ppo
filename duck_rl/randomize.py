"""域随机化：让策略在"参数不完美"的仿真里训练，缩小 sim2real gap。

== 为什么需要 ==
之前所有成绩都是在完全干净的仿真里拿的：
  地面摩擦固定 0.6、舵机 kp 分毫不差、每个 body 的质量精确到克。
真机上这三项全都会变——地面从瓷砖到橡胶垫、舵机批次差异 ±20%、打印件填充率不同。

== 怎么做（照抄 mujoco_playground 的 locomotion/g1 模式）==
不复制整个 model，只复制 7 个小数组，每个环境一份，其余字段共享。
显存代价约：
  (1 + 14 + 14 + 18 + 14 + 14 + 14) × 4 字节 × 512 环境 ≈ 180 KB
实测训练峰值显存 2 GB（总量 6 GB），完全够。

接入方式：brax 的 `ppo.train(randomization_fn=...)` 原生支持，
它会自己 `jax.random.split(key, num_envs)` 再 `functools.partial`。
我们只需要提供一个 `(model, rng) -> (model_v, in_axes)` 的函数。

== 两个必须踩对的坑 ==
① **字段操作要写在 `jax.vmap` 内部**
   vmap 会自动把每次调用返回的"完整字段"堆叠成带前导维的数组。
   如果写在 vmap 外面、对 `(ngeom, 3)` 的数组 set 一个 `(N,)` 的标量序列，
   JAX 不会自动加前导维，会直接报形状错（或不报错但语义错）。

② **kp 与 biasprm[1] 必须同步**
   position 执行器的力公式：
       force = gainprm[0]·ctrl + biasprm[0] + biasprm[1]·q + biasprm[2]·qd
   MuJoCo 对 position 类型自动设为：
       gainprm = [kp, 0, 0]，biasprm = [0, -kp, -kv]
   改 kp 时必须同时改 gainprm[0] 和 biasprm[1] = -kp。
   只改一个的话，伺服变成"只有前馈没有反馈"，力随 ctrl 线性增长——鸭子会飞出去。
"""
import jax
import mujoco
from mujoco import mjx

# 要随机化的字段名。in_axes 里只有这些设为 0（表示"这个维度是 per-env 的"），
# 其余全部 None（共享同一份）。漏写一个，那个字段就不会被 vmap，等于没随机化。
DR_FIELDS = (
    "geom_friction",        # 地面滑动摩擦
    "dof_frictionloss",     # 关节库仑摩擦
    "dof_armature",         # 转子惯量
    "body_mass",            # 各 body 质量
    "actuator_gainprm",     # 位置伺服刚度 kp
    "actuator_biasprm",     # 位置伺服反馈系数（必须与 kp 同步）
    "actuator_forcerange",  # 力矩上限
)

# 浮基占前 6 个 dof（3 平动 + 3 转动），关节从第 6 个开始。
# 注意 nq=21（浮基占 7）而 nv=20（浮基占 6），这里用的是 dof 索引，所以是 6。
FLOATING_BASE_DOF = 6


def make_randomization_fn(mj_model: mujoco.MjModel, cfg):
    """构造 randomization_fn。

    参数
    ----
    mj_model : mujoco.MjModel
        CPU 侧模型，只用来查静态信息（floor 的 geom id）。不参与随机化。
    cfg : ConfigDict
        需要 dr_* 那几个字段。

    返回
    ----
    fn : Callable[[mjx.Model, jax.Array], tuple]
        签名必须与 brax 期望一致：`(model, rng) -> (model_v, in_axes)`。
        rng 形状是 (num_envs, 2)，由 brax 用 `jax.random.split` 切好。
    """
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError(
            "模型里没有名为 'floor' 的 geom。"
            "确认加载的是 assets/scene_flat_terrain.xml 而不是本体 xml。"
        )

    friction_lo, friction_hi = cfg.dr_friction
    floss_lo, floss_hi = cfg.dr_frictionloss
    arm_lo, arm_hi = cfg.dr_armature
    mass_lo, mass_hi = cfg.dr_mass
    kp_lo, kp_hi = cfg.dr_kp
    force_lo, force_hi = cfg.dr_forcerange

    def randomization_fn(model: mjx.Model, rng: jax.Array):
        """构造 batch model。rng 形状 (N, 2)，vmap 后每次拿 (2,)。"""

        @jax.vmap
        def sample_and_patch(key):
            """采一组参数并直接打补丁到 model，返回 7 个完整字段。

            关键：**所有 .at[].set() 都在这里做**。
            vmap 会把每次返回的 (ngeom,3) 之类堆成 (N,ngeom,3)。
            """
            k = jax.random.split(key, 6)

            # ---- 采样 ----
            # ① 地面摩擦：一个标量
            friction = jax.random.uniform(
                k[0], (), minval=friction_lo, maxval=friction_hi)

            n_joint = model.nv - FLOATING_BASE_DOF
            # ② 关节库仑摩擦：逐关节独立，模拟齿轮批次差异与磨损
            floss_scale = jax.random.uniform(
                k[1], (n_joint,), minval=floss_lo, maxval=floss_hi)
            # ③ 转子惯量：辨识误差 + 个体差异。范围窄，它对稳定性影响很大
            arm_scale = jax.random.uniform(
                k[2], (n_joint,), minval=arm_lo, maxval=arm_hi)
            # ④ 各 body 质量：打印件填充率、电池容量、螺丝数量都会变
            mass_scale = jax.random.uniform(
                k[3], (model.nbody,), minval=mass_lo, maxval=mass_hi)
            # ⑤ 舵机刚度与力矩上限：STS3215 个体差异最大的一项
            kp_scale = jax.random.uniform(
                k[4], (model.nu,), minval=kp_lo, maxval=kp_hi)
            force_scale = jax.random.uniform(
                k[5], (model.nu,), minval=force_lo, maxval=force_hi)

            # ---- 打补丁（每次返回完整字段，vmap 负责堆叠）----

            # 地面摩擦：geom_friction 是 (ngeom, 3)，只动 floor 的滑动列
            geom_friction = model.geom_friction.at[floor_id, 0].set(friction)

            # 关节摩擦与惯量：只改关节段，浮基不动
            dof_frictionloss = model.dof_frictionloss.at[FLOATING_BASE_DOF:].set(
                model.dof_frictionloss[FLOATING_BASE_DOF:] * floss_scale)
            dof_armature = model.dof_armature.at[FLOATING_BASE_DOF:].set(
                model.dof_armature[FLOATING_BASE_DOF:] * arm_scale)

            # body 质量：floor 的 mass 是 0，乘完还是 0，无副作用
            body_mass = model.body_mass * mass_scale

            # 舵机 kp：biasprm[1] 必须同步为 -kp（见模块文档的推导）
            kp = model.actuator_gainprm[:, 0] * kp_scale           # (nu,)
            actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

            # 力矩上限：两列（下/上界）同比例缩放，保持对称
            actuator_forcerange = model.actuator_forcerange * force_scale[:, None]

            return (geom_friction, dof_frictionloss, dof_armature, body_mass,
                    actuator_gainprm, actuator_biasprm, actuator_forcerange)

        (geom_friction, dof_frictionloss, dof_armature, body_mass,
         actuator_gainprm, actuator_biasprm, actuator_forcerange) = sample_and_patch(rng)

        model_v = model.tree_replace({
            "geom_friction": geom_friction,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "body_mass": body_mass,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "actuator_forcerange": actuator_forcerange,
        })

        # in_axes：默认全部 None（共享一份），只有改过的字段设 0（按环境切片）
        in_axes = jax.tree.map(lambda _: None, model)
        in_axes = in_axes.tree_replace({f: 0 for f in DR_FIELDS})
        return model_v, in_axes

    return randomization_fn
