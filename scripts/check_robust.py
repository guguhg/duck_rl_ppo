"""鲁棒性对比：同一套策略，在不同物理参数下分别跑，看谁更抗扰动。

== 为什么需要这个脚本 ==
`check_all_dirs.py` / `check_policy_mjx.py` 都在**干净环境**里测（摩擦 0.6、质量精确）。
而域随机化的作用是"让策略在别的参数下也能走"，用干净环境测根本看不出差别——
exp8 和 dr1 在摩擦 0.6 下都会很好，分数甚至可能一样。

所以必须**改物理参数再测**，才能回答那个关键问题：
    加油漆的地板（0.35）和橡胶垫（0.95），哪个策略更不容易摔？

== 用 CPU MuJoCo 而不是 MJX ==
MJX 换 model 参数要重新 put_model + 可能触发重编译（每次几十秒）。
CPU 版改 `m.geom_friction` 立刻生效，250 步只要零点几秒。
代价是 CPU 和 MJX 的接触求解有细微差异（实测速度差约 0.04 m/s），
但**对比实验只关心相对值**，同一个引擎下比两个策略是有效的。

== 用法 ==
    # 单个策略，扫三档摩擦
    uv run python -u scripts/check_robust.py \
        --onnx checkpoints/policy_dr1.onnx --friction 0.3 0.6 0.95

    # 两个策略并排比（推荐）
    uv run python -u scripts/check_robust.py \
        --onnx checkpoints/policy_exp8.onnx checkpoints/policy_dr1.onnx \
        --friction 0.3 0.6 0.95
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from duck_rl import config as cfg
from duck_rl.constants import NUM_JOINTS

DEFAULT_XML = "assets/scene_flat_terrain.xml"


# ------------------------------------------------------------------ 观测拼装
# 和 watch_policy.py 里那份必须完全一致（53 维契约），否则喂给网络的是错数据。
def quat_to_gravity(q):
    """体坐标系下的重力方向 = -R[2,:]，R 是体→世界的旋转矩阵。"""
    w, x, y, z = q
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return -np.array([r20, r21, r22], dtype=np.float32)


def quat_to_yaw(q):
    """从四元数提取偏航角（绕 z 轴）。用 atan2 而非 asin，全角度范围都稳。"""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_one(sess, in_name, m, c, cmd, steps, floor_id, friction,
            kp_scale=1.0, mass_scale=1.0, base=None, idx=None):
    """在给定扰动下跑一段，返回统计。每次调用都从 keyframe 重新开始。

    base 是"未扰动"的原始参数快照（dict），用来做倍数缩放。
    每次都从 base 重新算，避免多次调用之间累积误差。
    """
    d = mujoco.MjData(m)
    m.geom_friction[floor_id, 0] = friction      # 改地面滑动摩擦

    if base is not None:
        # 舵机 kp：必须同时改 gainprm[0] 和 biasprm[1] = -kp。
        # position 执行器的力 = kp·ctrl - kp·q - kv·qd，
        # 只改一个会变成"只有前馈没有反馈"，力随 ctrl 线性增长，鸭子会飞出去。
        kp = base["gainprm"] * kp_scale
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        # 质量：整体缩放（floor 的 mass 是 0，乘完还是 0）
        m.body_mass[:] = base["mass"] * mass_scale
    # 必须显式加载 keyframe：MjData(m) 初始化的是 model.qpos0（建模悬空高度 0.22），
    # 不是 keyframe 里的蹲姿（0.15）。不加载的话鸭子会从半空落下，整段结果全废。
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # 关节位置用「按 actuator 顺序的索引」取，不能用固定的 qpos[7:7+14]。
    # 原因：只要模型在关节链里插入任何额外自由度（比如把每个关节拆成
    # "主关节 + 限位小关节"来模拟齿轮间隙），qpos 就会变成交错排列，
    # 固定切片会静默取错——不报错，但观测全乱。按 actuator 查地址则永远正确。
    qpos_idx, dof_idx = idx
    default_pos = np.array(d.qpos[qpos_idx], dtype=np.float32)
    d.ctrl[:] = default_pos                      # 伺服目标 = 当前姿态，无瞬态冲击
    mujoco.mj_forward(m, d)

    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = quat_to_yaw(d.qpos[3:7])
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    heights = []
    for t in range(steps):
        gyro = np.asarray(d.sensordata[0:3], dtype=np.float32) * c.obs_scale_gyro
        grav = quat_to_gravity(d.qpos[3:7])
        q = np.asarray(d.qpos[qpos_idx], dtype=np.float32) - default_pos
        qd = np.asarray(d.qvel[dof_idx], dtype=np.float32) * c.obs_scale_qd
        dyaw = quat_to_yaw(d.qpos[3:7]) - yaw0
        heading = np.array([np.sin(dyaw), np.cos(dyaw)], dtype=np.float32)

        obs = np.concatenate([gyro, grav, cmd, q, qd, last_action,
                              heading]).astype(np.float32)[None]
        action = sess.run(None, {in_name: obs})[0].reshape(-1)[:NUM_JOINTS]
        last_action = action.astype(np.float32)

        # 动作映射：默认姿态 + 缩放后的网络输出，再逐关节剪裁
        ctrl = default_pos + action * c.action_scale
        d.ctrl[:] = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

        # 10 个物理子步 = 一个控制周期（策略 50 Hz，物理 500 Hz）
        for _ in range(int(round(c.ctrl_dt / m.opt.timestep))):
            mujoco.mj_step(m, d)

        heights.append(float(d.qpos[2]))
        if float(d.qpos[2]) < c.min_height:      # 摔了就直接结束
            break

    return {
        "dx": float(d.qpos[0]) - x0,
        "dy": float(d.qpos[1]) - y0,
        "dyaw": np.degrees(quat_to_yaw(d.qpos[3:7]) - yaw0),
        "h_min": min(heights),
        "h_end": float(d.qpos[2]),
        "steps": len(heights),
        "fell": len(heights) < steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", nargs="+", required=True, help="一个或多个 ONNX，逐个对比")
    p.add_argument("--friction", nargs="+", type=float, default=[0.6],
                   help="地面滑动摩擦，可给多档（0.6 是训练基准）")
    p.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                   help="指令 [vx, vy, wz]，默认前进 0.4 m/s")
    p.add_argument("--steps", type=int, default=250, help="每档跑多少控制步（250 = 5 秒）")
    p.add_argument("--kp-scale", nargs="+", type=float, default=None,
                   help="舵机刚度倍数，可给多档（如 0.8 1.0 1.2）。"
                        "给了它就扫 kp，忽略 --friction 的多档（只用第一个摩擦值）")
    p.add_argument("--mass-scale", nargs="+", type=float, default=None,
                   help="整机质量倍数，可给多档（如 0.9 1.0 1.1）")
    p.add_argument("--force-kp", type=float, default=None,
                   help="强制把 kp 设成这个值，忽略模型自带的。用于隔离单一变量："
                        "换模型时若自带 kp 与训练值不同，不统一就没法判断差异来自哪个变量")
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--iterations", type=int, default=50,
                   help="CPU 求解器迭代数。MJCF 里是 1（给 MJX 用的），CPU 版必须调大")
    args = p.parse_args()

    c = cfg.default_config()
    m = mujoco.MjModel.from_xml_path(args.xml)
    m.opt.iterations = args.iterations
    m.opt.ls_iterations = args.iterations

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        sys.exit("模型里没有 floor geom，确认 xml 路径")
    base_friction = float(m.geom_friction[floor_id, 0])

    # 按 actuator 顺序建立 qpos/qvel 索引。
    # actuator_trnid[:,0] 给出每个执行器绑定的关节 id，顺序与 ctrl 完全一致。
    # 这样无论模型里插了多少额外自由度（比如 backlash），取出的 14 个数都对得上。
    act_jnt_ids = np.asarray(m.actuator_trnid[:, 0], dtype=int)
    qpos_idx = np.asarray(m.jnt_qposadr[act_jnt_ids], dtype=int)
    dof_idx = np.asarray(m.jnt_dofadr[act_jnt_ids], dtype=int)
    idx = (qpos_idx, dof_idx)

    n_extra = m.nq - (7 + NUM_JOINTS)
    if n_extra > 0:
        print(f"[robust] 注意：模型有 {m.nq} 维 qpos，比基础版多 {n_extra} 个自由度，"
              f"已按 actuator 顺序索引，观测不会错位")

    cmd = np.array(args.cmd, dtype=np.float32)
    theory = np.linalg.norm(cmd) * args.steps * c.ctrl_dt   # 理论位移（m）

    # --force-kp：把 kp 统一到指定值，隔离变量。
    # 必须放在取 base 快照之前，否则 base 里还是模型原值。
    if args.force_kp is not None:
        orig_kp = float(m.actuator_gainprm[0, 0])
        m.actuator_gainprm[:, 0] = args.force_kp
        m.actuator_biasprm[:, 1] = -args.force_kp   # 必须同步，否则伺服行为错
        print(f"[robust] 强制 kp = {args.force_kp:.2f}（模型自带 {orig_kp:.2f}）")

    # 未扰动参数的快照。每次都用它重算，避免多档扫描之间累积误差。
    base = {
        "gainprm": np.array(m.actuator_gainprm[:, 0], dtype=np.float64),
        "mass": np.array(m.body_mass, dtype=np.float64),
    }

    # ---- 决定这一轮扫哪个维度（一次只扫一个，否则会变成笛卡尔积）----
    if args.kp_scale:
        sweep, vals, key = "kp 倍数", args.kp_scale, "kp"
    elif args.mass_scale:
        sweep, vals, key = "质量倍数", args.mass_scale, "mass"
    else:
        sweep, vals, key = "摩擦", args.friction, "friction"

    print(f"[robust] xml={args.xml}  基准：摩擦={base_friction:.2f}  "
          f"kp={base['gainprm'][0]:.2f}  质量={base['mass'].sum():.3f} kg"
          f"  qpos={m.nq} nv={m.nv}")
    print(f"[robust] 扫描维度 = {sweep}，档位 {vals}")
    print(f"[robust] 每档 {args.steps} 步 = {args.steps * c.ctrl_dt:.1f}s，"
          f"指令 {cmd.tolist()} → 理论位移 {theory:.2f} m\n")

    rows = []
    for onnx_path in args.onnx:
        if not os.path.exists(onnx_path):
            sys.exit(f"找不到 {onnx_path}")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        n_in = sess.get_inputs()[0].shape[1]
        if n_in != 53:
            sys.exit(f"{onnx_path} 输入是 {n_in} 维，本脚本只支持 53 维（含 yaw）")

        name = os.path.basename(onnx_path).replace(".onnx", "")
        cells = []
        for v in vals:
            kw = {"kp_scale": 1.0, "mass_scale": 1.0}
            fric = base_friction
            if key == "friction":
                fric = v
            elif key == "kp":
                kw["kp_scale"] = v
            elif key == "mass":
                kw["mass_scale"] = v

            r = run_one(sess, in_name, m, c, cmd, args.steps, floor_id, fric,
                        base=base, idx=idx, **kw)
            cells.append(r)
            flag = "  ← 摔了" if r["fell"] else ""
            print(f"  {name:<18} {sweep} {v:.2f} → "
                  f"前进 {r['dx']:+.2f} m  侧 {r['dy']:+.2f}  偏航 {r['dyaw']:+6.1f}°  "
                  f"最低高度 {r['h_min']:.3f}  跑完 {r['steps']}/{args.steps}{flag}")
        rows.append((name, cells))
        # 复位，别影响下一个策略
        m.geom_friction[floor_id, 0] = base_friction
        m.actuator_gainprm[:, 0] = base["gainprm"]
        m.actuator_biasprm[:, 1] = -base["gainprm"]
        m.body_mass[:] = base["mass"]

    # ------------------------------------------------------------------ 汇总
    print("\n" + "=" * 78)
    header = "  " + f"{'策略':<20}" + "".join(f"{sweep.split()[0]}{v:<7.2f}" for v in vals)

    print(f"达成率（前进距离 / 理论位移）")
    print(header)
    for name, cells in rows:
        line = f"  {name:<20}"
        for r in cells:
            rate = r["dx"] / theory * 100 if theory > 1e-9 else 0.0
            mark = "✗" if r["fell"] else ("✓" if rate >= 80 else "⚠")
            line += f"{rate:6.1f}%{mark:<2}"
        print(line)

    print("\n最低高度（低于 0.10 就是摔了）")
    print(header)
    for name, cells in rows:
        line = f"  {name:<20}"
        for r in cells:
            line += f"{r['h_min']:7.3f}  "
        print(line)

    if len(vals) > 1:
        print(f"\n跨{sweep}的稳定性（达成率极差，越小越鲁棒）")
        for name, cells in rows:
            rates = [r["dx"] / theory * 100 if theory > 1e-9 else 0.0 for r in cells]
            fell = sum(1 for r in cells if r["fell"])
            print(f"  {name:<20} 最好 {max(rates):5.1f}%  最差 {min(rates):5.1f}%  "
                  f"极差 {max(rates) - min(rates):5.1f}  "
                  f"摔了 {fell}/{len(cells)} 档")

    print("\n判读：")
    if key == "friction":
        print("  摩擦对鸭子影响小是正常的（足底平面接触、低速、不靠蹬地推进）")
        print("  更值得测的是 --kp-scale 和动作延迟——那才是 sim2real 的主要 gap")
    else:
        print("  dr 版在偏离 1.0 的档位上更稳 → 域随机化确实生效，这就是要的证据")
        print("  两者都差不多 → 这一项对鸭子不敏感，换别的维度测")

if __name__ == "__main__":
    main()
