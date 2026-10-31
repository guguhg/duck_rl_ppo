#!/usr/bin/env python3
"""MJCF 模型自检：把一个陌生的 MuJoCo 模型拆开，逐项打印出来。

用法：
    uv run python scripts/check_model.py                       # 默认 assets/scene_flat_terrain.xml
    uv run python scripts/check_model.py assets/xxx.xml

设计意图：读 MJCF 原文时形成"我猜这模型是这样"的印象，用这个脚本核对。
换任何机器人模型（自己的、别人的 URDF 转的）都应该先跑一遍。

打印顺序刻意和读 XML 的顺序一致：
    规模 → body 树 → 关节 → 执行器 → 初始姿态 → 碰撞体 → 传感器
"""
import sys

import mujoco
import numpy as np


# ---------------------------------------------------------------- 小工具
def name_of(m, obj_type, i):
    """拿名字。自动生成的文件里偶尔有匿名元素，返回 None，统一转成占位串。"""
    n = mujoco.mj_id2name(m, obj_type, i)
    return n if n else f"<未命名{i}>"


def body_depth(m, i):
    """算 body 在树里的深度：往上数父节点直到 world（id=0）。用于缩进显示。"""
    d = 0
    while m.body_parentid[i] != 0:
        i = m.body_parentid[i]
        d += 1
    return d


# ---------------------------------------------------------------- 各段落
def print_overview(m, path):
    print(f"\n=== {path} ===")
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}  ngeom={m.ngeom}  nkey={m.nkey}")
    print(f"总质量 = {m.body_mass.sum():.4f} kg      物理步长 dt = {m.opt.timestep} s")
    print(f"重力 = {m.opt.gravity}")
    # nq/nv 的构成：有浮基时 7/6，其余是关节自由度
    print("提示：nq = 浮基7 + 关节数，nv = 浮基6 + 关节数（无浮基时不含这 7/6）")


def print_mass_com(m):
    """质量分布与质心高度。质量堆在高处 = 平衡更难，这是"能不能走"的物理前提。
    很多人只看总质量，但双足的难点是质心位置，不是重量。"""
    print("\n--- 质量分布（最重的 5 个 body）---")
    total = m.body_mass.sum()
    for i in np.argsort(-m.body_mass)[:5]:
        if m.body_mass[i] <= 0:
            continue
        print(f"  {name_of(m, mujoco.mjtObj.mjOBJ_BODY, i):<28} "
              f"{m.body_mass[i]:.5f} kg  ({100 * m.body_mass[i] / total:4.1f}%)")

    if m.nkey:
        # 用 keyframe 0 摆好姿态，正运动学算出整机质心
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        com = d.subtree_com[0]        # subtree_com[0] = 以 world 为根的整机质心
        print(f"\n  初始姿态整机质心 = {np.round(com, 4)}    质心高度 = {com[2]:.4f} m")
        print(f"  浮基质点高度 = {d.qpos[2]:.4f} m  → 质心相对躯干原点 {com[2] - d.qpos[2]:+.4f} m")
        print("  → 质心越高越难平衡；头/电池装错位置会让 sim2real 直接崩")


def print_body_tree(m):
    """打印 body 树。这是"运动链"，决定了 qpos 里的关节按什么顺序排列。"""
    print("\n--- body 树（缩进 = 层级；[joint] 是该 body 上挂的自由度）---")
    for i in range(m.nbody):
        if i == 0:
            print(f"{name_of(m, mujoco.mjtObj.mjOBJ_BODY, i)}   (world)")
            continue
        indent = "  " * body_depth(m, i)
        # 这个 body 上挂了哪些 joint：起始索引 + 个数
        jadr, jnum = m.body_jntadr[i], m.body_jntnum[i]
        joints = ""
        if jnum > 0:
            names = [name_of(m, mujoco.mjtObj.mjOBJ_JOINT, jadr + k) for k in range(jnum)]
            joints = f"   [{', '.join(names)}]"
        print(f"{indent}└ {name_of(m, mujoco.mjtObj.mjOBJ_BODY, i):<28s} "
              f"mass={m.body_mass[i]:.5f}{joints}")


def print_joints(m):
    """关节表：类型、限位、轴、在 qpos 里的位置。
    限位宽度排序能立刻回答"哪个关节最受限"。"""
    print("\n--- 关节 ---")
    print(f"{'id':>3} {'名称':<20} {'类型':<12} {'qposadr':>7} {'限位范围':<26} {'宽度':>6}  轴")
    rows = []
    for j in range(m.njnt):
        jt = mujoco.mjtJoint(m.jnt_type[j]).name.replace("mjJNT_", "")
        if jt == "FREE":  # 浮基没有限位，单独说明
            print(f"{j:>3} {name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j):<20} {jt:<12} "
                  f"{m.jnt_qposadr[j]:>7} {'—':<26} {'—':>6}  —")
            continue
        lo, hi = m.jnt_range[j]
        width = hi - lo
        ax = np.round(m.jnt_axis[j], 3)
        print(f"{j:>3} {name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j):<20} {jt:<12} "
              f"{m.jnt_qposadr[j]:>7} [{lo:+.4f}, {hi:+.4f}]      {width:>6.3f}  {ax}")
        rows.append((width, j))

    if rows:
        width, j = min(rows)
        print(f"\n限位最窄的关节：{name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j)}（宽度 {width:.3f} rad "
              f"= {np.degrees(width):.1f}°）")
        print("  → 最窄的通常是髋部 roll 或 yaw（限制侧向摆动），踝/膝一般有 ±90°。"
              "以实际数据为准，不要预设")


def print_actuators(m):
    """执行器表。重点是把 ctrlrange 和它绑定关节的 jnt_range 并排打，
    一眼就能看出 inheritrange="1" 有没有生效。"""
    print("\n--- 执行器 ---")
    print(f"{'id':>3} {'名称':<20} {'绑定关节':<20} {'kp':>7} {'kv':>7} "
          f"{'力矩上限':>10}  {'ctrl_range':<22} 与关节限位一致?")
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]            # 该执行器驱动的 joint id
        jname = name_of(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        kp = m.actuator_gainprm[a, 0]           # position 伺服：gainprm[0] 就是 kp
        kv = -m.actuator_biasprm[a, 2]          # biasprm[2] = -kv，取负还原
        flo, fhi = m.actuator_forcerange[a]
        clo, chi = m.actuator_ctrlrange[a]
        jlo, jhi = m.jnt_range[jid]
        same = np.allclose([clo, chi], [jlo, jhi], atol=1e-6)
        print(f"{a:>3} {name_of(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<20} {jname:<20} "
              f"{kp:>7.2f} {kv:>7.2f} [{flo:+.2f},{fhi:+.2f}]  "
              f"[{clo:+.3f},{chi:+.3f}]     {'✓ 是' if same else '✗ 否'}")
    print("\n  kp：位置刚度；kv：速度反馈（biasprm[2] 取负还原）")
    print("  '与关节限位一致'全为 ✓ → inheritrange=\"1\" 生效，env 里不必再设 ctrlrange")


def print_keyframe(m):
    """初始姿态。这是 reset 的唯一依据，也是判断"蹲姿还是站姿"的地方。"""
    print("\n--- keyframe（初始姿态）---")
    if m.nkey == 0:
        print("  [warn] 没有关键帧！说明你加载的是本体而不是 scene，reset 拿不到初始姿态")
        return
    for k in range(m.nkey):
        q = m.key_qpos[k]
        print(f"  [{k}] name={name_of(m, mujoco.mjtObj.mjOBJ_KEY, k)}  "
              f"qpos 维度={len(q)}")
        print(f"      浮基位置 = {np.round(q[:3], 4)}   （z 就是初始躯干高度）")
        print(f"      浮基四元数 = {np.round(q[3:7], 3)}")
        # 关节角：从第 7 位往后，按 joint 声明顺序映射
        print(f"      {'关节':<20} {'弧度':>8} {'角度':>9}")
        knee_max = 0.0
        for j in range(m.njnt):
            if mujoco.mjtJoint(m.jnt_type[j]).name.endswith("FREE"):
                continue
            adr = m.jnt_qposadr[j]
            v = float(q[adr])
            jn = name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"      {jn:<20} {v:>+8.4f} {np.degrees(v):>+8.1f}°")
            if "knee" in jn:
                knee_max = max(knee_max, abs(v))
        # 蹲姿判据：膝关节角度足够大就是蹲着
        if knee_max > 1.0:
            print(f"      → 膝关节最大 {knee_max:.3f} rad（{np.degrees(knee_max):.1f}°）"
                  f"，判定为【蹲姿】")
        else:
            print(f"      → 膝关节最大 {knee_max:.3f} rad，更接近【站姿】，注意确认是否符合预期")

        # ctrl 与 qpos 的关节部分是否相同：相同则说明 reset 时无瞬态冲击
        if m.key_ctrl.size:
            c = m.key_ctrl[k]
            same = np.allclose(c, q[7:7 + len(c)], atol=1e-6)
            print(f"      ctrl 与 qpos 的关节部分一致: {'是（reset 无瞬态冲击）' if same else '否（注意）'}")


def print_geoms(m):
    """几何统计。绝大多数是视觉件，真正参与碰撞的很少。"""
    print("\n--- geom 统计 ---")
    types = {}
    for g in range(m.ngeom):
        t = mujoco.mjtGeom(m.geom_type[g]).name.replace("mjGEOM_", "")
        types[t] = types.get(t, 0) + 1
    for t, n in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<12} {n}")

    # 参与碰撞的：contype 或 conaffinity 非 0
    colliding = [g for g in range(m.ngeom)
                 if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0]
    print(f"\n  参与碰撞的 geom：{len(colliding)} / {m.ngeom}")
    for g in colliding:
        n = name_of(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        print(f"    - {n:<28} contype={m.geom_contype[g]} "
              f"conaffinity={m.geom_conaffinity[g]} friction={np.round(m.geom_friction[g], 3)}")
    print("  → 只有这些算接触；域随机化改地面摩擦要去 scene 文件的 floor geom")


def print_sensors(m):
    """传感器清单。env 写观测时优先用它们，比自己从四元数推更不易错。"""
    print("\n--- 传感器 ---")
    if m.nsensor == 0:
        print("  无")
        return
    print(f"  {'id':>3} {'名称':<26} {'adr':>5} {'dim':>4}")
    for s in range(m.nsensor):
        print(f"  {s:>3} {name_of(m, mujoco.mjtObj.mjOBJ_SENSOR, s):<26} "
              f"{m.sensor_adr[s]:>5} {m.sensor_dim[s]:>4}")
    print("  → data.sensordata[m.sensor_adr[s] : +sensor_dim[s]] 就是该传感器的读数")


# ---------------------------------------------------------------- 主流程
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/scene_flat_terrain.xml"
    m = mujoco.MjModel.from_xml_path(path)

    print_overview(m, path)
    print_mass_com(m)
    print_body_tree(m)
    print_joints(m)
    print_actuators(m)
    print_keyframe(m)
    print_geoms(m)
    print_sensors(m)

    print("\n检查完这五样再动手写 env：总质量、关节限位、初始姿态、执行器五参数、碰撞体。")


if __name__ == "__main__":
    main()
