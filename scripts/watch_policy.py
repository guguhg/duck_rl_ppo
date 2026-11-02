"""用导出的 ONNX 策略在 CPU MuJoCo 里回放——终于能看到鸭子在走。

刻意走 ONNX 而不是 JAX：这条路径和真机部署完全一致，等于顺手验证了部署契约。

用法：
    # 出视频（不需要显示器，WSL 里用 egl 离屏渲染）
    MUJOCO_GL=egl uv run python scripts/watch_policy.py --onnx checkpoints/policy.onnx --video out.mp4

    # 开窗口实时看（WSLg）
    uv run python scripts/watch_policy.py --onnx checkpoints/policy.onnx --viewer
"""
import argparse
import os
import sys
import time

import mediapy as media
import mujoco
import mujoco.viewer          # 子模块，必须单独 import，否则 mujoco.viewer 不存在
import numpy as np
import onnxruntime as ort

from duck_rl import config as cfg
from duck_rl.constants import NUM_JOINTS, OBS_SIZE


def quat_to_gravity(q):
    """体坐标系下的重力方向。-R[2,:]，R 是体→世界的旋转矩阵。"""
    w, x, y, z = q
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return -np.array([r20, r21, r22], dtype=np.float32)


def quat_to_yaw(q):
    """四元数 → 偏航角（绕世界 z 轴，-π~π）。与 env.yaw_of 同一套公式。"""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--xml", default="assets/scene_flat_terrain.xml")
    p.add_argument("--steps", type=int, default=500, help="控制步数（每步 20 ms）")
    p.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                   help="指令 [vx, vy, wz]")
    p.add_argument("--video", default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--iterations", type=int, default=50,
                   help="求解器迭代次数。MJCF 里是 1（为 MJX 提速），"
                        "CPU MuJoCo 用 1 会穿模，必须调大")
    p.add_argument("--hold", action="store_true",
                   help="跑完后保持窗口不关，可以拖着看。否则 --viewer 会随脚本结束而关窗")
    p.add_argument("--cam", choices=["follow", "tracking", "fixed"], default="follow",
                   help="follow=跟位置不跟朝向（默认，能看出转向）；"
                        "tracking=MuJoCo 原生跟踪，会跟着鸭子转身，看着像原地踏步；"
                        "fixed=完全不动")
    p.add_argument("--cam-azimuth", type=float, default=120,
                   help="相机方位角：0=正前，90=正侧，180=正后")
    p.add_argument("--cam-distance", type=float, default=1.0,
                   help="相机到躯干的距离（米）。--cam fixed 时调大能看到更远的位移")
    p.add_argument("--cam-elevation", type=float, default=-15,
                   help="相机俯仰：0=平视，-30=略俯视，-60=接近正上方鸟瞰")
    p.add_argument("--cam-follow-gain", type=float, default=0.85,
                   help="跟随强度 0~0.99：越接近 1 越'懒'（镜头更稳），越小越紧跟")
    p.add_argument("--speed", type=float, default=1.0,
                   help="播放倍速：1.0=实时，0.5=半速慢放，0=不限速（视频模式总是全速）")
    args = p.parse_args()

    c = cfg.default_config()
    m = mujoco.MjModel.from_xml_path(args.xml)
    # 关键：XML 里的 iterations="1" 是给 MJX 用的，CPU MuJoCo 必须加大。
    # 静立时 1 次迭代勉强够（之前 view_model.py 就是这么站住的），但策略一发力、
    # 接触变复杂就解不出来，表现为躯干沉到地面以下（高度变负）。
    m.opt.iterations = args.iterations
    m.opt.ls_iterations = args.iterations
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    default_pos = np.array(d.qpos[7:7 + NUM_JOINTS], dtype=np.float32)
    yaw0 = quat_to_yaw(d.qpos[3:7])        # 起始航向：观测里的相对偏航以它为基准
    ctrl_lo = m.actuator_ctrlrange[:, 0].astype(np.float32)
    ctrl_hi = m.actuator_ctrlrange[:, 1].astype(np.float32)

    # 按名字找 gyro 的切片，和 env 里保持同一套做法
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
    g_adr, g_dim = m.sensor_adr[gid], m.sensor_dim[gid]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_dim = int(sess.get_inputs()[0].shape[-1])
    print(f"[onnx] 输入 {sess.get_inputs()[0].shape} → 输出 {sess.get_outputs()[0].shape}")

    # 自动适配新旧契约，免得拿旧 onnx 配新代码直接崩：
    #   53 维 = 带 yaw（当前契约）
    #   51 维 = 加 yaw 之前导出的旧模型
    use_yaw = in_dim == OBS_SIZE
    if in_dim == OBS_SIZE - 2:
        print("[onnx] 注意：这是 51 维的旧模型（没有 yaw 观测）。它会走歪，"
              "那是已知缺陷，不是脚本的问题。", flush=True)
    elif not use_yaw:
        raise SystemExit(
            f"[onnx] 输入 {in_dim} 维，既不是 {OBS_SIZE}（当前）也不是 {OBS_SIZE - 2}（旧），"
            f"和观测契约对不上。"
        )

    cmd = np.array(args.cmd, dtype=np.float32)
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    n_sub = int(round(c.ctrl_dt / c.sim_dt))

    # 相机。关键区别：
    #   TRACKING 模式下 azimuth 是**相对躯干朝向**的，鸭子一转身镜头也跟着转，
    #   于是画面里永远在"向前走"，看起来像原地踏步（这是踩过的坑）。
    #   FREE 模式 + 每步手动更新 lookat：只跟位置、不跟朝向，转向看得清清楚楚。
    trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly")
    cam = mujoco.MjvCamera()
    cam.trackbodyid = trunk_id
    cam.distance = args.cam_distance       # 相机到躯干的距离（米）
    cam.azimuth = args.cam_azimuth         # 侧后方，能同时看清前进方向和腿
    cam.elevation = args.cam_elevation     # 0=平视，负得越多越像从上往下看
    if args.cam == "tracking":
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.qpos[:3]         # 一开始对准鸭子

    renderer = mujoco.Renderer(m, height=360, width=480) if args.video else None
    frames = []
    x0 = float(d.qpos[0])
    heights = []

    viewer_ctx = mujoco.viewer.launch_passive(m, d) if args.viewer else None
    if viewer_ctx:
        viewer_ctx.__enter__()
        viewer_ctx.cam.type = cam.type          # 与上面同一套策略
        viewer_ctx.cam.trackbodyid = trunk_id
        viewer_ctx.cam.distance = cam.distance
        viewer_ctx.cam.azimuth = cam.azimuth
        viewer_ctx.cam.elevation = cam.elevation
        viewer_ctx.cam.lookat[:] = d.qpos[:3]

    t0 = time.time()
    ok = False          # 只有"正常跑完"才置 True；出错时不要进 --hold，否则像卡死
    try:
        for t in range(args.steps):
            # ---- 拼观测（顺序必须与 constants.OBS_LAYOUT 一致）----
            gyro = d.sensordata[g_adr:g_adr + g_dim] * c.obs_scale_gyro
            grav = quat_to_gravity(d.qpos[3:7])
            q = d.qpos[7:7 + NUM_JOINTS] - default_pos
            qd = d.qvel[6:6 + NUM_JOINTS] * c.obs_scale_qd
            # 相对航向两维（sin/cos），让策略知道自己偏了多少——不加它会走歪。
            # 旧模型（51 维）没有这两维，此时不拼，保持和它训练时一致。
            parts = [gyro, grav, cmd, q, qd, last_action]
            if use_yaw:
                dyaw = quat_to_yaw(d.qpos[3:7]) - yaw0
                parts.append(np.array([np.sin(dyaw), np.cos(dyaw)], dtype=np.float32))
            obs = np.concatenate(parts).astype(np.float32)[None]

            action = sess.run(None, {in_name: obs})[0].reshape(-1)[:NUM_JOINTS]
            last_action = action.astype(np.float32)

            d.ctrl[:] = np.clip(default_pos + action * c.action_scale, ctrl_lo, ctrl_hi)
            for _ in range(n_sub):
                mujoco.mj_step(m, d)

            heights.append(float(d.qpos[2]))

            # 相机跟随：只更新 lookat（位置），**不改 azimuth**，所以鸭子转身时
            # 画面不会跟着转。0.85 的一阶低通让镜头有惯性，看起来像跟拍而不是硬贴。
            if args.cam == "follow":
                cam.lookat[:] = args.cam_follow_gain * cam.lookat + (1 - args.cam_follow_gain) * d.qpos[:3]
                if viewer_ctx:
                    viewer_ctx.cam.lookat[:] = cam.lookat

            if renderer and t % 2 == 0:
                renderer.update_scene(d, camera=cam)   # 用跟随相机，否则出画
                frames.append(renderer.render())
            if viewer_ctx:
                viewer_ctx.sync()
                if not viewer_ctx.is_running():
                    break
                # 实时节流。不加的话物理跑多快画面就播多快（750 步 15 秒仿真 3 秒就完了）。
                # 按"这一帧应该出现在第几秒"来睡，误差不会累积。
                if args.speed > 0:
                    lag = (t + 1) * c.ctrl_dt / args.speed - (time.time() - t0)
                    if lag > 0:
                        time.sleep(lag)
            if (t + 1) % 50 == 0:
                print(f"  t={t * c.ctrl_dt:5.1f}s  高度 {heights[-1]:.3f}  "
                      f"前进 {float(d.qpos[0]) - x0:+.3f} m")
        ok = True                       # for 正常走完（或被 Esc 关窗 break），才算成功
    except KeyboardInterrupt:
        print("中断")
    finally:
        if viewer_ctx:
            # --hold：跑完后窗口不关，可以拖动着回看最后一帧。
            # 必须加 ok 判断——出错时若还停在这里等用户操作，表现为"卡死"。
            if args.hold and ok and viewer_ctx.is_running():
                print("\n[viewer] 已跑完，窗口保持中。左键拖动转视角；"
                      "按 Esc / 关窗，或 Ctrl+C 退出。")
                try:
                    while viewer_ctx.is_running():
                        viewer_ctx.sync()
                        time.sleep(0.02)
                except KeyboardInterrupt:
                    # 不在这里清理：MuJoCo 的 GL 上下文销毁在 WSLg 下会卡住，
                    # 见文件末尾那段 os._exit。直接跳出去。
                    print("\n[viewer] Ctrl+C，准备退出")
            viewer_ctx.__exit__(None, None, None)

    if heights:                     # 万一第一帧前就出错，heights 是空的，别去索引它
        h = np.array(heights)
        print(f"\n[统计] 高度 起 {h[0]:.3f} → 末 {h[-1]:.3f}（最低 {h.min():.3f}）")
        print(f"[统计] 前进 {float(d.qpos[0]) - x0:+.3f} m，用时 {args.steps * c.ctrl_dt:.1f}s "
              f"→ 平均 {((float(d.qpos[0]) - x0) / (args.steps * c.ctrl_dt)):+.2f} m/s"
              f"（指令 {args.cmd[0]:+.2f}）")

    if args.video and frames:
        media.write_video(args.video, frames, fps=int(1 / (c.ctrl_dt * 2)))
        print(f"[video] {len(frames)} 帧 → {args.video}")

    # MuJoCo 在解释器退出时会销毁 GL/EGL 上下文，WSLg 下这一步偶尔永久卡住
    # （症状：统计都打印了，但 shell 提示符不回来，Ctrl+C 也没用）。
    # flush 之后硬退出，跳过所有析构函数——这是最省事的绕法。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
