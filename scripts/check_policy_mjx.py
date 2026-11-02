"""用 ONNX 策略在 MJX 里跑（和训练同一个物理引擎）。

用途：把「策略本身好不好」和「MJX 与 CPU MuJoCo 的差异」分开。
  - MJX 里也倒  → 策略/观测有问题
  - MJX 里能站  → 是回放时的物理差异（CPU MuJoCo 配置或碰撞体）

观测直接用 env._get_obs 拼，所以和训练时逐位一致——这一步排除了手写拼装出错的可能。

用法：
    uv run python scripts/check_policy_mjx.py --onnx checkpoints/policy.onnx
"""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

from duck_rl.env import DuckEnv

# 编译缓存：同样的图第二次运行不用再编
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def _yaw(q):
    """四元数 (w,x,y,z) → 偏航角弧度。与 env.yaw_of 同一公式，这里要 float。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0])
    args = p.parse_args()

    env = DuckEnv()
    step_fn = jax.jit(env.step)          # 不 jit 每步几十次派发，会慢到以为卡死
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    print("[mjx] reset（首次含编译）...", flush=True)
    state = env.reset(jax.random.PRNGKey(0))
    jax.block_until_ready(state.obs)
    # 覆盖成固定指令，和控制台观测对齐
    state = state.replace(info={**state.info,
                                "command": jnp.asarray(args.cmd, dtype=jnp.float32)})
    print(f"[mjx] obs={env.observation_size} cmd={args.cmd} n_substeps={env.n_substeps}",
          flush=True)
    print(f"[mjx] 开始 {args.steps} 步：第一步含编译，可能要 60~90 秒，之后很快",
          flush=True)

    h0 = float(state.data.qpos[2])
    heights = [h0]
    xs = [float(state.data.qpos[0])]
    ys = [float(state.data.qpos[1])]          # 侧移看的是 y，不是 x
    # 累计偏航：yaw 本身在 ±180° 会跳变，所以按增量累加，才能算出"转了几圈"
    yaw_prev = _yaw(state.data.qpos[3:7])
    yaw_total = 0.0
    t0 = time.time()
    for i in range(args.steps):
        obs = np.asarray(state.obs, dtype=np.float32)[None]
        action = sess.run(None, {in_name: obs})[0].reshape(-1)
        state = step_fn(state, jnp.asarray(action, dtype=jnp.float32))
        if i == 0:
            jax.block_until_ready(state.obs)
            print(f"  [第一步完成] 高度 {float(state.data.qpos[2]):.3f}", flush=True)
        heights.append(float(state.data.qpos[2]))
        xs.append(float(state.data.qpos[0]))
        ys.append(float(state.data.qpos[1]))
        if (i + 1) % 50 == 0:
            cur = _yaw(state.data.qpos[3:7])
            d = (cur - yaw_prev + np.pi) % (2 * np.pi) - np.pi   # 归一到 ±π，避免跨圈跳变
            yaw_total += d
            yaw_prev = cur
            print(f"  t={(i + 1) * env.dt:5.1f}s 高度 {heights[-1]:.3f} "
                  f"前 {xs[-1] - xs[0]:+.3f} 侧 {ys[-1] - ys[0]:+.3f} "
                  f"偏航 {np.degrees(yaw_total):+7.1f}° done {float(state.done):.0f}")

    h = np.array(heights)
    t_total = args.steps * env.dt
    dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
    dist = float(np.hypot(dx, dy))
    print(f"\n[mjx] 高度 起 {h[0]:.3f} → 末 {h[-1]:.3f}（最低 {h.min():.3f}）")
    print(f"[mjx] 位移 前 {dx:+.3f} m / 侧 {dy:+.3f} m / 直线距离 {dist:.3f} m"
          f"　（{t_total:.1f}s → 平均 {dist / t_total:+.2f} m/s）")
    print(f"[mjx] 指令 [{args.cmd[0]:+.2f}, {args.cmd[1]:+.2f}, {args.cmd[2]:+.2f}]"
          f" → 理论位移 前 {args.cmd[0] * t_total:+.2f} / 侧 {args.cmd[1] * t_total:+.2f} m")
    print(f"[mjx] 累计偏航 {np.degrees(yaw_total):+.1f}°"
          f"　（指令 wz={args.cmd[2]:+.2f} rad/s × {t_total:.1f}s = "
          f"{np.degrees(args.cmd[2] * t_total):+.0f}°）")
    print(f"[mjx] 用时 {time.time() - t0:.1f}s（未 jit，仅供对比不看速度）")
    print("\n判读：")
    print("  高度最低 > 0.10 且位移明显 → 策略在 MJX 里是好的，问题出在 CPU 回放")
    print("  高度最低 < 0.05          → 策略本身不行，得继续训或调奖励")
    print("  位移方向与指令同向、量级接近 → 这条指令学会了")
    print("  指令要求侧移但「侧」接近 0 → 侧移没学会（鸭子横向移动本来就难，属正常）")


if __name__ == "__main__":
    main()
