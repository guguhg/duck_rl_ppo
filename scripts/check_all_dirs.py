"""六个方向一次测完：前进 / 后退 / 左移 / 右移 / 左转 / 右转。

为什么单独写这个：单方向跑一次要盯十几行输出，六个方向就得复制六次命令、还要人
脑合并结论。这里一次跑完，最后给一张表，哪个方向没学会一眼就能看出来。

判定用的是"达成率 = 实测 / 理论"（同号时），因为不同方向的量纲不同：
位移看米，转动看度。

用法：
    uv run python -u scripts/check_all_dirs.py --onnx checkpoints/policy.onnx
"""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

from duck_rl.env import DuckEnv

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

# (名称, 指令[vx,vy,wz], 用哪个量判定, 该量的理论速率)
DIRS = [
    ("前进", [0.4, 0.0, 0.0], "x", 0.4),
    ("后退", [-0.3, 0.0, 0.0], "x", -0.3),
    ("左移", [0.0, 0.3, 0.0], "y", 0.3),
    ("右移", [0.0, -0.3, 0.0], "y", -0.3),
    ("左转", [0.0, 0.0, 0.6], "yaw", 0.6),
    ("右转", [0.0, 0.0, -0.6], "yaw", -0.6),
]


def _yaw(q):
    """四元数 → 偏航角弧度。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_one(env, step_fn, sess, in_name, cmd, steps):
    """跑一个方向，返回 (dx, dy, 累计偏航度, 最低高度)。"""
    state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(info={**state.info,
                                "command": jnp.asarray(cmd, dtype=jnp.float32)})
    x0 = float(state.data.qpos[0])
    y0 = float(state.data.qpos[1])
    yaw_prev = _yaw(state.data.qpos[3:7])
    yaw_total = 0.0
    h_min = 9.9

    for i in range(steps):
        obs = np.asarray(state.obs, dtype=np.float32)[None]
        action = sess.run(None, {in_name: obs})[0].reshape(-1)
        state = step_fn(state, jnp.asarray(action, dtype=jnp.float32))
        h_min = min(h_min, float(state.data.qpos[2]))
        if (i + 1) % 25 == 0:                     # 每 0.5 s 累加一次偏航，避免跨圈跳变
            cur = _yaw(state.data.qpos[3:7])
            d = (cur - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_total += d
            yaw_prev = cur

    return (float(state.data.qpos[0]) - x0,
            float(state.data.qpos[1]) - y0,
            float(np.degrees(yaw_total)),
            h_min)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--steps", type=int, default=250, help="每个方向跑多少步（250 步 = 5 秒）")
    args = p.parse_args()

    env = DuckEnv()
    step_fn = jax.jit(env.step)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print(f"[all] obs={env.observation_size} 每方向 {args.steps} 步 "
          f"= {args.steps * env.dt:.1f}s", flush=True)
    print("[all] 首次含编译，约 60~90 秒；之后每个方向几秒\n", flush=True)

    T = args.steps * env.dt
    rows = []
    t0 = time.time()
    for name, cmd, axis, rate in DIRS:
        dx, dy, dyaw, hmin = run_one(env, step_fn, sess, in_name, cmd, args.steps)

        if axis == "x":
            actual, expect, unit = dx, rate * T, "m"
        elif axis == "y":
            actual, expect, unit = dy, rate * T, "m"
        else:
            actual, expect, unit = dyaw, np.degrees(rate * T), "°"

        ratio = actual / expect if abs(expect) > 1e-6 else 0.0
        mark = "✅" if ratio >= 0.8 else ("⚠️" if ratio >= 0.4 else "❌")
        rows.append((name, actual, expect, ratio, mark, hmin, dx, dy, dyaw))
        print(f"  {name}  实测 {actual:+8.2f}{unit}  理论 {expect:+8.2f}{unit}  "
              f"达成 {ratio * 100:5.1f}%  {mark}", flush=True)

    print(f"\n[all] 总耗时 {time.time() - t0:.1f}s\n")
    print("方向  实测      理论      达成率  判定   最低高度   前(m)    侧(m)   偏航(°)")
    for name, actual, expect, ratio, mark, hmin, dx, dy, dyaw in rows:
        print(f"{name}  {actual:+8.2f}  {expect:+8.2f}  {ratio * 100:6.1f}%  {mark:2s}  "
              f"  {hmin:.3f}   {dx:+6.3f}  {dy:+6.3f}  {dyaw:+7.1f}")

    print("\n判读：")
    print("  达成率 ≥ 80% → 学会了；40~80% → 会一点；< 40% → 没学会")
    print("  前后看「前」列，左右看「侧」列，转向看「偏航」列")
    print("  最低高度 < 0.10 的方向说明它在那个方向会摔")


if __name__ == "__main__":
    main()
