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
    heights, xs = [h0], [float(state.data.qpos[0])]
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
        if (i + 1) % 50 == 0:
            print(f"  t={(i + 1) * env.dt:5.1f}s 高度 {heights[-1]:.3f} "
                  f"前进 {xs[-1] - xs[0]:+.3f} m done {float(state.done):.0f}")

    h = np.array(heights)
    print(f"\n[mjx] 高度 起 {h[0]:.3f} → 末 {h[-1]:.3f}（最低 {h.min():.3f}）")
    print(f"[mjx] 前进 {xs[-1] - xs[0]:+.3f} m / {args.steps * env.dt:.1f}s "
          f"→ 平均 {(xs[-1] - xs[0]) / (args.steps * env.dt):+.2f} m/s")
    print(f"[mjx] 用时 {time.time() - t0:.1f}s（未 jit，仅供对比不看速度）")
    print("\n判读：")
    print("  最低高度 > 0.10 且前进量明显 → 策略在 MJX 里是好的，问题出在 CPU 回放")
    print("  最低高度 < 0.05           → 策略本身不行，得继续训或调奖励")


if __name__ == "__main__":
    main()
