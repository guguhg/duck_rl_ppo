"""奖励验收：站着不动能拿多少分？七项各占多少？

判读基准（指令 vx=0.3、鸭子原地站着）：
    站着约 3.5~4.0 分；其中 alive 1.0、height 0.5、upright 0.5 是"白给"的，
    tracking_lin 因为没动只有 0.7 左右（乘权重后 ~1.0），tracking_ang 取决于随机指令。
如果总分接近 0 或超过 6，说明权重或公式写错了。
"""
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from duck_rl.constants import NUM_JOINTS
from duck_rl.env import DuckEnv

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


env = DuckEnv()
reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

log("reset ...")
state = reset_fn(jax.random.PRNGKey(0))
jax.block_until_ready(state.obs)
cmd = state.info["command"]
log(f"本回合指令 = [vx={float(cmd[0]):+.2f}, vy={float(cmd[1]):+.2f}, wz={float(cmd[2]):+.2f}]")

# ---------------------------------------------------------------- 站着不动
print("\n=== 测试 1：零动作 200 步，看奖励量级 ===", flush=True)
zero = jp.zeros(NUM_JOINTS)
rewards, term_acc = [], {k: [] for k in ("tracking_lin", "tracking_ang", "upright",
                                         "height", "alive", "action_rate", "torque")}
for i in range(200):
    state = step_fn(state, zero)
    if i == 0:
        jax.block_until_ready(state.reward)
        log("第一步完成（含编译）")
    rewards.append(float(state.reward))
    # 一次性把 metrics 整个搬回主机，比逐个 float() 快得多
    mb = jax.device_get(state.metrics)
    for k in term_acc:
        term_acc[k].append(float(mb[f"rew/{k}"]))

import numpy as np

r = np.array(rewards)
print(f"单步奖励: 起 {r[0]:.3f} → 末 {r[-1]:.3f}   均值 {r.mean():.3f}   "
      f"最小 {r.min():.3f} 最大 {r.max():.3f}")
print(f"200 步回合总分 ≈ {r.sum():.1f}  ← 这是站着不动的收益")
print("\n各项均值（已乘权重后的贡献见括号）:")
c = env._config
weights = {"tracking_lin": c.w_tracking_lin, "tracking_ang": c.w_tracking_ang,
           "upright": c.w_upright, "height": c.w_height, "alive": c.w_alive,
           "action_rate": c.w_action_rate, "torque": c.w_torque}
for k, v in term_acc.items():
    mean = np.mean(v)
    print(f"  {k:<13} 原始 {mean:>8.4f}  ×权重 {weights[k]:>+8.4f} = {mean * weights[k]:>+8.4f}")

# ---------------------------------------------------------------- 乱动作
print("\n=== 测试 2：全 +1 动作，看摔之前能拿多少 ===", flush=True)
state = reset_fn(jax.random.PRNGKey(1))
one = jp.ones(NUM_JOINTS)
rs = []
for i in range(60):
    state = step_fn(state, one)
    rs.append(float(state.reward))
    if float(state.done) > 0.5:
        print(f"第 {i + 1} 步 done，此前均奖励 {np.mean(rs):.3f}")
        break
else:
    print(f"60 步内未触发 done，均奖励 {np.mean(rs):.3f}")

print("\n判读：")
print("  站着均值 3.5~4.0 → 权重配置合理")
print("  站着接近 0      → 检查 sigma（太小）/ 权重（太小）/ 观测（错位）")
print("  站着超过 6      → 权重叠加过多，alive 或 height 给太大")
