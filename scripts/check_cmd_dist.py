"""检查指令采样分布是否按预期工作。

改完采样策略后，不要只信代码——直接采 2 万次看统计。特别是"纯侧移"的占比，
它决定了策略到底有没有机会学会侧移。

用法：
    uv run python -u scripts/check_cmd_dist.py
"""
import jax
import numpy as np

from duck_rl.env import DuckEnv

N = 20000          # 采样次数（约等于 2 万个回合的指令）

env = DuckEnv()
keys = jax.random.split(jax.random.PRNGKey(0), N)
cmds = np.asarray(jax.vmap(env._sample_command)(keys))     # (N, 3)
vx, vy, wz = cmds[:, 0], cmds[:, 1], cmds[:, 2]

# 恰好两个分量为 0 → 单轴指令
zeros = np.isclose(cmds, 0.0, atol=1e-6)
n_single = int((zeros.sum(axis=1) == 2).sum())
# 纯侧移：只动 vy，且 |vy| 足够大（这种组合以前几乎采不到）
pure_side = (zeros[:, 0] & zeros[:, 2] & (np.abs(vy) >= 0.2))
# 纯转向
pure_turn = (zeros[:, 0] & zeros[:, 1] & (np.abs(wz) >= 0.4))

print(f"采样 {N} 次（{env._config.cmd_single_axis_prob:.0%} 单轴模式，"
      f"{env._config.cmd_vy_edge_prob:.0%} vy 边界加权）\n")

print("类别                     次数     占比")
for name, n in [
    ("单轴指令（两轴为 0）", n_single),
    ("纯侧移 |vy|≥0.2", int(pure_side.sum())),
    ("纯转向 |wz|≥0.4", int(pure_turn.sum())),
]:
    print(f"{name:<22} {n:>7}   {n / N * 100:5.1f}%")

print("\n各轴范围（min / mean / max）")
for name, a, rng in [("vx 前后", vx, env._config.cmd_lin_x),
                     ("vy 左右", vy, env._config.cmd_lin_y),
                     ("wz 转向", wz, env._config.cmd_ang_z)]:
    print(f"  {name}: {a.min():+.2f} / {a.mean():+.2f} / {a.max():+.2f}"
          f"   （配置范围 {rng[0]:+.2f} ~ {rng[1]:+.2f}）")

print("\n前 8 条样本：")
for c in cmds[:8]:
    tag = "单轴" if np.isclose(c, 0, atol=1e-6).sum() == 2 else "  三维"
    print(f"  [{c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}]  {tag}")

print("\n判读：")
print("  「纯侧移」占比明显大于 0.1% → 采样策略生效，可以开训")
print("  仍然接近 0 → 配置没被读到，检查 config 里那三个 cmd_* 参数")
