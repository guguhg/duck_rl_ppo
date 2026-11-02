"""Step 验收：能稳定站着（零动作），也能被判摔倒（乱动作）。

故意用 Python for 循环而不是 jax.lax.scan：
  - scan 要先把整段编译完才返回，编译期间黑屏几十秒，看着像卡死
  - for 循环每步都有输出，能看清进度，也更容易定位问题
  - 速度慢一点无所谓，这是验收脚本不是训练

首次调用 env.step 会触发 XLA 编译（几十秒），之后每步都是毫秒级。
"""
import time
from pathlib import Path

import jax
import jax.numpy as jp

from duck_rl.constants import NUM_JOINTS
from duck_rl.env import DuckEnv

# 开启编译缓存：同一个函数+同样的形状，第二次运行就不用再编译（省掉那 60+ 秒）
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


log("构造环境（首次会初始化 CUDA）...")
env = DuckEnv()
log(f"环境就绪 obs={env.observation_size} act={env.action_size} n_substeps={env.n_substeps}")

# 关键：把 reset / step 各编译成一个整体。
# 不 jit 的话，函数里几十个算子逐个派发 + 逐个同步，一步要上百毫秒；
# jit 之后一次派发跑完整个 step，正常 1~3 ms。训练时 Brax 会自己 jit，这里只是对齐它。
# 注意：jit 要放在第一次调用之前，否则第一次仍然走未编译的路径。
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

log("reset（首次含编译，可能 60~90 秒，正常）...")
state = reset_fn(jax.random.PRNGKey(0))
jax.block_until_ready(state.obs)          # 强制同步，确保这一步真的算完
log(f"reset 完成，初始高度 {float(state.data.qpos[2]):.4f}")

# ---------------------------------------------------------------- 1) 零动作
print("\n=== 测试 1：零动作 200 步（应稳定站立）===", flush=True)
zero = jp.zeros(NUM_JOINTS)
heights = []
t1 = time.time()
for i in range(200):
    state = step_fn(state, zero)
    if i == 0:
        jax.block_until_ready(state.obs)
        log(f"第一步完成（含编译），高度 {float(state.data.qpos[2]):.4f}")
    # 注意：这里每步都 float() 会强制同步，测量用；不需要看进度时可以攒到最后一次转
    heights.append(float(state.data.qpos[2]))
    if (i + 1) % 40 == 0:
        log(f"步 {i + 1:3d}  高度 {heights[-1]:.4f}  done {float(state.done):.0f}  "
            f"平均 {1000 * (time.time() - t1) / (i + 1):.1f} ms/步")

h = jp.array(heights)
log(f"200 步耗时 {time.time() - t1:.1f}s")
print(f"高度: 起 {heights[0]:.4f} → 末 {heights[-1]:.4f}  "
      f"最小 {float(h.min()):.4f} 最大 {float(h.max()):.4f}")
print(f"done 次数: {int(float(state.done))} （这是最后一步的值；整段应为 0 次）")
print("若 200 步后高度仍在 0.16 附近 → step 的数据流正确")

# ---------------------------------------------------------------- 2) 全 +1 动作
print("\n=== 测试 2：全 +1 动作，最多 300 步（应失稳并终止）===", flush=True)
state = reset_fn(jax.random.PRNGKey(1))
one = jp.ones(NUM_JOINTS)
first_done = -1
for i in range(300):
    state = step_fn(state, one)
    if i == 0:
        log("第一步完成（动作 +1，关节会顶到限位）")
    if (i + 1) % 40 == 0:
        log(f"步 {i + 1:3d}  高度 {float(state.data.qpos[2]):.4f}  done {float(state.done):.0f}")
    if float(state.done) > 0.5:
        first_done = i + 1
        break

print(f"首次 done 在第 {first_done} 步（-1 表示 300 步内都没倒）")
print(f"最终高度 {float(state.data.qpos[2]):.4f}")

print("\n判读：")
print("  测试 1 高度稳定 → action→ctrl→物理→观测 整条数据流通了")
print("  测试 2 能触发 done → 终止条件生效（-1 说明阈值太松，要调 min_height/min_upright）")
