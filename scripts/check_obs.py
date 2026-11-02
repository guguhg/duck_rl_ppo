"""观测验收：手动歪躯干，看重力三分量是否按预期变化。

这是抓「符号错误 / 行列弄反」最有效的一招——比看训练曲线快得多。

预期（绕 y 轴俯仰 θ，体坐标系重力 = R^T·(0,0,-1)）：
    θ=0°   → ( 0.000,  0.000, -1.000)
    θ=+30° → ( 0.500,  0.000, -0.866)   ← x 变正：躯干前倾时"下"落在身体前方
    θ=-30° → (-0.500,  0.000, -0.866)
若符号相反，说明用了 R 的第三列而不是第三行（互为转置）。
"""
import math

import jax
import jax.numpy as jp
import mujoco.mjx as mjx
import numpy as np

from duck_rl.env import DuckEnv, gravity_body

# 分阶段打印 + flush=True：卡在哪一步一眼可见。
# （Python 在终端里是行缓冲，正常会立刻显示；但如果导入阶段就卡住，什么都不会打印）
print("[1/3] 导入完成，开始构造环境（首次会初始化 CUDA，可能要十几秒）...", flush=True)
env = DuckEnv()
print(f"[2/3] 环境构造完成：obs={env.observation_size} act={env.action_size} "
      f"n_substeps={env.n_substeps}", flush=True)
state = env.reset(jax.random.PRNGKey(0))
print("[3/3] reset 完成，开始分析观测", flush=True)
obs = np.asarray(state.obs)

# ---------------------------------------------------------------- 1) 分段打印
print("=== 观测分段（对照 constants.OBS_LAYOUT）===")
off = 0
from duck_rl.constants import OBS_LAYOUT       # 直接读契约，加维度时这里不用改
for name, n in OBS_LAYOUT:
    seg = obs[off:off + n]
    print(f"[{off:>2}:{off + n:>2}] {name:<12} "
          f"min={seg.min():+.3f} max={seg.max():+.3f}  前三个 {np.round(seg[:3], 3)}")
    off += n

# ---------------------------------------------------------------- 2) 倾斜测试
print("\n=== 倾斜测试（只改浮基四元数，其余不动）===")
for deg in (0, 30, -30):
    th = math.radians(deg)
    # 绕 y 轴转 θ：q = (cos(θ/2), 0, sin(θ/2), 0)
    quat = jp.array([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0])
    data = state.data.replace(qpos=state.data.qpos.at[3:7].set(quat))
    data = mjx.forward(env.mjx_model, data)          # 重算运动学

    g = np.asarray(gravity_body(data.qpos[3:7]))
    sens = np.asarray(data.sensordata)
    up = sens[9:12]                                   # sensor "upvector"

    exp = np.array([math.sin(th), 0.0, -math.cos(th)])
    ok = np.allclose(g, exp, atol=2e-3)
    print(f"θ={deg:>+4}°  理论 {np.round(exp,3)}  实测 {np.round(g,3)}  "
          f"{'✓' if ok else '✗ 符号或行列有问题'}")
    print(f"           sensor upvector = {np.round(up,3)}   （对照它在竖直时的读数）")

print("\n结论要求：θ=0 时 gravity ≈ (0,0,-1)；θ=+30 时 x ≈ +0.5。")
print("若 upvector 在 θ=0 时是 (0,0,1)，说明它的定义是「体轴在世界系」，与重力方向差一个转置——")
print("不要混用这两个量，否则倾斜时观测错得悄无声息。")
