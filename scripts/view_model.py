"""在 MuJoCo 交互窗口里看模型——不用训练也能确认"摆得对不对"。

用法：
    uv run python scripts/view_model.py                       # 默认平地场景
    uv run python scripts/view_model.py assets/xxx.xml

它会做三件事：
1. 把模型摆到 keyframe 的 home 姿态（和 env.reset 完全一致）
2. 把位置伺服目标设成同样的姿态（ctrl = key_ctrl）
3. 以 1 倍实时速度跑物理，你可以看到它是"稳稳蹲住"还是"塌下去"

窗口操作：左键拖动旋转、右键平移、滚轮缩放、Esc 或 Ctrl+C 退出。
"""
import sys
import time

import mujoco
import mujoco.viewer

XML = sys.argv[1] if len(sys.argv) > 1 else "assets/scene_flat_terrain.xml"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# ---- 摆到初始姿态 ----
mujoco.mj_resetDataKeyframe(m, d, 0)   # 读 keyframe 0（home）：qpos 全部就位
mujoco.mj_forward(m, d)                 # 算一遍运动学，site/sensor 才有值

# ---- 位置伺服目标 = 初始姿态 ----
# key_ctrl 里存的就是 14 个关节目标角，与 env.reset 里 ctrl=default_pos 等价。
# 不设的话 ctrl 全 0，伺服会拼命把关节拉到 0 位，鸭子会抽搐着摊平。
if m.nkey and m.key_ctrl.size:
    d.ctrl[:] = m.key_ctrl[0]

z0 = float(d.qpos[2])
print(f"初始高度 {z0:.4f} m，质量 {m.body_mass.sum():.3f} kg")
print("窗口里看：它应该蹲住不动（有小幅下沉是正常的）")
print("如果直接塌下去 / 抽搐 / 穿地 → 模型或执行器参数有问题")

steps = 0
t0 = time.time()
try:
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            steps += 1

            # 按 1 倍实时速度播放（否则 500 Hz 会跑得飞快看不清）
            lag = steps * m.opt.timestep - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)

            # 每 0.5 秒打印一次高度，看它是否稳定
            if steps % 250 == 0:
                print(f"  t={steps * m.opt.timestep:5.2f}s  高度={float(d.qpos[2]):.4f} m")

            viewer.sync()
except KeyboardInterrupt:
    print("\n已退出")
