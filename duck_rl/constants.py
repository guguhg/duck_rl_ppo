"""观测契约与固定常量。

**这个文件一旦开始训练就冻结。**
改观测维度或顺序，必须同步改三处：env.py 的 _get_obs、导出脚本、真机部署端。
任何一处漏改，训练出来的策略上真机就是乱的，而且不报错——这是最难查的一类 bug。
"""
import mujoco

# ---------------------------------------------------------------- 关节
# 顺序 = MJCF 里 <actuator> 段的顺序（不是 qpos 顺序，虽然本模型恰好相同）
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
NUM_JOINTS = len(JOINT_NAMES)          # 14
assert NUM_JOINTS == 14, "关节数变了？模型换了吗？"

# ---------------------------------------------------------------- 观测
# (名称, 维度)。列表顺序 = 拼接顺序 = 契约本身
OBS_LAYOUT = [
    ("gyro", 3),           # 角速度（机体系），来自 sensor "gyro"
    ("gravity", 3),        # 重力方向（机体系），来自 sensor "upvector"
    ("command", 3),        # [vx, vy, wz] 速度指令
    ("q", NUM_JOINTS),     # 关节位置（相对默认姿态）
    ("qd", NUM_JOINTS),    # 关节速度
    ("last_action", NUM_JOINTS),   # 上一帧网络输出
]
OBS_SIZE = sum(n for _, n in OBS_LAYOUT)     # 51
assert OBS_SIZE == 51, f"观测维度变成了 {OBS_SIZE}，改了吗？"


def sensor_slice(m, name: str) -> slice:
    """按传感器名字取它在 data.sensordata 里的切片。

    为什么不用写死的下标：模型换了传感器顺序就变，写死的偏移会静默错位——
    程序照跑，观测全乱。用名字查，错了会立刻报错。

    @param m: mujoco.MjModel
    @param name: 传感器名（见 MJCF 的 <sensor> 段）
    @return: slice，可直接用于 data.sensordata[sl]
    """
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        avail = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
                 for i in range(m.nsensor)]
        raise ValueError(f"找不到传感器 {name!r}，可用的有：{avail}")
    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
    return slice(int(adr), int(adr + dim))


def print_contract() -> None:
    """打印观测契约。调试时对照它检查自己拼的向量。"""
    print(f"obs 总维度: {OBS_SIZE}")
    off = 0
    for name, n in OBS_LAYOUT:
        print(f"  [{off:>3d}:{off + n:>3d}] {name:<12s} {n:>2d} 维")
        off += n
    print(f"\n动作维度: {NUM_JOINTS}（= 14 个位置伺服执行器）")


if __name__ == "__main__":
    print_contract()
