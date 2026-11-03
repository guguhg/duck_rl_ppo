# Open Duck Mini v2 软件架构精读

> 本文基于实际源码与 ONNX 模型文件逐行核对，非二手转述。
> 核对版本：`Open_Duck_Mini_Runtime` 分支 `v2`，`Open_Duck_Mini` 分支 `v2`。

---

## 一、先建立一个正确的心智模型

这套软件**不是**"一个会推理的神经网络"，而是：

> **一个 20 ms 死循环，循环里做三件事：读传感器 → 拼 101 个数 → 喂给一个 0.22 M 参数的小 MLP → 拿回 14 个关节增量 → 写进舵机。**

所有复杂度都在这三件事的边界条件上，不在算法本身。

网络小到什么程度？**220,262 个参数，0.84 MB ONNX 文件**。

实测单次推理（开发机 CPU）**0.018 ms，P99 0.028 ms**，占 20 ms 预算的 **0.09%**。就算树莓派 Zero 2W 慢 15 倍，也只是 0.27 ms，占比 1.4%。

这个认知很重要：**你调不动的从来不是算力，是总线时序和舵机动力学。** 20 ms 周期里 98% 以上的时间都花在等那 14 个舵机的串行总线应答。

> 上述数字可用仓库内 `verify_policy.py` 自行复现：`python verify_policy.py BEST_WALK_ONNX_2.onnx`

---

## 二、四层架构

```
┌─────────────────────────────────────────────────────────────┐
│  L4  应用层  scripts/v2_rl_walk_mujoco.py  (RLWalk 类, 400 行) │
│      主循环、手柄映射、参数组织、异常退出清理                    │
├─────────────────────────────────────────────────────────────┤
│  L3  策略层  onnx_infer.py  (OnnxInfer, 43 行)                │
│      onnxruntime CPUExecutionProvider，裸推理，无预处理         │
├─────────────────────────────────────────────────────────────┤
│  L2  能力层  硬件抽象 + 传感器/执行器驱动                       │
│      rustypot_position_hwi.py  HWI      舵机总线 (166 行)      │
│      raw_imu.py                Imu      IMU 独立线程 (161 行)  │
│      feet_contacts.py          FeetContacts  足底 GPIO (34 行) │
│      xbox_controller.py        XBoxController 手柄线程 (220 行) │
│      poly_reference_motion.py  PRM      相位周期 (117 行)      │
├─────────────────────────────────────────────────────────────┤
│  L1  配置与工具层                                              │
│      duck_config.py   DuckConfig      关节零位/特性开关 (83 行) │
│      rl_utils.py      动作滤波 + 关节重排 + 四元数 (161 行)     │
└─────────────────────────────────────────────────────────────┘
```

**全部 Python，无框架，无 daemon，单进程。** 这一点和 Microduck（Rust 七进程 + JSON-RPC）是完全相反的设计取向。

---

## 三、主循环：20 ms 内到底发生了什么

这是整个项目的心脏。按 `v2_rl_walk_mujoco.py:199-328` 的 `run()` 逐行还原：

```
t0 ─ 取手柄命令（非阻塞，Queue.get(False)，拿不到就用上一帧）
   ├ 处理按键边沿：A=暂停  B=叫声  X=投影  LB=冲刺  DPad=相位微调
   └ 左右扳机 → 触角位置
   
   if paused: sleep(0.1); continue          ← 暂停时不读总线，省电

t1 ─ get_obs()                             ← 关键，见第四节
   ├ imu.get_data()          非阻塞取 Queue(maxsize=1) 最新帧
   ├ hwi.get_present_positions()   ★ 同步阻塞，走总线
   ├ hwi.get_present_velocities()  ★ 同步阻塞，走总线
   └ 足底 GPIO 直接读

t2 ─ 相位推进
   ├ imitation_i += (phase_frequency_factor + offset)
   ├ imitation_i %= nb_steps_in_period
   └ imitation_phase = [cos(2πi/N), sin(2πi/N)]

t3 ─ policy.infer(obs)                     ← 0.22M MLP，<1 ms
   
t4 ─ 动作后处理（顺序不能乱）
   ├ 动作历史平移：last→last_last→last_last_last
   ├ motor_targets = init_pos + action * action_scale(0.25)
   ├ 可选 LowPassActionFilter（默认 None，不启用）
   └ 头部叠加：targets[5:9] += last_commands[3:]

t5 ─ hwi.set_position_all()                ★ 一次批量写 14 个舵机
   
t6 ─ 计时核算
   ├ took > 20ms → 打印 "Policy control budget exceeded by X"
   └ sleep(max(0, 1/50 - took))            ← 用睡眠凑周期，非硬实时
```

**关键设计取舍：**

| 设计 | 做法 | 为什么 |
|---|---|---|
| 周期保证 | `sleep(剩余时间)` | 软实时。超时只打印警告，不补偿、不跳帧 |
| IMU 采样 | 独立线程 + `Queue(maxsize=1)` | 主循环绝不阻塞在 I2C 上；队列满自动丢旧帧 |
| 手柄采样 | 独立线程 @20 Hz + `Queue(maxsize=1)` | 人手 20 Hz 足够，主循环非阻塞取 |
| 舵机读写 | **同步阻塞，主循环内** | 无法避免，这是唯一真实延迟 |
| 异常处理 | 只有 `KeyboardInterrupt` | 摔倒/失联**没有任何保护逻辑** |

**总线波特率注意：** `HWI.__init__` 里是 `rustypot.feetech(usb_port, 1000000)` —— **1 Mbps**，不是常见的 500 kbps。这个值在 `experiments/v2/configure_motors.py` 里也有对应，改错会直接连不上舵机且不报明确错误。

---

## 四、观测 101 维：逐项拆解

这个数字是硬约束，网络输入写死 `[1, 101]`。构成（`get_obs` 的 `np.concatenate` 顺序）：

| # | 段 | 维度 | 实际计算 | 设计意图 |
|---|---|---|---|---|
| 1 | `gyro` | 3 | BNO055 陀螺仪 rad/s | 角速度，判断身体旋转趋势 |
| 2 | `accelero` | 3 | BNO055 加速度 m/s² | 重力方向，判断倾斜 |
| 3 | `cmds` | 7 | `[vx, vy, ωz, neck, head_pitch, head_yaw, head_roll]` | 速度指令 + 头部姿态指令 |
| 4 | `dof_pos - init_pos` | 14 | 实际角 − 初始角 | **相对偏差**，非绝对角 |
| 5 | `dof_vel × 0.05` | 14 | 角速度缩放 | 缩放到与位置同量级，利于网络 |
| 6 | `last_action` | 14 | 上一帧输出 | 动作平滑性约束 |
| 7 | `last_last_action` | 14 | 上上帧 | 二阶历史 |
| 8 | `last_last_last_action` | 14 | 上上上帧 | 三阶历史 |
| 9 | `motor_targets` | 14 | 当前 PD 目标角 | **指令前馈**，让策略知道自己下过什么命令 |
| 10 | `feet_contacts` | 2 | `[left, right]` bool→float | 触地状态 |
| 11 | `imitation_phase` | 2 | `[cos θ, sin θ]` | 步态相位时钟 |
| | **合计** | **101** | | |

### 三个值得注意的细节

**① 动作历史给了三层，不是一层。**
`last_action` / `last_last_action` / `last_last_last_action` 各 14 维，共占 42 维（41% 的观测）。目的是让策略知道自己的动作轨迹，从而输出平滑而不是抖动的控制。这是"无状态 MLP 模拟有状态控制器"的经典手法。

**② `motor_targets` 和 `last_action` 信息重复但物理意义不同。**
- `last_action` 是网络原始输出（[-1, 1] 区间）
- `motor_targets` 是 `init_pos + action × 0.25` 之后的**绝对目标角**

两者都喂进去，等于告诉网络"我想要什么"和"我实际下发了什么"。

**③ 有 IMU 但没有姿态四元数。**
只用了 gyro + accelero，**没有用 BNO055 的四元数输出**。这意味着策略得自己从加速度积分出姿态——这也解释了为什么它必须靠 `imitation_phase` 这个外部时钟来维持步态节律。Microduck 后来改用完整姿态观测，这是两代架构的重要差异。

---

## 五、动作 14 维：从网络输出到舵机

```
网络输出 action ∈ [-1, 1]^14      ← 最后一层 Tanh 保证
        ↓ × action_scale (0.25 rad ≈ 14.3°)
      action × 0.25
        ↓ + init_pos（蹲姿初值）
   motor_targets = init_pos + action × 0.25
        ↓ 头部叠加手柄命令
   targets[5:9] += last_commands[3:]
        ↓ 加零位偏移
   sent = target + joints_offsets[joint]
        ↓ rustypot 一次批量写
   舵机内部 PD（Kp=30，Kd=0）执行
```

### 关节顺序（顺序错了就摔，这是第一大坑）

`HWI.joints` 字典的定义顺序 = 观测顺序 = 动作顺序：

| 索引 | 关节名 | 舵机 ID | 初始角 init_pos (rad) |
|---|---|---|---|
| 0 | left_hip_yaw | 20 | 0.002 |
| 1 | left_hip_roll | 21 | 0.053 |
| 2 | left_hip_pitch | 22 | **−0.630** |
| 3 | left_knee | 23 | 1.368 |
| 4 | left_ankle | 24 | −0.784 |
| 5 | neck_pitch | 30 | 0.0 |
| 6 | head_pitch | 31 | 0.0 |
| 7 | head_yaw | 32 | 0.0 |
| 8 | head_roll | 33 | 0.0 |
| 9 | right_hip_yaw | 10 | −0.003 |
| 10 | right_hip_roll | 11 | −0.065 |
| 11 | right_hip_pitch | 12 | **+0.635** |
| 12 | right_knee | 13 | 1.379 |
| 13 | right_ankle | 14 | −0.796 |

**注意 `hip_pitch` 左右符号相反（−0.63 vs +0.635）。** 这不是笔误，是 URDF 里左右腿关节轴朝向镜像定义导致的。任何自己写驱动的人都会在这里栽一次。

**注意 init_pos 是蹲姿不是站姿。** 膝盖 1.368 rad ≈ 78°，踝 −0.784 rad ≈ −45°。所以上电瞬间鸭子是半蹲的，不是站直的。

### PD 参数

```python
pid = [30, 0, 0]        # 默认
kps = [30] * 14
kps[5:9] = [8, 8, 8, 8] # 头部 4 个关节降到 8
kds = [0] * 14          # 微分项全是 0
```

**Kd 全为 0，只有 P。** 阻尼完全依赖舵机固件默认值和机械摩擦。这也是为什么换舵机型号会直接毁掉步态。

---

## 六、ONNX 网络结构（实测，非推测）

用 `onnx.load()` 直接读 `BEST_WALK_ONNX_2.onnx`：

```
输入  obs: [1, 101]
  ↓ Sub（减均值）+ Mul（乘倒数）      ← 观测归一化烘焙在图内
Gemm  [101 → 512]  + Sigmoid + Mul   ← SiLU（Sigmoid Linear Unit）
Gemm  [512 → 256]  + Sigmoid + Mul
Gemm  [256 → 128]  + Sigmoid + Mul
Gemm  [128 → 28  ]
  ↓ Split(axis=-1) → [14 mean, 14 log_std]
  ↓ Tanh(mean)                        ← 只对 mean 做，log_std 丢弃
输出  continuous_actions: [1, 14]
```

| 项 | 值 |
|---|---|
| 参数量 | 220,262（0.22 M） |
| 文件大小 | 0.84 MB |
| 隐藏层 | 512 → 256 → 128 |
| 激活函数 | SiLU（Gemm+Sigmoid+Mul 三算子模拟） |
| 输出维度 | 28（stratified 双分支），部署只用前 14 |
| opset | 11 |
| metadata | **空**（无版本、无训练信息） |

### 三个关键事实

**① 归一化在模型内部。**
前两个算子 `Sub` + `Mul` 就是 `(obs - mean) / std`。所以**运行时传原始观测即可，不需要自己归一化**。但反过来：**你不能手改这个 ONNX，一旦归一化参数错位，网络会看到错尺度的输入，行为完全荒谬但不报错。**

**② 输出是 28 维不是 14 维。**
`Split` 把最后一层切成 `mean` 和 `log_std`。这是 PPO 训练时的随机策略结构，`log_std` 用于探索。**部署时 `log_std` 被直接丢弃**，只对 `mean` 做 Tanh。

`OnnxInfer.infer()` 返回 `outputs[0][0]` —— 取的是 `Split` 的第 0 个输出，恰好是 mean 分支。这个细节很容易被忽略。

实测验证：500 次随机观测下输出全局范围 **−0.9945 ~ +0.9962**，确认 Tanh 压缩生效，动作严格落在 `[-1, 1]`。

**③ metadata 是空的。**
对比 Microduck 的 `manifest.json` 强制记录 schema 版本、输入输出契约、归一化信息，Open Duck Mini 的 ONNX 是"裸文件"。这意味着**换一个 ONNX 进去，你无法从文件本身知道它期待什么观测布局**——只能靠读训练代码。这是这代架构最大的工程债。

### ④ 实测：各观测分段对输出的影响力

用全零观测为基准，把各段同时扰动 +0.3，看输出变化的平均幅度（`verify_policy.py` 第 5 项）：

| 分段 | 维度 | 输出变化量 | 解读 |
|---|---|---|---|
| dof_vel × 0.05 | 14 | **0.2078** | 关节速度最敏感 |
| dof_pos − init | 14 | **0.2054** | 关节位置同等敏感 |
| cmds | 7 | **0.2026** | 指令影响力与本体感知同级 |
| motor_targets | 14 | 0.1365 | 次之 |
| last_action | 14 | 0.0817 | 动作历史 |
| last2_last_action | 14 | 0.0719 | 三阶历史权重反而高于二阶 |
| gyro | 3 | 0.0488 | |
| imitation_phase | 2 | 0.0443 | 仅 2 维却有可测影响 |
| last_last_action | 14 | 0.0398 | 二阶历史权重最低 |
| feet_contacts | 2 | 0.0250 | 触地信息权重偏低 |
| accelero | 3 | 0.0050 | **几乎无效** |

两个反直觉的发现：

- **加速度段几乎不影响输出（0.005）**，而陀螺仪段是它的 10 倍。这说明策略主要靠角速度感知身体旋转，加速度只作微弱参考。考虑到只有 3 维加速度、没有完整姿态四元数，这解释了为什么它需要外部相位时钟来稳住步态。
- **`feet_contacts` 只有 2 维且权重偏低**，尽管物理上触地信息对双足极其关键。这是这一代架构的短板，Microduck 后来把它提升为完整接触观测。

> 注：灵敏度是**局部梯度性质的近似**，不代表因果重要性，但足以说明哪些输入通道几乎没被用起来。

---

## 七、线程模型

整个运行时只有 **3 个线程**：

```
主线程（50 Hz）
  ├─ 读手柄队列（非阻塞）
  ├─ 读 IMU 队列（非阻塞）
  ├─ ★ 读舵机位置+速度（同步阻塞 ~ 数 ms）
  ├─ 推理
  └─ ★ 写舵机目标（同步阻塞）
  
IMU 线程（50 Hz）              XBox 线程（20 Hz）
  loop:                          loop:
    读 I2C                        轮询 pygame
    put → Queue(maxsize=1)        put → Queue(maxsize=1)
    sleep 补足周期                 sleep 补足周期
```

**为什么用 `maxsize=1`？**
队列满时 `put()` 会阻塞——但这两个线程用的是"put 前不检查"的方式，实际效果是**最新数据覆盖**（因为消费者每帧都取）。这样主循环拿到的永远是最新数据，不会排队积压导致延迟累积。

**这是这套代码里最值得学的一处工程手法**：用单元素队列实现"只保留最新值"，比加锁共享变量更清晰，比无界队列更安全。

**没有的东西：**
- 没有 watchdog
- 没有摔倒检测
- 没有温度保护
- 没有通信超时处理（`get_present_positions` 异常直接返回 `None`，主循环 `continue` 跳过这帧——**这意味着舵机短暂掉线时策略会停摆但机器人还在动**）

---

## 八、相位机制：为什么需要一个外部时钟

```python
self.imitation_i += 1 * (phase_frequency_factor + offset)   # 默认 1.0
self.imitation_i %= self.PRM.nb_steps_in_period
phase = [cos(2πi/N), sin(2πi/N)]
```

这是**步态相位编码器**，本质是一个"虚拟节拍器"。用 cos/sin 对而非线性角度，是为了消除 2π 回绕处的跳变。

**为什么 RL 需要一个时钟？**
纯 RL 从零探索"走路"这个周期性动作极难——奖励稀疏、搜索空间巨大。加一个相位输入等于告诉网络"现在是步态的哪一段"，它只需要学"在相位 0.3 时该把髋关节转到多少"，任务从"发明步态"降级为"映射步态"。这是 **imitation learning / 参考运动引导** 的核心思路。

`PolyReferenceMotion` 加载的 `polynomial_coefficients.pkl` 里存了完整的多项式轨迹，但在 v2 里**只用了它的 `nb_steps_in_period`**——也就是只取了"一个步态周期多少帧"。真正的参考轨迹已经不用了（见第 103 行注释 `# TODO`）。

> 这行注释很说明问题：项目从"多项式参考运动驱动"演进到"相位 + RL 策略"，但旧模块没删干净，退化成了个常量提供器。

### 实测证明：相位确实是步态时钟

用 `phase_sweep.py` 做一个决定性实验：**把其他所有观测冻结成"静止站立"状态，只让相位从 0 扫到 2π**，观察策略输出。

如果相位是摆设，输出应该是一条平线。实测结果（`phase_sweep.py`，见 `phase_sweep.png`）：

| 关节 | 峰峰值 (rad) | 说明 |
|---|---|---|
| left_knee | **0.7062** | 摆动最大 |
| right_knee | **0.6616** | 与左膝反相 |
| right_hip_roll | 0.4952 | |
| right_hip_pitch | 0.3438 | |
| left_ankle | 0.3099 | |
| left_hip_pitch | 0.2583 | |
| left_hip_roll | 0.2379 | |
| **left_hip_yaw** | **0.0498** | 几乎不动（支撑腿不需要转向） |
| 14 关节峰峰值总和 | **4.0869** | |

**左右腿平均差异（同相位下）：**

| 关节对 | 平均差异 | 解读 |
|---|---|---|
| hip_pitch | **0.9238** | 强反相 → 交替步态 |
| knee | 0.4460 | 反相 |
| hip_roll | 0.2041 | 反相（重心左右转移） |
| hip_yaw | 0.0601 | 近似同相 |

**结论：`imitation_phase` 是真实的步态时钟。** 冻结一切本体感知、只转这个 2 维输入，策略就吐出一整套左右反相、膝踝配合的周期动作。

**两个值得注意的现象：**

1. **曲线不是纯正弦。** 看 `phase_sweep.png` 的右腿 knee 曲线，底部有明显的高频起伏。这说明策略不是"查表回放参考轨迹"，而是**以相位为基准做条件生成，再叠加反馈修正**——同样是相位 0.3，如果 IMU 说身体歪了，输出会不一样。这是 RL 策略相对传统 CPG 的关键优势。
2. **`left_hip_yaw` 几乎不动（0.05 rad ≈ 3°）。** 这不是 bug。该实验里速度指令为 0，处于原地站定状态，髋 yaw 是转向用的关节，没指令自然不动。这也侧面说明指令段 `cmds` 是被真实使用的（呼应第六节第 ④ 项的敏感度测试：`cmds` 影响量 0.20，与本体感知同级）。

---

## 九、这套架构的能力边界

| 能力 | 有 / 无 | 证据 |
|---|---|---|
| 走路 | ✅ | 单一策略，BEST_WALK_ONNX_2 |
| 站起 / 坐下 | ❌ | 无对应策略 |
| 摔倒保护 | ❌ | 无任何检测逻辑 |
| 策略热切换 | ❌ | 单策略硬编码，换策略要改命令行参数重跑 |
| 参数签名 / 校验 | ❌ | ONNX 无 metadata |
| OTA 升级 | ❌ | 手动 git pull + pip install |
| 进程守护 | ❌ | 单进程，崩了就没了 |
| 模块化服务 | ❌ | 全在一个进程里 |

对比 Microduck：7 个 Rust daemon、61 维固定观测契约（含 body_pose 6 维）、策略热插拔、签名 OTA + 健康检查回滚、`robotd` 独占电机控制权作为唯一安全关口。

**结论：Open Duck Mini 的架构是"能跑起来的最小实现"，Microduck 是"能上产线的实现"。** 前者适合学习，后者适合参考工程化。

---

## 十、读代码的推荐顺序

按这个顺序读，40 分钟能建立完整图景：

| 序 | 文件 | 行数 | 读什么 |
|---|---|---|---|
| 1 | `scripts/v2_rl_walk_mujoco.py` | 400 | `run()` 的 while 循环，从 199 行读起 |
| 2 | 同上 | — | `get_obs()` 第 123-172 行，对着第四节的表看 |
| 3 | `rustypot_position_hwi.py` | 166 | `joints` 字典顺序 + `init_pos` + `set_position_all` |
| 4 | `onnx_infer.py` | 43 | 全读，就这么点 |
| 5 | `raw_imu.py` | 161 | 只看线程模型（`imu_worker` 和 `get_data`） |
| 6 | `rl_utils.py` | 161 | `LowPassActionFilter` 和 `quat_rotate_inverse` |
| 7 | `duck_config.py` | 83 | `joints_offsets` 从哪来 |

**跳过：** `raw_imu.py` 的 BNO055 标定部分、`poly_reference_motion.py` 的多项式加载（v2 已退化）。

---

## 十一、不动硬件就能做的实验

这份代码最好的地方是**可以在没有机器人的情况下跑通大半**。

### 实验 1：验证 ONNX 输入输出（零依赖，现在就能做）

仓库里已经放了 `verify_policy.py`，一次跑完 6 项检查：

```bash
python verify_policy.py BEST_WALK_ONNX_2.onnx
```

它会输出：输入输出契约、网络层结构、推理耗时分布、输出范围、**各观测分段敏感度**、归一化是否内建。本文档所有实测数字都出自这个脚本。

最小手写版本：

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("BEST_WALK_ONNX_2.onnx",
                            providers=["CPUExecutionProvider"])
obs = np.zeros((1, 101), dtype=np.float32)   # 全零观测
out = sess.run(None, {"obs": obs})
print(out[0].shape, out[0])   # 期望 (1,14)，值在 [-1,1]
```

### 实验 2：观测重放（`--replay_obs` 参数就是为这个设计的）

原作者留了 `--save_obs` / `--replay_obs` 一对参数：真机跑一次存观测，之后在仿真或空跑里重放，**用来定位"是观测错了还是策略错了"**。这是很实用的调试手段。

### 实验 3：动作滤波器效果对比

`--cutoff_frequency 30` 打开低通滤波，对比开关时的动作平滑度。这是理解"为什么 RL 输出会有高频抖动"的直观演示。

### 实验 4：相位扫描（最有价值的一个）

```bash
python phase_sweep.py BEST_WALK_ONNX_2.onnx
```

冻结本体感知、只扫相位，会得到 `phase_sweep.png`：六张子图分别展示左右腿 pitch/knee/ankle、髋 roll、髋 yaw、头颈四关节在 2π 周期内的输出曲线。

**这张图就是"相位引导 RL"最直观的证据**——你能直接看到左右腿反相、膝踝配合的完整步态，而输入里除了 `[cos, sin]` 什么都没有。

### 实验 5：把观测打印出来看量级

在 `get_obs()` 里加一行，观察各段数值范围。你会发现：
- `dof_pos - init_pos` 在 ±0.5 量级
- `dof_vel × 0.05` 在 ±0.2 量级
- `cmds` 里 `vx` 只有 ±0.15（m/s，非常慢）

这个量级分布直接解释了为什么要做 `×0.05` 缩放——不缩放的话速度项会淹没位置项。

---

## 十二、给后续学习的三条判断

**① 想学 RL 本身，这套代码不是好教材。**
它只有推理侧，没有训练代码。训练在 `Open_Duck_Playground`（MuJoCo Playground）里，是另一套东西。想理解"策略怎么训练出来"，得去看 Pollen 的 `microduck_rl`，那里的 `tasks/mdp.py` 把奖励函数拆成了十几个具名项，才是真正的教材。

**② 想学嵌入式软件架构，这套代码水平中等偏上。**
IMU 单元素队列、动作历史栈、归一化烘焙进 ONNX —— 这三个手法值得学。但没有异常保护、没有参数校验、没有模块化，这些是减分项。

**③ 最有价值的认知：这整个项目是"用极简软件包裹极复杂物理"。**
软件部分 1800 行 Python，一个人一周能读懂。真正难的是让那个 0.22 M 的 MLP 输出能让 42 cm 高的双足在真实世界站稳——而那部分知识不在代码里，在 sim2real 的调参经验里。

---

*本文档所有行号、维度、参数量均来自实际源码与 ONNX 文件解析，可用第十一节的实验自行验证。*
