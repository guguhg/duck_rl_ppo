# 施工日志：从零搭建 `~/duck_rl`

> **本文记「怎么搭起来的」**：每一步的命令、实际输出、当时踩的坑、关键版本。
> 目标是换台机器能照着复现。
>
> 不放在这里的：训练调参的过程与结论（见《训练实验记录》）、代码结构讲解（见《代码导读》）、
> MJCF 逐行解释（见《MJCF逐段解析》）。
>
> 时间跨度：2026-10-31 ~ 11-03（约 3 天，含环境搭建）

---


## 进度总览

> 十步全部完成。表里的「51 维」「七项」是当时的版本号，最终是 **53 维 / 八项奖励**，见文末说明。

| Step | 内容 | 状态 | 时间 |
|---|---|---|---|
| 0 | 目录 + git + 忽略规则 | ✅ | 13:40–13:44 |
| 1 | `pyproject.toml` + `uv sync` | ✅ | 13:49 |
| 2 | 拷 MJCF 资产 + 最小集裁剪 + `check_model.py` 自检 | ✅ 自检已跑通，数据全部核对 | 13:54–14:56 |
| 3 | `config.py` + `constants.py`（观测契约 51 维） | ✅ | 15:22 |
| 4 | `env.py::reset`（obs 全零 + 画面验证） | ✅ | 15:34 |
| 5 | `env.py::_get_obs`（歪躯干验证） | ✅ | 16:10 |
| 6 | `env.py::step` + 终止（零动作会塌） | ✅ | 16:11 |
| 7 | `rewards.py` 七项接进 step | ✅ | 16:20 |
| 8 | `train.py` 接 Brax PPO（第 1 轮 18M 步） | ✅ | 16:23–17:21 |
| 9 | 导出 ONNX + 数值对齐 | ✅ 误差 4.9e-07 | 17:29 |
| 10 | 回放（窗口/视频） | ✅ 能走，实测 0.34~0.38 m/s | 11-02 12:10 |

> **最终状态（2026-11-03）**：Step 0~10 全部完成，九轮训练跑完。
> 观测契约 **53 维**（表里「51 维」是 Step 3 当时的版本，第 3 轮加了 yaw sin/cos）；
> 奖励 **八项**（表里「七项」是 Step 7 当时的版本，第 6 轮加了 heading）。
> 主策略 `policy_dr1.onnx`：六方向 **96.5%**、30 秒偏航 **3.0°**（`policy_exp8.onnx`）、质量 ±15% 稳 3.5 倍。

---

## 环境基线（2026-10-31 实测）

| 项 | 值 |
|---|---|
| 主机 | HASEE N960Kx，i5-11400（6C12T），16 GB |
| GPU | RTX 3060 Laptop 6 GB，驱动 617.42 / CUDA UMD 13.4 |
| 系统 | Windows 11 Build 26200 + WSL 3.0.3（VERSION 2）+ Ubuntu 22.04.5 |
| 发行版位置 | `G:\WSL\Ubuntu-22.04\ext4.vhdx`（`--location` 安装） |
| Linux 用户 | `gh` |
| 网络 | FlClash `127.0.0.1:7890`；`.wslconfig` 开 `networkingMode=mirrored`；tuna 走 `no_proxy` 直连 |
| Python | uv 0.13.6 + cpython 3.12.16（`uv python install 3.12`） |
| 参考仓库 | `~/Open_Duck_Playground`（只作参考与资产来源，不 import 其 env） |

---

## Step 0　项目骨架　（13:40 – 13:44）

### 目标
建空壳：目录结构 + git 仓库 + 忽略规则。**项目必须在 WSL 内部盘 `~`，不能放 `/mnt/`**（9p 跨文件系统，小文件慢一个数量级）。

### 执行的命令

```bash
cd ~
mkdir -p duck_rl/{duck_rl,scripts,assets,checkpoints}
cd duck_rl
git init

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
checkpoints/
*.mp4
*.onnx
EOF

touch assets/.gitkeep
git add .gitignore assets/.gitkeep
git commit -m "chore: 初始化项目骨架"
```

### 实际结果

| 项 | 结果 |
|---|---|
| 目录 | `assets/ checkpoints/ duck_rl/ scripts/` 四个 |
| git 初始分支 | **`master`**（安装默认），已 `git branch -m main` 改名 |
| 首个提交 | `d529d78 (HEAD -> main) chore: 初始化项目骨架`，2 文件 6 行 |
| 忽略规则验证 | `git check-ignore -v checkpoints/test.txt` → `.gitignore:4:checkpoints/` ✓ |

### 踩到的坑

1. **`Author identity unknown`**：新装的 Ubuntu 上 git 没有身份，首次 commit 直接失败。补：
   ```bash
   git config --global user.name "guguhg"
   git config --global user.email "3357697374@qq.com"
   ```
2. **默认分支是 `master` 不是 `main`**：`git branch -m main` 改掉；想以后自动用 main，可 `git config --global init.defaultBranch main`。
3. **git 不跟踪空目录**：`assets/`、`checkpoints/` 提交后不会出现在仓库里，所以放了 `assets/.gitkeep` 占位；`checkpoints/` 本来就要忽略，不需要占位。

### 知识点
`uv.lock` **不应**写进 `.gitignore`——它锁的是精确版本，复现环境靠它。很多人误以为 lock 文件是生成物就忽略掉，结果换机装出来版本全变。

---

## Step 1　依赖与虚拟环境　（13:49）

### 目标
用 `pyproject.toml` 声明依赖，由 uv 解析并装到项目自己的 `.venv`。

### 执行的命令

```bash
cat > pyproject.toml <<'EOF'
[project]
name = "duck-rl"
version = "0.1.0"
description = "从零手写的双足 RL 环境（MJX + Brax PPO）"
requires-python = ">=3.11"
dependencies = [
    "mujoco>=3.3.0",
    "mujoco-mjx>=3.3.0",
    "playground>=0.2.0",
    "brax>=0.14.0",
    "jax[cuda12]",
    "flax",
    "optax",
    "ml-collections",
    "numpy",
    "mediapy",
    "onnxruntime",
    "tensorflow",
    "tf2onnx",
]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["duck_rl"]
EOF

touch duck_rl/__init__.py
uv sync
```

### 实际结果

| 项 | 结果 |
|---|---|
| 解析 | `Resolved 126 packages in 1.54s` |
| 安装 | `Installed 116 packages in 2.07s`（走 `~/.cache/uv` 缓存，几乎全硬链接） |
| 虚拟环境 | `.venv` **8.2 GB** |
| 导入验证 | `mujoco 3.17.0` + `[CudaDevice(id=0)]` ✓ |

### 关键版本实录

| 包 | 版本 |
|---|---|
| jax / jaxlib / jax-cuda12-plugin | 0.12.0.1 |
| mujoco / mujoco-mjx | 3.17.0 |
| brax | 0.14.2 |
| playground（= Google DeepMind MuJoCo Playground） | 0.2.0 |
| flax / optax | 0.12.12 / 0.2.8 |
| tensorflow / tf2onnx / onnxruntime | 2.21.0 / 1.17.0 / 1.31.1 |
| numpy | 2.5.4 |
| cudnn / cublas（pip 自带，非系统安装） | 9.27 / 12.9 |

### 踩到的坑

1. **`packages = ["duck_rl"]` 要求目录里有 `__init__.py`**，否则 setuptools 报找不到包。空的 `touch duck_rl/__init__.py` 就够。
2. **`.venv` 8.2 GB 但磁盘不会掉 8.2 GB**：uv 从缓存硬链接，第二个项目复用同一批轮子。换机时这一点不成立（缓存为空），要预留时间和空间。
3. **`tensorflow` 与 `jax[cuda12]` 这次没冲突**（参考项目里也验证过）。若将来出现 CUDA 初始化失败，先怀疑它，换 `tensorflow-cpu` 即可（导出 ONNX 用不到 GPU）。

---

## 项目根目录说明（Step 1 之后长这样）

```
~/duck_rl/
├─ duck_rl/           ← 自己写的包（import duck_rl.xxx）
├─ scripts/           ← 一次性工具脚本
├─ assets/            ← MJCF + 网格（第 2 步拷入）
├─ checkpoints/       ← 训练产物（已 git 忽略）
├─ pyproject.toml     ← 依赖声明（手写）
├─ uv.lock            ← 精确版本锁（uv 生成，要提交）
├─ duck_rl.egg-info/  ← 构建元数据（setuptools 自动生成，应忽略）
├─ .venv/             ← 虚拟环境（uv 生成，8.2 GB，不提交）
└─ .git/              ← 版本库
```

| 条目 | 谁生成 | 是什么 | 提交？ | 删掉的后果 |
|---|---|---|---|---|
| `duck_rl/` | 自己写 | 包本体：config / constants / env / rewards / train… | ✅ | 代码全没 |
| `scripts/` | 自己写 | 一次性工具（模型自检、画曲线），不是库 | ✅ | 丢失工具脚本 |
| `assets/` | 从参考仓库拷 | MJCF 模型 + STL 网格 | ✅（几 MB，可接受） | 仿真起不来；需重新拷 |
| `checkpoints/` | 训练产生 | 参数、ONNX、视频 | ❌ 已忽略 | 可重新训练得到 |
| `pyproject.toml` | 手写 | 依赖声明 + 包配置 | ✅ | 无法 `uv sync` |
| `uv.lock` | uv 生成 | 每个包的精确版本与哈希 | ✅ | 换机装出不同版本，行为漂移 |
| `duck_rl.egg-info/` | setuptools | 构建元数据（`SOURCES.txt`、`PKG-INFO`） | ❌ 应加进忽略 | 下次 `uv sync` 自动重建 |
| `.venv/` | uv 生成 | 虚拟环境，8.2 GB | ❌ 已忽略 | 要重装（有缓存时几分钟） |
| `.git/` | `git init` | 版本库 | — | 历史全丢 |

**为什么会出现 `duck_rl.egg-info/`**：`pyproject.toml` 里声明了 `[tool.setuptools] packages = ["duck_rl"]`，uv 会把这个项目**当成一个包先构建再安装**（所以 `Installed … + duck-rl==0.1.0 (from file:///home/gh/duck_rl)`）。egg-info 就是这次构建留下的元数据，属于副产物，不该进仓库。

需要补一条忽略规则：

```bash
echo '*.egg-info/' >> .gitignore
git status --short   # 确认 egg-info 已不被跟踪
```

**三条容易混的命令**：

| 命令 | 作用 |
|---|---|
| `uv add <pkg>` | 改 `pyproject.toml` → 重新解析 → 更新 `uv.lock` → 安装 |
| `uv sync` | 按 `pyproject.toml` + `uv.lock` 把环境对齐（不新增依赖声明） |
| `uv run <cmd>` | 在 `.venv` 里执行命令，顺便确保环境是最新的 |

---

## Step 2　MJCF 资产与模型自检

### 资产清单（实际拷贝结果，2026-10-31 13:57）

```bash
cp -r ~/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/. ~/duck_rl/assets/
```

| 文件 | 作用 |
|---|---|
| `open_duck_mini_v2.xml` | **机器人本体**：body 树、14 关节、执行器、传感器 |
| `open_duck_mini_v2_backlash.xml` | 本体 + 舵机回差扰动版本 |
| `joints_properties.xml` / `sensors.xml` | 被本体 `<include>` 的片段（关节参数、传感器定义） |
| `scene_flat_terrain.xml` | **平地场景**（地面 + 光照）← **env 与自检都用这个** |
| `scene_flat_terrain_backlash.xml` | 平地 + 回差 |
| `scene_rough_terrain_backlash.xml` | 崎岖地形 + 回差 |
| `config.json` | onshape-to-robot 的导出配置（MJX 轻量化用） |
| `assets/`（子目录） | STL 网格 |

**与计划的偏差**：目录里**没有 `scene.xml`**，最贴近"平地"的是 `scene_flat_terrain.xml`。所以第 4 步 `DuckEnv` 的默认 xml 路径、第 10 步回放脚本的 `--xml` 都要指它，不能写成 `assets/scene.xml`。

### 2A　先读 XML，再写脚本（2026-10-31 14:05 调整顺序）

原本打算直接写 `check_model.py`，改为**先读原文形成假设，再用脚本核对**——脚本是验证工具，不是替代阅读。

#### include 链（决定该读哪几个文件）

```
scene_flat_terrain.xml
└─ <include file="open_duck_mini_v2.xml"/>   ← 本体，502 行
   ├─ <default class="open_duck_mini_v2">    通用默认（kp=50 dampratio=1）
   ├─ <default class="sts3215">              舵机关节默认（见下表）
   ├─ <worldbody> base(z=0.22) → trunk → hip → knee → ankle
   ├─ <actuator> 第 486~501 行，14 个 <position class="sts3215" inheritrange="1">
   └─ <sensor> 第 26 行起，gyro / accelerometer / upvector / local_linvel …
```

**重要发现：`open_duck_mini_v2.xml` 里 `<include>` 数量为 0。** 也就是说 `joints_properties.xml` 和 `sensors.xml` 这两个文件**根本没被引用**，是别的代号留下的遗留物（它们的参数是 kp=17.8 / damping 0.60，与本体里 kp=13.37 / damping 0.56 不一致，读错了会一路错下去）。教训：**看到文件别当真，先追 include 链。**

#### 读出来的关键参数

| 项 | 值 | 来源 |
|---|---|---|
| 单位 / 网格目录 | `angle="radian" meshdir="assets"` | `<compiler>` 第 9 行 |
| 舵机关节默认 | `damping=0.56`、`frictionloss=0.068`、`armature=0.027` | `<default class="sts3215">` 第 45~47 行 |
| 位置伺服 | `kp=13.37`、`kv=0`、`forcerange=-3.23 3.23`（上一版 kp=17.12 被注释掉） | 第 49 行 |
| 控制范围 | 未显式写，靠 `inheritrange="1"` **自动继承关节限位** | `<actuator>` 各条 |
| 浮基 | `<freejoint name="floating_base"/>` | 第 59 行 |
| 关节限位 | hip_yaw ±0.524（30°）、hip_roll ±0.436（25°）、hip_pitch −1.222~0.524、knee ±1.571、ankle ±1.571 | 各 `<joint>` |
| 躯干初始高度 | `base pos="0 0 0.22"` | 第 58 行 |
| IMU 位置 | site `imu` 在 `(-0.08, 0, 0.05)` | 第 60 行 |
| 地面 | plane，`friction=0.6`、`condim=3`、`contype=1 conaffinity=0` | scene 文件 |
| 关键帧 | **在 scene 文件里**，不在本体；`home` 里 knee≈1.379 rad（≈79°）→ **蹲姿** | scene 文件末尾 |

**两个会直接导致后续 bug 的点**：

1. **keyframe 在 scene 文件里**。如果只加载 `open_duck_mini_v2.xml`，`nkey=0`，`reset` 拿不到姿态；必须加载 `scene_flat_terrain.xml`。
2. **仿真 kp=13.37 与笔记里的"真机 Kp=30"不是一回事**。前者是 MJCF 里 BAM 辨识出的等效位置增益，后者是真机 Runtime 的姿态控制参数。仿真以 MJCF 为准，`env.py` 里不要想当然写 30。

### 2A 附　为什么用完整模型、哪些能删（决策记录）

| 类别 | 内容 | 处理 |
|---|---|---|
| 物理必需 | `open_duck_mini_v2.xml`（本体）、被引用的 **28 个 STL**、`scene_flat_terrain.xml`（地面 + keyframe） | **不能删** |
| 完全未被引用 | `joints_properties.xml`、`sensors.xml` | 可删（留着但别看） |
| 仅切片元数据 | 54 个 `.part` 文件（本体里 `.part` 引用数为 **0**） | 可删，省空间 |
| 备用变体 | `*_backlash.xml` 三份、`config.json` | 留着，做 sim2real 对照与重新导出用 |

**自由度构成**：14 = 左腿 5 + 右腿 5 + 头颈 **4**（`neck_pitch`、`head_pitch`、`head_yaw`、`head_roll`）。

**想简化时的正确做法**：删 **joint**、保留 **body**。joint 定义自由度，body 承载质量/惯量/几何；直接删 body 会改变总质量与质心，动力学全变。删掉 4 个头颈 joint 后是 10 DoF，同时要改 `constants.py` 的 `NUM_JOINTS` 与 actuator 段。

**视觉 mesh 不要删**：它们 `contype=0 conaffinity=0`，不参与碰撞，只用于渲染。删了就没法在 viewer 里看、也录不了视频。

**当前决定（2026-10-31 14:15 修订）：按"从零开发"的最小集裁剪。** 保留打开模型必需的文件，其余删掉，需要时从参考仓库随时取回（一条 `cp` 即可，因为本来就是拷来的）。

#### 裁剪方案

| 保留（30 个） | 说明 |
|---|---|
| `assets/scene_flat_terrain.xml` | 入口场景 |
| `assets/open_duck_mini_v2.xml` | 本体 |
| `assets/assets/*.stl`（28 个） | 被 `<asset>` 引用的网格，一个都不能少（已用引用清单与目录逐项比对，28 对 28 完全吻合） |

| 删除 | 理由 |
|---|---|
| `joints_properties.xml`、`sensors.xml` | 未被 include，参数还是旧版 |
| `open_duck_mini_v2_backlash.xml`、`scene_flat_terrain_backlash.xml`、`scene_rough_terrain_backlash.xml` | 备用变体，用到再取 |
| `config.json` | onshape-to-robot 导出配置，重新导出模型时才需要 |
| `assets/assets/*.part`（28 个） | 切片元数据，MJCF 引用数为 0 |
| `assets/assets/hfield.png` | 只有崎岖地形场景用，平地不需要 |

取回命令（任何被删文件）：

```bash
cp ~/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/<文件名> ~/duck_rl/assets/
```

#### 裁剪后的验证（2026-10-31 14:19）

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('assets/scene_flat_terrain.xml'); \
print('nq',m.nq,'nv',m.nv,'nu',m.nu,'nbody',m.nbody,'ngeom',m.ngeom,'nkey',m.nkey)"
# → nq 21  nv 20  nu 14  nbody 18  ngeom 47  nkey 1
```

| 数字 | 含义 |
|---|---|
| `nq=21` / `nv=20` | 浮基 7/6 + 关节 14 |
| `nu=14` | 14 个位置伺服执行器 |
| `nbody=18` | floor + base + trunk + 两腿各 5 + 头颈 4 + 附属 |
| `ngeom=47` | 28 视觉 mesh + 足底碰撞体 + 地面 + 附属 |
| `nkey=1` | `home` 关键帧存在 |

最终 `assets/` 只剩：2 个 xml + 28 个 stl + `SOURCE.md`。





### 开发环境：VS Code Remote-WSL

| 项 | 状态 |
|---|---|
| VS Code（Windows 侧） | ✅ 已装：`D:\Users\ahs\AppData\Local\Programs\Microsoft VS Code` |
| WSL 扩展 | ❌ 未装（必装，否则连不进 WSL） |
| Python 扩展 | ❌ 未装（含 Pylance） |

用法：WSL 里 `cd ~/duck_rl && code .`，首次会连到远程窗口并在 WSL 内装 `vscode-server`。
**不要**用 Windows 端打开 `\\wsl.localhost\Ubuntu-22.04\...`——那样走 9p，IO 慢、扩展也跑在 Windows 侧。

---

## Step 2B　`check_model.py` 自检结果（2026-10-31 14:56）

自检脚本跑通，六段全部有输出。记录关键数据与**两处事前猜错的地方**。

### 模型事实（以数据为准）

| 项 | 值 | 备注 |
|---|---|---|
| 规模 | `nq21 nv20 nu14 nbody18 ngeom47 nkey1` | 与 XML 阅读结论一致 |
| **总质量** | **2.1071 kg** | ❗事前估计 0.7~1.0 kg **错了**（见下） |
| 质量分布 | trunk 0.699 kg（33%）、**head_assembly 0.407 kg（19%）**、hip_pitch body 0.124 kg、两腿各 0.414 kg | 头部接近全机 1/5，质心偏高 |
| 限位最窄 | **left_hip_roll 0.873 rad（50°）** | ❗事前猜 ankle **错了**；ankle/knee 都是 ±90° |
| 关节轴 | 14 个 hinge **全部是 `[0,0,1]` 默认轴** | 数据证明"朝向由 body 的 quat 决定，不是 axis" |
| 左右镜像 | hip_pitch `[-1.222,+0.524]` vs `[-0.524,+1.222]`；knee/ankle 两侧相同 | 印证 keyframe 里 `-0.630 / +0.635` |
| 执行器 | 14 个 `kp=13.37`、`kv=0`、`forcerange=±3.23` | 与 `sts3215` class 一致 |
| `inheritrange` | **14 行 ctrlrange 与 jnt_range 逐行相等** | 第三题结案：env 里不必再设 ctrlrange |
| 初始姿态 | z=0.15 m，knee 78.4°/79.0° → **蹲姿** | `ctrl` 与 `qpos` 关节部分完全相同 |
| 碰撞 | 47 个 geom 里**只有 3 个参与碰撞**：左右足底 + 地面 | 足底 friction 默认 1.0，地面 0.6 且 `priority=1` |
| 传感器 | 15 个共 46 维；`upvector` 在 adr=9、`gyro` 在 0、`local_linvel` 在 3 | 观测可直接读，不用自己推四元数 |
| **质心** | 整机质心 `[-0.0256, -0.0006, 0.1940]`，高度 **0.194 m**；浮基原点 0.15 m → 质心高出 0.044 m | 质心几乎与髋轴同高（base 0.15 + 髋偏移 0.0459 = 0.196），**略偏后 2.6 cm** |

### 两条教训

1. **质量、限位这类数字不能猜。** 我按"小鸭子应该很轻"估了 0.7~1.0 kg，实际 2.1 kg——光 14 个 STS3215 舵机就 0.8 kg 以上，加上打印件与电池，2.1 kg 合理。限位也一样：以为是 ankle 最窄，实际是髋部 roll（侧向摆动受限 50°）。
2. **"没写 axis" ≠ "没有轴"。** 全部 hinge 都是默认 `[0,0,1]`，真实朝向由各层 body 的 `quat` 决定。这一点不看打印出来的 `jnt_axis` 很容易误判。

### 脚本增强

加了 `print_mass_com()`：打印最重的 5 个 body + 用 keyframe 摆好姿态后正运动学算出的**整机质心位置与高度**。质心高度是"能不能走"的物理前提，比总质量更关键。


### 质心数据的三个用处

1. **判摔倒的阈值要用浮基高度，不是质心。** keyframe 里 base z=0.15 m，`config.py` 的 `min_height` 取 0.09~0.10（低 5~6 cm 就算倒），比用质心直观。
2. **质心略靠后 2.6 cm。** 静止时重心在脚掌中心之后，策略必须主动前倾才能站住；这也是"高度/姿态奖励不能省"的原因。
3. **env 不需要改 MJCF 的执行器参数。** MJCF 里已经是 position 伺服（kp=13.37、kv=0、forcerange=±3.23、inheritrange=1），BAM 辨识值就在里面。`env.py` 只负责 `action → ctrl` 的映射与剪裁，**不要覆盖 gainprm/biasprm/forcerange**，否则等于扔掉辨识结果。

---

## Step 3　`config.py` + `constants.py`（15:22 完成）

| 项 | 结果 |
|---|---|
| 观测契约 | `print_contract()` 输出 **51 维**，六段区间 `[0:3][3:6][6:9][9:23][23:37][37:51]` |
| 传感器切片 | `upvector → slice(9,12)`、`gyro → slice(0,3)`，与 check_model 输出一致 |
| config | 22 个字段正常展开（6 组：时间/动作/观测缩放/指令/奖励/终止） |
| commit | `c4724ab feat: 配置与观测契约（51 维）` |

**这一步的两个决定**（写下来的理由，防止以后自己都忘了）：

1. **不加步态相位（phase 两维）**。phase 是程序生成的时钟不是传感器，加了学步态更快但步频被锁死；先让策略从关节速度/角速度/重力里自己学节奏。若训出"双足同跳"，再补回来。
2. **新增 4 个 `sigma_*` 参数**。原来 `exp(-误差²/0.25)` 里的分母写死在公式里，提出来变成可调——调奖励时经常要动，但大多数人忘了它在公式里。

---

## Step 4　`env.py` 骨架与 `reset`（15:34 完成）

### 验收输出

```
obs_size 51 | action_size 14 | n_substeps 10
obs shape  (51,)
qpos[:7]   [0.   0.   0.15 1.   0.   0.   0.  ]
height     0.15
ctrl[0:3]  [ 0.002  0.053 -0.63 ]
command    [ 0.2946  -0.2455   0.4026 ]   ← 随机采样，每次 reset 都不同
```

`n_substeps=10` 就是"策略 50 Hz ÷ 物理 500 Hz"的实现；`ctrl[0:3]` 与 keyframe 左腿前三个关节角完全一致，说明 `default_pos = key_qpos[0][7:]` 取对了位置。

commit：`ae6fa94 feat: env 骨架与 reset`

### 踩坑：CUDA 启动时 OOM 警告

```
Failed to allocate device memory of 4.50GiB: CUDA_ERROR_OUT_OF_MEMORY
Failed to allocate device memory of 4.05GiB: CUDA_ERROR_OUT_OF_MEMORY
```

**不是错误，是 JAX 在降级**：它默认预分配 75% 显存（4.5 GB），而当时空闲只有 4994 MiB（Windows 桌面占用 1001 MiB：explorer、企业微信 FlutterPlugins、NVIDIA Overlay 等），于是它一路降档重试，最后跑通了。

代价是显存碎片化，训练时可能变成真 OOM。解决：

```bash
echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false' >> ~/.bashrc
source ~/.bashrc
```

生效后可用显存从 4968 → 5056 MiB。训练前习惯性看一眼：`nvidia-smi --query-gpu=memory.free --format=csv`，低于 4 GB 就把 envs 降到 512。

### 新增工具：`scripts/view_model.py`（看画面）

```bash
uv run python scripts/view_model.py            # 默认 assets/scene_flat_terrain.xml
```

它做三件事：摆到 keyframe 姿态 → `ctrl = key_ctrl[0]`（位置伺服目标设为当前姿态）→ 按 1 倍实时速度跑物理。窗口操作：左键旋转、右键平移、滚轮缩放、Esc 退出。

**为什么必须设 `ctrl`**：不设的话默认全 0，伺服会拼命把 14 个关节拉到 0 位，鸭子抽搐着摊平——看起来像模型坏了，其实是没给目标。

**判读表**（看到什么 → 查什么）：

| 画面 | 结论 | 查 |
|---|---|---|
| 蹲住不动（轻微下沉） | 全对 | 继续下一步 |
| 慢慢塌下去 | 伺服无力 / ctrl 未设 | 打印 `d.ctrl[:]` 是否全 0 |
| 开局抽搐抖动 | 初始姿态与伺服目标不一致 / 穿模 | 对比 `d.qpos[7:]` 与 `d.ctrl` |
| 穿过地面下坠 | 地面没加载 | 是否加载了 scene 而非本体 xml |

### 实测：高度曲线藏着一个细节

```
t=0.00s 高度=0.1500（keyframe 值）
t=0.50s 高度=0.1660   ← 先弹起来 1.6 cm
t=1.00s 高度=0.1600   ← 然后回落
t=4.00s 高度=0.1600   ← 稳定
```

原因：keyframe 里 z=0.15 时脚底其实**略微插进地面约 1 cm**，接触力把它弹起来，最终稳态是 **0.160 m**。

长时观察（连续 100 秒以上）稳态为 **0.1599 m**，抖动不超过 1 mm：

```
t= 96.00s 高度=0.1599   ...   t=102.50s 高度=0.1599
```

**两条推论（已落地）**：
1. 稳态高度是 **0.16 而不是 keyframe 的 0.15**。已在 `config.py` 增加 `height_ref = 0.16`，第 7 步"高度维持"奖励用它当参考——拿 0.15 当参考，策略一开局就被白扣 1 分。
2. `min_height=0.10` 相对稳态有 6 cm 余量，合理，不改。
3. 连续 100 秒高度不漂移，同时验证了四件事：模型质量/惯量正确、执行器 kp=13.37 撑得住、初始姿态可用、足底接触稳定。

---

## 训练纪要（详细数据见《训练实验记录》）

九轮训练的过程、假设、公式推导都在《训练实验记录-duck_rl.md》，这里只留一张索引表，
方便回溯"某轮大概是什么时候跑的、产物叫什么"。

| 轮 | 日期 | 步数 | 一句话 | 产物 |
|---|---|---|---|---|
| 1 | 10-31 17:21 | 18.0M | 缺归一化，只会站 | `params_18022400.pkl` |
| 2 | 11-02 11:40 | 32.7M | 修归一化，0.38 m/s，但 30 秒掉头 | `policy_exp2.onnx` |
| 3 | 11-02 15:30 | 40.96M | 加 yaw（53 维），30 秒不歪 | `policy_exp3.onnx` |
| 4 | 11-02 17:00 | 40.96M | 改采样，六方向全过，但前进走弧线 | `policy_exp4.onnx` |
| 5 | 11-02 18:30 | 中断 | 只调采样，证明治不了漂移 | — |
| 6 | 11-02 19:05 | 44.24M | 加 heading，直线 1.6°，转向死了 | `policy_exp6.onnx` |
| 7 | 11-02 20:20 | 36.0M | heading 门控，六方向 84~104% | `policy_exp7.onnx` |
| 8 | 11-03 11:10 | 40.96M | sigma 回 0.1，直线 3.0° + 转向 93.7% | `policy_exp8.onnx` |
| 9 | 11-03 13:31 | 65.5M | 域随机化，质量维度稳 3.5 倍 | `policy_dr1.onnx` |

### 训练期踩过的六个坑（按被发现的时间）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | `carry input/output must have same pytree structure`（0→2 键） | `reset` 返回 `metrics={}`，`step` 返回两键 | 两边都走 `_metrics()` |
| 2 | 同样的错（10→9 键，差 `reward`） | brax 包装层往 metrics 加了 `reward`，我们整字典替换 | 改成 `{**state.metrics, **新字典}` |
| 3 | `jax.device_put_replicated is deprecated` | brax 0.14.2 用了 jax 0.12 删掉的 API | `train.py` 顶部打兼容补丁 |
| 4 | 导出"对齐通过"但策略不会走 | `verify` 的 JAX 参考也没归一化 → 两边都错 = **假阳性** | 参考实现必须与训练路径一致 |
| 5 | `NameError: name 'jnp' is not defined` | `env.py` 用 `jp`，其余文件用 `jnp`，复制代码撞车 | 统一为 `jp` |
| 6 | 首次 `reset`/`step` 编译要 60~160 秒 | XLA 编译，非卡死 | 开 `jax_compilation_cache_dir` |


## 附：工具链迭代记录

### 代码逐行注释（6 个核心文件）

6 个核心文件全部加了行内注释，**直接写在 WSL 的代码里**，不另开文档：

`constants.py` · `config.py` · `rewards.py` · `env.py` · `train.py` · `export_onnx.py`

**校验方法（可复用）**：用 `tokenize` 提 token，过滤掉 COMMENT 和 STRING，再和备份逐 token 比对。
6 个文件结果都是「代码 token 完全一致」——证明只加了注释、没动一行逻辑。

- 校验脚本：`Z:/E_data/qr_ggh/duck/.workbuddy/tmp/chk.py`
- 原文件备份：`Z:/E_data/qr_ggh/duck/.workbuddy/backup_20261102/`（要回滚直接 cp 回去）


### 训练输出增强（心跳线程）

原状态：brax 只在 eval 时回调一次，间隔 30 秒到 3 分钟，看起来像卡住。

| 改动 | 参数 | 效果 |
|---|---|---|
| eval 次数 | `--num_evals`（默认 50，原 10） | 输出频率 ×5 |
| eval 环境数 | `--num_eval_envs`（默认 64，原 128） | eval 开销减半，抵消上面的代价 |
| 心跳 | `--heartbeat`（默认 2 秒） | 守护线程，每 2 秒一行进度条 + 步数 + sps + ETA + 显存温度 |
| 明细 | 每次 eval | 多打**回合长度** + **七项奖励分解** |

**为什么不逐帧 print**：`training_step` / `actor_step` 被 `lax.scan` 编译成一张 GPU 图，
运行期没有 Python 语句可执行，print 进不去。心跳线程（纯 Python，读共享变量）是唯一干净的办法。

**回合长度是最有用的读数**：站着不动也能跑满 1000，所以它只说明"不摔"，不说明"会走"。
判断会不会走得看实测速度。


### `watch_policy.py` 的五次迭代

回放脚本从"能看"到"好看"，一共改了五轮，每轮都是被一个具体现象逼出来的：

| 现象 | 原因 | 修法 |
|---|---|---|
| 窗口一闪就没 | 脚本跑完 500 步（10 秒）就 `__exit__` 关窗 | 加 `--hold`，跑完保持窗口 |
| 鸭子走两步出画 | 相机固定在世界原点 | 改成跟随躯干位置 |
| 视角跟着鸭子转，像原地踏步 | `mjCAMERA_TRACKING` 的 azimuth 是**相对躯干朝向**的 | 换 `mjCAMERA_FREE` + 每步只更新 `lookat`，方向固定在世界系 → `--cam follow` |
| 播放速度过快 | 脚本里**根本没有节流** | 加 `--speed`（1.0 实时 / 0.5 慢放 / 0 不限速） |
| Ctrl+C 后卡住不返回提示符 | MuJoCo 退出时销毁 EGL 上下文，WSLg 下偶发永久阻塞 | 末尾 `sys.stdout.flush()` + `os._exit(0)`，跳过所有析构 |

新增参数总览：`--cam {follow,tracking,fixed}`、`--cam-azimuth`、`--cam-elevation`、
`--cam-distance`、`--cam-follow-gain`、`--speed`、`--hold`。

**相机跟随的原理**（值得记住）：tracking 模式会让镜头随躯干旋转，所以看起来永远在"向前走"；
要看清转向，必须自己拿 free 相机、只同步 `lookat`。


### 当时发现的问题：30 秒后掉头

30 秒回放（`--cmd 0.4`）的数据：

```
t=15.0s  前进 +4.163 m
t=17.0s  前进 +4.217 m   ← 峰值
t=30.0s  前进 +1.298 m   ← 退回去 3 米
```

高度全程 0.149~0.156，**没摔，是掉头走回去了**。

**根因在观测契约**：策略看不见自己朝哪。

| 观测分量 | 对偏航（yaw）旋转的反应 |
|---|---|
| `gravity`（3 维） | **完全不变**——绕 z 轴转多少度，重力在体系里的投影都一样 |
| `gyro`（3 维） | 只给瞬时角速度，不给累积角度 |
| `q` / `qd` / `last_action` | 关节量，不含躯干朝向 |

所以微小偏航偏差没有任何信息能纠正它，走十几秒就累积成掉头。这是双足 RL 的经典缺陷。

**解法（需改契约 + 重训）**：观测里加 yaw 的 sin/cos 两维（相对回合起点），`OBS_SIZE` 51 → 53。

| 位置 | 改什么 |
|---|---|
| `constants.py` | `OBS_LAYOUT` 加 `("yaw", 2)`，`OBS_SIZE` 改 53，assert 同步 |
| `env.py::reset` | `info` 里记起始 yaw |
| `env.py::_get_obs` | 算当前 yaw − 起始 yaw，取 sin/cos 拼进去 |
| 部署端 | 真机也要有同样的累积量（IMU 积分或里程计） |


### 2026-11-02 动过的文件

| 文件 | 改动 |
|---|---|
| `duck_rl/constants.py` `config.py` `rewards.py` `env.py` `train.py` | 逐行注释（逻辑未变） |
| `duck_rl/train.py` | 加心跳线程、`--num_evals`、`--num_eval_envs`、`--heartbeat`、`--dump-metrics`；eval 输出七项分解 |
| `duck_rl/export_onnx.py` | **默认带归一化**；verify 传 `running_statistics.normalize`；`--normalize` → `--no-normalize` |
| `scripts/watch_policy.py` | 相机三种模式、跟随强度、俯仰/距离/角度、倍速、`--hold`、硬退出 |
| `assets/` | 无变化 |

---

## 续写约定

1. 每完成一步：**追加时间、命令、关键输出（3~5 行，不贴整屏）、遇到的问题**
2. 验收没过的步骤不要标 ✅，宁可留着说明卡在哪
3. 版本号变了（重装、升级）就在「环境基线」里更新，别覆盖旧记录

---
