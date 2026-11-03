# Open Duck Mini 仿真训练环境搭建（本机 Windows 实操版）

> **本文是「装环境」**：从零到"能跑仿真"的完整命令，含本机实测的偏差与加速方案。
> 所有命令都是你复制执行，不含代跑。
>
> 读法：**先看下面的「状态」节**（实际结果与偏差），再按需翻第三～八节的命令。
> 不想一步步看的话，直接跳第八节的镜像与代理配置——那是国内网络下最省时间的一段。

---

## 状态：已完成（2026-10-31 11:15）

下面第一~七节是**当时的操作方案与命令**，可作存档与换机复现。实际落地的结果与偏差记录在此，读文档时以这一节为准：

| 项 | 计划 | 实际落地 |
|---|---|---|
| 安装位置 | `--location G:\WSL\Ubuntu-22.04` | ✅ 生效，vhdx 在 `G:\WSL\Ubuntu-22.04\ext4.vhdx`，未用 `--move` |
| WSL 版本 | 期望 2.x | 实测 **3.0.3**，支持 `--location`，WSLg 内置 |
| Ubuntu | 22.04 | 22.04.5 LTS，Linux 用户 `gh` |
| uv 安装 | `curl astral.sh \| sh` | ❌ 报 `SSL: unexpected eof`；✅ 改 `python3 -m pip install --user -i 清华源 uv`，得 **uv 0.13.6** |
| Python | 3.12 | 3.12.16（`uv python install`） |
| 依赖 | jax[cuda12] + TF | 121 包，**7 分 51 秒**，`.venv` 8.3 GB |
| tensorflow 冲突 | 预判需换 tf-cpu | ✅ 未发生，**tensorflow 2.21.0 与 jax 0.12.0.1 共存正常** |
| GPU | `[CudaDevice(id=0)]` | ✅ 达标；`which nvidia-smi` → `/usr/lib/wsl/lib/nvidia-smi` |
| 回放 | MuJoCo 窗口出小鸭 | ✅ 达标（`mujoco_infer.py -o BEST_WALK_ONNX_2.onnx`） |

**过程中发现的额外关键点**（原文档未覆盖）：

1. **下载慢的真因是代理没镜像**：Windows 侧 FlClash 在 `127.0.0.1:7890`，WSL 默认 NAT 模式访问不到宿主 localhost。已在 `.wslconfig` 加 `networkingMode=mirrored`，并在 `~/.bashrc` 设 `http_proxy/https_proxy`，同时把 `mirrors.tuna.tsinghua.edu.cn` 放进 `no_proxy` 直连（国内镜像绕代理反而更快）。
2. **`--location` 参数确实存在**（微软文档：*"指定要将 WSL 分发版安装到哪个文件夹"*），比"先装再 `--move`"省一步。
3. **在 `/mnt/c/WINDOWS/system32` 下敲 `nvidia-smi` 会跑到 Windows 的 exe**，证明不了 WSL 里的 GPU 可用；必须 `cd ~` 后用 `which nvidia-smi` 确认指向 `/usr/lib/wsl/lib/`。

---

## 一、本机硬件实测结论

| 项目 | 实测值 | 对训练的影响 |
|---|---|---|
| CPU | Intel i5-11400，6 核 12 线程 @2.6 GHz | 够用；并行环境数由 GPU 决定，不吃 CPU |
| 内存 | 16 GB（2×8 GB DDR4-3200） | **偏紧**，WSL2 必须限内存，否则训练时主机卡死 |
| 显卡 | RTX 3060 Laptop，**6144 MiB 显存**，95 W | 可训练，但 **4096 envs 会 OOM**，起手 1024 |
| 显卡驱动 | 617.42，CUDA UMD 13.4，**未装 CUDA Toolkit** | 无需装 Toolkit，pip 版 jax 自带 CUDA 运行时 |
| 磁盘 | C 37.2 / D 39.3 / E 22 / **G 99.9 GB（已清空，专供 WSL）** / Z 407.9 | WSL 发行版独占 G: |
| 系统 | Windows 11 家庭版 | 支持 WSL2 + WSLg（图形窗口） |
| WSL 状态 | **功能已启用，但未安装任何发行版** | 只需装发行版，**不用重启、不用开 Hyper-V** |
| VBS/内核隔离 | 正在运行（VirtualizationBasedSecurityStatus=2） | 正常，WSL2 不受影响 |
| 已有虚拟化软件 | VMware Workstation 16.1.0 | 不用于本任务，见第二节 |
| 工具链 | Python 3.13 / 3.11 / 3.12，git 2.55，**无 uv** | 装 uv 即可 |

---

## 二、为什么是 WSL2，不是虚拟机

| 方案 | GPU 加速 | 结论 |
|---|---|---|
| **WSL2** | 官方直通（`/dev/dxg` + Windows 驱动），JAX CUDA 可用，WSLg 能弹 MuJoCo 窗口 | **选它** |
| VMware Workstation 16 | 只有虚拟 3D 加速，**不支持 CUDA**；6 GB 显存也不够切给虚拟机 | 否决 |
| 原生 Windows Python | **JAX 官方明确不支持 Windows 上的 CUDA**（"use WSL2 for CUDA support"） | 只能 CPU，3e8 步跑不动 |
| 云 GPU（AutoDL 等） | 完整 4090/A100 | 备选，见第六节 |

补充：WSL 的 `VirtualMachinePlatform` 与 `Microsoft-Windows-Subsystem-Linux` 两个功能**已启用**，只缺发行版，所以这条路成本最低。

训练栈的真实依赖（已核对仓库 `pyproject.toml` 与源码）：

- 框架：**mjlab 那条线的真身是 `Open_Duck_Playground`** —— 基于 `mujoco_playground`（MJX，JAX 版 MuJoCo）+ **Brax PPO**（`brax.training.agents.ppo`），不是 PyTorch/rsl_rl，没有 `.pth`，checkpoint 是 orbax
- 依赖：`jax[cuda12]`、`flax`、`brax`、`mujoco_playground`、`tensorflow` + `tf2onnx`（导出 ONNX 用）
- 仓库自带 `BEST_WALK_ONNX_2.onnx` 回放脚本：`mujoco_infer.py`

---

## 三、分步操作

### Step 1　安装 Ubuntu-22.04 到 G 盘（管理员 PowerShell）

> **先纠正一个常见误解**：`ubuntu-22.04.5-desktop-amd64.iso` 是给**物理机/VMware** 用的安装盘，**WSL 完全用不到它**（WSL 只认 rootfs 的 `.tar.gz` 或应用商店分发的 appx）。那张 ISO 留着以后装物理机或给别人装系统用。若硬要用 ISO，只能走 VMware/裸机双系统，而虚拟机的 CUDA 通路是不通的（见第二节）。

本机现状：Win11 Build 26200，WSL 功能已启用但**没装任何发行版**，Store 可用。

**路径 A（推荐）：一条命令直接装到 G**

```powershell
# --location 指定发行版落地目录（WSL 2.4.4+ 支持；文件夹不存在会自动建）
wsl --install -d Ubuntu-22.04 --web-download --location G:\WSL\Ubuntu-22.04

# 首次会提示重启，重启后 Ubuntu 自动启动，让你建 Linux 用户名和密码（记住它，sudo 要用）
# 验收
wsl -l -v
dir G:\WSL\Ubuntu-22.04\ext4.vhdx
```

**注意：`cd G:` 对安装位置没有任何影响**，安装路径由参数决定，与当前工作目录无关。另外 PowerShell 的 `cd G:` 只切了 G 盘的工作目录，盘符跟着变，但下一次再 `cd` 一个相对路径就会跳回 C 盘，容易造成"我在 G 盘"的错觉。

若 `--location` 报"无法识别的参数"（旧版 WSL），退化成"先装再搬"：

```powershell
wsl --install -d Ubuntu-22.04 --web-download
wsl --shutdown
wsl --manage Ubuntu-22.04 --move G:\WSL\Ubuntu-22.04
```

**路径 B（备选）：rootfs 直接导入到 G，不依赖 Store、不占 C 盘**

适合 Store 下载卡住、或想要完全可控的安装位置：

```powershell
mkdir G:\WSL
curl.exe -L -o G:\WSL\u2204.tar.gz https://cloud-images.ubuntu.com/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz
# 文件 341 MB，Ubuntu 官方 WSL 专用 rootfs

wsl --import Ubuntu-22.04 G:\WSL\Ubuntu-22.04 G:\WSL\u2204.tar.gz --version 2
wsl -d Ubuntu-22.04
```

> `--import` 之后默认以 **root** 登录。要恢复正常用户，先建用户再写 `/etc/wsl.conf`：
> ```bash
> sudo adduser duck            # 建普通用户
> sudo usermod -aG sudo duck
> sudo tee /etc/wsl.conf > /dev/null <<'EOF'
> [user]
> default=duck
> [boot]
> systemd=true
> EOF
> ```
> 然后 Windows 侧 `wsl --shutdown`，重新 `wsl -d Ubuntu-22.04` 即以 `duck` 登录。

两条路选一条即可，**A 更省事，B 更可控**。装完都做 Step 2。

> **C 盘上不可避免的部分**：WSL 本体（`wsl.exe` 这层运行时）固定在 `C:\Program Files\WSL`，约 150 MB，无法迁移也不值得迁移；能搬到 G 盘的只有发行版那块 `ext4.vhdx`（也就是整个 Ubuntu 系统 + 后续所有 Python 环境，几十 GB 级）。

### Step 2　限制 WSL 内存（16 GB 主机必做）

在 Windows 侧新建 `C:\Users\ahs\.wslconfig`：

```ini
[wsl2]
memory=10GB
processors=6
swap=8GB
```

改完 `wsl --shutdown` 生效。不设这个，JAX 预分配显存 + 编译缓存容易把主机内存吃干。

### Step 3　验证 GPU 直通（在 Ubuntu 终端里）

```bash
nvidia-smi
```

应能看到 `NVIDIA GeForce RTX 3060 Laptop GPU` 和 6144 MiB。

> **关键**：WSL 里**绝对不要装 NVIDIA 驱动**，驱动由 Windows 侧提供，装了必崩。看不到卡就更新 Windows 显卡驱动。

### Step 3.5　图形界面（WSLg）与无窗口备选

Win11 自带 **WSLg**，单个 Linux GUI 程序可以直接弹窗（X11 + Wayland），不需要装 VcXsrv/X410 之类。但它**不是完整桌面环境**（没有 GNOME/KDE 面板），只需要显示 MuJoCo 窗口的话完全够用。前提：装的是 Store 版 WSL（本方案即如此）+ Windows 侧有支持 vGPU 的显卡驱动（617.42 满足）。

验证三条：

```bash
wsl --version            # 输出里应有 WSLg 版本号（在 Windows 侧终端执行）
ls /mnt/wslg             # 存在即 WSLg 已挂载
echo $DISPLAY            # 应为 :0
sudo apt install -y x11-apps && xeyes    # 弹出眼睛窗口 = GUI 通路 OK
```

OpenGL 走 Mesa 的 d3d12 驱动（映射到 DirectX 12）获得硬件加速。笔记本是双显卡，WSLg 默认可能绑到 Intel 核显渲染窗口——这对显示没有影响，**CUDA 训练仍然只用 3060**。

**如果 MuJoCo 窗口起不来**，不必死磕，改成无窗口渲染：

```bash
MUJOCO_GL=egl uv run playground/open_duck_mini_v2/mujoco_infer.py -o xxx.onnx   # 或 osmesa 纯软渲染
```

或直接让训练脚本把 rollout 录成 mp4（仓库本身用 mediapy 出视频），在 Windows 侧播放。**训练过程本来就不需要任何图形窗口**，GUI 只在你"想看小鸭走两步"时才需要。

### Step 4　装基础工具与 uv

```bash
sudo apt update && sudo apt install -y git build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # 或重开终端
uv --version
uv python install 3.12           # Ubuntu 22.04 自带 3.10，仓库要求 >=3.11
```

### Step 5　取代码并装依赖（放 WSL 内部盘，别放 /mnt/ 下的 Windows 盘）

```bash
cd ~
git clone https://github.com/apirrone/Open_Duck_Playground.git
cd Open_Duck_Playground
uv sync                          # 会拉 jax[cuda12]，约 4~6 GB
```

验证 JAX 认到 GPU：

```bash
uv run python -c "import jax; print(jax.devices())"
# 期望输出里有 CudaDevice(id=0)
```

> **坑**：`tensorflow`（GPU 版）和 `jax[cuda12]` 会抢 CUDA 库。若上面这行报 CUDA 初始化错误，把 TF 换成 CPU 版即可（导出 ONNX 用不到 GPU）：
> ```bash
> uv pip install --force-reinstall --no-deps "tensorflow-cpu==2.18.*"
> ```

### Step 6　先验证仿真环境（不训练，用官方 ONNX 回放）

```bash
cd ~/Open_Duck_Playground
curl -LO https://github.com/apirrone/Open_Duck_Mini/raw/main/BEST_WALK_ONNX_2.onnx
uv run playground/open_duck_mini_v2/mujoco_infer.py -o BEST_WALK_ONNX_2.onnx
```

弹出 MuJoCo 窗口、小鸭能走 → **环境验收通过**，再进训练。

可用参数（已核对源码）：
- `-o/--onnx_model_path`：必填
- `--model_path`：默认 `playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml`
- `--standing`：测站立策略
- `--reference_data`：默认 `data/polynomial_coefficients.pkl`

### Step 7　正式训练

先做小规模 smoke test（20M 步，约几十分钟量级，验证不 OOM 再放大）：

```bash
cd ~/Open_Duck_Playground
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain --num_timesteps 20000000
```

跑通后按 README 的"当前最优"配置正式训：

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

- 显存不够时：改 `playground/common/runner.py` 里 `ppo_training_params` 的 `num_envs`（1024 → 512），或调小 `batch_size`
- 监控：`uv run tensorboard --logdir=<输出目录>`（输出目录由 `--output_dir` 决定，默认 `checkpoints`）
- 训练会自动调用 `export_onnx(...)`（`common/runner.py` 里已 import 并调用），产物即 ONNX

### Step 8　回放自训模型

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <你自己导出的.onnx>
```

对比官方 `BEST_WALK_ONNX_2.onnx`，能稳定走完一段再考虑买硬件。

---

## 四、预期与取舍

| 项目 | 判断 |
|---|---|
| 6 GB 显存能跑多大 | 1024 envs 稳，2048 需试，4096 基本会 OOM |
| 训练时长 | 1.5e8 步量级在 3060 Laptop 上预计 **十几小时**（未实测，先按 smoke test 校准） |
| 是否影响日常用 | 训练时 GPU 满载，主机绘图会卡，建议夜里跑 |
| 磁盘占用 | 代码 + 环境约 10 GB（WSL 内），checkpoint 按需几百 MB~几 GB |

---

## 五、坑位清单

1. WSL 里装 NVIDIA 驱动 → 必崩，驱动只装在 Windows
2. 在 `/mnt/g`、`/mnt/z` 等 Windows 盘上跑训练 → 跨文件系统 IO 慢一个数量级，代码必须放 WSL 内 `~`
3. Ubuntu 22.04 默认 Python 3.10 → 仓库要求 >=3.11，用 `uv python install 3.12`
4. 忘了搬盘 → 默认装 C 盘，C 只剩 37 GB，装完环境就满；G 盘是给 WSL 专用的
5. 直接上 4096 envs → 6 GB 显存 OOM
6. 以为是 PyTorch → 这条线是 JAX/Brax，没有 `.pth`，checkpoint 是 orbax
7. `tensorflow` 与 `jax[cuda12]` 抢库 → 换 `tensorflow-cpu`
8. 不改 `.wslconfig` → 16 GB 主机训练时容易整机假死

---

## 六、备选：云 GPU

若本机训练太慢（十几小时/轮、调奖励要跑好几轮）：

- 租 AutoDL / 恒源云 4090（约 2~3 元/小时），单轮训练压到 1~2 小时，调参成本几十元
- 本机保留为"开发 + 回放验证 + ONNX 调试"机，训练上云
- 上传只需代码 + ONNX 产物，仓库小，无需传大文件

---

## 七、下一步的分工

| 谁做 | 做什么 |
|---|---|
| 你 | Step 1~3（装 WSL、限内存、验证 nvidia-smi） |
| 你 | Step 4~6（装 uv、拉代码、回放官方 ONNX） |
| 一起 | Step 7 调 `num_envs` / 奖励参数，看训练曲线 |
| 你 | 达标后再启动选型采购（Open Duck Mini v2 套件） |

Step 6 出结果（小鸭在窗口里走起来）就说明整条链路通了，之后每一轮训练-回放都可以在本地闭环。

---

## 八、下载加速（国内网络实测）

慢通常发生在三个不同的地方，先分清卡在哪：

| 卡点 | 特征 | 对策 |
|---|---|---|
| **WSL 运行时本体**（几百 MB） | `C:\Program Files\WSL` 不存在、`%LOCALAPPDATA%\Packages` 下没有 Canonical 目录 | 改用 Store 手动装 "Windows Subsystem for Linux"（Store 有国内 CDN），或 GitHub Release 的 `.msi` |
| **Ubuntu 发行版**（约 600 MB） | 已有 `C:\Program Files\WSL`，但 `wsl -l` 仍为空 | **绕开 Store，用清华镜像的 rootfs 直接 import**（见下） |
| **Python 依赖**（jax cuda 轮子 4~6 GB） | `uv sync` 阶段 | 给 uv 配 PyPI 镜像（见下） |

**发行版走清华镜像**（与官方同一文件，实测本机 3.3 MB/s，341 MB 约 2 分钟）：

```powershell
mkdir G:\WSL
curl.exe -L -o G:\WSL\u2204.tar.gz https://mirrors.tuna.tsinghua.edu.cn/ubuntu-cloud-images/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz
wsl --import Ubuntu-22.04 G:\WSL\Ubuntu-22.04 G:\WSL\u2204.tar.gz --version 2
```

**apt 与 PyPI 换源**（装完立刻做，后面 `uv sync` 才是真正的大头）：

```bash
# apt → 清华
sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
sudo apt update

# uv/PyPI → 清华（写进 ~/.bashrc 持久化）
echo 'export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc
source ~/.bashrc
```

> 判断"慢"还是"卡死"：盯着目标目录的体积是否在增长（`C:\Program Files\WSL`、`%LOCALAPPDATA%\Packages`、`G:\WSL`）。十分钟一点没变才算卡死，否则只是带宽问题。
