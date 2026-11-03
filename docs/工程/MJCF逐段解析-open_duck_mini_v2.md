# MJCF 逐段解析：`open_duck_mini_v2.xml`（502 行）

> 对照对象：`~/duck_rl/assets/open_duck_mini_v2.xml`
> 配套：`scene_flat_terrain.xml`（75 行，包一层地面 + keyframe）
> 生成方式：第 2 行注明 `Generated using onshape-to-robot` —— **自动生成，不是人手写的**，所以满屏 `1e-17` 这种浮点噪声很正常，别去纠结。

---

> **本文是「读模型」**：`open_duck_mini_v2.xml` 每一段在说什么、哪些行可以跳过、
> URDF 过来的人最容易误解哪六处。附场景文件 `scene_flat_terrain.xml` 的逐段解析。
>
> 不需要从头读到尾——**先看第一节（URDF↔MJCF 差异）和第二节（文件骨架），
> 再按需要跳到对应行段**。视觉 mesh 那 400 行可以直接跳过。
>
> 模型参数的实际数值、自检输出见《施工日志》Step 2。


## 一、先解决认知落差：URDF 与 MJCF 的六处根本差异

| 概念 | URDF | MJCF | 后果 |
|---|---|---|---|
| **树怎么表达** | `<link>` + `<joint>` 两个平级元素，joint 用 `parent`/`child` 属性连 | `<body>` **嵌套**表示父子，`<joint>` 写在**子 body 内部** | MJCF 里搜不到 `parent=`，层级靠缩进 |
| **关节朝向** | joint 有 `origin`(rpy) + `axis` | body 有 `pos` + `quat`；joint **常常不写 axis**，默认是 body 自身的 z 轴 | 左右镜像的符号差异来自 body 的 `quat`，不是 axis |
| **驱动** | 没有（靠 `ros2_control`/`transmission`） | 独立的 `<actuator>` 段，`joint` 只定义自由度 | 换执行器不动结构——这正是 BAM 参数的落点 |
| **默认参数** | 无继承 | `<default class>` **继承体系**，子 class 只写差异 | 改一处 `sts3215`，14 个关节一起变 |
| **视觉 / 碰撞** | `visual` 与 `collision` 两套 link | 同一个 `<geom>`，靠 `class` + `contype`/`conaffinity` 区分 | 视觉件不参与碰撞（本例只有 3 个碰撞 geom） |
| **初始状态 / 传感器** | 不存在（靠外部） | `<keyframe>` + `<sensor>` 内建 | `nkey`、`sensordata` 都是模型的一部分 |

一句话：**URDF 描述"机构的连接关系"，MJCF 描述"仿真里的一切"**（含求解器、驱动、传感器、初始姿态）。所以同一个机器人，MJCF 一定比 URDF 长、且信息更多。

---

## 二、文件骨架

| 行 | 段落 | 作用 |
|---|---|---|
| 1–4 | 文件头 | XML 声明、生成器注释、`<mujoco model=...>` |
| 6–8 | `<option>` | 求解器迭代次数、积分器 flag |
| 9 | `<compiler>` | 单位（弧度）、网格目录 |
| 13–24 | `<default class="open_duck_mini_v2">` | 通用默认：joint 参数、position 默认、`visual`/`collision` 两个子 class |
| 26–42 | `<sensor>` | 16 个传感器，全部基于 `imu` 与足底 site |
| 44–56 | `<default class="sts3215">` + `backlash` | **舵机五参数**（最关键的一段） |
| 57–60 | `base` + `freejoint` + `imu` site | 浮动基座 |
| 62–113 | `trunk_assembly` | 躯干、外壳、`trunk` site |
| 114–219 | **左腿 5 个 body** | hip_yaw → hip_roll → hip_pitch → knee → ankle |
| 220–318 | **头颈 4 个 body** | neck_pitch → head_pitch → head_yaw → head_roll |
| 319–485 | 右腿（镜像）+ `<asset>` | 28 个 mesh + 材质 |
| 486–501 | `<actuator>` | 14 条位置伺服 |
| 502 | `<equality/>` | 空（未用等式约束） |

---

## 三、逐段讲解

### 1–4　文件头

```xml
<?xml version="1.0"?>
<!-- Generated using onshape-to-robot -->
<!-- Onshape document_id: 64074dfcfa379b37d8a47762 -->
<mujoco model="open_duck_mini_v2">
```

`model="open_duck_mini_v2"` 是模型名，`mujoco.viewer` 标题栏显示的就是它。注释里的 Onshape 文档 id 说明几何源在哪——将来改设计要回那个文档。

### 6–9　`<option>` 与 `<compiler>`

```xml
<option iterations="1" ls_iterations="5">
  <flag eulerdamp="disable"/>
</option>
<compiler angle="radian" meshdir="assets"/>
```

| 属性 | 含义 | 为什么这么设 |
|---|---|---|
| `iterations="1"` | 主求解器迭代 1 次（MuJoCo 默认 100） | **为 MJX 提速**：GPU 上每步迭代越少越快；代价是接触精度略降 |
| `ls_iterations="5"` | 线搜索迭代 5 次 | 配合上面，防止单步发散 |
| `eulerdamp="disable"` | 关闭欧拉积分的隐式阻尼 | 阻尼已在 joint 上显式给了，避免重复叠加 |
| `angle="radian"` | 所有角度按弧度解释 | **必须的**，否则 30 会被当成 30° |
| `meshdir="assets"` | 网格路径前缀 | 所以 STL 在 `assets/assets/`（相对本体 xml 所在目录） |

这三行是"为训练速度做的工程取舍"，不是物理必需。以后你如果发现接触抖动，先把 `iterations` 调回 2~3 试试。

### 13–24　通用 default class

```xml
<default>
  <default class="open_duck_mini_v2">
    <joint frictionloss="0.1" armature="0.005"/>
    <position kp="50" dampratio="1"/>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="collision">
      <geom group="3"/>
    </default>
  </default>
</default>
```

- **继承机制**：`<default class="X">` 里再嵌 `<default class="Y">`，Y 继承 X 的全部属性，只写差异。这解决了"14 个关节 × 10 个参数"的重复。
- `frictionloss=0.1`、`armature=0.005`、`kp=50`：这是**通用默认**，不是舵机的。真正驱动鸭子的 14 个关节用的是下面 `sts3215` 那套。
- `visual` 子 class：`contype="0" conaffinity="0"` = **不参与任何碰撞**，`group="2"` = 显示分组（viewer 里可以按 group 开关）。
- `collision` 子 class：`group="3"`，可见性另一组。

**这就是为什么 ngeom=47 里绝大多数是"纯装饰"**——它们只是给渲染看的。

#### 追问：`<joint>` 和 `<position>` 写在 `<default>` 里是什么意思

```xml
<default class="open_duck_mini_v2">
  <joint frictionloss="0.1" armature="0.005"/>
  <position kp="50" dampratio="1"/>
```

这两行**不产生任何实体**——它们不是"一个关节"和"一个执行器"，而是**属性模板**。MJCF 的 `<default>` 可以容纳绝大多数元素类型（joint、geom、site、tendon，以及 actuator 类的 general/motor/position/velocity 等），写在这里的属性会成为"后面所有没显式写该属性的同类元素"的默认值。

| 参数 | 单位 | 物理含义 | 直观后果 |
|---|---|---|---|
| `frictionloss=0.1` | N·m（关节空间力矩） | 库仑摩擦 / 静摩擦死区 | 力矩小于它时关节**推不动**；小舵机死区大，这项很重要 |
| `armature=0.005` | kg·m² | 电机转子惯量折算到关节侧，加到质量矩阵对角 | 太小 → 关节"太轻太快"，仿真能做真机做不到的动作 |
| `kp=50` | N·m/rad | 位置伺服刚度：目标与当前差 1 rad 时输出 50 N·m | 太硬抖、太软塌 |
| `dampratio=1` | 无量纲 | **阻尼比**。写了它就不用写 `kv`，MuJoCo 自动算 `kv = 2·ζ·√(kp·m)`（m 为该关节有效惯量） | ζ=1 是临界阻尼：不振荡、不超调 |

`dampratio` 是个很实用的写法：**用"我想要多稳"来间接指定阻尼**，而不是凭手感猜 `kv`。临界阻尼（1.0）意味着关节走到目标位置时刚好停住、不会来回晃。

**关键：这两行在这份文件里其实没有生效。** 因为 14 个 `<joint>` 全部写了 `class="sts3215"`，14 个 `<actuator>` 也全部写了 `class="sts3215"`——后者覆盖了这里的通用默认。真正起作用的是第 45–49 行那套：

| 参数 | 通用默认（未生效） | 实际生效（sts3215） |
|---|---|---|
| frictionloss | 0.1 | **0.068** |
| armature | 0.005 | **0.027** |
| kp | 50（+ dampratio=1） | **13.37**（+ kv=0） |
| damping | 由 dampratio 自动算 | **0.56**（显式给） |

这个"两套并存"恰好暴露了文件的分层：**通用默认是 onshape-to-robot 自动生成的兜底，`sts3215` 那套是人工辨识（BAM）后覆盖上去的真实值**（连注释掉的 `kp=17.12` 都留着，说明这组数被迭代过）。读 MJCF 时遇到"一个参数出现两次"，先判断谁覆盖谁。


### 26–42　`<sensor>`：模型已经替你算好了物理量

```xml
<gyro site="imu" name="gyro"/>
<velocimeter site="imu" name="local_linvel"/>
<accelerometer site="imu" name="accelerometer"/>
<framezaxis objtype="site" objname="imu" name="upvector"/>
<framexaxis objtype="site" objname="imu" name="forwardvector"/>
<framelinvel objtype="site" objname="imu" name="global_linvel"/>
<frameangvel objtype="site" objname="imu" name="global_angvel"/>
<framepos objtype="site" objname="imu" name="position"/>
<framequat objtype="site" objname="imu" name="orientation"/>
... 足底 site 的 6 个 ...
```

| 传感器 | 输出 | 用途 |
|---|---|---|
| `gyro` | 角速度（site 坐标系） | 判断"我正在往哪边转" |
| `velocimeter` | 线速度（site 坐标系） | 机体速度 |
| `accelerometer` | 加速度（**含重力**） | 真机 IMU 也含重力，所以一致 |
| `framezaxis`（叫 upvector） | 世界 z 轴投到 site 坐标系的三个分量 | **这就是重力方向**，竖直时 ≈ `(0,0,1)` 或 `(0,0,-1)`，取决于定义 |
| `framexaxis`（叫 forwardvector） | 世界 x 轴投影 | 判断朝向 |
| `framelinvel` / `frameangvel` / `framepos` / `framequat` | 世界系速度、位置、姿态 | 奖励与观测 |
| 足底 `framelinvel` / `framexaxis` / `framepos` | 足端速度、朝向、位置 | 抬脚高度、触地判断 |

**最实用的结论**：写 `env.py` 的观测时，`gravity` 直接从 `upvector` 拿，不用自己从四元数推。`data.sensordata` 是一个平坦数组，按定义顺序排列，偏移用 `m.sensor_adr[i]` 查。

### 44–56　`sts3215`：整份文件最重要的六行

```xml
<default class="sts3215">
  <geom contype="0" conaffinity="0"/>
  <joint damping="0.56" frictionloss="0.068" armature="0.027"/>
  <!-- <position kp="17.12" kv="0.0" forcerange="-3.23 3.23"/> -->
  <position kp="13.37" kv="0.0" forcerange="-3.23 3.23"/>
</default>
```

| 参数 | 值 | 物理含义 | 错了会怎样 |
|---|---|---|---|
| `damping` | 0.56 | 粘性阻尼（速度相关） | 动作"太活"，抖动 |
| `frictionloss` | 0.068 | 库仑摩擦 / 静摩擦死区 | 小动作推不动，或反过来过于顺滑 |
| `armature` | 0.027 | 电机转子惯量折算到关节侧 | **最常被忽略**；不填关节"太轻太快"，真机做不出来 |
| `kp` | 13.37 | 位置伺服等效刚度 | 太硬则抖、太软则软塌 |
| `kv` | 0.0 | 速度反馈增益 | 为 0 → 阻尼全靠 damping+frictionloss，这就是"Kd=0"在仿真里的体现 |
| `forcerange` | ±3.23 N·m | STS3215 的扭矩上限 | 填大了仿真能做到真机做不到的动作 |

两个细节：
1. **注释掉的 `kp=17.12` 是上一轮辨识结果**，现在的 13.37 是迭代后的值——说明这组参数被反复调过，别拿网上抄的数覆盖它。
2. **13.37 ≠ 真机 Runtime 的 Kp=30**。真机固件有自己的控制律，仿真要的是"等效行为"，两个数本来就不该相等。仿真参数只以 MJCF 为准。

`backlash` class（±0.0087 rad = ±0.5°）在这份文件里**没有 body 用到**，只有 `*_backlash.xml` 会引用。

### 57–60　浮动基座

```xml
<worldbody>
  <body name="base" pos="0 0 0.22">
    <freejoint name="floating_base"/>
    <site name="imu" pos="-0.08 -0.0 0.05"/>
```

- `<body name="base" pos="0 0 0.22">`：根 body，初始悬空在 0.22 m，靠自重落地。**这里的 z 与 keyframe 里的 z 不同**，实际初始状态由 keyframe 决定。
- `<freejoint>`：URDF 里要造假 world→base 的虚拟六自由度关节，MJCF 有专门元素。**它贡献了 `nq` 的 7（3 位置 + 4 四元数）和 `nv` 的 6（3 线速度 + 3 角速度）**，这就是 `nq=21`、`nv=20` 的来源：21−7=14，20−6=14。
- `<site>`：**没有质量、没有碰撞、只有坐标系**。16 个传感器全部挂在 `imu` 和足底这两个 site 上。IMU 装在躯干后部（x=−0.08）——将来真机装错位置，观测就系统性偏差。

### 62–113　`trunk_assembly`（躯干）

- `childclass="open_duck_mini_v2"`：从这一层往后，所有子 body 默认继承该 class。
- 大量 `<geom type="mesh" class="visual">` = 外壳、电池盒、装饰件，纯视觉。
- `<site name="trunk" pos="-0.024 0 0.088">`：躯干坐标系，算世界速度/位置时用。
- 注释掉的 imu site（`-0.087, 0, 0.042`）是旧的安装位置，**说明这机器人的 IMU 位置改过**——这种历史痕迹对 sim2real 很有价值，读数不对时先怀疑 site 位置。

### 114–219　左腿：五个 body，五个关节

```
hip_roll_assembly          left_hip_yaw    [-0.524, +0.524]   ← ±30°
└ left_roll_to_pitch_assembly  left_hip_roll   [-0.436, +0.436]   ← ±25°
  └ knee_and_ankle_assembly    left_hip_pitch  [-1.222, +0.524]
    └ knee_and_ankle_assembly_2  left_knee      [-1.571, +1.571]
      └ foot_assembly            left_ankle     [-1.571, +1.571]
```

三个必须看懂的细节：

**① 每个 body 都有 `<inertial>`**

```xml
<inertial pos="0.000795081 -5.46541e-06 -0.03306" mass="0.06648"
  fullinertia="2.44411e-05 2.81818e-05 1.42687e-05 2.35658e-09 -3.81059e-07 -7.50036e-09"/>
```

- `mass`：由**切片软件按实际填充率估重**，不是按材料密度算的（3D 打印件的真实质量比名义值小）
- `pos`：质心相对该 body 原点
- `fullinertia`：完整惯量张量 6 分量（Ixx Iyy Izz Ixy Ixz Iyz）
- 这四样东西决定"摆起来有多费劲"，是步态真实感的地基

**② 所有 `<joint>` 都没写 `axis`**

```xml
<joint name="left_hip_yaw" type="hinge" range="-0.524 0.524" class="sts3215"/>
```

14 个关节全部没有 `axis` 属性（我数过：0 处带 axis）。MuJoCo 的 `hinge` 默认轴是 body 自身的 **z 轴**，所以**关节朝向完全由各层 body 的 `quat` 决定**。这就是为什么你在 URDF 里熟悉的"给 joint 写 axis"在这里看不到。

**③ 左右镜像的符号差异来自 `range`**

| 关节 | 左 | 右 |
|---|---|---|
| `hip_pitch` | `[-1.222, +0.524]` | `[-0.524, +1.222]` |
| `knee` | `[-1.571, +1.571]` | `[-1.571, +1.571]` |

左腿 hip_pitch 的负向大（能往后摆 70°），右腿正好反过来。这印证了笔记里那句"hip_pitch 左右符号相反"——**不是 axis 不同，是限位区间镜像**。写 `env.py` 时如果你对 action 统一处理而不区分左右，一条腿会往前、一条往后。

### 203–205　全机唯一的碰撞体

```xml
<geom type="mesh" class="collision" pos="0.01656 0.2228 0.10955"
  quat="0.707107 -0.707107 -0 1.72496e-18" mesh="foot_bottom_tpu"
  material="foot_bottom_tpu_material" name="left_foot_bottom_tpu"/>
```

整份文件里 `class="collision"` 的 geom **只有 3 个**（两只脚 + 躯干底），其余全是 `contype=0 conaffinity=0` 的视觉件。这意味着：

- 接触计算量很小（对 MJX 是好事）
- **想让鸭子"摔得真实"，改的是足底形状和地面摩擦，不是别处**
- 域随机化里的地面摩擦 `friction=0.6` 改在 scene 文件，不在本体

### 220–318　头颈四个 body

```
neck_pitch_assembly  neck_pitch  [-0.349, +1.134]
└ head_pitch_to_yaw   head_pitch  [-0.785, +0.785]
  └ neck_yaw_assembly  head_yaw   [-2.793, +2.793]
    └ head_assembly    head_roll  [-0.524, +0.524]
```

**名字和内容对不上**：叫 `head_pitch_to_yaw` 的 body 挂的是 `head_pitch`，叫 `neck_yaw_assembly` 的挂的是 `head_yaw`。自动生成的名字不直观，**以 `<joint name=...>` 为准，别信 body 名**。

这 4 个关节也是 `sts3215` class（同款舵机），所以 `nu=14 = 5 + 5 + 4`。

### 319–485　右腿 + `<asset>`

右腿结构与左腿镜像，body 名带 `_2` 后缀。`<asset>` 段定义 28 个 `<mesh file="xxx.stl">` 和一堆 `<material>`——这些就是那 28 个 STL 被引用的地方，删任何一个都会加载失败。

### 486–501　`<actuator>`：驱动与结构解耦

```xml
<position class="sts3215" name="left_hip_yaw" joint="left_hip_yaw" inheritrange="1"/>
```

- `class="sts3215"`：套用上面那套 kp/damping/forcerange
- `joint="left_hip_yaw"`：绑定到哪个自由度
- `inheritrange="1"`：**`ctrlrange` 自动等于该关节的 `jnt_range`** → `env.py` 里不需要再手设控制范围，剪裁用关节限位即可

**这是 MJCF 最漂亮的设计**：`joint` 只负责"能不能动"，`actuator` 负责"怎么驱动"。换执行器只改 actuator，动力学结构完全不动——BAM 参数辨识改的就是这里。

### 502　`<equality/>`

空的，说明没用等式约束（比如四杆机构、耦合关节）。将来要加闭链机构才需要它。

---

## 四、读这份文件只需要抓五样东西

1. `nq / nv / nu / nkey` → 21 / 20 / 14 / 1
2. 14 个 `jnt_range` → 动作幅度上限
3. `sts3215` 那 6 行 → 执行器五参数
4. `class="collision"` 的 geom → 只有足底在算接触
5. `freejoint` + `keyframe` → 初始状态怎么来的

其余 400 多行是网格与材质，扫一眼即可。

---

## 五、从 URDF 过来最容易踩的五个坑

| # | 坑 | 真相 |
|---|---|---|
| 1 | 找 `parent=` / `child=` 属性 | MJCF 没有，父子靠 **body 嵌套** |
| 2 | 找 joint 的 `axis` / `rpy` | 多数不写；朝向由 body 的 `quat` 决定，`hinge` 默认绕自身 z |
| 3 | 改一个关节参数，结果 14 个全变 | 因为改的是 `<default class>`，不是那个 `<joint>` |
| 4 | 以为 mesh 会参与碰撞 | `contype=0 conaffinity=0` 的全是装饰；全机只有 3 个碰撞 geom |
| 5 | 把仿真 `kp=13.37` 当真机 `Kp=30` | 前者是辨识出的等效刚度，后者是真机固件参数，不必相等 |

---

# 附：`scene_flat_terrain.xml` 逐段解析（75 行）

> 这份文件才是 `env` 真正加载的入口。它做三件事：把本体嵌进来、定义地面、定义初始姿态。

## 1–2　模型名与 include

```xml
<mujoco model="scene">
    <include file="open_duck_mini_v2.xml"/>
```

- `include` 会**就地展开**被包含文件的内容：本体的 `<default>`、`<asset>`、`<worldbody>`、`<actuator>`、`<sensor>` 全部并进来，与本文件的同类元素合并。所以你加载 scene 得到的模型，`nu=14`、`nbody=18`，和本体一样多，只是多了地面。
- 路径相对 **scene 文件所在目录**（不是当前工作目录），所以两个 xml 必须放同一层。
- `<mujoco model="scene">` 是最终模型名，`viewer` 标题栏显示 `MuJoCo : scene` 就是这个。

## 4–8　`<visual>`：只影响观感，不影响物理

```xml
<visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20"/>
</visual>
```

| 属性 | 作用 |
|---|---|
| `headlight` | 跟随相机的头灯（diffuse 漫反射、ambient 环境光） |
| `haze` | 远处雾化颜色（让地面边缘不突兀） |
| `azimuth`/`elevation` | 相机初始方位角/俯仰角 |

**这三行对训练毫无影响**——MJX 不渲染。改它只会改变你录视频时好不好看。

## 9–23　两段被注释掉的旧代码

```xml
<!-- <asset> ... rgb1="0.2 0.3 0.4" ... </asset> -->
<!-- <worldbody><body name="floor"><light .../><geom ... size="0 0 0.05" .../></body></worldbody> -->
```

历史痕迹，两个信息：

1. 旧版是**深色地面 + 独立光源**，现在是**纯白地面 + 头灯**。深色版更像演示视频，纯白版更适合看轨迹。
2. 旧版 `floor` 里有 `<light>`，新版删掉了 → 现在完全靠 `<headlight>` 照明。所以如果你把 `headlight` 关了，画面会全黑。

## 26–31　`<asset>`：天空盒与地面贴图

```xml
<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
```

- 天空盒与地面**都是纯白**（`rgb1 = rgb2 = 1 1 1`），只用 `markrgb="0 0 0"` 画黑色网格线。
- `texrepeat="5 5"`：贴图在平面上重复 5×5 → 黑线间距 = 平面尺寸 / 10，用来看"鸭子走了多远"。
- `reflectance="0"`：不反射。渲染成本低。

**这个配色是给训练/录视频用的，不是美化版**——纯色背景便于看清关节动作。想做得好看，改这里就行，不影响任何物理。

## 33–38　`<worldbody>`：地面（**四个关键属性**）

```xml
<body name="floor">
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"
          contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
</body>
```

| 属性 | 值 | 含义 |
|---|---|---|
| `type="plane"` | — | **物理上无限大**的平面。`size` 的前两个数只影响渲染网格范围，第三个数是网格线间距（0.01 m）；碰撞与尺寸无关 |
| `contype` / `conaffinity` | `1` / `0` | 碰撞位掩码。两个 geom 碰撞的条件是 `(contype_a & conaffinity_b) ‖ (contype_b & conaffinity_a)`。地面 `contype=1, conaffinity=0`，足底是默认 `1/1` → 能碰；视觉件是 `0/0` → 永远不碰 |
| `priority="1"` | — | 接触参数冲突时**以高 priority 一方为准**。地面给 1，足底给 0 → **地面摩擦 0.6 会覆盖足底自己的摩擦设置** |
| `friction="0.6"` | — | 只写一个数 = 滑动摩擦系数；另外两个（扭转、滚动）用默认 |
| `condim="3"` | — | 摩擦锥维度。3 = **只有切向滑动摩擦**；4 加扭转摩擦，6 加滚动摩擦。3 是最省的，也够走路用 |

**对域随机化最直接的含义**：想随机化地面摩擦，改的是**这里**（scene 文件），不是本体 xml。而且因为 `priority=1`，改这里一定生效。

## 39–75　`<keyframe>`：初始姿态（本节最重要的部分）

```xml
<keyframe>
    <key name="home" qpos="0 0 0.15  1 0 0 0  0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"
         ctrl="0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.003 -0.065 0.635 1.379 -0.796"/>
</keyframe>
```

### qpos 的 21 个数怎么分

| 位置 | 个数 | 含义 | 本例值 |
|---|---|---|---|
| 1–3 | 3 | 浮基位置 `x y z` | `0 0 0.15` |
| 4–7 | 4 | 浮基姿态四元数 `w x y z` | `1 0 0 0`（无旋转） |
| 8–21 | 14 | 14 个关节角 | 见下表 |

### 14 个关节角与关节的对应（按 `<joint>` 声明顺序）

| # | 关节 | 角度 (rad) | 度数 | 解读 |
|---|---|---|---|---|
| 1 | `left_hip_yaw` | 0.002 | 0.1° | 几乎为零，双腿平行 |
| 2 | `left_hip_roll` | 0.053 | 3.0° | 轻微外展 |
| 3 | `left_hip_pitch` | **−0.630** | −36.1° | 大腿前摆 |
| 4 | `left_knee` | **1.368** | **78.4°** | **深蹲** |
| 5 | `left_ankle` | **−0.784** | −44.9° | 踝背屈，保持脚掌贴地 |
| 6 | `neck_pitch` | 0 | 0° | 头颈中位 |
| 7 | `head_pitch` | 0 | 0° | |
| 8 | `head_yaw` | 0 | 0° | |
| 9 | `head_roll` | 0 | 0° | |
| 10 | `right_hip_yaw` | −0.003 | −0.2° | |
| 11 | `right_hip_roll` | −0.065 | −3.7° | |
| 12 | `right_hip_pitch` | **+0.635** | +36.4° | **与左侧符号相反** |
| 13 | `right_knee` | 1.379 | 79.0° | 同样深蹲 |
| 14 | `right_ankle` | **−0.796** | −45.6° | |

三个必须记住的结论：

1. **这是蹲姿，不是站姿**。膝盖 78°，躯干高度只有 **0.15 m**（本体 xml 里 base 写的 0.22 只是建模时的悬空高度，实际初始状态由 keyframe 覆盖）。
2. **`hip_pitch` 左右符号相反**（−0.630 vs +0.635），而 **`knee` 和 `ankle` 左右同号**（1.368/1.379、−0.784/−0.796）。这直接来自之前看到的限位镜像。写 `env.py` 时若对 action 统一处理，一条腿会往前、一条往后。
3. **`ctrl` 与 `qpos` 的 14 个数完全相同**。含义：reset 时位置伺服的目标就是初始姿态，关节不需要"走"到目标，第一帧没有瞬态冲击。

### 顺序陷阱

`qpos` 的顺序由 **`<joint>` 在 XML 中的声明顺序**决定；`actuator` 的顺序由 **`<actuator>` 段**决定；`sensordata` 的顺序由 **`<sensor>` 段**决定。三者**不同源**，本例恰好一致纯属生成器按同序输出。

正确做法：不要假定顺序，用 `m.jnt_qposadr[joint_id]` 和 `m.actuator_trnid[act_id, 0]` 建立映射，`check_model.py` 打印出来看一眼。

## 对 `env.py` 的三条直接推论

| 推论 | 依据 |
|---|---|
| 初始姿态取自 `m.key_qpos[0][7:]`，共 14 个 | keyframe 在本文件里，加载 scene 才有 `nkey=1` |
| 起始高度 0.15 m 是"脚刚好触地"的蹲姿高度 | 改高会悬空落地砸一下，改低会插进地面 |
| 随机化地面摩擦改 `floor` 的 `friction`，且必定生效 | `priority="1"` 让地面参数覆盖足底 |
