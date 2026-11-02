"""奖励项：七个纯函数，只负责算 0~1 的「满意度」，权重在 config 里。

为什么拆成独立模块：
1. 每一项都能单独测试、单独替换，不用碰 env
2. 以后想加"足底腾空时间""关节限位惩罚"，只在这里加函数 + config 加权重
3. 权重和公式分离：调参时只改 config，不动代码

约定：
- 所有跟踪项都用 exp(-误差²/sigma) 的形式：误差 0 得 1.0，误差大平滑趋 0，永不为负。
  为什么不用 -误差：那样"已经跑偏"时再多偏一点损失也差不多，策略会破罐破摔。
- 惩罚项（action_rate / torque）返回的是「代价」，调用方用负权重去乘。
"""
import jax                          # 只为类型标注 jax.Array；缺了它，定义函数时就报 NameError
import jax.numpy as jp              # jp.exp / jp.sum / jp.square


def tracking_lin_vel(v_body: jax.Array, cmd: jax.Array, sigma: float) -> jax.Array:
    """线速度跟踪：机体实际速度与指令的差距。

    v_body 是体坐标系下的 [vx, vy]（前向、左向），cmd 是 [vx*, vy*]。
    sigma 越小越苛刻：0.25 表示速度差 0.5 m/s 时约得 0.37 分。
    """
    err = (v_body[0] - cmd[0]) ** 2 + (v_body[1] - cmd[1]) ** 2   # 二维平方误差
    return jp.exp(-err / sigma)                                   # 误差 0 → 1.0，误差大 → 趋 0


def tracking_ang_vel(gyro_z: jax.Array, cmd_z: jax.Array, sigma: float) -> jax.Array:
    """角速度跟踪：转向指令。

    用陀螺仪 z 轴（机体系）而不是世界系角速度：接近直立时两者几乎相同，
    而陀螺仪是策略真的能测到的量，真机上一致。
    """
    return jp.exp(-((gyro_z - cmd_z) ** 2) / sigma)               # 一维平方误差，同上的 exp 形式


def upright(g_body: jax.Array, sigma: float) -> jax.Array:
    """躯干竖直：重力在体坐标系 z 轴的投影接近 -1 就得高分。

    为什么必须有：躯干一歪，质心就偏出支撑面。这一项在"摔倒"发生之前就开始扣分，
    相当于给策略一个"快要倒了"的预警信号。
    """
    return jp.exp(-((g_body[2] + 1.0) ** 2) / sigma)              # 竖直时 g_z=-1，偏 0.1 掉到 0.82


def keep_height(z: jax.Array, z_ref: float, sigma: float) -> jax.Array:
    """高度维持：浮基高度接近参考值。

    z_ref 用实测稳态高度 0.16（不是 keyframe 的 0.15，见 config 注释）。
    sigma=0.01 很小，意思是"差 10 cm 基本就没分了"——它对塌陷特别敏感。
    """
    return jp.exp(-((z - z_ref) ** 2) / sigma)                    # 差 3 cm 就掉到 0.91


def action_rate(action: jax.Array, prev_action: jax.Array) -> jax.Array:
    """动作平滑代价：相邻两帧动作的平方差。

    返回的是「代价」（越大越差），调用方用负权重乘。
    作用不是扣分，而是在两组得分相近的方案里挑更平滑的那个——抑制高频抖动。
    """
    return jp.sum(jp.square(action - prev_action))                # 14 个关节的差平方求和


def torque_cost(tau: jax.Array) -> jax.Array:
    """扭矩代价：Στ²。让策略少用力、省电、减少舵机发热。

    权重给得很小（-0.0002）：14 个关节的 τ² 求和大约 3~5，乘完只有 -0.001 量级，
    属于打破平局，不是主要约束。
    """
    return jp.sum(jp.square(tau))                                 # 站立时约 0.6，乱动时 3~5


def alive() -> jax.Array:
    """活着：每步白给 1.0。

    ⚠ 这一项占总分约 23%，给太大会让"站着不动"变成最优解：
       站着每步还能拿 alive + height + upright ≈ 2.0，
       走起来拿 4.3 但会摔——如果摔的代价不够大，它宁愿站着。
       （摔的代价靠 done 终止回合实现，不是扣分。）
    """
    return jp.array(1.0)                                          # 常数，没有任何输入
