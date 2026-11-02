"""训练入口：Brax PPO。

第一轮目标不是"训出好步态"，而是确认三件事：
  1. 能跑起来、不 OOM
  2. eval 奖励在涨（不是 0、不是 NaN）
  3. checkpoint 能存下来

超参策略：借 mujoco_playground 里 BerkeleyHumanoid 的现成配置，只覆盖
num_envs 与 num_timesteps。理由——这份配置的字段名与已装的 brax 版本一定匹配，
手填容易撞上版本差异；而且它本来就是一个双足行走任务的成熟超参。

输出设计（--heartbeat）：
  brax 只在 eval 时回调，间隔可能几十秒到几分钟。为了"随时能看到它在动"，
  另起一个守护线程每 N 秒打一行进度条 + 步数 + sps + ETA + 显存；eval 时再打明细。
"""
import argparse
import functools
import pickle
import subprocess                 # 读 nvidia-smi（心跳里的显存/温度）
import threading                  # 心跳线程
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks   # 网络定义
from brax.training.agents.ppo import train as ppo               # PPO 训练主函数
from mujoco_playground import wrapper                            # vmap 并行包装层
from mujoco_playground.config import locomotion_params           # 现成超参

from duck_rl.env import DuckEnv, REWARD_TERMS                    # 环境 + 八项名字


# ---------------------------------------------------------------- 心跳用的共享状态
# 主线程（brax 训练循环）写，心跳线程读。只读写几个数字，GIL 足够安全。
STATUS = {"step": 0, "reward": None, "sps": None, "length": None}
STOP = threading.Event()          # 主线程结束时置位，心跳线程退出


def _hms(sec):
    """秒 → HH:MM:SS。"""
    sec = int(max(float(sec), 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(frac, width=22):
    """进度条：按比例拼实心块和空心块。"""
    frac = min(max(float(frac), 0.0), 1.0)
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def _gpu_line():
    """读一次显存/利用率/温度。失败返回空串（没有 nvidia-smi 也不影响训练）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return ""
        used, total, util, temp = [x.strip() for x in out.stdout.strip().split(",")[:4]]
        return f"  GPU {int(used) / 1024:.1f}/{int(total) / 1024:.1f}G {util}% {temp}°C"
    except Exception:
        return ""                 # 任何异常都静默：心跳不能影响训练


def _heartbeat(num_timesteps, interval):
    """守护线程：每 interval 秒打一行，让终端一直在滚。"""
    t_start = time.time()
    while not STOP.wait(interval):          # 被 set 就立刻返回 True → 退出循环
        step = STATUS["step"]
        frac = step / max(num_timesteps, 1)
        line = (f"[{_hms(time.time() - t_start)}] {_bar(frac)} {frac * 100:5.1f}% "
                f"step {step:>10,}/{num_timesteps:,}")
        sps = STATUS["sps"]
        if sps:                              # sps 有了才算得出 ETA
            line += f"  {sps:>7,.0f} sps  ETA {_hms((num_timesteps - step) / sps)}"
        elif step == 0:
            line += "  编译中（首次 3~10 分钟无输出是正常的）"
        if STATUS["reward"] is not None:     # 最近一次 eval 的总分
            line += f"  reward {STATUS['reward']:.1f}"
        line += _gpu_line()                  # 显存 + 利用率 + 温度
        try:
            print(line, flush=True)
        except Exception:
            return                           # 终端被关掉时安静退出


# ---------------------------------------------------------------- 兼容层
# brax 0.14.2（当前最新）在 train() 里无条件调用 jax.device_put_replicated，
# 而 jax 0.12 已经把这个 API 从顶层删掉了（deprecations 里直接抛 AttributeError）。
# 语义很简单：把同一份 pytree 复制到给定设备上，结果是带「前导设备维」的 pytree。
# 单卡时等价于给每个叶子加一个长度为 1 的维度。
# 注意：这是版本错配的权宜之计，brax 更新后应当删掉。
def _device_put_replicated(tree, devices):
    devices = list(devices)                                   # 设备可迭代对象先转 list

    def put(x):
        if x is None:                                         # pytree 里可能有 None 叶子
            return None
        return jnp.stack([jax.device_put(x, d) for d in devices])   # 每块设备放一份再堆叠

    return jax.tree_util.tree_map(put, tree)                  # 逐叶子处理，保持树结构


if not hasattr(jax, "device_put_replicated"):    # 只有缺了才补；将来 brax 修好会自动失效
    jax.device_put_replicated = _device_put_replicated

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax"))  # 编译缓存放哪
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)  # 编译超 1 秒就落盘


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_timesteps", type=int, default=2_000_000)   # 总环境步（会被量化）
    p.add_argument("--num_envs", type=int, default=512)              # 并行环境数，6 GB 的安全值
    p.add_argument("--out", type=str, default="checkpoints")         # 产物目录
    p.add_argument("--num_evals", type=int, default=50,
                   help="全程 eval 次数；越大输出越频繁（eval 本身也有开销）")
    p.add_argument("--num_eval_envs", type=int, default=64,
                   help="eval 用多少个并行环境；越小 eval 越快，但评估方差大")
    p.add_argument("--heartbeat", type=float, default=2.0,
                   help="心跳间隔秒数，0 = 关闭（只要 eval 输出）")
    p.add_argument("--dump-metrics", action="store_true",
                   help="第一次 eval 时打印所有可用指标键名，用来确认八项分解的键")
    args = p.parse_args()

    out = Path(args.out)                     # 转成 Path 方便拼路径
    out.mkdir(parents=True, exist_ok=True)   # 不存在就建，存在也不报错

    env = DuckEnv()                          # 构造环境（加载模型 + 读 keyframe）
    print(f"[env] obs={env.observation_size} act={env.action_size} "
          f"dt={env.dt}s n_substeps={env.n_substeps}", flush=True)     # 确认 53/14/0.02/10

    # 借现成超参，再覆盖环境数/步数/eval 次数
    ppo_cfg = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    params = dict(ppo_cfg)                   # 复制一份，别改上游的配置对象
    if "network_factory" in ppo_cfg:         # 配置里是 dict，要转成可调用对象
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_cfg["network_factory"]
        )
        del params["network_factory"]        # 已单独处理，避免重复传参报错
    else:
        network_factory = ppo_networks.make_ppo_networks   # 没配就用默认工厂

    params["num_timesteps"] = args.num_timesteps   # 覆盖总步数
    params["num_envs"] = args.num_envs             # 覆盖环境数
    params["num_evals"] = args.num_evals           # 覆盖 eval 次数（输出频率）
    params["num_eval_envs"] = args.num_eval_envs   # 覆盖 eval 并行环境数
    print(f"[ppo] num_envs={params['num_envs']} num_timesteps={params['num_timesteps']} "
          f"batch={params.get('batch_size')} unroll={params.get('unroll_length')}", flush=True)
    print(f"[ppo] num_evals={args.num_evals}（约每 "
          f"{args.num_timesteps // max(args.num_evals, 1):,} 环境步输出一段明细）"
          f"  心跳每 {args.heartbeat:g}s 一行", flush=True)

    t0 = time.time()                          # 计时起点，给日志用

    def progress(step, metrics):
        """brax 每次 eval 回调一次：总分 + 回合长度 + 八项奖励分解。"""
        el = time.time() - t0

        def fmt(key, spec="{:.1f}"):
            """取一个指标并格式化；没有就返回横杠，避免 KeyError/格式化崩。"""
            v = metrics.get(key)
            if v is None:
                return "—"
            try:
                return spec.format(float(v))
            except (TypeError, ValueError):
                return str(v)

        r = metrics.get("eval/episode_reward")     # 评估回合总分
        sps = metrics.get("training/sps")          # 每秒环境步数，反映真实吞吐

        # 先把最新值写给心跳线程
        STATUS["step"] = int(step)
        if r is not None:
            STATUS["reward"] = float(r)
        if sps is not None:
            STATUS["sps"] = float(sps)

        print("", flush=True)                      # 空行，把心跳行和明细隔开
        print(f"[{_hms(el)}] ===== eval @ step {step:,} =====", flush=True)
        print(f"  奖励 {fmt('eval/episode_reward')}    "
              f"回合长度 {fmt('eval/avg_episode_length', '{:.0f}')}    "
              f"吞吐 {fmt('training/sps', '{:,.0f}')} sps", flush=True)

        # 八项分解：EpisodeWrapper 会把 env.metrics 里的键累加成
        # eval/episode_rew/xxx，等于"这一回合该项贡献了多少分"
        parts = []
        for name in REWARD_TERMS:
            key = f"eval/episode_rew/{name}"
            if key in metrics:
                try:
                    parts.append(f"{name}={float(metrics[key]):+.1f}")
                except (TypeError, ValueError):
                    pass
        if not parts:                              # 键名不对就退化为扫描
            for k, v in metrics.items():
                if k.startswith("eval/") and "rew/" in k:
                    try:
                        parts.append(f"{k.split('rew/')[-1]}={float(v):+.1f}")
                    except (TypeError, ValueError):
                        pass
        if parts:
            print("  八项  " + "  ".join(parts), flush=True)

        if args.dump_metrics:                      # 只打一次，方便确认键名
            print("  全部指标键: " + ", ".join(sorted(metrics.keys())), flush=True)
            args.dump_metrics = False

    def save_params(step, make_policy, policy_params):
        """每个 eval 周期存一次参数。ONNX 导出留到 Step 9。"""
        path = out / f"params_{step}.pkl"      # 文件名带步数，方便挑 checkpoint
        with open(path, "wb") as f:
            pickle.dump(policy_params, f)      # 只存策略参数（含 normalizer）
        print(f"  [ckpt] {path}", flush=True)

    # 起心跳线程（daemon：主进程退出时自动结束）
    if args.heartbeat > 0:
        threading.Thread(target=_heartbeat, args=(args.num_timesteps, args.heartbeat),
                         daemon=True).start()

    print("[ppo] 开始训练。首次编译 3~10 分钟，期间只有心跳行在滚，属正常。", flush=True)

    try:
        _, final_params, _ = ppo.train(
            environment=env,                                # 我们的 env
            wrap_env_fn=wrapper.wrap_for_brax_training,     # 这层负责 vmap 并行与批处理
            network_factory=network_factory,                # 512-256-128，swish
            progress_fn=progress,                           # 上面那个打印回调
            policy_params_fn=save_params,                   # 上面那个存档回调
            **params,                                       # 其余超参全部来自 BerkeleyHumanoid
        )
    finally:
        STOP.set()                                          # 不管正常结束还是 Ctrl+C，都停心跳

    with open(out / "params_final.pkl", "wb") as f:     # 训练正常结束时再存一份
        pickle.dump(final_params, f)
    print(f"[done] 最终参数 → {out / 'params_final.pkl'}", flush=True)


if __name__ == "__main__":
    main()
