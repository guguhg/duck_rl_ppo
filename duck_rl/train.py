"""训练入口：Brax PPO。

第一轮目标不是"训出好步态"，而是确认三件事：
  1. 能跑起来、不 OOM
  2. eval 奖励在涨（不是 0、不是 NaN）
  3. checkpoint 能存下来

超参策略：借 mujoco_playground 里 BerkeleyHumanoid 的现成配置，只覆盖
num_envs 与 num_timesteps。理由——这份配置的字段名与已装的 brax 版本一定匹配，
手填容易撞上版本差异；而且它本来就是一个双足行走任务的成熟超参。
"""
import argparse                     # 命令行参数
import functools                    # functools.partial 把配置字典变成可调用对象
import pickle                        # 存/读 .pkl 参数
import time                          # 打印耗时
from pathlib import Path             # 路径拼装

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks   # 网络定义
from brax.training.agents.ppo import train as ppo               # PPO 训练主函数
from mujoco_playground import wrapper                            # vmap 并行包装层
from mujoco_playground.config import locomotion_params           # 现成超参

from duck_rl.env import DuckEnv                                  # 我们自己的环境


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
    args = p.parse_args()

    out = Path(args.out)                     # 转成 Path 方便拼路径
    out.mkdir(parents=True, exist_ok=True)   # 不存在就建，存在也不报错

    env = DuckEnv()                          # 构造环境（加载模型 + 读 keyframe）
    print(f"[env] obs={env.observation_size} act={env.action_size} "
          f"dt={env.dt}s n_substeps={env.n_substeps}", flush=True)     # 确认 51/14/0.02/10

    # 借现成超参，再覆盖环境数/步数
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
    print(f"[ppo] num_envs={params['num_envs']} num_timesteps={params['num_timesteps']} "
          f"batch={params.get('batch_size')} unroll={params.get('unroll_length')}", flush=True)

    t0 = time.time()                          # 计时起点，给日志用

    def progress(step, metrics):
        """brax 每次 eval 回调一次。这里只打印关键量。"""
        r = metrics.get("eval/episode_reward")     # 评估回合总分
        sps = metrics.get("training/sps")          # 每秒环境步数，反映真实吞吐
        print(f"[{time.time() - t0:7.1f}s] step {step:>10,}  "
              f"eval_reward {r}  sps {sps}", flush=True)

    def save_params(step, make_policy, policy_params):
        """每个 eval 周期存一次参数。ONNX 导出留到 Step 9。"""
        path = out / f"params_{step}.pkl"      # 文件名带步数，方便挑 checkpoint
        with open(path, "wb") as f:
            pickle.dump(policy_params, f)      # 只存策略参数（含 normalizer）
        print(f"[ckpt] {path}", flush=True)

    _, final_params, _ = ppo.train(
        environment=env,                                # 我们的 env
        wrap_env_fn=wrapper.wrap_for_brax_training,     # 这层负责 vmap 并行与批处理
        network_factory=network_factory,                # 512-256-128，swish
        progress_fn=progress,                           # 上面那个打印回调
        policy_params_fn=save_params,                   # 上面那个存档回调
        **params,                                       # 其余超参全部来自 BerkeleyHumanoid
    )

    with open(out / "params_final.pkl", "wb") as f:     # 训练正常结束时再存一份
        pickle.dump(final_params, f)
    print(f"[done] 最终参数 → {out / 'params_final.pkl'}", flush=True)


if __name__ == "__main__":
    main()
