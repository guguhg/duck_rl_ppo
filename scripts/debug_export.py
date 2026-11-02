"""导出诊断：暴力枚举几种可能的结构差异，看哪种能和 JAX 对上。

只在导出对不上时用。原理：把「归一化 / 激活函数 / 取前 14 还是后 14」三种选择
两两组合，各算一遍 numpy 前向，和 JAX 策略的输出比最大误差。
误差最小的那个组合就是真实结构。

用法：
    uv run python scripts/debug_export.py --params checkpoints/params_18022400.pkl
"""
import argparse
import itertools
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from duck_rl.export_onnx import _layers, _stat, _unwrap


def act_fn(name, x):
    if name == "swish":
        return x * (1.0 / (1.0 + np.exp(-x)))
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "none":
        return x
    raise ValueError(name)


def forward(obs, mean, std, layers, use_act, pick, use_tanh, normalize=True):
    x = (obs - mean) / std if normalize else obs
    for i, (_, w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = act_fn(use_act, x)
    act = x[:, :14] if pick == "first" else x[:, 14:]
    return np.tanh(act) if use_tanh else act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--obs", type=int, default=53)
    p.add_argument("--act", type=int, default=14)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    args = p.parse_args()

    with open(args.params, "rb") as f:
        params = pickle.load(f)
    normalizer, policy_params = _unwrap(params)
    mean, std = _stat(normalizer.mean), _stat(normalizer.std)
    layers = _layers(policy_params)
    print(f"[info] 层: {[(n, w.shape) for n, w, _ in layers]}")
    print(f"[info] mean 前 5: {np.round(mean[:5], 4)}")
    print(f"[info] std  前 5: {np.round(std[:5], 4)}")
    print(f"[info] std  范围: {std.min():.4f} ~ {std.max():.4f}")

    # JAX 侧参考输出
    from brax.training.agents.ppo import networks as ppo_networks
    net = ppo_networks.make_ppo_networks(
        observation_size=args.obs, action_size=args.act,
        policy_hidden_layer_sizes=tuple(args.hidden),
        value_hidden_layer_sizes=tuple(args.hidden),
    )
    policy = ppo_networks.make_inference_fn(net)(params, deterministic=True)

    rng = np.random.default_rng(0)
    obs = (rng.standard_normal((8, args.obs)) * 1.5).astype(np.float32)
    jax_out = np.stack([np.asarray(policy(jnp.asarray(o)[None],
                                          jax.random.PRNGKey(0))[0]).reshape(-1)
                        for o in obs])
    print(f"\n[JAX] 输出范围 [{jax_out.min():+.3f}, {jax_out.max():+.3f}]")
    print(f"[JAX] 第一条动作: {np.round(jax_out[0], 3)}")

    print("\n=== 组合扫描（最大绝对误差，越小越对）===")
    print(f"{'归一化':<6}{'激活':<8}{'取μ':<8}{'tanh':<7}{'最大误差':>10}")
    best = None
    for norm, act, pick, tanh_on in itertools.product(
            [True, False], ["swish", "relu", "tanh", "none"], ["first", "second"],
            [True, False]):
        try:
            out = forward(obs, mean, std, layers, act, pick, tanh_on, norm)
        except Exception as e:                       # noqa: BLE001
            print(f"{norm!s:<6}{act:<8}{pick:<8}{tanh_on!s:<7}  失败 {e}")
            continue
        if out.shape != jax_out.shape:
            continue
        err = float(np.abs(out - jax_out).max())
        mark = "  ←" if best is None or err < best[0] else ""
        if best is None or err < best[0]:
            best = (err, norm, act, pick, tanh_on)
        print(f"{norm!s:<6}{act:<8}{pick:<8}{tanh_on!s:<7}{err:>10.3e}{mark}")

    print()
    if best and best[0] < 1e-4:
        print(f"✓ 找到匹配结构：normalize={best[1]} act={best[2]} pick={best[3]} tanh={best[4]}")
        print("  按这个组合改 duck_rl/export_onnx.py 的 build_onnx 即可")
    elif best:
        print(f"✗ 最好的一组也有 {best[0]:.3e} 误差，说明差异不在这些维度上。")
        print("  下一步该查：网络是否带 LayerNorm / 权重是否要先转置 / 是否还有 extra 状态")
        # 额外提示：把 numpy 前向的最后一层输出和 JAX 的原始 28 维对比不方便，
        # 就打印前两个样本的数值供人工比对
        print(f"  JAX  样本0: {np.round(jax_out[0][:6], 3)}")
        out = forward(obs, mean, std, layers, best[2], best[3], best[4], best[1])
        print(f"  numpy 样本0: {np.round(out[0][:6], 3)}")


if __name__ == "__main__":
    main()
