"""把训练好的 brax 参数导出成 ONNX，并与 JAX 策略做数值对齐。

为什么不用 TF/Keras + tf2onnx：
  参考实现靠 `tf_model.get_layer(name)` 按层名搬权重，一旦 brax 内部层命名变了就
  "打印一句 not found 然后跳过"，导出的模型少一层还不报错——这是最难查的一类 bug。
  这里直接按权重数组拼 ONNX 图：层数、顺序、激活都是显式的，导错了会立刻在
  数值对齐那步暴露。

用法：
    uv run python -m duck_rl.export_onnx --params checkpoints/params_18022400.pkl \
        --out checkpoints/policy.onnx
"""
import argparse                       # 命令行参数
import pickle                         # 读 brax 存的 .pkl

import jax
import jax.numpy as jnp
import numpy as np
import onnx                           # 建图 + 保存
import onnxruntime as ort             # 推理，用来做数值对齐
from onnx import TensorProto, helper  # 常量张量 / 节点 / 图 的构造器


# ---------------------------------------------------------------- 参数解包
def _is_map(x):
    """判断是不是「字典类」对象。flax 的 FrozenDict 不是 dict 子类，但有 keys()。"""
    return hasattr(x, "keys") and hasattr(x, "__getitem__")   # 有 keys() 和取下标就当字典


def _stat(x):
    """取归一化统计量。

    两种结构都见过：
      - 观测是扁平向量 → normalizer.mean 直接是 (obs,) 数组
      - 观测是 dict（多路输入）→ normalizer.mean["state"] 才是数组
    """
    if _is_map(x):                                    # 是字典 → 还要按 key 取
        # 常见键名依次尝试
        for k in ("state", "obs", "observation"):
            if k in x.keys():
                return np.asarray(x[k])
        raise KeyError(f"归一化统计量是多键字典，但没有 state/obs 键：{list(x.keys())}")
    return np.asarray(x)                              # 扁平观测，直接用


def _unwrap(params):
    """从 pickle 里取出 (normalizer, policy_params)。

    brax 的 params 结构在不同版本里略有差异：可能是 tuple，也可能是带键的对象。
    这里两种都试，取不到就报清晰的错。
    """
    def get(obj, key):
        if _is_map(obj) and key in obj.keys():        # 当字典取
            return obj[key]
        if hasattr(obj, key):                         # 当属性取
            return getattr(obj, key)
        return None                                   # 都没有 → None，交给上层判断

    if isinstance(params, (tuple, list)):             # 结构 A：元组 (normalizer, policy)
        normalizer, policy = params[0], params[1]
    else:                                             # 结构 B：带键对象
        normalizer = get(params, "normalizer_params")
        policy = get(params, "policy") or get(params, "policy_params")

    # policy 可能是 {'params': {...}} 或直接是 {...}
    if _is_map(policy) and "params" in policy.keys():     # 剥第一层
        policy = policy["params"]
    if _is_map(policy) and "policy" in policy.keys():     # 再剥一层
        policy = policy["policy"]
        if _is_map(policy) and "params" in policy.keys(): # 里面还可能再套 params
            policy = policy["params"]
    return normalizer, policy


def _layers(policy_params):
    """按顺序取出每层的 (kernel, bias)。"""
    out = []
    for name, p in policy_params.items():             # 按 dict 顺序遍历（= 网络层顺序）
        if "kernel" in p:                             # 只认有 kernel 的，跳过其他参数
            out.append((name, np.asarray(p["kernel"]), np.asarray(p["bias"])))
    return out                                        # [(name, W, b), ...] 共 4 层


# ---------------------------------------------------------------- 拼 ONNX
def build_onnx(mean, std, layers, obs_size, act_size, out_path, normalize=True):
    """构图：obs → [归一化] → [MatMul+Add+Swish]*n → 最后一层不激活
            → 取前 act_size 维（μ） → tanh → action

    ⚠ normalize 必须与训练时的 `normalize_observations` 完全一致：
      - 训练归一化了而导出没带 → 真机读数没归一化，策略看到另一种量纲，直接失控
      - 训练没归一化而导出带了 → 输入被压两次，同样错
      本项目借用的 BerkeleyHumanoid 配置在 locomotion_params.py:33 写了
      `normalize_observations=True`，brax 因此把 running_statistics.normalize
      挂到了策略前面，**所以这里必须带归一化**，默认 True。
    """
    nodes, inits = [], []                             # nodes = 计算节点；inits = 常量张量

    def init(name, arr):
        """把一个 numpy 数组注册成 ONNX 常量，返回它的名字（供节点引用）。"""
        arr = np.asarray(arr, dtype=np.float32)        # ONNX 图统一 float32
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                                        arr.flatten().tolist()))   # 要展平成一维列表
        return name

    if normalize:                                      # 只有训练开归一化时才加这两步
        inits_mean = init("mean", mean)                # 常量 mean
        inits_std = init("std", std)                   # 常量 std
        nodes.append(helper.make_node("Sub", ["obs", inits_mean], ["x0"]))    # obs - mean
        nodes.append(helper.make_node("Div", ["x0", inits_std], ["x1"]))      # / std
        cur = "x1"                                     # 当前张量名
    else:
        cur = "obs"                                    # 不做归一化，直接用原始输入

    n = len(layers)                                    # 层数（4）
    for i, (name, w, b) in enumerate(layers):
        wn = init(f"W{i}", w)                          # 第 i 层权重常量
        bn = init(f"B{i}", b)                          # 第 i 层偏置常量
        mm = f"mm{i}"
        nodes.append(helper.make_node("MatMul", [cur, wn], [mm]))    # y = x · W
        ab = f"add{i}"
        nodes.append(helper.make_node("Add", [mm, bn], [ab]))        # y = y + b
        if i == n - 1:
            cur = ab                                   # 最后一层不加激活（输出 28 维：μ 和 logσ）
        else:
            sg = f"sig{i}"
            sw = f"sw{i}"
            nodes.append(helper.make_node("Sigmoid", [ab], [sg]))    # σ(x)
            # swish(x) = x * sigmoid(x)
            nodes.append(helper.make_node("Mul", [ab, sg], [sw]))    # 逐元素相乘
            cur = sw                                   # 本层输出接下一层输入

    # 只取前 act_size 维（μ）。后一半是 logσ，部署时丢掉。
    nodes.append(helper.make_node(
        "Slice", [cur, "starts", "ends", "axes"], ["loc"]))          # 沿第 1 轴切 [0:14]
    for nm, val in (("starts", [0]), ("ends", [act_size]), ("axes", [1])):
        inits.append(helper.make_tensor(nm, TensorProto.INT64, [1], val))   # Slice 的三个参数
    nodes.append(helper.make_node("Tanh", ["loc"], ["action"]))      # 压到 [-1, 1]

    graph = helper.make_graph(
        nodes, "duck_policy",                                            # 图名
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_size])],     # 输入
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, act_size])],  # 输出
        inits,                                                           # 所有常量
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])    # opset 17
    model.ir_version = 8                               # 兼容 onnxruntime 1.x
    onnx.checker.check_model(model)                    # 结构自检，图不合法立刻报
    onnx.save(model, out_path)                         # 落盘
    print(f"[onnx] 已保存 {out_path}（{n} 层，输入 {obs_size} → 输出 {act_size}）")
    return model


# ---------------------------------------------------------------- 数值对齐
def verify_onnx(model_path, params, mean, std, obs_size, act_size, hidden, normalize=True):
    """同一份 obs 分别喂 JAX 策略和 ONNX，比最大绝对误差。

    ⚠ 这里的 JAX 参考**必须带上与训练相同的 preprocess**（归一化）。
      否则会得到"两边都没归一化"的假阳性：误差看起来 1e-7，实际两边都是错的。
      这个坑之前踩过一次，所以参数从 build_onnx 一路透传下来。
    """
    from brax.training.acme import running_statistics                 # 归一化函数
    from brax.training.agents.ppo import networks as ppo_networks   # 局部导入，避免顶层依赖

    kwargs = dict(                                   # 网络结构的公共参数
        observation_size=obs_size, action_size=act_size,
        policy_hidden_layer_sizes=hidden, value_hidden_layer_sizes=hidden,
    )
    if normalize:                                    # 只有归一化时才挂 preprocess
        kwargs["preprocess_observations_fn"] = running_statistics.normalize
    net = ppo_networks.make_ppo_networks(**kwargs)    # 重建与训练完全相同的网络
    make_policy = ppo_networks.make_inference_fn(net)  # 包成推理函数
    policy = make_policy(params, deterministic=True)   # 确定性策略（取 μ 不采样）

    rng = np.random.default_rng(0)                     # 固定种子，结果可复现
    # 用真实量纲造 16 组输入：归一化后的 obs 大致落在 [-3, 3]
    obs = (rng.standard_normal((16, obs_size)) * 1.5).astype(np.float32)

    jax_actions = []
    for o in obs:
        a, _ = policy(jnp.asarray(o)[None], jax.random.PRNGKey(0))   # JAX 侧推理
        jax_actions.append(np.asarray(a).reshape(-1))
    jax_actions = np.stack(jax_actions)                # (16, 14)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])   # ONNX 侧
    onnx_actions = np.stack([
        sess.run(None, {"obs": o[None]})[0].reshape(-1) for o in obs
    ])                                                 # (16, 14)

    diff = np.abs(jax_actions - onnx_actions)          # 逐元素绝对误差
    print(f"[verify] JAX 输出范围 [{jax_actions.min():+.3f}, {jax_actions.max():+.3f}]")
    print(f"[verify] ONNX 输出范围 [{onnx_actions.min():+.3f}, {onnx_actions.max():+.3f}]")
    print(f"[verify] 最大绝对误差 = {diff.max():.3e}  平均 = {diff.mean():.3e}")
    if diff.max() > 1e-4:                              # 超过这个阈值说明图拼错了
        print("[verify] ✗ 超过 1e-4，导出的图和 JAX 不一致——检查层顺序/激活/归一化")
    else:
        print("[verify] ✓ 对齐通过（< 1e-4）")          # 这一行出现才算能上真机
    return diff.max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="params_xxxx.pkl")   # 训练产物
    p.add_argument("--out", default="checkpoints/policy.onnx")          # 输出路径
    p.add_argument("--obs", type=int, default=53)                       # 输入维度，与 constants 一致
    p.add_argument("--act", type=int, default=14)                       # 输出维度
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])   # 隐层宽度
    p.add_argument("--no-normalize", action="store_true",
                   help="不把 (x-mean)/std 烘进图。仅在训练确实没开 normalize_observations 时才用")
    p.add_argument("--skip-verify", action="store_true")                # 跳过对齐（提速用）
    args = p.parse_args()

    with open(args.params, "rb") as f:
        params = pickle.load(f)                        # brax 存的是 (normalizer, policy) 之类

    normalizer, policy_params = _unwrap(params)        # 剥掉外层结构
    mean = _stat(normalizer.mean)                      # 统计量（本项目未实际参与计算）
    std = _stat(normalizer.std)
    print(f"[info] 归一化: mean{mean.shape} std{std.shape}")

    layers = _layers(policy_params)                    # 按顺序取 4 层权重
    print(f"[info] 权重层: {[(n, w.shape) for n, w, _ in layers]}")
    assert layers, "没找到任何 kernel 权重，params 结构可能和预期不同"   # 空列表 = 解包失败

    build_onnx(mean, std, layers, args.obs, args.act, args.out,
               normalize=not args.no_normalize)      # 默认带归一化（与训练一致）

    if not args.skip_verify:
        if args.no_normalize:
            print("[verify] 注意：已手动关闭归一化，JAX 参考也用恒等预处理")
        verify_onnx(args.out, params, mean, std, args.obs, args.act,
                    tuple(args.hidden), normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
