"""域随机化验收：训练前先确认参数真的随机了、范围对、而且能跑。

分两段：
  [1] 纯数值检查（秒级，不编译）
      - 每个字段有没有前导维 num_envs
      - 数值范围是否落在 config 设定的区间
      - **kp 与 biasprm[1] 是否保持 -kp 关系**（错了鸭子会飞出去）
  [2] 真跑一步（含编译，60~90 秒）
      - 用 brax 的包装层走一遍 reset + step，确认不 NaN、不崩

用法：
    uv run python -u scripts/check_dr.py            # 两段都跑
    uv run python -u scripts/check_dr.py --fast     # 只跑第 1 段
"""
import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from duck_rl.env import DuckEnv
from duck_rl.randomize import DR_FIELDS, FLOATING_BASE_DOF, make_randomization_fn

N = 8   # 验证用的小批量。真实训练是 512，但数值检查 8 个就够看范围了


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="只做数值检查，不编译不跑")
    args = p.parse_args()

    t0 = time.time()
    print("[1/2] 构造环境 ...", flush=True)
    env = DuckEnv()
    cfg = env._config
    print(f"       obs={env.observation_size} act={env.action_size} nv={env.mjx_model.nv} "
          f"nbody={env.mjx_model.nbody} nu={env.mjx_model.nu}", flush=True)

    fn = make_randomization_fn(env.mj_model, cfg)
    rng = jax.random.split(jax.random.PRNGKey(0), N)          # (N, 2)，模拟 brax 的做法
    model_v, in_axes = fn(env.mjx_model, rng)
    jax.block_until_ready(model_v.geom_friction)              # 强制算完

    base = env.mjx_model      # 未随机化的原模型，用来对比

    # ---------------------------------------------------------------- 形状
    print("\n=== 形状：改过的字段应有前导维 N，其余不该有 ===")
    ok_shape = True
    for f in DR_FIELDS:
        got = getattr(model_v, f).shape
        want0 = N
        flag = "✓" if got[0] == want0 else "✗"
        if got[0] != want0:
            ok_shape = False
        print(f"  {flag} {f:<22} {got}")

    # 挑两个不该有前导维的字段确认
    for f in ("qpos0", "jnt_range"):
        got = getattr(model_v, f).shape
        flag = "✓" if got[0] == getattr(base, f).shape[0] else "✗"
        print(f"  {flag} {f:<22} {got}  （不应有前导维）")

    # ---------------------------------------------------------------- 范围
    print("\n=== 范围：每个字段应在 config 设定区间内 ===")
    import mujoco
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    fr = np.asarray(model_v.geom_friction[:, floor_id, 0])
    print(f"  地面摩擦      min={fr.min():.3f} max={fr.max():.3f}  "
          f"期望 [{cfg.dr_friction[0]}, {cfg.dr_friction[1]}]  "
          f"{'✓' if cfg.dr_friction[0] - 1e-6 <= fr.min() and fr.max() <= cfg.dr_friction[1] + 1e-6 else '✗'}")

    fl = np.asarray(model_v.dof_frictionloss[:, FLOATING_BASE_DOF:])
    b_fl = np.asarray(base.dof_frictionloss[FLOATING_BASE_DOF:])
    print(f"  关节摩擦倍数  min={(fl.min(1)/b_fl).min():.3f} max={(fl.max(1)/b_fl).max():.3f}  "
          f"期望 [{cfg.dr_frictionloss[0]}, {cfg.dr_frictionloss[1]}]")

    kp = np.asarray(model_v.actuator_gainprm[:, :, 0])
    b_kp = np.asarray(base.actuator_gainprm[:, 0])
    print(f"  舵机 kp 倍数  min={(kp.min(1)/b_kp).min():.3f} max={(kp.max(1)/b_kp).max():.3f}  "
          f"期望 [{cfg.dr_kp[0]}, {cfg.dr_kp[1]}]  （基准 kp={b_kp[0]:.2f}）")

    mass = np.asarray(model_v.body_mass).sum(1)
    b_mass = float(np.asarray(base.body_mass).sum())
    print(f"  总质量        min={mass.min():.3f} max={mass.max():.3f}  "
          f"期望 [{b_mass*cfg.dr_mass[0]:.3f}, {b_mass*cfg.dr_mass[1]:.3f}]  （基准 {b_mass:.3f} kg）")

    fmax = np.abs(np.asarray(model_v.actuator_forcerange)).max()
    b_fmax = float(np.abs(np.asarray(base.actuator_forcerange)).max())
    print(f"  力矩上限 max  min={np.abs(np.asarray(model_v.actuator_forcerange)).max(axis=(1,2)).min():.3f} "
          f"max={fmax:.3f}  （基准 ±{b_fmax:.2f}）")

    # ---------------------------------------------------------------- 最关键：kp 与 biasprm 同步
    print("\n=== kp 与 biasprm[1] 必须互为相反数 ===")
    bias = np.asarray(model_v.actuator_biasprm[:, :, 1])
    err = np.abs(bias + kp).max()
    print(f"  max|biasprm[1] + kp| = {err:.3e}   {'✓ 同步正确' if err < 1e-5 else '✗ 不同步！鸭子会飞出去'}")
    print(f"  对照：第一环境前 3 个舵机")
    print(f"    kp        = {kp[0, :3]}")
    print(f"    biasprm[1]= {bias[0, :3]}")

    # ---------------------------------------------------------------- 环境间的差异
    print("\n=== 环境间差异（要真随机，不能所有 env 一样）===")
    print(f"  地面摩擦 8 个环境: {np.round(fr, 3)}")
    print(f"  kp 8 个环境      : {np.round(kp[:, 0], 3)}")
    n_unique = len(np.unique(np.round(fr, 6)))
    print(f"  不同摩擦值个数 = {n_unique} / {N}   {'✓' if n_unique == N else '✗ 有重复，可能没真随机'}")

    print(f"\n数值检查用时 {time.time()-t0:.1f}s")
    if not ok_shape:
        print("\n!! 形状不对，先修 randomize.py 的 in_axes，别往下跑")
        return

    if args.fast:
        print("\n--fast：跳过第 2 段（真跑一步）")
        return

    # ---------------------------------------------------------------- 真跑一步
    print("\n[2/2] 用 brax 包装层真跑 reset + step（首次含编译，60~90 秒）...", flush=True)
    from mujoco_playground import wrapper

    wrapped = wrapper.wrap_for_brax_training(
        env,
        episode_length=1000,
        randomization_fn=functools.partial(fn, rng=rng),
    )
    key = jnp.zeros((N, 2), dtype=jnp.uint32)          # 同一 key，只让随机化产生差异
    t1 = time.time()
    state = jax.jit(wrapped.reset)(key)
    jax.block_until_ready(state.obs)
    print(f"  reset 完成（{time.time()-t1:.1f}s），obs {state.obs.shape}", flush=True)

    action = jnp.zeros((N, env.action_size))
    t2 = time.time()
    state2 = jax.jit(wrapped.step)(state, action)
    jax.block_until_ready(state2.obs)
    print(f"  step 完成（{time.time()-t2:.1f}s）", flush=True)

    h = np.asarray(state2.data.qpos[:, 2])
    d = np.asarray(state2.done)
    print(f"\n  8 个环境的高度: {np.round(h, 4)}")
    print(f"  高度范围 {h.min():.4f} ~ {h.max():.4f}（有差异说明摩擦/质量真的不同）")
    print(f"  done: {d}")
    print(f"  NaN 检查: obs {np.isnan(np.asarray(state2.obs)).any()}  "
          f"qpos {np.isnan(np.asarray(state2.qpos)).any()}")

    print("\n判读：")
    print("  ✓ 形状/范围/kp 同步都对 → 可以开训")
    print("  ✗ 有 NaN 或 done 全 1 → 随机化太狠或字段改错，先缩范围（比如 kp 改 0.9~1.1）")


if __name__ == "__main__":
    main()
