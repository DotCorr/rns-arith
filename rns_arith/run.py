"""Benchmark the RNS integer MLP: exactness vs int64, accuracy vs float, op/energy model."""
import numpy as np
import torch

from baseline import mlp
from common.bench import Timer, best_of
from rns_arith.rns import MODULI, M, QuantMLP

# Energy per op in pJ, 45 nm, from M. Horowitz, "Computing's energy problem", ISSCC 2014.
# 9-bit lane costs are scaled from the 8-bit figures (mult ~ bits^2, add ~ bits). These are
# modelled estimates for dedicated hardware, not measurements on this CPU.
PJ = {"fp32_mul": 3.7, "fp32_add": 0.9, "int32_mul": 3.1, "int32_add": 0.1, "int8_mul": 0.2, "int8_add": 0.03}
PJ["int9_mul"] = PJ["int8_mul"] * (9 / 8) ** 2
PJ["int24_add"] = PJ["int32_add"] * 24 / 32


def energy_model(n_in, n_hid, n_out):
    macs = n_in * n_hid + n_hid * n_out
    outs = n_hid + n_out
    fp32 = macs * (PJ["fp32_mul"] + PJ["fp32_add"])
    int8 = macs * (PJ["int8_mul"] + PJ["int32_add"]) + n_hid * PJ["int32_mul"]  # + requant mult
    lanes = len(MODULI)
    # per lane per MAC: 9-bit residue multiply + 24-bit deferred accumulate. Reductions mod
    # 2^9 / 2^9 +- 1 once per 64 terms are an end-around add (~one 24-bit add): amortised.
    rns_mac = macs * lanes * (PJ["int9_mul"] + PJ["int24_add"] * (1 + 1 / 64))
    # CRT per output: 3 residue x constant multiplies + 3 adds (+ mod M), modelled as int32 ops
    rns_crt = outs * lanes * (PJ["int32_mul"] + PJ["int32_add"])
    # inputs and requantised activations are uint8 < 511, so forward conversion is free
    rns = rns_mac + rns_crt + n_hid * PJ["int32_mul"]
    return {"macs": macs, "fp32_nJ": fp32 / 1e3, "int8_nJ": int8 / 1e3, "rns_nJ": rns / 1e3,
            "lane_ops": macs * lanes, "crt_ops": outs}


def run(xtr, ytr, xte, yte):
    with Timer() as t_tr:
        model, _ = mlp.train(xtr, ytr)
        q = QuantMLP(model, xtr[:10000])
    with torch.no_grad():
        f_pred = model(torch.from_numpy(xte.reshape(len(xte), -1).astype(np.float32) / 255.0)).argmax(1).numpy()

    (acc1_i, logit_i), t_int = best_of(lambda: q.forward_int(xte), repeats=2)
    (acc1_r, logit_r), t_rns = best_of(lambda: q.forward_rns(xte), repeats=2)
    exact = bool((acc1_i == acc1_r).all() and (logit_i == logit_r).all())
    assert exact, "RNS result differs from plain integer arithmetic"
    e = energy_model(784, q.W1.shape[0], 10)
    acc_q = float((logit_r.argmax(1) == yte).mean())
    return [{
        "model": f"RNS int8 MLP, moduli {MODULI}",
        "acc": acc_q,
        "train_s": t_tr.s,
        "infer_s": t_rns,
        "mem_bytes": q.mem_bytes(),
        "note": (f"fp32 acc of same net {float((f_pred == yte).mean()):.4f}; RNS == int64 on "
                 f"{acc1_i.size + logit_i.size:,} values: {exact}; numpy int64 ref {t_int:.2f}s"),
        "extra": {**e, "float_acc": float((f_pred == yte).mean()), "exact": exact, "int_ref_s": t_int,
                  "dynamic_range": M, "max_abs_acc": int(max(np.abs(acc1_i).max(), np.abs(logit_i).max()))},
    }]
