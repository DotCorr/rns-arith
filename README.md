# rns-arith

An integer-quantised MLP whose matrix multiplies run in a **Residue Number System** (RNS). Each dot
product is split into three small, carry-free arithmetic lanes and recombined exactly via the Chinese
Remainder Theorem. The repo checks exactness against plain integer arithmetic, reports accuracy and
timing on MNIST and Fashion-MNIST, and estimates per-image energy against float32 and plain int8.

## How it works ([`rns_arith/rns.py`](rns_arith/rns.py))

- **Model:** a 784-256-10 MLP trained in float with PyTorch (Adam, 5 epochs), then quantised after
  training: int8 per-row weights, uint8 activations, int32 biases. Between layers it uses an
  integer multiply-and-shift requantisation.
- **Moduli:** {2^9−1, 2^9, 2^9+1} = {511, 512, 513}. They are pairwise coprime, with dynamic range
  M = 134,217,216. Each modulus has a cheap hardware reduction (end-around carry, truncation,
  end-around borrow). The code asserts that every layer's worst-case accumulator fits in ±M/2.
- **Lanes:** every affine map runs independently mod each m_i, with a modular reduction every 64
  terms, so a lane accumulator never needs more than 24 bits.
- **CRT:** reconstruction to signed integers happens only where a non-modular step is needed (ReLU and
  requantisation, then argmax). Inputs and activations are uint8 < 511, so converting them into
  residues costs nothing.

## Results

Written by `run_all.py` from an actual run. Everything runs on **one CPU thread**. The RNS row's train
time includes training the float MLP it is quantised from.

<!-- RESULTS:START -->
Measured 2026-11-08 14:31 on `arm64 arm / macOS-27.0-arm64-arm-64bit-Mach-O`, Python 3.13.15, numpy 2.5.4, torch 2.15.0, **1 CPU thread** for everything. Train time is wall-clock for the whole fit; inference is the best of repeated runs over the 10,000 test images.

### MNIST

| Model | Test acc | Train (s) | Infer 10k (s) | Model memory |
|---|---:|---:|---:|---:|
| MLP 784-256-10 fp32 (Adam, 5 ep) | 97.47% | 3.63 | 0.008 | 814 kB |
| MLP 784-32-10 fp32 (Adam, 5 ep) | 95.29% | 1.60 | 0.003 | 102 kB |
| RNS int8 MLP, moduli (511, 512, 513) | 97.34% | 3.15 | 3.690 | 205 kB |

RNS check: float32 accuracy of the same trained net 97.47%; residue result identical to int64 arithmetic: **True** (max |accumulator| 1,612,259 vs dynamic range M = 134,217,216). Modelled energy per image (203,264 MACs): fp32 935 nJ, RNS 3x9-bit 204 nJ, plain int8/int32 62 nJ.

### Fashion-MNIST

| Model | Test acc | Train (s) | Infer 10k (s) | Model memory |
|---|---:|---:|---:|---:|
| MLP 784-256-10 fp32 (Adam, 5 ep) | 87.22% | 3.22 | 0.007 | 814 kB |
| MLP 784-32-10 fp32 (Adam, 5 ep) | 85.32% | 1.16 | 0.003 | 102 kB |
| RNS int8 MLP, moduli (511, 512, 513) | 86.17% | 2.70 | 3.499 | 205 kB |

RNS check: float32 accuracy of the same trained net 87.22%; residue result identical to int64 arithmetic: **True** (max |accumulator| 3,108,829 vs dynamic range M = 134,217,216). Modelled energy per image (203,264 MACs): fp32 935 nJ, RNS 3x9-bit 204 nJ, plain int8/int32 62 nJ.
<!-- RESULTS:END -->

## What the numbers say

- **Exactness holds.** The residue result matches int64 arithmetic on every one of 2.66 M accumulator
  values.
- **Quantisation costs little accuracy:** about 0.1–1 point against the float network.
- **The energy claim is mixed.** The *modelled* energy (Horowitz, ISSCC 2014, 45 nm per-op figures,
  not a measurement) puts the RNS datapath about 4.5× below fp32, but about 3× *above* a plain
  int8×int8→int32 multiply-accumulate. For a dynamic range this small, three 9-bit lanes do more work
  than one 8-bit multiplier feeding a 32-bit adder.
- **Where RNS could pay off:** exact wide-precision accumulation, or cases where carry-free
  independent lanes matter (fault isolation, very wide words). It does not pay off at int8 MNIST
  scale.
- **Wall-clock in numpy is slow:** integer matmul has no BLAS path, and it runs once per lane.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py          # downloads MNIST + Fashion-MNIST into data/ on first run
.venv/bin/python run_all.py --quick  # MNIST only, does not touch README
```

## Next steps

Test RNS where it should win: wider accumulators (int16 weights, larger layers), or hardware where a
residue lane is a natural unit.

Part of a set of three edge-ML prototypes, alongside
[lookup-grid](https://github.com/DotCorr/lookup-grid) and
[closed-form-nets](https://github.com/DotCorr/closed-form-nets).
