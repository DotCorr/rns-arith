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
