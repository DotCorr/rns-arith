"""Integer-quantised MLP whose matmuls run in a Residue Number System.

Moduli {2^9-1, 2^9, 2^9+1} = {511, 512, 513}: pairwise coprime, M = 134,217,216, and each has
a cheap hardware reduction (end-around carry / truncation / end-around borrow). Every dot
product is computed independently in three 9-bit residue lanes with no carries between them;
CRT reconstruction to a signed integer happens only where a non-modular operation is needed
(ReLU + requantisation between layers, and argmax at the output).
"""
import numpy as np

MODULI = (511, 512, 513)
M = int(np.prod(MODULI, dtype=np.int64))
# CRT constants: x = sum_i r_i * Mi * inv(Mi mod mi)  (mod M)
_MI = [M // m for m in MODULI]
_CRT = [Mi * pow(Mi, -1, m) % M for Mi, m in zip(_MI, MODULI)]


def to_rns(x):
    """int64 array -> list of residue arrays (one per modulus), values in [0, m)."""
    return [np.mod(x, m).astype(np.int64) for m in MODULI]


def from_rns(res):
    """CRT reconstruction to signed integers in [-M/2, M/2)."""
    # r < 513 and _CRT < M ~ 2^27, so each product < 2^36: int64 is safe; sum mod M stepwise
    acc = np.zeros(res[0].shape, dtype=np.int64)
    for r, c in zip(res, _CRT):
        acc = (acc + r * c) % M
    return np.where(acc >= M // 2, acc - M, acc)


def rns_matmul(xr, wr, br):
    """Per-lane modular affine map: (x @ W.T + b) mod m_i for each lane. A lane's dot product
    is accumulated with a modular reduction every 64 terms so the accumulator never needs
    more than 9 + 9 + 6 = 24 bits, which is what a 9-bit-lane MAC with deferred reduction
    would hold in hardware."""
    out = []
    for x, w, b, m in zip(xr, wr, br, MODULI):
        acc = np.zeros((x.shape[0], w.shape[0]), dtype=np.int64)
        for s in range(0, x.shape[1], 64):
            acc = (acc + x[:, s:s + 64] @ w[:, s:s + 64].T) % m
        out.append((acc + b) % m)
    return out


class QuantMLP:
    """784-H-10 int8-weight / uint8-activation MLP built by post-training quantisation."""

    def __init__(self, torch_model, x_calib):
        l1, l2 = torch_model[0], torch_model[2]
        W1, b1 = l1.weight.detach().numpy().astype(np.float64), l1.bias.detach().numpy().astype(np.float64)
        W2, b2 = l2.weight.detach().numpy().astype(np.float64), l2.bias.detach().numpy().astype(np.float64)
        sx = 1.0 / 255.0
        s1 = np.abs(W1).max(1) / 127.0
        self.W1 = np.round(W1 / s1[:, None]).astype(np.int64)
        self.b1 = np.round(b1 / (s1 * sx)).astype(np.int64)
        # calibrate the hidden activation scale on training data (99.99th percentile)
        h = np.maximum(x_calib.reshape(len(x_calib), -1) / 255.0 @ W1.T + b1, 0)
        sh = np.percentile(h, 99.99) / 255.0
        # requant: h_q = (relu(acc) * mult + 2^23) >> 24, an integer multiply-and-shift
        self.mult = np.round(s1 * sx / sh * 2 ** 24).astype(np.int64)
        s2 = np.abs(W2).max(1) / 127.0
        self.W2 = np.round(W2 / s2[:, None]).astype(np.int64)
        self.b2 = np.round(b2 / (s2 * sh)).astype(np.int64)
        for W, b, xmax in ((self.W1, self.b1, 255), (self.W2, self.b2, 255)):
            bound = (np.abs(W).sum(1) * xmax + np.abs(b)).max()
            assert 2 * bound < M, f"accumulator bound {bound} exceeds RNS dynamic range"
        self.rW1, self.rb1 = [r for r in to_rns(self.W1)], to_rns(self.b1)
        self.rW2, self.rb2 = [r for r in to_rns(self.W2)], to_rns(self.b2)

    def _requant(self, acc1):
        return np.clip((np.maximum(acc1, 0) * self.mult + (1 << 23)) >> 24, 0, 255)

    def forward_int(self, x):
        """Reference: plain int64 arithmetic. Returns (layer-1 acc, logits)."""
        x = x.reshape(len(x), -1).astype(np.int64)
        acc1 = x @ self.W1.T + self.b1
        return acc1, self._requant(acc1) @ self.W2.T + self.b2

    def forward_rns(self, x):
        x = x.reshape(len(x), -1).astype(np.int64)
        acc1 = from_rns(rns_matmul(to_rns(x), self.rW1, self.rb1))
        logits = from_rns(rns_matmul(to_rns(self._requant(acc1)), self.rW2, self.rb2))
        return acc1, logits

    def mem_bytes(self):
        # int8 weights; int32 biases and requant multipliers
        return self.W1.size + self.W2.size + 4 * (self.b1.size + self.b2.size + self.mult.size)
