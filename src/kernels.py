"""Numba-accelerated kernels for Elman RNN forward and backward passes.

Activation is selected via an integer flag to stay numba-compatible:
    0 = tanh,  1 = sigmoid,  2 = relu
"""

from __future__ import annotations

import numpy as np
import numba as nb

# Activation flag constants (importable from Python-land)
ACT_TANH: int = 0
ACT_SIGMOID: int = 1
ACT_RELU: int = 2

ACTIVATION_NAMES = {"tanh": ACT_TANH, "sigmoid": ACT_SIGMOID, "relu": ACT_RELU}


def activation_flag(name: str) -> int:
    """Convert a human-readable activation name to a numba-friendly int flag."""
    name = name.lower().strip()
    if name not in ACTIVATION_NAMES:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Choose from {list(ACTIVATION_NAMES.keys())}."
        )
    return ACTIVATION_NAMES[name]

@nb.njit(cache=True, inline="always")
def _apply_activation(x: np.ndarray, act: int) -> np.ndarray:
    """Apply activation element-wise, returning a new array."""
    out = np.empty_like(x)
    for i in range(x.size):
        v = x.flat[i]
        if act == 0:       # tanh
            out.flat[i] = np.tanh(v)
        elif act == 1:     # sigmoid
            out.flat[i] = 1.0 / (1.0 + np.exp(-v))
        else:              # relu
            out.flat[i] = v if v > 0.0 else 0.0
    return out.reshape(x.shape)

@nb.njit(cache=True, inline="always")
def _activation_deriv(h: np.ndarray, act: int) -> np.ndarray:
    """Derivative of activation w.r.t. pre-activation, expressed via h=phi(a)."""
    out = np.empty_like(h)
    for i in range(h.size):
        v = h.flat[i]
        if act == 0:       # tanh: 1 - h^2
            out.flat[i] = 1.0 - v * v
        elif act == 1:     # sigmoid: h*(1-h)
            out.flat[i] = v * (1.0 - v)
        else:              # relu: 1 if h>0 else 0
            out.flat[i] = 1.0 if v > 0.0 else 0.0
    return out.reshape(h.shape)

@nb.njit(cache=True)
def _forward_pass(
    x: np.ndarray,       # (T, B, D)
    h0: np.ndarray,      # (B, H)
    W_xh: np.ndarray,    # (H, D)
    W_hh: np.ndarray,    # (H, H)
    b_h: np.ndarray,     # (H,)
    W_hy: np.ndarray,    # (O, H)
    b_y: np.ndarray,     # (O,)
    act: int = 0,
):
    """Forward pass returning hidden states and linear outputs.

    Parameters
    ----------
    act : int
        0 = tanh, 1 = sigmoid, 2 = relu

    Returns
    -------
    h   : (T, B, H) hidden states
    out : (T, B, O) linear outputs
    """
    T, B, D = x.shape
    H = W_xh.shape[0]
    O = W_hy.shape[0]

    h = np.empty((T, B, H), dtype=np.float64)
    out = np.empty((T, B, O), dtype=np.float64)

    h_prev = h0.copy()
    for t in range(T):
        preact = x[t] @ W_xh.T + h_prev @ W_hh.T + b_h
        h_t = _apply_activation(preact, act)
        o_t = h_t @ W_hy.T + b_y
        h[t] = h_t
        out[t] = o_t
        h_prev = h_t

    return h, out

@nb.njit(cache=True)
def _backward_pass(
    x: np.ndarray,        # (T, B, D)
    y: np.ndarray,        # (T, B, O)
    h0: np.ndarray,       # (B, H)
    h: np.ndarray,        # (T, B, H)
    out: np.ndarray,      # (T, B, O)
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    W_hy: np.ndarray,
    act: int = 0,
):
    r"""BPTT for MSE loss = mean_t mean_b  0.5 \|out_t - y_t\|^2.

    Returns
    -------
    dW_xh, dW_hh, db_h, dW_hy, db_y, dh0
    """
    T, B, O = y.shape
    H = h.shape[2]

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros(H, dtype=np.float64)
    dW_hy = np.zeros_like(W_hy)
    db_y = np.zeros(O, dtype=np.float64)

    dh_next = np.zeros((B, H), dtype=np.float64)

    for t in range(T - 1, -1, -1):
        h_t = h[t]
        h_prev = h0 if t == 0 else h[t - 1]

        # dL/d(out_t) = (out_t - y_t) / (T * B)
        d_out = (out[t] - y[t]) / (T * B)

        # Output layer
        dW_hy += d_out.T @ h_t
        for b in range(B):
            for o in range(O):
                db_y[o] += d_out[b, o]

        # Into hidden
        dh = d_out @ W_hy + dh_next

        # Through activation
        dphi = _activation_deriv(h_t, act)
        dpreact = dh * dphi

        dW_xh += dpreact.T @ x[t]
        dW_hh += dpreact.T @ h_prev
        for b in range(B):
            for hh in range(H):
                db_h[hh] += dpreact[b, hh]

        dh_next = dpreact @ W_hh

    dh0 = dh_next
    return dW_xh, dW_hh, db_h, dW_hy, db_y, dh0
