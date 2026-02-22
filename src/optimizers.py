"""Optimizers and gradient utilities for the RNN."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rnn import RNNGrads, RNNParams

def global_grad_norm(grads: RNNGrads, dH0: Optional[np.ndarray] = None) -> float:
    """L2 norm over all gradient arrays concatenated."""
    s = 0.0
    for g in (grads.W_xh, grads.W_hh, grads.b_h, grads.W_hy, grads.b_y):
        s += float(np.sum(g * g))
    if dH0 is not None:
        s += float(np.sum(dH0 * dH0))
    return float(np.sqrt(s))

def clip_grads(
    grads: RNNGrads,
    max_norm: float,
    dH0: Optional[np.ndarray] = None,
) -> tuple[RNNGrads, Optional[np.ndarray], float]:
    """Global-norm gradient clipping.

    Returns ``(grads, dH0, original_norm)``.  If *max_norm* ≤ 0 no
    clipping is applied.
    """
    norm = global_grad_norm(grads, dH0)
    if max_norm <= 0 or norm <= max_norm:
        return grads, dH0, norm
    scale = max_norm / (norm + 1e-12)
    clipped = RNNGrads(
        W_xh=grads.W_xh * scale,
        W_hh=grads.W_hh * scale,
        b_h=grads.b_h * scale,
        W_hy=grads.W_hy * scale,
        b_y=grads.b_y * scale,
    )
    clipped_dH0 = dH0 * scale if dH0 is not None else None
    return clipped, clipped_dH0, norm

class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(self, lr: float = 1e-2, clip_norm: float = 0.0) -> None:
        self.lr = lr
        self.clip_norm = clip_norm

    def step(
        self,
        params: RNNParams,
        grads: RNNGrads,
        dH0: Optional[np.ndarray] = None,
    ) -> tuple[RNNParams, Optional[np.ndarray], float]:
        """Apply one SGD update.

        Returns ``(new_params, updated_H0_or_None, pre_clip_grad_norm)``.
        """
        grads, dH0, gnorm = clip_grads(grads, self.clip_norm, dH0)
        new_params = RNNParams(
            W_xh=params.W_xh - self.lr * grads.W_xh,
            W_hh=params.W_hh - self.lr * grads.W_hh,
            b_h=params.b_h - self.lr * grads.b_h,
            W_hy=params.W_hy - self.lr * grads.W_hy,
            b_y=params.b_y - self.lr * grads.b_y,
        )
        return new_params, dH0, gnorm

class Adam:
    """Adam optimizer (Kingma & Ba, 2015)."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip_norm: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self.t = 0
        self._m: dict[str, np.ndarray] | None = None
        self._v: dict[str, np.ndarray] | None = None
        # Separate moment buffers for H0
        self._m_h0: np.ndarray | None = None
        self._v_h0: np.ndarray | None = None

    def _init_state(self, params: RNNParams) -> None:
        self._m = {
            "W_xh": np.zeros_like(params.W_xh),
            "W_hh": np.zeros_like(params.W_hh),
            "b_h": np.zeros_like(params.b_h),
            "W_hy": np.zeros_like(params.W_hy),
            "b_y": np.zeros_like(params.b_y),
        }
        self._v = {
            "W_xh": np.zeros_like(params.W_xh),
            "W_hh": np.zeros_like(params.W_hh),
            "b_h": np.zeros_like(params.b_h),
            "W_hy": np.zeros_like(params.W_hy),
            "b_y": np.zeros_like(params.b_y),
        }

    def step(
        self,
        params: RNNParams,
        grads: RNNGrads,
        dH0: Optional[np.ndarray] = None,
        H0_params: Optional[np.ndarray] = None,
        batch_indices: Optional[np.ndarray] = None,
    ) -> tuple[RNNParams, Optional[np.ndarray], float]:
        """Apply one Adam update.

        Parameters
        ----------
        params, grads : shared-weight params and their gradients.
        dH0 : (num_sequences, H)  accumulated H0 gradients (from scatter-add).
        H0_params : the current H0 array to update in-place (or returns new).
        batch_indices : which sequence IDs were in the batch (for count-based
            averaging of dH0).

        Returns
        -------
        ``(new_params, new_H0_or_None, pre_clip_grad_norm)``
        """
        grads, dH0, gnorm = clip_grads(grads, self.clip_norm, dH0)

        if self._m is None:
            self._init_state(params)
        assert self._m is not None and self._v is not None

        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)

        new = {}
        for name in ("W_xh", "W_hh", "b_h", "W_hy", "b_y"):
            g = getattr(grads, name)
            self._m[name] = b1 * self._m[name] + (1 - b1) * g
            self._v[name] = b2 * self._v[name] + (1 - b2) * (g * g)
            new[name] = (
                getattr(params, name)
                - lr_t * self._m[name] / (np.sqrt(self._v[name]) + eps)
            )

        new_H0 = None
        if dH0 is not None and H0_params is not None:
            # Average dH0 by the number of times each sequence appeared
            if batch_indices is not None:
                counts = np.bincount(
                    batch_indices, minlength=H0_params.shape[0]
                ).reshape(-1, 1).astype(np.float64)
                counts[counts == 0] = 1.0
                dH0 = dH0 / counts

            if self._m_h0 is None:
                self._m_h0 = np.zeros_like(H0_params)
                self._v_h0 = np.zeros_like(H0_params)

            self._m_h0 = b1 * self._m_h0 + (1 - b1) * dH0
            self._v_h0 = b2 * self._v_h0 + (1 - b2) * (dH0 * dH0)
            new_H0 = H0_params - lr_t * self._m_h0 / (
                np.sqrt(self._v_h0) + eps
            )

        return RNNParams(**new), new_H0, gnorm
