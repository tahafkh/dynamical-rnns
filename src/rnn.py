"""Elman RNN with numba-accelerated forward and backward passes.

Supports:
- Configurable nonlinearity (tanh / sigmoid / relu).
- Per-sequence learnable initial hidden states (``H0_params``) for
  multi-attractor training.
- Returns hidden states and outputs directly so you can inspect /
  plot them without helper functions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from .kernels import (
    _forward_pass,
    _backward_pass,
    activation_flag,
)

@dataclass
class RNNParams:
    """Shared weight arrays of an Elman RNN."""

    W_xh: np.ndarray  # (H, D)
    W_hh: np.ndarray  # (H, H)
    b_h: np.ndarray    # (H,)
    W_hy: np.ndarray   # (O, H)
    b_y: np.ndarray    # (O,)

    @property
    def hidden_size(self) -> int:
        return self.W_xh.shape[0]

    @property
    def input_size(self) -> int:
        return self.W_xh.shape[1]

    def copy(self) -> "RNNParams":
        return RNNParams(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

@dataclass
class RNNGrads:
    """Gradients that mirror :class:`RNNParams`."""

    W_xh: np.ndarray
    W_hh: np.ndarray
    b_h: np.ndarray
    W_hy: np.ndarray
    b_y: np.ndarray

@dataclass
class ForwardResult:
    """Everything returned by :meth:`ElmanRNN.forward`.

    Attributes
    ----------
    loss : float                MSE loss (if targets were provided, else NaN).
    h : np.ndarray              (T, B, H) hidden states at every step.
    out : np.ndarray            (T, B, O) linear outputs at every step.
    h_last : np.ndarray         (B, H) last hidden state.
    cache : dict                Opaque dict needed by :meth:`backward`.
    """

    loss: float
    h: np.ndarray
    out: np.ndarray
    h_last: np.ndarray
    cache: dict


class ElmanRNN:
    """Elman RNN with linear output + MSE loss.

    Parameters
    ----------
    input_dim : int
        Input feature dimension *D*.
    hidden_size : int
        Number of recurrent units *H*.
    output_dim : int or None
        Output dimension *O*  (defaults to ``input_dim``).
    num_sequences : int or None
        If set, allocates a learnable initial hidden state ``H0_params``
        of shape ``(num_sequences, H)`` - one per attractor / sequence ID.
    activation : str
        ``"tanh"`` (default), ``"sigmoid"``, or ``"relu"``.
    weight_scale : float
        Std-dev for Gaussian weight initialisation.
    rng : int or Generator
        Random seed / generator.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: Optional[int] = None,
        num_sequences: Optional[int] = None,
        activation: str = "tanh",
        weight_scale: float = 0.01,
        rng: int | np.random.Generator = 0,
    ) -> None:
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        self.rng = rng

        D = input_dim
        H = hidden_size
        O = output_dim if output_dim is not None else D

        self.activation = activation
        self._act_flag = activation_flag(activation)

        self.params = RNNParams(
            W_xh=weight_scale * rng.standard_normal((H, D)),
            W_hh=weight_scale * rng.standard_normal((H, H)),
            b_h=np.zeros(H, dtype=np.float64),
            W_hy=weight_scale * rng.standard_normal((O, H)),
            b_y=np.zeros(O, dtype=np.float64),
        )

        # Per-sequence initial hidden states
        self.num_sequences = num_sequences
        if num_sequences is not None:
            self.H0_params: np.ndarray = weight_scale * rng.standard_normal(
                (num_sequences, H)
            )
        else:
            self.H0_params = None  # type: ignore[assignment]

    def forward(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        h0: Optional[np.ndarray] = None,
        batch_indices: Optional[np.ndarray] = None,
    ) -> ForwardResult:
        """Run the forward pass.

        Parameters
        ----------
        x : (T, B, D)   inputs.
        y : (T, B, O)   targets.  If *None* loss is ``NaN`` but outputs /
            hidden states are still computed (useful for inference).
        h0 : (B, H)     explicit initial hidden state (overrides H0_params).
        batch_indices : (B,) int array  - index into ``H0_params`` to select
            per-sequence initial hidden states.  Ignored when ``h0`` is given
            or ``H0_params`` is *None*.

        Returns
        -------
        ForwardResult with ``.loss``, ``.h``, ``.out``, ``.h_last``, ``.cache``.
        """
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        T, B, D = x.shape
        H = p.hidden_size

        # Resolve initial hidden state
        if h0 is not None:
            h0 = np.asarray(h0, dtype=np.float64)
        elif batch_indices is not None and self.H0_params is not None:
            h0 = self.H0_params[batch_indices]  # (B, H)
        else:
            h0 = np.zeros((B, H), dtype=np.float64)

        h, out = _forward_pass(
            x, h0, p.W_xh, p.W_hh, p.b_h, p.W_hy, p.b_y, self._act_flag
        )

        # Loss
        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            diff = out - y
            loss = float(0.5 * np.mean(np.sum(diff ** 2, axis=-1)))
        else:
            y_placeholder = np.empty_like(out)  # not used
            loss = float("nan")

        cache = {
            "x": x,
            "y": y,
            "h0": h0,
            "h": h,
            "out": out,
            "batch_indices": batch_indices,
        }
        return ForwardResult(
            loss=loss,
            h=h,
            out=out,
            h_last=h[-1].copy(),
            cache=cache,
        )

    def backward(self, cache: dict) -> tuple[RNNGrads, Optional[np.ndarray]]:
        """BPTT backward pass.

        Returns
        -------
        grads : RNNGrads
            Gradients for the shared weights.
        dH0 : np.ndarray or None
            If ``H0_params`` exists, an array of shape ``(num_sequences, H)``
            with accumulated gradients for each sequence-specific h0
            (use ``np.add.at`` scatter-add semantics).  *None* otherwise.
        """
        p = self.params
        dW_xh, dW_hh, db_h, dW_hy, db_y, dh0 = _backward_pass(
            cache["x"],
            cache["y"],
            cache["h0"],
            cache["h"],
            cache["out"],
            p.W_xh,
            p.W_hh,
            p.W_hy,
            self._act_flag,
        )
        grads = RNNGrads(
            W_xh=dW_xh, W_hh=dW_hh, b_h=db_h, W_hy=dW_hy, b_y=db_y
        )

        # Scatter-add dh0 into per-sequence gradient table
        batch_indices = cache.get("batch_indices")
        if batch_indices is not None and self.H0_params is not None:
            dH0 = np.zeros_like(self.H0_params)
            np.add.at(dH0, batch_indices, dh0)
            return grads, dH0

        return grads, None

    def predict(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
        batch_indices: Optional[np.ndarray] = None,
    ) -> ForwardResult:
        """Forward without targets - same return type as :meth:`forward`
        but ``loss`` will be ``NaN``."""
        return self.forward(x, y=None, h0=h0, batch_indices=batch_indices)
