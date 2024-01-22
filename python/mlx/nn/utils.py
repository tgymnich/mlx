# Copyright © 2023 Apple Inc.

from contextlib import contextmanager
from typing import Any, Callable

import mlx.core as mx


@contextmanager
def updated_model(model: "mlx.nn.Module", *parameters: Any):
    old_state = model.parameters()
    try:
        for p in parameters:
            model.update(p)
        yield model
    finally:
        model.update(old_state)


def value_and_grad(model: "mlx.nn.Module", fn: Callable, compile: bool = False):
    """Transform the passed function ``fn`` to a function that computes the
    gradients of ``fn`` wrt the model's trainable parameters and also its
    value.

    Args:
        model (mlx.nn.Module): The model whose trainable parameters to compute
                               gradients for
        fn (Callable): The scalar function to compute gradients for
        compile (bool): Whether to "compile" the function before returning it.
                        Default: ``False``.

    Returns:
        A callable that returns the value of ``fn`` and the gradients wrt the
        trainable parameters of ``model``
    """

    def inner_fn(trainable_parameters, other_parameters, *args, **kwargs):
        with updated_model(model, other_parameters, trainable_parameters):
            return fn(*args, **kwargs)

    value_grad_fn = mx.value_and_grad(inner_fn)
    if compile:
        value_grad_fn = mx.compile(value_grad_fn)

    def wrapped_value_grad_fn(*args, **kwargs):
        value, grad = value_grad_fn(
            model.trainable_parameters(),
            model.non_trainable_parameters(),
            *args,
            **kwargs,
        )
        return value, grad

    return wrapped_value_grad_fn
