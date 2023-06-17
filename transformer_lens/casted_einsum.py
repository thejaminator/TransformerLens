from fancy_einsum import einsum
from torch import Tensor


def einsum_cast(equation: str, *operands: Tensor) -> Tensor:
    """Evaluates the Einstein summation convention on the operands.
    casts the operands to the same type before calling einsum

    See:
      https://pytorch.org/docs/stable/generated/torch.einsum.html
      https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
    """
    new_operands = []
    first_dtype = operands[0].dtype
    for operand in operands:
        if operand.dtype != first_dtype:
            new_operands.append(operand.to(first_dtype))
        else:
            new_operands.append(operand)
    return einsum(equation, *new_operands)
