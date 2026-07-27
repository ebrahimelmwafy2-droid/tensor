#ifndef B_ARITHMETIC_H
#define B_ARITHMETIC_H

#include "tensor.h"

/* Tensor-Tensor Arithmetic */

Tensor* Tensor_Add(
    const Tensor* a,
    const Tensor* b);

Tensor* Tensor_Sub(
    const Tensor* a,
    const Tensor* b);

Tensor* Tensor_Mul(
    const Tensor* a,
    const Tensor* b);

Tensor* Tensor_Div(
    const Tensor* a,
    const Tensor* b);

/* Tensor-Scalar Arithmetic */

Tensor* Tensor_AddScalar(
    const Tensor* tensor,
    float scalar);

Tensor* Tensor_SubScalar(
    const Tensor* tensor,
    float scalar);

Tensor* Tensor_MulScalar(
    const Tensor* tensor,
    float scalar);

Tensor* Tensor_DivScalar(
    const Tensor* tensor,
    float scalar);

#endif /* B_ARITHMETIC_H */