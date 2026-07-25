#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*=========================================================
    Arithmetic
=========================================================*/

Tensor Tensor_Add(const Tensor *a, const Tensor *b);

Tensor Tensor_Sub(const Tensor *a, const Tensor *b);

Tensor Tensor_Mul(const Tensor *a, const Tensor *b);

Tensor Tensor_Div(const Tensor *a, const Tensor *b);

/*=========================================================
    Scalar
=========================================================*/

Tensor Tensor_AddScalar(const Tensor *tensor, float scalar);

Tensor Tensor_SubScalar(const Tensor *tensor, float scalar);

Tensor Tensor_MulScalar(const Tensor *tensor, float scalar);

Tensor Tensor_DivScalar(const Tensor *tensor, float scalar);

/*=========================================================
    Utilities
=========================================================*/

Tensor Tensor_Copy(const Tensor *tensor);

Tensor Tensor_Clone(const Tensor *tensor);

void Tensor_Fill(Tensor *tensor, float value);

void Tensor_Zero(Tensor *tensor);

void Tensor_Ones(Tensor *tensor);

void Tensor_Random(Tensor *tensor);

int Tensor_Equal(const Tensor *a, const Tensor *b);

void Tensor_Print(const Tensor *tensor);

/*=========================================================
    Linear Algebra
=========================================================*/

Tensor Tensor_Dot(const Tensor *a, const Tensor *b);

Tensor Tensor_MatMul(const Tensor *a, const Tensor *b);

Tensor Tensor_Transpose(const Tensor *tensor);

Tensor Tensor_Reshape(
    const Tensor *tensor,
    size_t ndim,
    const size_t *shape);

Tensor Tensor_View(
    const Tensor *tensor,
    size_t offset,
    size_t ndim,
    const size_t *shape);

Tensor Tensor_Slice(
    const Tensor *tensor,
    size_t start,
    size_t length);

Tensor Tensor_Squeeze(const Tensor *tensor);

Tensor Tensor_Unsqueeze(
    const Tensor *tensor,
    size_t axis);

#ifdef __cplusplus
}
#endif

#endif