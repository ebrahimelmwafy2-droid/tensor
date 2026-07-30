#ifndef UTILITY_H
#define UTILITY_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*=========================================================
    Utilities
=========================================================*/

Tensor* Tensor_Copy(const Tensor* tensor);

Tensor* Tensor_Clone(const Tensor* tensor);

void Tensor_Fill(
    Tensor* tensor,
    float value);

void Tensor_Zero(
    Tensor* tensor);

void Tensor_Ones(
    Tensor* tensor);

void Tensor_Random(
    Tensor* tensor);

int Tensor_Equal(
    const Tensor* a,
    const Tensor* b);

void Tensor_Print(
    const Tensor* tensor);
Tensor* Tensor_Clone(const Tensor* tensor);

int Tensor_Equal(
    const Tensor* a,
    const Tensor* b);

void Tensor_Print(
    const Tensor* tensor);
#ifdef __cplusplus
}
#endif

#endif