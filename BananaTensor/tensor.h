#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "banana_storage.h"
#include "shape.h"

/*==========================
        Data Types
==========================*/

typedef enum
{
    B_FLOAT32,
    B_FLOAT64,

    B_INT8,
    B_UINT8,

    B_INT16,
    B_UINT16,

    B_INT32,
    B_UINT32,

    B_INT64,
    B_UINT64

} B_DataType;

/*==========================
          Tensor
==========================*/

typedef struct Tensor
{
    Storage* storage;

    Shape shape;

    Strides strides;

    size_t offset;

    B_DataType dtype;

    bool requires_grad;

} Tensor;

/*==========================
        Utilities
==========================*/

size_t
Tensor_ElementSize(
    B_DataType type);

/*==========================
    Creation / Destruction
==========================*/

Tensor*
Tensor_Create(
    size_t ndim,
    const size_t* shape,
    B_DataType dtype,
    bool requires_grad);

Tensor*
Tensor_CreateScalar(
    B_DataType dtype);

void
Tensor_Destroy(
    Tensor* tensor);

/*==========================
        Information
==========================*/

size_t
Tensor_Numel(
    const Tensor* tensor);

size_t
Tensor_Rank(
    const Tensor* tensor);

size_t
Tensor_Dimension(
    const Tensor* tensor,
    size_t axis);

/*==========================
      Memory Access
==========================*/

void*
Tensor_Data(
    const Tensor* tensor);

/*==========================
        Reductions
==========================*/

Tensor*
Tensor_Sum(
    const Tensor* tensor);

Tensor*
Tensor_Mean(
    const Tensor* tensor);

Tensor*
Tensor_Min(
    const Tensor* tensor);

Tensor*
Tensor_Max(
    const Tensor* tensor);

/*==========================
      Internal Helpers
==========================*/

void
Tensor_ComputeStrides(
    Tensor* tensor);

void
Tensor_ComputeOffset(
    Tensor* tensor);

#endif