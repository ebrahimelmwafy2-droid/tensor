#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include "banana_storage.h"
#include <stdint.h>
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
typedef struct Tensor
{
    Storage *storage;

    size_t *shape;
    size_t *strides;

    size_t ndim;
    size_t offset;

    B_DataType dtype;

    struct B_Node *creator;

    int requires_grad;

} Tensor;
size_t Tensor_ElementSize(B_DataType type);


/*==========================
    Creation / Destruction
==========================*/

Tensor Tensor_Create(size_t ndim, const size_t *shape);

void Tensor_Destroy(Tensor *tensor);

/*==========================
        Information
==========================*/

size_t Tensor_Numel(const Tensor *tensor);

size_t Tensor_Rank(const Tensor *tensor);

size_t Tensor_Dimension(const Tensor *tensor, size_t axis);

/*==========================
      Memory Access
==========================*/

float *Tensor_Data(const Tensor *tensor);

/*==========================
      Internal Helpers
==========================*/

void Tensor_ComputeStrides(Tensor *tensor);

#endif