#include "tensor.h"

#include <stdlib.h>
#include <string.h>
size_t Tensor_ElementSize(B_DataType type)
{
    switch (type)
    {
        case B_FLOAT32: return sizeof(float);
        case B_FLOAT64: return sizeof(double);

        case B_INT8:   return sizeof(int8_t);
        case B_UINT8:  return sizeof(uint8_t);

        case B_INT16:  return sizeof(int16_t);
        case B_UINT16: return sizeof(uint16_t);

        case B_INT32:  return sizeof(int32_t);
        case B_UINT32: return sizeof(uint32_t);

        case B_INT64:  return sizeof(int64_t);
        case B_UINT64: return sizeof(uint64_t);

        default:
            return 0;
    }
}
size_t Tensor_Numel(const Tensor *tensor)
{
    size_t numel = 1;

    for (size_t i = 0; i < tensor->ndim; i++)
        numel *= tensor->shape[i];

    return numel;
}

size_t Tensor_Rank(const Tensor *tensor)
{
    return tensor->ndim;
}

size_t Tensor_Dimension(const Tensor *tensor, size_t axis)
{
    if (axis >= tensor->ndim)
        return 0;

    return tensor->shape[axis];
}

float *Tensor_Data(const Tensor *tensor)
{
    return (float *)tensor->storage->data;
}

void Tensor_ComputeStrides(Tensor *tensor)
{
    tensor->strides = malloc(sizeof(size_t) * tensor->ndim);

    tensor->strides[tensor->ndim - 1] = 1;

    for (int i = (int)tensor->ndim - 2; i >= 0; i--)
    {
        tensor->strides[i] =
            tensor->strides[i + 1] * tensor->shape[i + 1];
    }
}

Tensor Tensor_Create(size_t ndim, const size_t *shape)
{
    Tensor tensor;

    tensor.ndim = ndim;
    tensor.offset = 0;

    tensor.shape = malloc(sizeof(size_t) * ndim);
    memcpy(tensor.shape, shape, sizeof(size_t) * ndim);

    Tensor_ComputeStrides(&tensor);

    size_t numel = Tensor_Numel(&tensor);

    tensor.storage = Storage_Create(numel * sizeof(float));

    return tensor;
}

void Tensor_Destroy(Tensor *tensor)
{
    if (tensor == NULL)
        return;

    Storage_Release(tensor->storage);

    free(tensor->shape);
    free(tensor->strides);

    tensor->storage = NULL;
    tensor->shape = NULL;
    tensor->strides = NULL;

    tensor->ndim = 0;
    tensor->offset = 0;
}
