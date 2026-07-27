#include "tensor.h"
#include "iterator.h"
#include "memory.h"

/*==========================
        Utilities
==========================*/

size_t
Tensor_ElementSize(B_DataType type)
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

/*==========================
        Information
==========================*/

size_t
Tensor_Numel(const Tensor* tensor)
{
    size_t numel = 1;

    for (size_t i = 0; i < tensor->shape.ndim; i++)
    {
        numel *= tensor->shape.dims[i];
    }

    return numel;
}

size_t
Tensor_Rank(const Tensor* tensor)
{
    return tensor->shape.ndim;
}

size_t
Tensor_Dimension(
    const Tensor* tensor,
    size_t axis)
{
    if (axis >= tensor->shape.ndim)
        return 0;

    return tensor->shape.dims[axis];
}

/*==========================
      Memory Access
==========================*/

void*
Tensor_Data(
    const Tensor* tensor)
{
    return (char*)tensor->storage->data +
           tensor->offset *
           Tensor_ElementSize(tensor->dtype);
}

/*==========================
      Internal Helpers
==========================*/

void
Tensor_ComputeStrides(
    Tensor* tensor)
{
    Strides_Compute(
        &tensor->strides,
        &tensor->shape);
}

/*==========================
    Creation / Destruction
==========================*/

Tensor*
Tensor_Create(
    size_t ndim,
    const size_t* shape,
    B_DataType dtype,
    bool requires_grad)
{
    if (ndim > 0 && shape == NULL)
        return NULL;

    Tensor* tensor =
        B_Malloc(sizeof(Tensor));

    if (tensor == NULL)
        return NULL;

    tensor->shape = Shape_Create(ndim, shape);
    tensor->strides = Strides_Create(ndim);

    if (ndim > 0 &&
        (tensor->shape.dims == NULL ||
         tensor->strides.values == NULL))
    {
        Shape_Destroy(&tensor->shape);
        Strides_Destroy(&tensor->strides);
        B_Free(tensor);
        return NULL;
    }

    tensor->offset = 0;
    tensor->dtype = dtype;
    tensor->requires_grad = requires_grad;

    Tensor_ComputeStrides(tensor);

    tensor->storage =
        Storage_Create(
            Tensor_Numel(tensor) *
            Tensor_ElementSize(dtype));

    if (tensor->storage == NULL)
    {
        Shape_Destroy(&tensor->shape);
        Strides_Destroy(&tensor->strides);
        B_Free(tensor);
        return NULL;
    }

    return tensor;
}

Tensor*
Tensor_CreateScalar(
    B_DataType dtype)
{
    return Tensor_Create(
        0,
        NULL,
        dtype,
        false);
}

void
Tensor_Destroy(
    Tensor* tensor)
{
    if (tensor == NULL)
        return;

    Storage_Release(tensor->storage);

    Shape_Destroy(&tensor->shape);
    Strides_Destroy(&tensor->strides);

    B_Free(tensor);
}

/*==========================
        Reductions
==========================*/

Tensor*
Tensor_Sum(
    const Tensor* tensor)
{
    if (tensor == NULL)
        return NULL;

    Tensor* result =
        Tensor_CreateScalar(
            tensor->dtype);

    if (result == NULL)
        return NULL;

    TensorIterator* it =
        Iterator_Create(tensor);

    if (it == NULL)
    {
        Tensor_Destroy(result);
        return NULL;
    }

    switch (tensor->dtype)
    {
        case B_FLOAT32:
        {
            double sum = 0.0;

            while (Iterator_Next(it))
            {
                sum += *(float*)Iterator_Data(it);
            }

            *(float*)Tensor_Data(result) = (float)sum;
            break;
        }

        case B_FLOAT64:
        {
            double sum = 0.0;

            while (Iterator_Next(it))
            {
                sum += *(double*)Iterator_Data(it);
            }

            *(double*)Tensor_Data(result) = sum;
            break;
        }

        default:
        {
            Iterator_Destroy(it);
            Tensor_Destroy(result);
            return NULL;
        }
    }

    Iterator_Destroy(it);

    return result;
}