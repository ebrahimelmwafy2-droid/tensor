#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
Tensor* Tensor_Copy(const Tensor* tensor)
{
    if (tensor == NULL)
        return NULL;

    Tensor* copy = Tensor_Create(
        tensor->shape.ndim,
        tensor->shape.dims,
        tensor->dtype,
        tensor->requires_grad);

    if (copy == NULL)
        return NULL;

  size_t bytes =
    Tensor_Numel(tensor) *
    Tensor_ElementSize(tensor->dtype);

memcpy(
    Tensor_Data(copy),
    Tensor_Data(tensor),
    bytes);

    return copy;
}
void Tensor_Fill(
    Tensor* tensor,
    float value)
{
    if (tensor == NULL)
        return;

    size_t n = Tensor_Numel(tensor);

    switch (tensor->dtype)
    {
        case B_FLOAT32:
        {
         void* ptr = Tensor_Data(tensor);
         float* data = (float*)ptr;
            for (size_t i = 0; i < n; ++i)
                data[i] = value;

            break;
        }

        case B_FLOAT64:
        {
            double* data = (double*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (double)value;

            break;
        }

        case B_INT8:
        {
            int8_t* data = (int8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int8_t)value;

            break;
        }

        case B_UINT8:
        {
            uint8_t* data = (uint8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint8_t)value;

            break;
        }

        case B_INT16:
        {
            int16_t* data = (int16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int16_t)value;

            break;
        }

        case B_UINT16:
        {
            uint16_t* data = (uint16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint16_t)value;

            break;
        }

        case B_INT32:
        {
            int32_t* data = (int32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int32_t)value;

            break;
        }

        case B_UINT32:
        {
            uint32_t* data = (uint32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint32_t)value;

            break;
        }

        case B_INT64:
        {
            int64_t* data = (int64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int64_t)value;

            break;
        }

        case B_UINT64:
        {
            uint64_t* data = (uint64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint64_t)value;

            break;
        }

        default:
            break;
    }
}
void Tensor_Zero(Tensor* tensor)
{
    Tensor_Fill(tensor, 0.0f);
}

void Tensor_Ones(Tensor* tensor)
{
    Tensor_Fill(tensor, 1.0f);
}
void Tensor_Random(Tensor* tensor)
{
    if (tensor == NULL)
        return;

    static int initialized = 0;

    if (!initialized)
    {
        srand((unsigned)time(NULL));
        initialized = 1;
    }

    size_t n = Tensor_Numel(tensor);

    switch (tensor->dtype)
    {
        case B_FLOAT32:
        {
void* ptr = Tensor_Data(tensor);
float* data = (float*)ptr;
            for (size_t i = 0; i < n; ++i)
                data[i] = (float)rand() / RAND_MAX;

            break;
        }

        case B_FLOAT64:
        {
            double* data = (double*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (double)rand() / RAND_MAX;

            break;
        }

        case B_INT8:
        {
            int8_t* data = (int8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int8_t)rand();

            break;
        }

        case B_UINT8:
        {
            uint8_t* data = (uint8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint8_t)rand();

            break;
        }

        case B_INT16:
        {
            int16_t* data = (int16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int16_t)rand();

            break;
        }

        case B_UINT16:
        {
            uint16_t* data = (uint16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint16_t)rand();

            break;
        }

        case B_INT32:
        {
            int32_t* data = (int32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int32_t)rand();

            break;
        }

        case B_UINT32:
        {
            uint32_t* data = (uint32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint32_t)rand();

            break;
        }

        case B_INT64:
        {
            int64_t* data = (int64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (int64_t)rand();

            break;
        }

        case B_UINT64:
        {
            uint64_t* data = (uint64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                data[i] = (uint64_t)rand();

            break;
        }

        default:
            return;
    }
}
Tensor* Tensor_Clone(const Tensor* tensor)
{
    return Tensor_Copy(tensor);
}
int Tensor_Equal(
    const Tensor* a,
    const Tensor* b)
{
    if (a == NULL || b == NULL)
        return 0;

    if (a->dtype != b->dtype)
        return 0;

    if (a->shape.ndim != b->shape.ndim)
        return 0;

    for (size_t i = 0; i < a->shape.ndim; ++i)
    {
        if (a->shape.dims[i] != b->shape.dims[i])
            return 0;
    }

    size_t bytes =
        Tensor_Numel(a) *
        Tensor_ElementSize(a->dtype);

    return memcmp(
        Tensor_Data(a),
        Tensor_Data(b),
        bytes) == 0;
}
void Tensor_Print(const Tensor* tensor)
{
    if (tensor == NULL)
    {
        printf("Tensor(NULL)\n");
        return;
    }

    size_t n = Tensor_Numel(tensor);

    printf("[ ");

    switch (tensor->dtype)
    {
        case B_FLOAT32:
        {
            float* data = (float*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%f ", data[i]);

            break;
        }

        case B_FLOAT64:
        {
            double* data = (double*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%lf ", data[i]);

            break;
        }

        case B_INT8:
        {
            int8_t* data = (int8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%d ", data[i]);

            break;
        }

        case B_UINT8:
        {
            uint8_t* data = (uint8_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%u ", data[i]);

            break;
        }

        case B_INT16:
        {
            int16_t* data = (int16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%d ", data[i]);

            break;
        }

        case B_UINT16:
        {
            uint16_t* data = (uint16_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%u ", data[i]);

            break;
        }

        case B_INT32:
        {
            int32_t* data = (int32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%d ", data[i]);

            break;
        }

        case B_UINT32:
        {
            uint32_t* data = (uint32_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%u ", data[i]);

            break;
        }

        case B_INT64:
        {
            int64_t* data = (int64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%lld ", (long long)data[i]);

            break;
        }

        case B_UINT64:
        {
            uint64_t* data = (uint64_t*)Tensor_Data(tensor);

            for (size_t i = 0; i < n; ++i)
                printf("%llu ", (unsigned long long)data[i]);

            break;
        }

        default:
            printf("Unsupported dtype");
            break;
    }

    printf("]\n");
}