#include "arithmetic.h"

#include <stdint.h>
#include <stddef.h>
typedef double (*BinaryFunc)(double, double);

static double AddOp(double a, double b)
{
    return a + b;
}

static double SubOp(double a, double b)
{
    return a - b;
}

static double MulOp(double a, double b)
{
    return a * b;
}

static double DivOp(double a, double b)
{
    return a / b;
}

static Tensor* Tensor_BinaryOp(
    const Tensor* a,
    const Tensor* b,
    BinaryFunc op)
{
    if (a == NULL || b == NULL)
        return NULL;

    if (a->dtype != b->dtype)
        return NULL;

    if (a->shape.ndim != b->shape.ndim)
        return NULL;

    for (size_t i = 0; i < a->shape.ndim; ++i)
    {
        if (a->shape.dims[i] != b->shape.dims[i])
            return NULL;
    }

    Tensor* result = Tensor_Create(
        a->shape.ndim,
        a->shape.dims,
        a->dtype,
        a->requires_grad || b->requires_grad);

    if (result == NULL)
        return NULL;

    size_t n = Tensor_Numel(a);

    void* pa = Tensor_Data(a);
    void* pb = Tensor_Data(b);
    void* pr = Tensor_Data(result);

   switch (a->dtype)
{
    case B_FLOAT32:
    {
        float* x = (float*)pa;
        float* y = (float*)pb;
        float* z = (float*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (float)op(x[i], y[i]);

        break;
    }

    case B_FLOAT64:
    {
        double* x = (double*)pa;
        double* y = (double*)pb;
        double* z = (double*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = op(x[i], y[i]);

        break;
    }

    case B_INT8:
    {
        int8_t* x = (int8_t*)pa;
        int8_t* y = (int8_t*)pb;
        int8_t* z = (int8_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (int8_t)op(x[i], y[i]);

        break;
    }

    case B_UINT8:
    {
        uint8_t* x = (uint8_t*)pa;
        uint8_t* y = (uint8_t*)pb;
        uint8_t* z = (uint8_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (uint8_t)op(x[i], y[i]);

        break;
    }

    case B_INT16:
    {
        int16_t* x = (int16_t*)pa;
        int16_t* y = (int16_t*)pb;
        int16_t* z = (int16_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (int16_t)op(x[i], y[i]);

        break;
    }

    case B_UINT16:
    {
        uint16_t* x = (uint16_t*)pa;
        uint16_t* y = (uint16_t*)pb;
        uint16_t* z = (uint16_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (uint16_t)op(x[i], y[i]);

        break;
    }

    case B_INT32:
    {
        int32_t* x = (int32_t*)pa;
        int32_t* y = (int32_t*)pb;
        int32_t* z = (int32_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (int32_t)op(x[i], y[i]);

        break;
    }

    case B_UINT32:
    {
        uint32_t* x = (uint32_t*)pa;
        uint32_t* y = (uint32_t*)pb;
        uint32_t* z = (uint32_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (uint32_t)op(x[i], y[i]);

        break;
    }

    case B_INT64:
    {
        int64_t* x = (int64_t*)pa;
        int64_t* y = (int64_t*)pb;
        int64_t* z = (int64_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (int64_t)op(x[i], y[i]);

        break;
    }

    case B_UINT64:
    {
        uint64_t* x = (uint64_t*)pa;
        uint64_t* y = (uint64_t*)pb;
        uint64_t* z = (uint64_t*)pr;

        for (size_t i = 0; i < n; ++i)
            z[i] = (uint64_t)op(x[i], y[i]);

        break;
    }

    default:
    {
        Tensor_Destroy(result);
        return NULL;
    }
}

    return result;
}

static Tensor* Tensor_ScalarOp(
    const Tensor* tensor,
    double scalar,
    BinaryFunc op)
{
    if (tensor == NULL)
        return NULL;

    Tensor* result = Tensor_Create(
        tensor->shape.ndim,
        tensor->shape.dims,
        tensor->dtype,
        tensor->requires_grad);

    if (result == NULL)
        return NULL;

    size_t n = Tensor_Numel(tensor);

    void* pt = Tensor_Data(tensor);
    void* pr = Tensor_Data(result);

    switch (tensor->dtype)
    {
        case B_FLOAT32:
        {
            float* x = (float*)pt;
            float* z = (float*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (float)op(x[i], scalar);

            break;
        }

        case B_FLOAT64:
        {
            double* x = (double*)pt;
            double* z = (double*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = op(x[i], scalar);

            break;
        }

        case B_INT8:
        {
            int8_t* x = (int8_t*)pt;
            int8_t* z = (int8_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (int8_t)op(x[i], scalar);

            break;
        }

        case B_UINT8:
        {
            uint8_t* x = (uint8_t*)pt;
            uint8_t* z = (uint8_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (uint8_t)op(x[i], scalar);

            break;
        }

        case B_INT16:
        {
            int16_t* x = (int16_t*)pt;
            int16_t* z = (int16_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (int16_t)op(x[i], scalar);

            break;
        }

        case B_UINT16:
        {
            uint16_t* x = (uint16_t*)pt;
            uint16_t* z = (uint16_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (uint16_t)op(x[i], scalar);

            break;
        }

        case B_INT32:
        {
            int32_t* x = (int32_t*)pt;
            int32_t* z = (int32_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (int32_t)op(x[i], scalar);

            break;
        }

        case B_UINT32:
        {
            uint32_t* x = (uint32_t*)pt;
            uint32_t* z = (uint32_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (uint32_t)op(x[i], scalar);

            break;
        }

        case B_INT64:
        {
            int64_t* x = (int64_t*)pt;
            int64_t* z = (int64_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (int64_t)op(x[i], scalar);

            break;
        }

        case B_UINT64:
        {
            uint64_t* x = (uint64_t*)pt;
            uint64_t* z = (uint64_t*)pr;

            for (size_t i = 0; i < n; ++i)
                z[i] = (uint64_t)op(x[i], scalar);

            break;
        }

        default:
        {
            Tensor_Destroy(result);
            return NULL;
        }
    }

    return result;
}
Tensor* Tensor_Add(
    const Tensor* a,
    const Tensor* b)
{
    return Tensor_BinaryOp(a, b, AddOp);
}

Tensor* Tensor_Sub(
    const Tensor* a,
    const Tensor* b)
{
    return Tensor_BinaryOp(a, b, SubOp);
}

Tensor* Tensor_Mul(
    const Tensor* a,
    const Tensor* b)
{
    return Tensor_BinaryOp(a, b, MulOp);
}

Tensor* Tensor_Div(
    const Tensor* a,
    const Tensor* b)
{
    return Tensor_BinaryOp(a, b, DivOp);
}

Tensor* Tensor_AddScalar(
    const Tensor* tensor,
    double scalar)
{
    return Tensor_ScalarOp(tensor, scalar, AddOp);
}

Tensor* Tensor_SubScalar(
    const Tensor* tensor,
    double scalar)
{
    return Tensor_ScalarOp(tensor, scalar, SubOp);
}

Tensor* Tensor_MulScalar(
    const Tensor* tensor,
    double scalar)
{
    return Tensor_ScalarOp(tensor, scalar, MulOp);
}

Tensor* Tensor_DivScalar(
    const Tensor* tensor,
    double scalar)
{
    return Tensor_ScalarOp(tensor, scalar, DivOp);
}