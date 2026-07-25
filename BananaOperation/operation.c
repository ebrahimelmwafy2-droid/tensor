#include "operation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int Tensor_SameShape(const Tensor *a, const Tensor *b)
{
    if (a->ndim != b->ndim)
        return 0;

    for (size_t i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
            return 0;
    }

    return 1;
}

Tensor Tensor_Copy(const Tensor *tensor)
{
    Tensor copy = Tensor_Create(
        tensor->ndim,
        tensor->shape);

    float *src = Tensor_Data(tensor);
    float *dst = Tensor_Data(&copy);

    memcpy(
        dst,
        src,
        Tensor_Numel(tensor) * sizeof(float));

    return copy;
}

Tensor Tensor_Clone(const Tensor *tensor)
{
    return Tensor_Copy(tensor);
}

void Tensor_Fill(Tensor *tensor, float value)
{
    float *data = Tensor_Data(tensor);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
        data[i] = value;
}

void Tensor_Zero(Tensor *tensor)
{
    Tensor_Fill(tensor, 0.0f);
}

void Tensor_Ones(Tensor *tensor)
{
    Tensor_Fill(tensor, 1.0f);
}

void Tensor_Random(Tensor *tensor)
{
    static int initialized = 0;

    if (!initialized)
    {
        srand((unsigned)time(NULL));
        initialized = 1;
    }

    float *data = Tensor_Data(tensor);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
    {
        data[i] =
            ((float)rand() / RAND_MAX) - 0.5f;
    }
}

int Tensor_Equal(
    const Tensor *a,
    const Tensor *b)
{
    if (!Tensor_SameShape(a, b))
        return 0;

    size_t n = Tensor_Numel(a);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);

    for (size_t i = 0; i < n; i++)
    {
        if (pa[i] != pb[i])
            return 0;
    }

    return 1;
}

void Tensor_Print(const Tensor *tensor)
{
    float *data = Tensor_Data(tensor);

    size_t n = Tensor_Numel(tensor);

    printf("[ ");

    for (size_t i = 0; i < n; i++)
    {
        printf("%f ", data[i]);
    }

    printf("]\n");
}
Tensor Tensor_Add(
    const Tensor *a,
    const Tensor *b)
{
    if (!Tensor_SameShape(a, b))
    {
        fprintf(stderr, "Tensor_Add: shape mismatch\n");
        exit(EXIT_FAILURE);
    }

    Tensor result = Tensor_Create(
        a->ndim,
        a->shape);

    size_t n = Tensor_Numel(a);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);
    float *pr = Tensor_Data(&result);

    for (size_t i = 0; i < n; i++)
    {
        pr[i] = pa[i] + pb[i];
    }

    return result;
}

Tensor Tensor_Sub(
    const Tensor *a,
    const Tensor *b)
{
    if (!Tensor_SameShape(a, b))
    {
        fprintf(stderr, "Tensor_Sub: shape mismatch\n");
        exit(EXIT_FAILURE);
    }

    Tensor result = Tensor_Create(
        a->ndim,
        a->shape);

    size_t n = Tensor_Numel(a);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);
    float *pr = Tensor_Data(&result);

    for (size_t i = 0; i < n; i++)
    {
        pr[i] = pa[i] - pb[i];
    }

    return result;
}

Tensor Tensor_Mul(
    const Tensor *a,
    const Tensor *b)
{
    if (!Tensor_SameShape(a, b))
    {
        fprintf(stderr, "Tensor_Mul: shape mismatch\n");
        exit(EXIT_FAILURE);
    }

    Tensor result = Tensor_Create(
        a->ndim,
        a->shape);

    size_t n = Tensor_Numel(a);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);
    float *pr = Tensor_Data(&result);

    for (size_t i = 0; i < n; i++)
    {
        pr[i] = pa[i] * pb[i];
    }

    return result;
}

Tensor Tensor_Div(
    const Tensor *a,
    const Tensor *b)
{
    if (!Tensor_SameShape(a, b))
    {
        fprintf(stderr, "Tensor_Div: shape mismatch\n");
        exit(EXIT_FAILURE);
    }

    Tensor result = Tensor_Create(
        a->ndim,
        a->shape);

    size_t n = Tensor_Numel(a);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);
    float *pr = Tensor_Data(&result);

    for (size_t i = 0; i < n; i++)
    {
        pr[i] = pa[i] / pb[i];
    }

    return result;
}
Tensor Tensor_AddScalar(
    const Tensor *tensor,
    float scalar)
{
    Tensor result = Tensor_Copy(tensor);

    float *data = Tensor_Data(&result);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
    {
        data[i] += scalar;
    }

    return result;
}

Tensor Tensor_SubScalar(
    const Tensor *tensor,
    float scalar)
{
    Tensor result = Tensor_Copy(tensor);

    float *data = Tensor_Data(&result);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
    {
        data[i] -= scalar;
    }

    return result;
}

Tensor Tensor_MulScalar(
    const Tensor *tensor,
    float scalar)
{
    Tensor result = Tensor_Copy(tensor);

    float *data = Tensor_Data(&result);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
    {
        data[i] *= scalar;
    }

    return result;
}

Tensor Tensor_DivScalar(
    const Tensor *tensor,
    float scalar)
{
    Tensor result = Tensor_Copy(tensor);

    float *data = Tensor_Data(&result);

    size_t n = Tensor_Numel(tensor);

    for (size_t i = 0; i < n; i++)
    {
        data[i] /= scalar;
    }

    return result;
}
Tensor Tensor_Dot(
    const Tensor *a,
    const Tensor *b)
{
    if (Tensor_Numel(a) != Tensor_Numel(b))
    {
        fprintf(stderr, "Tensor_Dot: size mismatch\n");
        exit(EXIT_FAILURE);
    }

    size_t shape[1] = {1};

    Tensor result = Tensor_Create(1, shape);

    float *pa = Tensor_Data(a);
    float *pb = Tensor_Data(b);
    float *pr = Tensor_Data(&result);

    size_t n = Tensor_Numel(a);

    pr[0] = 0.0f;

    for (size_t i = 0; i < n; i++)
    {
        pr[0] += pa[i] * pb[i];
    }

    return result;
}
Tensor Tensor_Transpose(
    const Tensor *tensor)
{
    if (tensor->ndim != 2)
    {
        fprintf(stderr,
                "Tensor_Transpose: requires 2D tensor\n");
        exit(EXIT_FAILURE);
    }

    size_t shape[2];

    shape[0] = tensor->shape[1];
    shape[1] = tensor->shape[0];

    Tensor result =
        Tensor_Create(2, shape);

    float *src = Tensor_Data(tensor);
    float *dst = Tensor_Data(&result);

    size_t rows = tensor->shape[0];
    size_t cols = tensor->shape[1];

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            dst[j * rows + i] =
                src[i * cols + j];
        }
    }

    return result;
}
Tensor Tensor_MatMul(
    const Tensor *a,
    const Tensor *b)
{
    if (a->ndim != 2 ||
        b->ndim != 2)
    {
        fprintf(stderr,
                "Tensor_MatMul: requires matrices\n");
        exit(EXIT_FAILURE);
    }

    size_t m = a->shape[0];
    size_t n = a->shape[1];
    size_t p = b->shape[1];

    if (n != b->shape[0])
    {
        fprintf(stderr,
                "Tensor_MatMul: shape mismatch\n");
        exit(EXIT_FAILURE);
    }

    size_t shape[2] = {m, p};

    Tensor result =
        Tensor_Create(2, shape);

    float *A = Tensor_Data(a);
    float *B = Tensor_Data(b);
    float *C = Tensor_Data(&result);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            float sum = 0.0f;

            for (size_t k = 0; k < n; k++)
            {
                sum +=
                    A[i * n + k] *
                    B[k * p + j];
            }

            C[i * p + j] = sum;
        }
    }

    return result;
}
Tensor Tensor_Reshape(
    const Tensor *tensor,
    size_t ndim,
    const size_t *shape)
{
    size_t old_numel =
        Tensor_Numel(tensor);

    size_t new_numel = 1;

    for (size_t i = 0; i < ndim; i++)
    {
        new_numel *= shape[i];
    }

    if (old_numel != new_numel)
    {
        fprintf(stderr,
                "Tensor_Reshape: invalid shape\n");
        exit(EXIT_FAILURE);
    }

    Tensor result =
        Tensor_Create(ndim, shape);

    memcpy(
        Tensor_Data(&result),
        Tensor_Data(tensor),
        old_numel * sizeof(float));

    return result;
}
Tensor Tensor_View(
    const Tensor *tensor,
    size_t offset,
    size_t ndim,
    const size_t *shape)
{
    Tensor view;

    view.storage = tensor->storage;

    Storage_Retain(view.storage);

    view.ndim = ndim;

    view.offset = offset;

    view.shape =
        malloc(ndim * sizeof(size_t));

    view.strides =
        malloc(ndim * sizeof(size_t));

    memcpy(
        view.shape,
        shape,
        ndim * sizeof(size_t));

    Tensor_ComputeStrides(&view);

    return view;
}Tensor Tensor_Slice(
    const Tensor *tensor,
    size_t start,
    size_t length)
{
    if (tensor->ndim != 1)
    {
        fprintf(stderr, "Tensor_Slice: only 1D tensors supported\n");
        exit(EXIT_FAILURE);
    }

    if (start + length > tensor->shape[0])
    {
        fprintf(stderr, "Tensor_Slice: out of bounds\n");
        exit(EXIT_FAILURE);
    }

    size_t shape[1] = { length };

    Tensor slice;

    slice.storage = tensor->storage;
    Storage_Retain(slice.storage);

    slice.ndim = 1;
    slice.offset = tensor->offset + start;

    slice.shape = malloc(sizeof(size_t));
    slice.strides = malloc(sizeof(size_t));

    slice.shape[0] = length;
    slice.strides[0] = 1;

    return slice;
}

Tensor Tensor_Squeeze(const Tensor *tensor)
{
    size_t ndim = 0;

    for (size_t i = 0; i < tensor->ndim; i++)
    {
        if (tensor->shape[i] != 1)
            ndim++;
    }

    if (ndim == 0)
        ndim = 1;

    size_t *shape = malloc(ndim * sizeof(size_t));

    size_t j = 0;

    for (size_t i = 0; i < tensor->ndim; i++)
    {
        if (tensor->shape[i] != 1)
            shape[j++] = tensor->shape[i];
    }

    if (j == 0)
        shape[0] = 1;

    Tensor result = Tensor_View(
        tensor,
        tensor->offset,
        ndim,
        shape);

    free(shape);

    return result;
}

Tensor Tensor_Unsqueeze(
    const Tensor *tensor,
    size_t axis)
{
    if (axis > tensor->ndim)
    {
        fprintf(stderr, "Tensor_Unsqueeze: invalid axis\n");
        exit(EXIT_FAILURE);
    }

    size_t ndim = tensor->ndim + 1;

    size_t *shape = malloc(ndim * sizeof(size_t));

    size_t j = 0;

    for (size_t i = 0; i < ndim; i++)
    {
        if (i == axis)
            shape[i] = 1;
        else
            shape[i] = tensor->shape[j++];
    }

    Tensor result =
        Tensor_View(
            tensor,
            tensor->offset,
            ndim,
            shape);

    free(shape);

    return result;
}