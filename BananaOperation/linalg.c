#include "linalg.h"
#include "broadcast.h"
#include "shape.h"
#include "utility.h"
static Shape GetBatchShape(
    const Tensor* tensor)
{
    if (tensor->shape.ndim <= 2)
    {
        return Shape_Create(0, NULL);
    }

    return Shape_Create(
        tensor->shape.ndim - 2,
        tensor->shape.dims);
}
static int CheckMatMul(
    const Tensor* a,
    const Tensor* b)
{
    if (a == NULL || b == NULL)
        return 0;

    if (a->dtype != b->dtype)
        return 0;

    if (a->shape.ndim < 2)
        return 0;

    if (b->shape.ndim < 2)
        return 0;

    size_t k1 = a->shape.dims[a->shape.ndim - 1];
    size_t k2 = b->shape.dims[b->shape.ndim - 2];

    if (k1 != k2)
        return 0;

    Shape batchA = GetBatchShape(a);
    Shape batchB = GetBatchShape(b);

    int ok = Broadcast_CanBroadcast(
        &batchA,
        &batchB);

    Shape_Destroy(&batchA);
    Shape_Destroy(&batchB);

    return ok;
}
static Shape MatMulOutputShape(
    const Tensor* a,
    const Tensor* b)
{
    Shape batchA = GetBatchShape(a);
    Shape batchB = GetBatchShape(b);

    Shape batch = Broadcast_Shape(
        &batchA,
        &batchB);

    Shape output = Shape_Create(batch.ndim + 2, NULL);

    if (output.ndim == 0)
    {
        Shape_Destroy(&batchA);
        Shape_Destroy(&batchB);
        Shape_Destroy(&batch);

        return output;
    }

    for (size_t i = 0; i < batch.ndim; ++i)
    {
        output.dims[i] = batch.dims[i];
    }

    output.dims[batch.ndim] =
        a->shape.dims[a->shape.ndim - 2];

    output.dims[batch.ndim + 1] =
        b->shape.dims[b->shape.ndim - 1];

    Shape_Destroy(&batchA);
    Shape_Destroy(&batchB);
    Shape_Destroy(&batch);

    return output;
}
static void MatMulKernel(
    const Tensor* a,
    const Strides* a_view,
    const Tensor* b,
    const Strides* b_view,
    Tensor* out,
    const Strides* out_view);

void MatMulKernel(const Tensor *a, const Strides *a_view, const Tensor *b, const Strides *b_view, Tensor *out, const Strides *out_view)
{
}
