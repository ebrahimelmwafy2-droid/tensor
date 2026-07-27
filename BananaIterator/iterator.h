#include "iterator.h"

#include "memory.h"

TensorIterator*
Iterator_Create(
    const Tensor* tensor)
{
    if (tensor == NULL)
        return NULL;

    TensorIterator* iterator =
        B_Malloc(sizeof(TensorIterator));

    if (iterator == NULL)
        return NULL;

    iterator->tensor = tensor;

    iterator->linear = 0;

    iterator->total = Tensor_Numel(tensor);

    iterator->index =
        B_Calloc(
            tensor->ndim,
            sizeof(size_t));

    if (iterator->index == NULL)
    {
        B_Free(iterator);

        return NULL;
    }

    return iterator;
}
void
Iterator_Destroy(
    TensorIterator* iterator)
{
    if (iterator == NULL)
        return;

    B_Free(iterator->index);

    B_Free(iterator);
}
bool
Iterator_Next(
    TensorIterator* iterator)
{
    if (iterator == NULL)
        return false;

    if (iterator->linear >= iterator->total)
        return false;

    if (iterator->linear > 0)
    {
        size_t dim = iterator->tensor->ndim;

        while (dim--)
        {
            iterator->index[dim]++;

            if (iterator->index[dim] <
                iterator->tensor->shape[dim])
            {
                break;
            }

            iterator->index[dim] = 0;
        }
    }

    iterator->linear++;

    return true;
}
void*
Iterator_Data(
    const TensorIterator* iterator)
{
    if (iterator == NULL)
        return NULL;

    const Tensor* tensor = iterator->tensor;

    size_t offset = tensor->offset;

    for (size_t i = 0; i < tensor->shape.ndim; i++)
    {
        offset +=
            iterator->index[i] *
            tensor->strides.values[i];
    }

    return (char*)tensor->storage->data +
           offset * Tensor_ElementSize(tensor->dtype);
}