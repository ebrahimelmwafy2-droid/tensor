#include "broadcast.h"
#include "memory.h" 
bool Broadcast_CanBroadcast(
    const Shape* a,
    const Shape* b)
{
    size_t i = a->ndim;
    size_t j = b->ndim;

    while (i > 0 || j > 0)
    {
        size_t da = 1;
        size_t db = 1;

        if (i > 0)
            da = a->dims[--i];

        if (j > 0)
            db = b->dims[--j];

        if (da != db && da != 1 && db != 1)
            return false;
    }

    return true;
}
Shape Broadcast_Shape(
    const Shape* a,
    const Shape* b)
{
    size_t ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;

    size_t* dims = B_Malloc(ndim * sizeof(size_t));

    size_t ia = a->ndim;
    size_t ib = b->ndim;

    for (size_t k = ndim; k > 0; --k)
    {
        size_t da = 1;
        size_t db = 1;

        if (ia > 0)
            da = a->dims[--ia];

        if (ib > 0)
            db = b->dims[--ib];

        dims[k - 1] = (da > db) ? da : db;
    }

    Shape result = Shape_Create(ndim, dims);

    B_Free(dims);

    return result;
}
Strides Broadcast_Strides(
    const Shape* shape,
    const Strides* strides,
    const Shape* target)
{
    Strides result = Strides_Create(target->ndim);

    size_t is = shape->ndim;
    size_t it = target->ndim;

    while (it > 0)
    {
        --it;

        if (is == 0)
        {
            result.values[it] = 0;
            continue;
        }

        --is;

        if (shape->dims[is] == 1)
        {
            result.values[it] = 0;
        }
        else
        {
            result.values[it] = strides->values[is];
        }
    }

    return result;
}
