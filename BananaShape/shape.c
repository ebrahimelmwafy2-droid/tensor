#include "shape.h"

#include <stdlib.h>
#include <string.h>

/*==========================
        Shape
==========================*/

Shape Shape_Create(size_t ndim, const size_t* dims)
{
    Shape shape;

    shape.ndim = ndim;
    shape.dims = NULL;

    if (ndim == 0)
        return shape;

    if (dims == NULL)
    {
        shape.ndim = 0;
        return shape;
    }

    shape.dims = (size_t*)malloc(sizeof(size_t) * ndim);

    if (shape.dims == NULL)
    {
        shape.ndim = 0;
        return shape;
    }

    memcpy(shape.dims, dims, sizeof(size_t) * ndim);

    return shape;
}

void Shape_Destroy(Shape* shape)
{
    if (shape == NULL)
        return;

    free(shape->dims);

    shape->dims = NULL;
    shape->ndim = 0;
}

/*==========================
        Strides
==========================*/

Strides Strides_Create(size_t ndim)
{
    Strides strides;

    strides.ndim = ndim;
    strides.values = NULL;

    if (ndim == 0)
        return strides;

    strides.values = (size_t*)malloc(sizeof(size_t) * ndim);

    if (strides.values == NULL)
    {
        strides.ndim = 0;
    }

    return strides;
}

void Strides_Destroy(Strides* strides)
{
    if (strides == NULL)
        return;

    free(strides->values);

    strides->values = NULL;
    strides->ndim = 0;
}

void Strides_Compute(Strides* strides, const Shape* shape)
{
    if (strides == NULL || shape == NULL)
        return;

    if (strides->values == NULL)
        return;

    if (strides->ndim != shape->ndim)
        return;

    if (shape->ndim == 0)
        return;

    strides->values[shape->ndim - 1] = 1;

    for (int i = (int)shape->ndim - 2; i >= 0; --i)
    {
        strides->values[i] =
            strides->values[i + 1] * shape->dims[i + 1];
    }
}