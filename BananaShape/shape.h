#ifndef SHAPE_H
#define SHAPE_H

#include <stddef.h>

/*==========================
            Shape
==========================*/

typedef struct
{
    size_t ndim;
    size_t* dims;
} Shape;

/*==========================
           Strides
==========================*/

typedef struct
{
    size_t ndim;
    size_t* values;
} Strides;

/*==========================
      Creation / Destroy
==========================*/

Shape Shape_Create(size_t ndim, const size_t* dims);

void Shape_Destroy(Shape* shape);

Strides Strides_Create(size_t ndim);

void Strides_Destroy(Strides* strides);
void Strides_Compute(Strides* strides, const Shape* shape);
#endif