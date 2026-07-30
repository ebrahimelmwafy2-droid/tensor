#ifndef BROADCAST_H
#define BROADCAST_H

#include <stddef.h>
#include <stdbool.h>
#include "shape.h"

bool Broadcast_CanBroadcast(
    const Shape* a,
    const Shape* b);

Shape Broadcast_Shape(
    const Shape* a,
    const Shape* b);
Strides Broadcast_Strides(
    const Shape* shape,
    const Strides* strides,
    const Shape* target);
#endif