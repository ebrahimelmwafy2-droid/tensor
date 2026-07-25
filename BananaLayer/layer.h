#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>
#include "neuron.h"

typedef struct
{
    Neuron *neurons;

    size_t input_size;
    size_t output_size;

} DenseLayer;

/* Create / Destroy */

DenseLayer DenseLayer_Create(size_t input_size, size_t output_size);

void DenseLayer_Destroy(DenseLayer *layer);

/* Forward */

Tensor DenseLayer_Forward(DenseLayer *layer, const Tensor *input);

#endif