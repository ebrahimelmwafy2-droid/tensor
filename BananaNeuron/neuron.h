#ifndef NEURON_H
#define NEURON_H

#include <stddef.h>
#include "tensor.h"

typedef struct
{
    Tensor weights;
    float bias;
} Neuron;

/* Create / Destroy */
Neuron Neuron_Create(size_t input_size);
void Neuron_Destroy(Neuron *neuron);

/* Forward */
float Neuron_Forward(Neuron *neuron, const Tensor *input);

#endif