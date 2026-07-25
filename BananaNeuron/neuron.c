#include "neuron.h"

#include <stdlib.h>

Neuron Neuron_Create(size_t input_size)
{
    Neuron neuron;

    size_t shape[1] = { input_size };

    neuron.weights = Tensor_Create(1, shape);

    float *weights = Tensor_Data(&neuron.weights);

    for (size_t i = 0; i < input_size; i++)
    {
        weights[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }

    neuron.bias = 0.0f;

    return neuron;
}

void Neuron_Destroy(Neuron *neuron)
{
    if (neuron == NULL)
        return;

    Tensor_Destroy(&neuron->weights);

    neuron->bias = 0.0f;
}