#include "layer.h"

#include <stdlib.h>

DenseLayer DenseLayer_Create(size_t input_size, size_t output_size)
{
    DenseLayer layer;

    layer.input_size = input_size;
    layer.output_size = output_size;

    layer.neurons = malloc(sizeof(Neuron) * output_size);

    if (layer.neurons == NULL)
    {
        layer.input_size = 0;
        layer.output_size = 0;
        return layer;
    }

    for (size_t i = 0; i < output_size; i++)
    {
        layer.neurons[i] = Neuron_Create(input_size);
    }

    return layer;
}

void DenseLayer_Destroy(DenseLayer *layer)
{
    if (layer == NULL)
        return;

    for (size_t i = 0; i < layer->output_size; i++)
    {
        Neuron_Destroy(&layer->neurons[i]);
    }

    free(layer->neurons);

    layer->neurons = NULL;
    layer->input_size = 0;
    layer->output_size = 0;
}