#include "activation.h"
#include <math.h>

float ReLU(float x)
{
    return x > 0 ? x : 0;
}

float Sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float Tanh(float x)
{
    return tanhf(x);
}