#include "vector.h"
#include <stdlib.h>

Vector Vector_Create(int size)
{
    Vector V;

    V.size = size;
    V.data = (float *)malloc(size * sizeof(float));

    return V;
}

void Vector_Destroy(Vector *v)
{
    free(v->data);

    v->data = NULL;
    v->size = 0;
}

void Vector_Set(Vector *v, int index, float value)
{
    v->data[index] = value;
}

float Vector_Get(Vector *v, int index)
{
    return v->data[index];
}
float Vector_Dot(Vector A, Vector B)
{
    float sum = 0.0f;

    for (int i = 0; i < A.size; i++)
    {
        sum += A.data[i] * B.data[i];
    }

    return sum;
}