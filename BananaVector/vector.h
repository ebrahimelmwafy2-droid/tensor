#ifndef VECTOR_H
#define VECTOR_H

typedef struct
{
    int size;
    float *data;
} Vector;

Vector Vector_Create(int size);
void Vector_Destroy(Vector *v);

void Vector_Set(Vector *v, int index, float value);
float Vector_Get(Vector *v, int index);
float Vector_Dot(Vector A, Vector B);
#endif