#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    int rows;
    int cols;
    float *data;
} Matrix;

Matrix Matrix_Create(int rows, int cols);
void Matrix_Destroy(Matrix *m);
void Matrix_Set(Matrix *m, int row, int col, float value
);
float Matrix_Get(Matrix *m, int row, int col);
Matrix Matrix_Add(Matrix A, Matrix B);
Matrix Matrix_Subtract(Matrix A, Matrix B);
Matrix Matrix_Multiply(Matrix A, Matrix B);
#endif