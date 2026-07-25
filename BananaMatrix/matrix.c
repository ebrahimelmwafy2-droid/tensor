#include "matrix.h"
#include <stdlib.h>

Matrix Matrix_Create(int rows, int cols)
{
    Matrix A;

    A.rows = rows;
    A.cols = cols;
    A.data = (float *)malloc(rows * cols * sizeof(float));

    return A;
}

void Matrix_Set(Matrix *m, int row, int col, float value)
{
    m->data[row * m->cols + col] = value;
}

float Matrix_Get(Matrix *m, int row, int col)
{
    return m->data[row * m->cols + col];
}

void Matrix_Destroy(Matrix *m)
{
    free(m->data);

    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}
Matrix Matrix_Add(Matrix A, Matrix B)
{
    Matrix C = Matrix_Create(A.rows, A.cols);

    int size = A.rows * A.cols;

    for (int i = 0; i < size; i++)
    {
        C.data[i] = A.data[i] + B.data[i];
    }

    return C;
}
Matrix Matrix_Subtract(Matrix A, Matrix B)
{
    Matrix C = Matrix_Create(A.rows, A.cols);

    int size = A.rows * A.cols;

    for (int i = 0; i < size; i++)
    {
        C.data[i] = A.data[i] - B.data[i];
    }

    return C;
}
Matrix Matrix_Multiply(Matrix A, Matrix B)
{
    Matrix C = Matrix_Create(A.rows, B.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < B.cols; j++)
        {
            float sum = 0;

            for (int k = 0; k < A.cols; k++)
            {
                sum += Matrix_Get(&A, i, k) * Matrix_Get(&B, k, j);
            }

            Matrix_Set(&C, i, j, sum);
        }
    }

    return C;
}