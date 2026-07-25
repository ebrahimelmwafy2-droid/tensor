#include <stdio.h>

#include "tensor.h"
#include "operation.h"

int main(void)
{
    printf("=====================================\n");
    printf("        BananaAI Demo v0.1\n");
    printf("=====================================\n\n");

    size_t shape2d[2] = {2, 2};

    Tensor A = Tensor_Create(2, shape2d);
    Tensor B = Tensor_Create(2, shape2d);

    float *a = Tensor_Data(&A);
    float *b = Tensor_Data(&B);

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;

    b[0] = 5;
    b[1] = 6;
    b[2] = 7;
    b[3] = 8;

    printf("Tensor A\n");
    Tensor_Print(&A);

    printf("\nTensor B\n");
    Tensor_Print(&B);

    printf("\n----------------------------\n");
    printf("Addition\n");

    Tensor C = Tensor_Add(&A, &B);
    Tensor_Print(&C);

    printf("\nSubtraction\n");

    Tensor D = Tensor_Sub(&A, &B);
    Tensor_Print(&D);

    printf("\nMultiplication\n");

    Tensor E = Tensor_Mul(&A, &B);
    Tensor_Print(&E);

    printf("\nDivision\n");

    Tensor F = Tensor_Div(&B, &A);
    Tensor_Print(&F);

    printf("\n----------------------------\n");
    printf("Scalar Operations\n");

    Tensor G = Tensor_AddScalar(&A, 10.0f);
    Tensor_Print(&G);

    Tensor H = Tensor_SubScalar(&A, 1.0f);
    Tensor_Print(&H);

    Tensor I = Tensor_MulScalar(&A, 2.0f);
    Tensor_Print(&I);

    Tensor J = Tensor_DivScalar(&A, 2.0f);
    Tensor_Print(&J);

    printf("\n----------------------------\n");
    printf("Dot Product\n");

    Tensor Dot = Tensor_Dot(&A, &B);
    Tensor_Print(&Dot);

    printf("\n----------------------------\n");
    printf("Matrix Multiplication\n");

    Tensor Mat = Tensor_MatMul(&A, &B);
    Tensor_Print(&Mat);

    printf("\n----------------------------\n");
    printf("Transpose\n");

    Tensor T = Tensor_Transpose(&A);
    Tensor_Print(&T);

    printf("\n----------------------------\n");
    printf("Copy\n");

    Tensor Copy = Tensor_Copy(&A);
    Tensor_Print(&Copy);

    printf("\nClone\n");

    Tensor Clone = Tensor_Clone(&A);
    Tensor_Print(&Clone);

    printf("\n----------------------------\n");
    printf("Fill\n");

    Tensor Fill = Tensor_Copy(&A);
    Tensor_Fill(&Fill, 9.0f);
    Tensor_Print(&Fill);

    printf("\nZero\n");

    Tensor Zero = Tensor_Copy(&A);
    Tensor_Zero(&Zero);
    Tensor_Print(&Zero);

    printf("\nOnes\n");

    Tensor Ones = Tensor_Copy(&A);
    Tensor_Ones(&Ones);
    Tensor_Print(&Ones);

    printf("\nRandom\n");

    Tensor Random = Tensor_Copy(&A);
    Tensor_Random(&Random);
    Tensor_Print(&Random);

    printf("\n----------------------------\n");
    printf("Reshape\n");

    size_t shape1d[1] = {4};

    Tensor R = Tensor_Reshape(&A, 1, shape1d);

    Tensor_Print(&R);

    printf("\nView\n");

    Tensor View = Tensor_View(&A, 0, 1, shape1d);

    Tensor_Print(&View);

    printf("\nSlice\n");

    Tensor Slice = Tensor_Slice(&R, 1, 2);

    Tensor_Print(&Slice);

    printf("\nSqueeze\n");

    Tensor Sq = Tensor_Squeeze(&View);

    Tensor_Print(&Sq);

    printf("\nUnsqueeze\n");

    Tensor Uq = Tensor_Unsqueeze(&R, 0);

    Tensor_Print(&Uq);

    printf("\n----------------------------\n");

    if (Tensor_Equal(&A, &Copy))
        printf("Tensor_Equal : PASS\n");
    else
        printf("Tensor_Equal : FAIL\n");

    printf("\n=====================================\n");
    printf("All demonstrations finished.\n");
    printf("=====================================\n");

    Tensor_Destroy(&A);
    Tensor_Destroy(&B);

    Tensor_Destroy(&C);
    Tensor_Destroy(&D);
    Tensor_Destroy(&E);
    Tensor_Destroy(&F);

    Tensor_Destroy(&G);
    Tensor_Destroy(&H);
    Tensor_Destroy(&I);
    Tensor_Destroy(&J);

    Tensor_Destroy(&Dot);
    Tensor_Destroy(&Mat);
    Tensor_Destroy(&T);

    Tensor_Destroy(&Copy);
    Tensor_Destroy(&Clone);

    Tensor_Destroy(&Fill);
    Tensor_Destroy(&Zero);
    Tensor_Destroy(&Ones);
    Tensor_Destroy(&Random);

    Tensor_Destroy(&R);
    Tensor_Destroy(&View);
    Tensor_Destroy(&Slice);
    Tensor_Destroy(&Sq);
    Tensor_Destroy(&Uq);

    return 0;
}