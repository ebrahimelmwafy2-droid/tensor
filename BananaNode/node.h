#ifndef NODE_H
#define NODE_H

#include "tensor.h"

typedef enum
{
    B_OP_NONE,

    B_OP_ADD,
    B_OP_SUB,
    B_OP_MUL,
    B_OP_DIV,

    B_OP_MATMUL,

    B_OP_RELU,
    B_OP_SIGMOID,
    B_OP_TANH

} B_Operation;

typedef struct B_Node
{
    B_Operation op;

    Tensor **inputs;
    size_t input_count;

    Tensor *output;

    void *context;

    void (*backward)(struct B_Node *);

} B_Node;

B_Node *B_Node_Create(B_Operation op, size_t input_count);

void B_Node_Destroy(B_Node *node);

#endif