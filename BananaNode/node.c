#include "node.h"
#include "memory.h"

B_Node *Node_Create(
    B_Operation op,
    Tensor **inputs,
    size_t input_count,
    Tensor *output)
{
    B_Node *node = B_Malloc(sizeof(B_Node));

    if (!node)
        return NULL;

    node->operation = op;
    node->input_count = input_count;
    node->output = output;
    node->backward = NULL;
    node->context = NULL;

    if (input_count > 0)
    {
        node->inputs = B_Malloc(sizeof(Tensor *) * input_count);

        if (!node->inputs)
        {
            B_Free(node);
            return NULL;
        }

        for (size_t i = 0; i < input_count; i++)
            node->inputs[i] = inputs[i];
    }
    else
    {
        node->inputs = NULL;
    }

    if (output)
        output->creator = node;

    return node;
}