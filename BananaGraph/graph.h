#ifndef BANANA_GRAPH_H
#define BANANA_GRAPH_H

#include "node.h"

typedef struct
{
    B_Node **nodes;

    size_t count;
    size_t capacity;

} B_Graph;

B_Graph *Graph_Create(void);

void Graph_Destroy(B_Graph *graph);

int Graph_AddNode(
    B_Graph *graph,
    B_Node *node
);

#endif