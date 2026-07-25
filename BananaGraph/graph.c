#include "graph.h"
B_Graph *Graph_Create(void)
{
    B_Graph *graph = B_Malloc(sizeof(B_Graph));

    if (!graph)
        return NULL;

    graph->nodes = NULL;
    graph->count = 0;
    graph->capacity = 0;

    return graph;
}