#include "memory.h"

#include <stdlib.h>
#include <string.h>

void Memory_Copy(void *dest, const void *src, size_t bytes)
{
    memcpy(dest, src, bytes);
}

void Memory_Move(void *dest, const void *src, size_t bytes)
{
    memmove(dest, src, bytes);
}

void Memory_Set(void *dest, int value, size_t bytes)
{
    memset(dest, value, bytes);
}

int Memory_Compare(const void *a, const void *b, size_t bytes)
{
    return memcmp(a, b, bytes);
}

void *B_Malloc(size_t size)
{
    return malloc(size);
}

void *B_Calloc(size_t count, size_t size)
{
    return calloc(count, size);
}

void *B_Realloc(void *ptr, size_t size)
{
    return realloc(ptr, size);
}

void B_Free(void *ptr)
{
    free(ptr);
}