#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

void Memory_Copy(void *dest, const void *src, size_t bytes);
void Memory_Move(void *dest, const void *src, size_t bytes);
void Memory_Set(void *dest, int value, size_t bytes);
int Memory_Compare(const void *a, const void *b, size_t bytes);

void *B_Malloc(size_t size);
void *B_Calloc(size_t count, size_t size);
void *B_Realloc(void *ptr, size_t size);
void B_Free(void *ptr);

#endif