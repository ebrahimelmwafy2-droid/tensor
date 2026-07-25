#ifndef STORAGE_H
#define STORAGE_H

#include <stddef.h>

typedef struct
{
    void *data;         // Pointer to raw memory
    size_t bytes;       // Total allocated bytes
    size_t ref_count;   // Reference counter
} Storage;

/* Create Storage */
Storage *Storage_Create(size_t bytes);
void Storage_Destroy(Storage *storage);

void Storage_Retain(Storage *storage);

void Storage_Release(Storage *storage);
#endif