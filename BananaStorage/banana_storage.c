#include "banana_storage.h"
#include <stdlib.h>
#include <string.h>

Storage *Storage_Create(size_t bytes)
{
    Storage *storage = malloc(sizeof(Storage));

    if (storage == NULL)
        return NULL;

    storage->bytes = bytes;
    storage->ref_count = 1;
    storage->data = calloc(1, bytes);

    return storage;
}

void Storage_Destroy(Storage *storage)
{
    if (storage == NULL)
        return;

    free(storage->data);

    storage->data = NULL;
    storage->bytes = 0;
    storage->ref_count = 0;
}


void Storage_Retain(Storage *storage)
{
    if (storage == NULL)
        return;

    storage->ref_count++;
}

void Storage_Release(Storage *storage)
{
    if (storage == NULL)
        return;

    if (--storage->ref_count == 0)
    {
        Storage_Destroy(storage);
        free(storage);
    }
}