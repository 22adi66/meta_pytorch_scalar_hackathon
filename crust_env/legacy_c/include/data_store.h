#ifndef DATA_STORE_H
#define DATA_STORE_H

#include "math_ops.h"
#include "string_ops.h"

/* Simple fixed-capacity key-value store — intermediate node */
/* Depends on: math_ops.h (for clamp), string_ops.h (for str_equals, str_copy) */

#define MAX_ENTRIES 64
#define KEY_SIZE    32
#define VAL_SIZE    64

typedef struct {
    char key[KEY_SIZE];
    int  value;
    int  in_use;
} Entry;

typedef struct {
    Entry entries[MAX_ENTRIES];
    int   count;
} DataStore;

/* Initialize an empty store */
void store_init(DataStore *store);

/* Insert or update a key. Returns 1 on success, 0 if store is full. */
int store_set(DataStore *store, const char *key, int value);

/* Retrieve value by key. Returns 1 on success (value written to *out), 0 if not found. */
int store_get(const DataStore *store, const char *key, int *out);

/* Delete a key. Returns 1 if deleted, 0 if not found. */
int store_delete(DataStore *store, const char *key);

/* Return number of entries currently in the store */
int store_count(const DataStore *store);

#endif /* DATA_STORE_H */
