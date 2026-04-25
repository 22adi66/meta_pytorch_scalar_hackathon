#include "data_store.h"

/* Intermediate node — depends on math_ops.h + string_ops.h */

void store_init(DataStore *store) {
    store->count = 0;
    for (int i = 0; i < MAX_ENTRIES; i++) {
        store->entries[i].in_use = 0;
        store->entries[i].key[0] = '\0';
        store->entries[i].value  = 0;
    }
}

int store_set(DataStore *store, const char *key, int value) {
    /* First, check if key already exists — update in place */
    for (int i = 0; i < MAX_ENTRIES; i++) {
        if (store->entries[i].in_use && str_equals(store->entries[i].key, key)) {
            store->entries[i].value = value;
            return 1;
        }
    }

    /* Find a free slot */
    for (int i = 0; i < MAX_ENTRIES; i++) {
        if (!store->entries[i].in_use) {
            str_copy(store->entries[i].key, key);
            store->entries[i].value  = clamp(value, -1000000, 1000000);
            store->entries[i].in_use = 1;
            store->count = add(store->count, 1);
            return 1;
        }
    }

    return 0;  /* Store is full */
}

int store_get(const DataStore *store, const char *key, int *out) {
    for (int i = 0; i < MAX_ENTRIES; i++) {
        if (store->entries[i].in_use && str_equals(store->entries[i].key, key)) {
            *out = store->entries[i].value;
            return 1;
        }
    }
    return 0;
}

int store_delete(DataStore *store, const char *key) {
    for (int i = 0; i < MAX_ENTRIES; i++) {
        if (store->entries[i].in_use && str_equals(store->entries[i].key, key)) {
            store->entries[i].in_use = 0;
            store->entries[i].key[0] = '\0';
            store->count = subtract(store->count, 1);
            return 1;
        }
    }
    return 0;
}

int store_count(const DataStore *store) {
    return store->count;
}
