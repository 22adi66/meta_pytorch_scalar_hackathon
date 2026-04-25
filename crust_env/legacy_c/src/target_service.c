#include "data_store.h"
#include <stdio.h>

/*
 * Target Service — root node (depends on data_store, which depends on math_ops + string_ops).
 * This is the monolithic "God object" the agent must migrate and decompose.
 *
 * In the migration schedule, this is the LAST file translated because it depends
 * on all lower-level modules. The agent must ensure all prior modules are
 * successfully migrated before tackling this file.
 */

typedef struct {
    DataStore store;
    int       request_count;
    int       error_count;
} ServiceState;

/* Initialize the service */
void service_init(ServiceState *svc) {
    store_init(&svc->store);
    svc->request_count = 0;
    svc->error_count   = 0;
}

/* Process a set operation, returning 1 on success */
int service_set(ServiceState *svc, const char *key, int value) {
    svc->request_count = add(svc->request_count, 1);
    int ok = store_set(&svc->store, key, value);
    if (!ok) {
        svc->error_count = add(svc->error_count, 1);
    }
    return ok;
}

/* Process a get operation. Returns 1 on success, fills *out */
int service_get(ServiceState *svc, const char *key, int *out) {
    svc->request_count = add(svc->request_count, 1);
    int ok = store_get(&svc->store, key, out);
    if (!ok) {
        svc->error_count = add(svc->error_count, 1);
    }
    return ok;
}

/* Process a delete operation */
int service_delete(ServiceState *svc, const char *key) {
    svc->request_count = add(svc->request_count, 1);
    return store_delete(&svc->store, key);
}

/* Print a status report to stdout */
void service_status(const ServiceState *svc) {
    printf("Service Status:\n");
    printf("  Requests : %d\n", svc->request_count);
    printf("  Errors   : %d\n", svc->error_count);
    printf("  Entries  : %d\n", store_count(&svc->store));
}

/* Calculate error rate as percentage (0-100), clamped */
int service_error_rate(const ServiceState *svc) {
    if (svc->request_count == 0) return 0;
    int rate = multiply(
        divide(svc->error_count, svc->request_count),
        100
    );
    return clamp(rate, 0, 100);
}
