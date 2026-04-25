#ifndef STRING_OPS_H
#define STRING_OPS_H

#include <stddef.h>

/* String utility operations — leaf node, no internal dependencies */

/* Returns length of string (excluding null terminator) */
size_t str_len(const char *s);

/* Convert ASCII string to uppercase in-place */
void to_upper(char *s);

/* Convert ASCII string to lowercase in-place */
void to_lower(char *s);

/* Returns 1 if strings are equal, 0 otherwise */
int str_equals(const char *a, const char *b);

/* Copy src into dest (caller responsible for buffer size) */
void str_copy(char *dest, const char *src);

#endif /* STRING_OPS_H */
