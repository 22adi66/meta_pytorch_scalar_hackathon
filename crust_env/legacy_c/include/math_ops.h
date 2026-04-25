#ifndef MATH_OPS_H
#define MATH_OPS_H

/* Basic arithmetic operations — leaf node, no dependencies */

int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);

/* Returns -1 on divide-by-zero */
int divide(int a, int b);

/* Clamp value to [min_val, max_val] */
int clamp(int value, int min_val, int max_val);

#endif /* MATH_OPS_H */
