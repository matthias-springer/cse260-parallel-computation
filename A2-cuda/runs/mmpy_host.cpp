#include "types.h"
void
matMulHost(_DOUBLE_  *C, const _DOUBLE_  *A, const _DOUBLE_  *B, unsigned int m, unsigned int n)
{
    for (unsigned int i = 0; i < m; i++)
        for (unsigned int j = 0; j < n; j++) {
            _DOUBLE_ sum = 0;
            for (unsigned int k = 0; k < n; k++) 
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = (_DOUBLE_) sum;
        }
}
