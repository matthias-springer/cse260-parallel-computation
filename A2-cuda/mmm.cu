#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCK_X 16
#define BLOCK_Y 16

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    const unsigned int bx = BLOCK_X, by = BLOCK_Y;
    const unsigned int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx;
    const unsigned int I16 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int J16 = (blockIdx.x+gridDim.x)*bx + tx;
    const unsigned int gx = gridDim.x, gy = gridDim.y * 2;

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y][2];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y][2];

        _DOUBLE_ c[4] = {0};

        for (int k=0; k < gy; k++){
            __syncthreads();
            a[ty][tx][0] = A[I*N+k*by+tx];
            a[ty][tx][1] = A[I16*N+k*by+tx];
            b[ty][tx][0] = B[J+N*(k*bx+ty)];
            b[ty][tx][1] = B[J16+N*(k*bx+ty)];
            __syncthreads();

            for (int kk=0; kk<bx; kk++) {
                c[0] += a[ty][kk][0]*b[kk][tx][0];
                c[1] += a[ty][kk][1]*b[kk][tx][0];
                c[2] += a[ty][kk][0]*b[kk][tx][1];
                c[3] += a[ty][kk][1]*b[kk][tx][1];

            }
        }
        C[I * N + J] = c[0];
        C[I16 * N + J] = c[1];
        C[I * N + J16] = c[2];
        C[I16 * N + J16] = c[3];
}

