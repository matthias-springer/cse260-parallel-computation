// Matrix multiply device code
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
    const unsigned int I1 = blockIdx.y*by + ty, J = blockIdx.x*bx + tx;
    const unsigned int I2 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int I3 = (blockIdx.y+gridDim.y*2)*by + ty;
    const unsigned int I4 = (blockIdx.y+gridDim.y*3)*by + ty;
    const unsigned int gx = gridDim.x, gy = gridDim.y * 4; 

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y][4];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y];

    if((I4 < N) && (J < N)){
        _DOUBLE_ c[4] = {0,0,0,0};
        
        for (int k=0; k < gy; k++){ 
            a[ty][tx][0] = A[I1*N+k*by+tx]; 
            a[ty][tx][1] = A[I2*N+k*by+tx];
            a[ty][tx][2] = A[I3*N+k*by+tx];
            a[ty][tx][3] = A[I4*N+k*by+tx];
            b[ty][tx] = B[J+N*(k*bx+ty)]; 
            __syncthreads(); 
            
#pragma unroll
            for (int kk=0; kk< BLOCK_X; kk++) { 
                c[0] += a[ty][kk][0]*b[kk][tx];
                c[1] += a[ty][kk][1]*b[kk][tx];
                c[2] += a[ty][kk][2]*b[kk][tx];
                c[3] += a[ty][kk][3]*b[kk][tx];
            }
            
            __syncthreads();  
        }
        C[I1 * N + J] = c[0];
        C[I2 * N + J] = c[1];
        C[I3 * N + J] = c[2];
        C[I4 * N + J] = c[3];
    }
}
