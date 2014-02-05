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
    const unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx; 
    const unsigned int gx = gridDim.x, gy = gridDim.y; 

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y];

    if((I < N) && (J < N)){
        _DOUBLE_ c = 0;
        
        for (int k=0; k < gy; k++){ 
            a[ty][tx] = A[I*N+k*by+tx]; 
            b[ty][tx] = B[J+N*(k*bx+ty)]; 
            __syncthreads(); 
            
            for (int kk=0; kk< bx; kk++) { 
                c += a[ty][kk]*b[kk][tx]; 
            }
            
            __syncthreads();  
        }
        C[I * N + J] = c;
    }
}
