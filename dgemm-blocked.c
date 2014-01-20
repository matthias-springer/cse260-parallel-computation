/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <emmintrin.h>
#include <malloc.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
//#define BLOCK_SIZE 32
#endif

int BLOCK_SIZE = 32;

int BLOCK_SIZE_I = 16;
int BLOCK_SIZE_J = 32;
int BLOCK_SIZE_K = 32;

#define min(a,b) (((a)<(b))?(a):(b))

#define do_block_not_unrolled(i, j, k, M, N, K) \
        do_block_1(lda, (M), (N), (K), A + (i)*lda + (k), B + (j)*lda + (k), C + (i)*lda + (j));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
#define DO_BLOCK_EXPAND(odd_increment) \
static void do_block_##odd_increment (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) \
{ \
  for (int i = 0; i < M; ++i) \
    for (int j = 0; j < N; ++j) \
    { \
      double cij = C[i*lda+j]; \
      for (int k = 0; k < K; ++k) \
			{ \
				cij += A[i*lda+k] * B[j*(lda+(odd_increment))+k]; \
			} \
\
      C[i*lda+j] = cij; \
    } \
}

DO_BLOCK_EXPAND(0)
DO_BLOCK_EXPAND(1)

double A_buffered[32*800*sizeof(double)] __attribute__((aligned(16)));
double B_transposed[32*32*sizeof(double)] __attribute__((aligned(16)));
double B_global_transpose[800*800*sizeof(double)] __attribute__((aligned(16)));

#define DO_BLOCK_UNROLLED_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) \
static void do_block_unrolled_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	for (int i = 0; i < (block_size_i); ++i) \
		for (int j = 0; j < (block_size_j); ++j) \
    { \
      double cij = C[i*lda+j]; \
 \
			for (int k = 0; k < (block_size_k); k += 8) { \
				int index_A = i*(lda + (odd_increment)) + k; \
				int index_B = j*(lda + (odd_increment)) + k; \
 \
				__m128d a1 = _mm_load_pd(A + index_A); \
				__m128d a2 = _mm_load_pd(A + index_A + 2); \
        __m128d a3 = _mm_load_pd(A + index_A + 4); \
        __m128d a4 = _mm_load_pd(A + index_A + 6); \
 \
        __m128d b1 = _mm_load_pd(B + index_B); \
        __m128d b2 = _mm_load_pd(B + index_B + 2); \
        __m128d b3 = _mm_load_pd(B + index_B + 4); \
        __m128d b4 = _mm_load_pd(B + index_B + 6); \
 \
				__m128d d1 = _mm_mul_pd(a1, b1); \
				__m128d d2 = _mm_mul_pd(a2, b2); \
        __m128d d3 = _mm_mul_pd(a3, b3); \
        __m128d d4 = _mm_mul_pd(a4, b4); \
 \
				__m128d c1 = _mm_add_pd(d1, d2); \
				__m128d c2 = _mm_add_pd(d3, d4); \
				__m128d c3 = _mm_add_pd(c1, c2); \
 \
				cij += _mm_cvtsd_f64(c3) + _mm_cvtsd_f64(_mm_unpackhi_pd(c3, c3)); \
			} \
 \
			C[i*lda+j] = cij; \
		} \
}

DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 32)

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */ 
#define SQUARE_DGEMM_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) \
void square_dgemm_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	int fringe_start_i = lda / (block_size_i) * (block_size_i); \
	int fringe_start_j = lda / (block_size_j) * (block_size_j); \
	int fringe_start_k = lda / (block_size_k) * (block_size_k); \
 \
  for (int i = 0; i < fringe_start_i; i += (block_size_i)) { \
		for(int x = i; x < 32 + i; x++) { \
			memcpy(A_buffered + (x-i)*(lda+(odd_increment)), A + x*lda, lda*sizeof(double)); \
		} \
    \
		for (int j = 0; j < fringe_start_j; j += (block_size_j)) { \
      for (int k = 0; k < fringe_start_k; k += (block_size_k)) \
      { \
				do_block_unrolled_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(lda, A_buffered + i%32 * (lda+(odd_increment)) + k, B + j*(lda + (odd_increment)) + k, C + i*lda + j); \
				/* do_block_##odd_increment(lda, (block_size_i), (block_size_j), (block_size_k), A + i*lda + k, B + j*(lda+(odd_increment)) + k, C + i*lda + j); */ \
      } \
		} \
	} \
	/* For the fringe cases */ \
	/* {i} */ \
	{ \
		int i = fringe_start_i; \
		for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
			for (int k = 0; k < fringe_start_k; k += (block_size_k)) \ 
			{ \
				do_block_##odd_increment(lda, lda-i, (block_size_j), (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
			} \
	} \
 \
	/*( {j} */ \
  { \
    int j = fringe_start_j; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \
      for (int k = 0; k < fringe_start_k; k += (block_size_k)) \
      { \
				do_block_##odd_increment(lda, (block_size_i), lda-j, (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
      } \
  } \
 \
	/* {k} */ \
  { \
    int k = fringe_start_k; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \
      for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
      { \
				do_block_##odd_increment(lda, (block_size_i), (block_size_j), lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
      } \
  } \
 \
	/* {i, j} */ \
	{ \
		int i = fringe_start_i; \
		int j = fringe_start_j; \
		for (int k = 0; k < fringe_start_k; k += (block_size_k)) \ 
		{ \
			do_block_##odd_increment(lda, lda-i, lda-j, (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
		} \
	} \
 \
	/* {j, k} */ \
  { \
    int j = fringe_start_j; \
    int k = fringe_start_k; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \ 
    { \
			do_block_##odd_increment(lda, (block_size_i), lda-j, lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
    } \
  } \
 \
	/* {i, k} */ \
  { \
    int i = fringe_start_i; \
    int k = fringe_start_k; \
    for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
    { \
			do_block_##odd_increment(lda, lda-i, (block_size_j), lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
    } \
  } \
 \
	/* {i, j, k} */ \
	{ \
		int i = fringe_start_i; \
		int j = fringe_start_j; \
		int k = fringe_start_k; \
		\
		do_block_##odd_increment(lda, lda-i, lda-j, lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
	} \
}

SQUARE_DGEMM_EXPAND(1, 32, 32, 32)
SQUARE_DGEMM_EXPAND(0, 32, 32, 32)


void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C) {
  if (lda % 2 == 0) {
    for (int i = 0; i < lda; i += 32) {
      for (int j = 0; j < lda; j += 32) {
        for (int i0 = i; i0 < min(i + 32, lda); ++i0) {
          for (int j0 = j; j0 < min(j + 32, lda); ++j0) {
            B_global_transpose[i0*lda + j0] = B[j0*lda + i0];
          }
        }
      }
    }
		
		square_dgemm_0_32_32_32(lda, A, B_global_transpose, C);
  }
  else {
    for (int i = 0; i < lda; i += 32) {
      for (int j = 0; j < lda; j += 32) {
        for (int i0 = i; i0 < min(i + 32, lda); ++i0) {
          for (int j0 = j; j0 < min(j + 32, lda); ++j0) {
            B_global_transpose[i0*(lda + 1) + j0] = B[j0*lda + i0];
          }
        }
      }
    }
		
		square_dgemm_1_32_32_32(lda, A, B_global_transpose, C);
  }
}

