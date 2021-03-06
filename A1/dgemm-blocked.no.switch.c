/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

/* CSE 260 Assignment 1 
 * Andrew Conegliano, Matthias Springer
 */

#include <emmintrin.h>
// #include <pmmintrin.h>
// #include <malloc.h>
// #include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

/* Default values for block size */
// #define BLOCK_SIZE_I 24
// #define BLOCK_SIZE_J 24
// #define BLOCK_SIZE_K 64

#define MIN_HELPER(a,b) (((a)<(b))?(a):(b))
#define min(a, b) MIN_HELPER(a, b)

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

// Aligned buffers for A and B
double A_buffered[32*800*sizeof(double)] __attribute__((aligned(16)));
double B_global_transpose[800*800*sizeof(double)] __attribute__((aligned(16)));

#define DO_BLOCK_UNROLLED_WITHOUT_I_HELPER(odd_increment, block_size_i, block_size_j, block_size_k) \
static void do_block_unrolled_without_i_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	for (int i = 0; i < (block_size_i); ++i) \
	{ \
		register int index_A = i*(lda + (odd_increment)); \
		register int index_C = i*lda; \
		\
		for (int j = 0; j < (block_size_j); j += 2) \
    { \
      __m128d cij_1 = _mm_setzero_pd(); \
			__m128d cij_2 = _mm_setzero_pd(); \
			register int index_B_1 = j*(lda + (odd_increment)); \
			register int index_B_2 = index_B_1 + (lda + (odd_increment)); \
 \
			for (int k = 0; k < (block_size_k); k += 4) { \
				__m128d a1 = _mm_load_pd(A + index_A); \
				index_A += 2; \
        __m128d b1 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				__m128d a2 = _mm_load_pd(A + index_A); \
				index_A += 2; \
        __m128d b2 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				cij_1 = _mm_add_pd(cij_1, _mm_add_pd(_mm_mul_pd(a1, b1), _mm_mul_pd(a2, b2))); \
 \
        __m128d b3 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        __m128d b4 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        cij_2 = _mm_add_pd(cij_2, _mm_add_pd(_mm_mul_pd(a1, b3), _mm_mul_pd(a2, b4))); \
			} \
 \
			__m128d cij_1_u = _mm_unpackhi_pd(cij_1, cij_1); \
			C[index_C] += _mm_cvtsd_f64(cij_1) + _mm_cvtsd_f64(cij_1_u); \
      __m128d cij_2_u = _mm_unpackhi_pd(cij_2, cij_2); \
      C[index_C + 1] += _mm_cvtsd_f64(cij_2) + _mm_cvtsd_f64(cij_2_u); \
 \
			index_A -= (block_size_k); \
			index_C += 2; \
		} \
	} \
}

#define DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) DO_BLOCK_UNROLLED_WITHOUT_I_HELPER(odd_increment, block_size_i, block_size_j, block_size_k)

#define DO_BLOCK_UNROLLED_HELPER(odd_increment, block_size_i, block_size_j, block_size_k) \
static void do_block_unrolled_with_i_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	for (int i = 0; i < (block_size_i); i += 2) \
	{ \
		register int index_A_1 = i*(lda + (odd_increment)); \
		register int index_A_2 = index_A_1 + lda + (odd_increment); \
		register int index_C = i*lda; \
		\
		for (int j = 0; j < (block_size_j); j += 2) \
    { \
      __m128d cij_1 = _mm_setzero_pd(); \
			__m128d cij_2 = _mm_setzero_pd(); \
			__m128d cij_a2_1 = _mm_setzero_pd(); \
			__m128d cij_a2_2 = _mm_setzero_pd(); \
			register int index_B_1 = j*(lda + (odd_increment)); \
			register int index_B_2 = index_B_1 + (lda + (odd_increment)); \
 \
			for (int k = 0; k < (block_size_k); k += 4) { \
				__m128d a1 = _mm_load_pd(A + index_A_1); \
				index_A_1 += 2; \
        __m128d b1 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				__m128d a2 = _mm_load_pd(A + index_A_1); \
				index_A_1 += 2; \
        __m128d b2 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				cij_1 = _mm_add_pd(cij_1, _mm_add_pd(_mm_mul_pd(a1, b1), _mm_mul_pd(a2, b2))); \
 \
        __m128d b3 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        __m128d b4 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        cij_2 = _mm_add_pd(cij_2, _mm_add_pd(_mm_mul_pd(a1, b3), _mm_mul_pd(a2, b4))); \
 \
        __m128d a3 = _mm_load_pd(A + index_A_2); \
        index_A_2 += 2; \
        __m128d a4 = _mm_load_pd(A + index_A_2); \
        index_A_2 += 2; \
				cij_a2_1 = _mm_add_pd(cij_a2_1, _mm_add_pd(_mm_mul_pd(a3, b1), _mm_mul_pd(a4, b2))); \
				cij_a2_2 = _mm_add_pd(cij_a2_2, _mm_add_pd(_mm_mul_pd(a3, b3), _mm_mul_pd(a4, b4))); \
			} \
 \
			__m128d cij_1_u = _mm_unpackhi_pd(cij_1, cij_1); \
			C[index_C] += _mm_cvtsd_f64(cij_1) + _mm_cvtsd_f64(cij_1_u); \
      __m128d cij_2_u = _mm_unpackhi_pd(cij_2, cij_2); \
      C[index_C + 1] += _mm_cvtsd_f64(cij_2) + _mm_cvtsd_f64(cij_2_u); \
      cij_1_u = _mm_unpackhi_pd(cij_a2_1, cij_a2_1); \
      C[index_C + lda] += _mm_cvtsd_f64(cij_a2_1) + _mm_cvtsd_f64(cij_1_u); \
      cij_2_u = _mm_unpackhi_pd(cij_a2_2, cij_a2_2); \
      C[index_C + lda + 1] += _mm_cvtsd_f64(cij_a2_2) + _mm_cvtsd_f64(cij_2_u); \
 \
			index_A_1 -= (block_size_k); \
			index_A_2 -= (block_size_k); \
			index_C += 2; \
		} \
	} \
}

#define DO_BLOCK_UNROLLED_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) DO_BLOCK_UNROLLED_HELPER(odd_increment, block_size_i, block_size_j, block_size_k)

/* Expand for default case */
// DO_BLOCK_UNROLLED_EXPAND(0, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)
// DO_BLOCK_UNROLLED_EXPAND(1, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)

#define SQUARE_DGEMM_CALL_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) square_dgemm_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k

#define SQUARE_DGEMM(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) SQUARE_DGEMM_CALL_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k)

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */ 
#define SQUARE_DGEMM_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) \
void square_dgemm_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	const register int fringe_start_i = lda / (block_size_i) * (block_size_i); \
	const register int fringe_start_j = lda / (block_size_j) * (block_size_j); \
	const register int fringe_start_k = lda / (block_size_k) * (block_size_k); \
 \
  for (int i = 0; i < fringe_start_i; i += (block_size_i)) { \
		for(int x = i; x < (block_size_i) + i; x++) { \
			memcpy(A_buffered + (x-i)*(lda+(odd_increment)), A + x*lda, lda*sizeof(double)); \
		} \
    \
		for (int j = 0; j < fringe_start_j; j += (block_size_j)) { \
      for (int k = 0; k < fringe_start_k; k += (block_size_k)) \
      { \
				do_block_unrolled_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(lda, A_buffered + i%(block_size_i) * (lda+(odd_increment)) + k, B + j*(lda + (odd_increment)) + k, C + i*lda + j); \
				/* do_block_##odd_increment(lda, (block_size_i), (block_size_j), (block_size_k), A + i*lda + k, B + j*(lda+(odd_increment)) + k, C + i*lda + j); */ \
      } \
		} \
	} \
	if (fringe_start_i == lda && fringe_start_j == lda && fringe_start_k == lda) return; \
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

#define SQUARE_DGEMM_EXPAND(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) SQUARE_DGEMM_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k)

/* Expand for default case */
// SQUARE_DGEMM_EXPAND(1, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)
// SQUARE_DGEMM_EXPAND(0, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)


#define TRANSPOSE_B_HELPER(odd_increment, block_size_i, block_size_j) \
void transpose_B_##odd_increment##_##block_size_i##_##block_size_j (int lda, double* B) \
{ \
const register int fringe_start_i = lda / (block_size_i) * (block_size_i); \
const register int fringe_start_j = lda / (block_size_j) * (block_size_j); \
for (int i = 0; i < fringe_start_i; i += (block_size_i)) { \
  for (int j = 0; j < fringe_start_j; j += (block_size_i)) { \
    for (int i0 = i; i0 < i + (block_size_i); ++i0) { \
			register int index_L = i0*(lda + (odd_increment)) + j; \
			register int index_R = i0 + j*lda; \
      for (int j0 = j; j0 < j + (block_size_j); ++j0) { \
        B_global_transpose[index_L] = B[index_R]; \
				index_L++; \
				index_R += lda; \
      } \
    } \
  } \
} \
if (fringe_start_i == lda && fringe_start_j == lda) return; \
 \
{ \
	for (int i = fringe_start_i; i < lda; i++) { \
		for (int j = 0; j < fringe_start_j; j++) { \
			B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
		} \
	} \
} \
{ \
  for (int i = 0; i < fringe_start_i; i++) { \
    for (int j = fringe_start_j; j < lda; j++) { \
      B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
    } \
  } \
} \
{ \
  for (int i = fringe_start_i; i < lda; i++) { \
    for (int j = fringe_start_j; j < lda; j++) { \
      B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
    } \
  } \
} \
} 

#define TRANSPOSE_B_EXPAND(odd_increment, block_size_i, block_size_j) TRANSPOSE_B_HELPER(odd_increment, block_size_i, block_size_j)

#define DGEMM_SELECT(lda, do_block_function, odd_increment, block_size_i_j, block_size_k) \
case (lda): \
	transpose_B_##odd_increment##_##block_size_i_j##_##block_size_i_j ((lda), B); \
	square_dgemm_##do_block_function##_##odd_increment##_##block_size_i_j##_##block_size_i_j##_##block_size_k ((lda), A, B_global_transpose, C); \
	break;

/* Expand macros for several block sizes */
TRANSPOSE_B_EXPAND(0, 8, 8);
TRANSPOSE_B_EXPAND(1, 8, 8);
TRANSPOSE_B_EXPAND(0, 16, 16);
TRANSPOSE_B_EXPAND(1, 16, 16);
TRANSPOSE_B_EXPAND(0, 24, 24);
TRANSPOSE_B_EXPAND(1, 24, 24);
TRANSPOSE_B_EXPAND(0, 32, 32);
TRANSPOSE_B_EXPAND(1, 32, 32);
TRANSPOSE_B_EXPAND(0, 40, 40);
TRANSPOSE_B_EXPAND(1, 40, 40);
TRANSPOSE_B_EXPAND(0, 48, 48);
TRANSPOSE_B_EXPAND(1, 48, 48);
TRANSPOSE_B_EXPAND(0, 56, 56);
TRANSPOSE_B_EXPAND(1, 56, 56);
TRANSPOSE_B_EXPAND(0, 64, 64);
TRANSPOSE_B_EXPAND(1, 64, 64);

DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(0, 32, 32, 32)
DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(1, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 64)

SQUARE_DGEMM_EXPAND(without_i, 0, 32, 32, 32)
SQUARE_DGEMM_EXPAND(without_i, 1, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 64)

void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C) {
	switch (lda) {


		default:
		  if (lda % 2 == 0) {
		    transpose_B_0_32_32(lda, B);
		    SQUARE_DGEMM(with_i, 0, 8, 8, 64)(lda, A, B_global_transpose, C);
		  }
		  else {
		    transpose_B_0_32_32(lda, B);
		    SQUARE_DGEMM(with_i, 1, 8, 8, 64)(lda, A, B_global_transpose, C);
			} 
	}
}

