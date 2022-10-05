#ifndef NN_UTILS_H

/**
 * This header consists of macros that
 * defines a structure representing a column-major, 2- or N-dimensional array
 * and related utilities.
 * The structure itself is made up with
 * the number of dimensions, the pointer to the length of each dimension,
 * and the pointer to the data sequence.

 * C-style arrays (foo[]) are FORBIDDEN in this project per the assignment requirements,
 * and is thus not used here.
 **/

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "idx.h"

// 2-dimensional arrays

#define ARR2D_TYPE(T) arr2d_##T

#define DEFINE_ARR2D_TYPE(T) \
  typedef struct ARR2D_TYPE(T) { \
    size_t dim0, dim1; \
    T* p_data; \
  } ARR2D_TYPE(T)
#define ARR2D_ALLOC(p_arr) ALLOC_TO(&(p_arr)->p_data, (p_arr)->dim0*(p_arr)->dim1)
#define ARR2D_FREE(p_arr) free((p_arr)->p_data)
#define ARR2D_PIDX(p_arr, i0, i1) PIDX((p_arr)->p_data, i0 + i1 * (p_arr)->dim0)

#define ARR2D_NELEM(p_arr) ((p_arr)->dim0 * (p_arr)->dim1)

#define ARR2D_REPEAT(p_arr, X) \
  for (size_t i_elem = 0; i_elem < ARR2D_NELEM(p_arr); ++i_elem) { \
    IDX((p_arr)->p_data, i_elem) = (X); \
  }

/// Left product onto a column vector plus a bias: Av + b
///
/// The shape of A is m * n, v n * 1, b n * 1, and Av 1 * m
///
/// If the bias pointer is NULL, b = 0 is assumed
/// p_out should be allocated to fill in p_arr->dim0 elements
/// prior to calling
#define ARR2D_LPROD_BIASED(p_arr, p_out, p_in, b_in) \
  do { \
    if (b_in) { \
      MEMSETN(p_out, 0, (p_arr)->dim0); \
    } else { \
      MEMCPYN(p_out, b_in, (p_arr)->dim0); \
    } \
    for (size_t i = 0; i < (p_arr)->dim1; ++i) { \
      for (size_t j = 0; j < (p_arr)->dim0; ++j) { \
        IDX(p_out, j) += *ARR2D_PIDX(p_arr, j, i) * IDX(p_in, i); \
      } \
    } \
  } while (0)

/// Left product onto a column vector: Av
///
/// The shape of A is m * n, v n * 1, and Av 1 * m
/// p_out should be allocated to fill in p_arr->dim0 elements
/// prior to calling
#define ARR2D_LPROD(p_arr, p_out, p_in) ARR2D_LPROD_BIASED(p_arr, p_out, p_in, NULL)

/// Right product onto a row vector: vA
///
/// The shape of A is m * n, v 1 * m, and vA n * 1
/// p_out should be allocated to fill in p_arr->dim1 elements
/// prior to calling
#define ARR2D_RPROD(p_arr, p_out, p_in) \
  do { \
    MEMSETN(p_out, 0, (p_arr)->dim1); \
    for (size_t j = 0; j < (p_arr)->dim1; ++j) { \
      for (size_t i = 0; i < (p_arr)->dim0; ++i) { \
        IDX(p_out, j) += IDX(p_in, i) * *ARR2D_PIDX(p_arr, i, j); \
      } \
    } \
  } while (0)

#ifndef DEFINED_ARR2D_DOUBLE
#define DEFINED_ARR2D_DOUBLE
DEFINE_ARR2D_TYPE(double);
#endif /* DEFINED_ARR2D_DOUBLE */

#ifndef DEFINED_ARR2D_FLOAT
#define DEFINED_ARR2D_FLOAT
DEFINE_ARR2D_TYPE(float);
#endif /* DEFINED_ARR2D_FLOAT */

// N-dimensional arrays

#define ARRND_TYPE(T) arrnd_##T

#define DEFINE_ARRND_TYPE(T) \
  typedef struct ARRND_TYPE(T) { \
    size_t n_dim; \
    size_t* p_dims; \
    T* p_data; \
  } ARRND_TYPE(T)

#define ARRND_ALLOC_FN(T) arrnd_alloc_##T
#define DECL_ARRND_ALLOC_FN(T) void ARRND_ALLOC_FN(T)(ARRND_TYPE(T) * p_arr, const size_t n_dim, ...)
#define DEFINE_ARRND_ALLOC_FN(T) \
  DECL_ARRND_ALLOC_FN(T) { \
    va_list ap; \
    va_start(ap, n_dim); \
    p_arr->n_dim = n_dim; \
    p_arr->p_dims = (size_t*)malloc(sizeof(size_t) * n_dim); \
    if (!n_dim) { \
      p_arr->p_data = (T*)malloc(0); \
    } \
    size_t n_data = 1; \
    for (size_t i = 0; i < n_dim; ++i) { \
      *(p_arr->p_dims + sizeof(size_t) * i) = va_arg(ap, size_t); \
      n_data *= IDX(p_arr->p_dims, i); \
    } \
    DEBUG_PRINTF(1, "n_data: %zu\n", n_data); \
    va_end(ap); \
    p_arr->p_data = (T*)malloc(sizeof(T) * n_data); \
  }

#define ARRND_PIDX_FN(T) arrnd_pidx_##T
#define DECL_ARRND_PIDX_FN(T) T* ARRND_PIDX_FN(T)(ARRND_TYPE(T) * p_arr, ...)
#define DEFINE_ARRND_PIDX_FN(T) \
  DECL_ARRND_PIDX_FN(T) { \
    va_list ap; \
    va_start(ap, p_arr); \
    size_t dim_prev_prod = 1; \
    T* result = p_arr->p_data; \
    DEBUG_PRINTF(1, "p_arr->p_data: %p\n", p_arr->p_data); \
    for (size_t i = 0; i < p_arr->n_dim; ++i) { \
      const size_t idx = va_arg(ap, size_t); \
      DEBUG_PRINTF(1, "i: %zu, dim_len: %zu, idx: %zu, dim_prev_prod: %zu\n", i, p_arr->p_dims + sizeof(size_t) * i, idx, dim_prev_prod); \
      result += sizeof(T) * dim_prev_prod * idx; \
      DEBUG_PRINTF(1, "result: %p\n", result); \
      dim_prev_prod *= IDX(p_arr->p_dims, i); \
    } \
    va_end(ap); \
    return result; \
  }

#define ARRND_PWIDX_FN(T) arrnd_pwidx_##T
#define DECL_ARRND_PWIDX_FN(T) T* ARRND_PWIDX_FN(T)(ARRND_TYPE(T) * p_arr, ...)
#define DEFINE_ARRND_PWIDX_FN(T) \
  DECL_ARRND_PWIDX_FN(T) { \
    va_list ap; \
    va_start(ap, p_arr); \
    size_t dim_prev_prod = 1; \
    size_t idx_tot = 0; \
    for (size_t i = 0; i < p_arr->n_dim; ++i) { \
      const ptrdiff_t idx_signed = va_arg(ap, ptrdiff_t); \
      const size_t idx = (idx_signed < 0) ? IDX(p_arr->p_dims, i) + idx_signed : idx_signed; \
      idx_tot += dim_prev_prod * idx; \
      dim_prev_prod *= IDX(p_arr->p_dims, i); \
    } \
    va_end(ap); \
    return PIDX(p_arr->p_data, idx_tot); \
  }

#define ARRND_FREE(p_arr) \
  free((p_arr)->p_data); \
  free((p_arr)->p_dims)

#define ACQUIRE_ARRND_UTILS(T) \
  DEFINE_ARRND_TYPE(T); \
  DEFINE_ARRND_ALLOC_FN(T); \
  DEFINE_ARRND_PIDX_FN(T); \
  DEFINE_ARRND_PWIDX_FN(T)

#ifndef ACQUIRED_ARRND_UTILS_DOUBLE
#define ACQUIRED_ARRND_UTILS_DOUBLE
ACQUIRE_ARRND_UTILS(double);
#endif

#ifndef ACQUIRED_ARRND_UTILS_FLOAT
#define ACQUIRED_ARRND_UTILS_FLOAT
ACQUIRE_ARRND_UTILS(float);
#endif

#endif  // NN_UTILS_H
