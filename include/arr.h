#ifndef NN_UTILS_H

/**
 * This header consists of macros that
 * defines a structure representing a column-based N-dimensional array
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

#include "debug.h"
#include "idx.h"

#define NDARR_TYPE(T) struct ndarr_##T

#define DEFINE_NDARR_TYPE(T) \
  NDARR_TYPE(T) { \
    size_t n_dim; \
    size_t* p_dims; \
    T* p_data; \
  }

#define NDARR_ALLOC_FN(T) ndarr_alloc_##T
#define DECL_NDARR_ALLOC_FN(T) void NDARR_ALLOC_FN(T)(NDARR_TYPE(T) * parr, const size_t n_dim, ...)
#define DEFINE_NDARR_ALLOC_FN(T) \
  DECL_NDARR_ALLOC_FN(T) { \
    va_list ap; \
    va_start(ap, n_dim); \
    parr->n_dim = n_dim; \
    parr->p_dims = (size_t*)malloc(sizeof(size_t) * n_dim); \
    if (!n_dim) { \
      parr->p_data = (T*)malloc(0); \
    } \
    size_t n_data = 1; \
    for (size_t i = 0; i < n_dim; ++i) { \
      *(parr->p_dims + sizeof(size_t) * i) = va_arg(ap, size_t); \
      n_data *= IDX(parr->p_dims, i); \
    } \
    DEBUG_PRINTF(1, "n_data: %zu\n", n_data); \
    va_end(ap); \
    parr->p_data = (T*)malloc(sizeof(T) * n_data); \
  }

#define NDARR_PIDX_FN(T) ndarr_pidx_##T
#define DECL_NDARR_PIDX_FN(T) T* NDARR_PIDX_FN(T)(NDARR_TYPE(T) * parr, ...)
#define DEFINE_NDARR_PIDX_FN(T) \
  DECL_NDARR_PIDX_FN(T) { \
    va_list ap; \
    va_start(ap, parr); \
    size_t dim_prev_prod = 1; \
    T* result = parr->p_data; \
    DEBUG_PRINTF(1, "parr->p_data: %p\n", parr->p_data); \
    for (size_t i = 0; i < parr->n_dim; ++i) { \
      const size_t idx = va_arg(ap, size_t); \
      DEBUG_PRINTF(1, "i: %zu, dim_len: %zu, idx: %zu, dim_prev_prod: %zu\n", i, parr->p_dims + sizeof(size_t) * i, idx, dim_prev_prod); \
      result += sizeof(T) * dim_prev_prod * idx; \
      DEBUG_PRINTF(1, "result: %p\n", result); \
      dim_prev_prod *= IDX(parr->p_dims, i); \
    } \
    va_end(ap); \
    return result; \
  }

#define NDARR_PSIDX_FN(T) ndarr_psidva_endx_##T
#define DECL_NDARR_PSIDX_FN(T) T* NDARR_PSIDX_FN(T)(NDARR_TYPE(T) * parr, ...)
#define DEFINE_NDARR_PSIDX_FN(T) \
  DECL_NDARR_PSIDX_FN(T) { \
    va_list ap; \
    va_start(ap, parr); \
    size_t dim_prev_prod = 1; \
    T* result = parr->p_data; \
    for (size_t i = 0; i < parr->n_dim; ++i) { \
      const ptrdiff_t idx_signed = va_arg(ap, ptrdiff_t); \
      const size_t idx = (idx_signed < 0) ? IDX(parr->p_dims, i) + idx_signed : idx_signed; \
      result += sizeof(T) * dim_prev_prod * idx; \
      dim_prev_prod *= IDX(parr->p_dims, i); \
    } \
    va_end(ap); \
    return result; \
  }

#define NDARR_FREE(parr) \
  free((parr)->p_data); \
  free((parr)->p_dims)

#define ACQUIRE_NDARR_UTILS(T) \
  DEFINE_NDARR_TYPE(T); \
  DEFINE_NDARR_ALLOC_FN(T); \
  DEFINE_NDARR_PIDX_FN(T); \
  DEFINE_NDARR_PSIDX_FN(T)

#ifndef ACQUIRED_NDARRAY_UTILS_DOUBLE
#define ACQUIRED_NDARRAY_UTILS_DOUBLE
ACQUIRE_NDARR_UTILS(double);
#endif

#ifndef ACQUIRED_NDARRAY_UTILS_FLOAT
#define ACQUIRED_NDARRAY_UTILS_FLOAT
ACQUIRE_NDARR_UTILS(float);
#endif

#endif  // NN_UTILS_H
