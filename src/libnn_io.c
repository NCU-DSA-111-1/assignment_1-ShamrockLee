#include "libnn_io.h"

#include <assert.h>
#include <string.h>

#include "iee754_float.h"
#include "libnn.h"

// Ref endian:
// https://stackoverflow.com/questions/13994674/how-to-write-endian-agnostic-c-c-code

static inline void encode_uint32_be(char* pbegin, const uint32_t x) {
  *(pbegin + 0) = x & 0xff000000;
  *(pbegin + 1) = x & 0x00ff0000;
  *(pbegin + 2) = x & 0x0000ff00;
  *(pbegin + 3) = x & 0x000000ff;
}

static inline uint32_t decode_uint32_be(const char* const pbegin) {
  // clang-format off
  return
      ((uint32_t) * (pbegin + 0) << 24)
    | ((uint32_t) * (pbegin + 1) << 16)
    | ((uint32_t) * (pbegin + 2) << 8)
    | ((uint32_t) * (pbegin + 3) << 0);
  // clang-format off
}

int write_model_data(
    FILE* pf_out,
    const size_t n_layer,
    const size_t* const p_dims,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases) {
  RETURN_WHEN_FALSE(fputs(NN_FILE_MAGIC, pf_out), , "Failed to write magic.\n");
  char* buf = NULL;
  {
    // buf should be n_dim p_dims,
    // and the length of p_dims is (n_dim + 1),
    // so n_layer + 2
    const size_t len_buf = sizeof(uint32_t) * (n_layer + 2);
    buf = realloc(buf, len_buf);
    encode_uint32_be(buf, n_layer);
    for (size_t i_dim = 0; i_dim < n_layer + 1; ++i_dim) {
      encode_uint32_be(buf + sizeof(uint32_t) * (1 + i_dim), IDX(p_dims, i_dim));
    }
    RETURN_WHEN_FALSE(fwrite(buf, len_buf, 1, pf_out), free(buf), "Failed to write dims.\n");
  }
  size_t n_elem_bias_all_tot = 0;
  size_t n_elem_weight_all_tot = 0;
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    n_elem_weight_all_tot += IDX(p_dims, i_layer) * IDX(p_dims, i_layer + 1);
    n_elem_bias_all_tot += IDX(p_dims, i_layer + 1);
  }
  {
    const size_t len_buf = sizeof(double) * n_elem_weight_all_tot;
    buf = realloc(buf, len_buf);
    size_t n_elem_weight_prevall_tot = 0;
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      const ARR2D_TYPE(double)* const pa_weight_now = PIDX(pa_weights, i_layer);
      const size_t n_elem_weight_now_tot = pa_weight_now->dim0 * pa_weight_now->dim1;
      for (size_t i_elem = 0; i_elem < n_elem_weight_now_tot; ++i_elem) {
        IEE754_binary64_encode(IDX(pa_weight_now->p_data, i_elem), buf + sizeof(double) * (n_elem_weight_prevall_tot + i_elem));
      }
      n_elem_weight_prevall_tot += n_elem_weight_now_tot;
    }
    RETURN_WHEN_FALSE(fwrite(buf, len_buf, 1, pf_out), free(buf), "Failed to write weights.\n");
  }
  {
    const size_t len_buf = sizeof(double) * n_elem_bias_all_tot;
    buf = realloc(buf, len_buf);
    size_t n_elem_bias_prevall_tot = 0;
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      const double* const p_bias_now = IDX(pp_biases, i_layer);
      const size_t n_elem_bias_now = IDX(p_dims, i_layer + 1);
      for (size_t i_elem = 0; i_elem < n_elem_bias_now; ++i_elem) {
        IEE754_binary64_encode(IDX(p_bias_now, i_elem), buf + sizeof(double) * (n_elem_bias_prevall_tot + i_elem));
      }
      n_elem_bias_prevall_tot += n_elem_bias_now;
    }
    RETURN_WHEN_FALSE(fwrite(buf, len_buf, 1, pf_out), free(buf), "Failed to write biases.\n");
  }
  free(buf);
  return 1;
}

int read_model_data(
  FILE* pf_in,
  size_t* pn_layer,
  size_t* p_dims,
  ARR2D_TYPE(double) * pa_weights,
  double** pp_biases) {
  {
    char* magic_read;
    // magic_read should be magic '\0', so LEN_NN_FILE_MAGIC + 1
    magic_read = calloc(LEN_NN_FILE_MAGIC + 1, sizeof(char));
    RETURN_WHEN_FALSE(fread(magic_read, LEN_NN_FILE_MAGIC, 1, pf_in), free(magic_read), "Failed to read magic.\n");
    RETURN_WHEN_FALSE(strcmp(magic_read, NN_FILE_MAGIC) != 0, free(magic_read), "Model data file magic mismatched. Expect %s; got %s.\n", NN_FILE_MAGIC, magic_read);
    free(magic_read);
  }
  char* buf = NULL;
  {
    // buf should be *pn_layer, so 1.
    buf = realloc(buf, sizeof(uint32_t));
    RETURN_WHEN_FALSE(fread(buf, 1, 1, pf_in) == 1, free(buf), "Failed to read n_layer.\n");
    *pn_layer = decode_uint32_be(buf);
  }
  ALLOC_TO(p_dims, *pn_layer + 1);
  {
    // buf should be p_dims, and the length of p_dims is *pn_layer + 1, so *pn_layer + 1
    const size_t len_buf = *pn_layer + 1;
    buf = realloc(buf, len_buf);
    RETURN_WHEN_FALSE(fread(buf, len_buf, 1, pf_in), free(buf), "Failed to read dims.\n");
    for (size_t i_dim = 0; i_dim < *pn_layer + 1; ++i_dim) {
      IDX(p_dims, i_dim) = decode_uint32_be(buf + sizeof(u_int32_t) * i_dim);
    }
  }
  size_t n_elem_bias_all_tot = 0;
  size_t n_elem_weight_all_tot = 0;
  for (size_t i_layer = 0; i_layer < *pn_layer; ++i_layer) {
    n_elem_weight_all_tot += IDX(p_dims, i_layer) * IDX(p_dims, i_layer + 1);
    n_elem_bias_all_tot += IDX(p_dims, i_layer + 1);
  }
  init_param_weight(pa_weights, *pn_layer, p_dims, 0);
  init_param_bias(pp_biases, *pn_layer, p_dims, 0);
  {
    const size_t len_buf = sizeof(double) * n_elem_weight_all_tot;
    buf = realloc(buf, len_buf);
    RETURN_WHEN_FALSE(fread(buf, len_buf, 1, pf_in), free(buf), "Failed to read weights.\n");
    size_t n_elem_weight_prevall_tot = 0;
    for (size_t i_layer = 0; i_layer < *pn_layer; ++i_layer) {
      ARR2D_TYPE(double) * const p_weight_now= PIDX(pa_weights, i_layer);
      const size_t n_elem_weight_now_tot = p_weight_now->dim0 * p_weight_now->dim1;
      for (size_t i_elem = 0; i_elem < n_elem_weight_now_tot; ++i_elem) {
        IDX(p_weight_now->p_data, i_elem) = IEE754_binary64_decode(buf + sizeof(double) * (n_elem_weight_prevall_tot + i_elem));
      }
      n_elem_weight_prevall_tot += n_elem_weight_now_tot;
    }
  }
  {
    const size_t len_buf = sizeof(double) * n_elem_bias_all_tot;
    buf = realloc(buf, len_buf);
    RETURN_WHEN_FALSE(fread(buf, len_buf, 1, pf_in), free(buf), "Failed to read biases.\n");
    size_t n_elem_bias_prevall_tot = 0;
    for (size_t i_layer = 0; i_layer < *pn_layer; ++i_layer) {
      double* const p_bias_now = IDX(pp_biases, i_layer);
      const size_t n_elem_bias_now = IDX(p_dims, i_layer + 1);
      for (size_t i_elem = 0; i_elem < n_elem_bias_now; ++i_elem) {
        IDX(p_bias_now, i_elem) = IEE754_binary64_decode(buf + sizeof(double) * (n_elem_bias_prevall_tot + i_elem));
      }
      n_elem_bias_prevall_tot += n_elem_bias_now;
    } 
  }
  free(buf);
  return 1;
}
