#ifndef LIBNN_IO_H
#define LIBNN_IO_H

#include <stdint.h>
#include <stdio.h>

#include "idx.h"
#include "arr.h"

#define NN_FILE_MAGIC "NN4D"
#define LEN_NN_FILE_MAGIC 4

/**
 * This header declares functions to read and write
 * NN models used by libnn.h
 * to and from a big-endian, size-integer-is-4-byte-long, IEE754_double file.
 **/

int write_model_data(
    FILE* pf_out,
    const size_t n_layer,
    const size_t* const p_dims,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases);

int read_model_data(
  FILE* pf_in,
  size_t* pn_layer,
  size_t* p_dims,
  ARR2D_TYPE(double) * pa_weights,
  double** pp_biases);

#endif /* LIBNN_IO_H */
