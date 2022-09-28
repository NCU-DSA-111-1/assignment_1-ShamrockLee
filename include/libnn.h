#ifndef LIBNN_H
#define LIBNN_H

#include <stddef.h>

#include "arr.h"

void forward(
    double (*const fn_activation)(const double),
    double (*const fn_loss)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double** pp_layers,
    double** pp_layers_activated);

void backward(
    double (*const fn_diff_activation)(const double),
    double (*const fn_diff_loss)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const p_inputs,
    const double* const* const pp_layers,
    const double* const* const pp_layers_activated,
    ARR2D_TYPE(double) * pa_grad_weights,
    double** pp_grad_biases);

#endif /* LIBNN_H */
