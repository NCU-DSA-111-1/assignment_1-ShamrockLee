#ifndef LIBNN_H
#define LIBNN_H

#include <stdbool.h>
#include <stddef.h>

#include "arr.h"

inline double logistic_shifted(const double x);

void init_param_weight(
    ARR2D_TYPE(double) * pa_weights,
    const size_t n_layer,
    const size_t* p_dims,
    const bool zeroize);

void free_param_weight(
    ARR2D_TYPE(double) * const pa_weights,
    const size_t n_layer);

void init_param_bias(
    double** pp_biases,
    const size_t n_layer,
    const size_t* p_dims,
    const bool zeroize);

void init_param_bias_skiplast(
    double** pp_biases,
    double* const p_output,
    const size_t n_layer,
    const size_t* p_dims,
    const bool zeroize);

void free_param_bias(
    double** const pp_biases,
    const size_t n_layer);

void init_dims_from_weights(
    size_t* p_dims,
    const size_t n_layer,
    const ARR2D_TYPE(double) * pa_weights);

void forward(
    double (*const fn_activation)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double* const* const pp_layers,
    double* const* const pp_layers_activated);

void backward(
    double (*const fn_diff_activation)(const double),
    double (*const fn_diff_loss)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const p_inputs,
    const double* const* const pp_layers,
    const double* const* const pp_layers_activated,
    ARR2D_TYPE(double) * const pa_grad_weights,
    double* const* const pp_grad_biases);

int train(
    double (*const fn_activation)(const double),
    double (*const fn_diff_activation)(const double),
    double (*const fn_loss)(const double),
    double (*const fn_diff_loss)(const double),
    const size_t n_layer,
    const size_t n_data_train,
    const ARR2D_TYPE(double) * pa_inputs,
    const ARR2D_TYPE(double) * pa_answers,
    const double learning_rate,
    ARR2D_TYPE(double) * const pa_weights,
    double* const* const pp_biases);

void predict_raw(
    double (*const fn_activation)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double* const p_output);

#endif /* LIBNN_H */
