#ifndef LIBNN_H
#define LIBNN_H

#include <stdbool.h>
#include <stddef.h>

#include "arr.h"

double logistic_shifted(const double x);
double diff_logistic_shifted(const double x);

double square_error(const double x, const double y);
double diff_square_error(const double x, const double y);

void init_param_weight(
    ARR2D_TYPE(double) * pa_weights,
    const size_t n_layer,
    const size_t* const p_dims,
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
    double (*const fn_diff_loss)(const double, const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const p_inputs,
    const double* const p_answers,
    const double* const* const pp_layers,
    const double* const* const pp_layers_activated,
    ARR2D_TYPE(double) * const pa_grad_weights,
    double* const* const pp_grad_biases);

int batch_train(
    double (*const fn_activation)(const double),
    double (*const fn_diff_activation)(const double),
    double (*const fn_loss)(const double, const double),
    double (*const fn_diff_loss)(const double, const double),
    const size_t n_layer,
    const size_t n_data_train,
    const ARR2D_TYPE(double) * pa_inputs,
    const ARR2D_TYPE(double) * pa_answers,
    const double learning_rate,
    const double max_loss,
    const double max_delta_loss,
    const size_t max_retrial,
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
