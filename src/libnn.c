#include "libnn.h"

#include <assert.h>
#include <math.h>

double logistic(const double x) {
  return (tanh(x * 0.5) + 1.) * 0.5;
}

void forward(
    double (*const fn_activation)(const double),
    double (*const fn_loss)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double** pp_layers,
    double** pp_layers_activated) {
  // The "previous activated layer"
  // Set the "previous activated layer" to the inputs
  double* p_layer_activated_prev = *pp_layers_activated;
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    // Filling IDX(pp_layers, i_layer)
    // Multiply the weight
    ARR2D_LPROD(PIDX(pa_weights, i_layer), IDX(pp_layers, i_layer), p_layer_activated_prev);
    // Plus the bias
    for (size_t i_elem = 0; i_elem < PIDX(pa_weights, i_layer)->dim0; ++i_elem) {
      IDX(IDX(pp_layers, i_layer), i_elem) += IDX(IDX(pp_biases, i_layer), i_elem);
    }
    // IDX(pp_layers, i_layer) filled
    // Filling IDX(pp_layers_activated, i_layer)
    // Apply the activation function
    for (size_t i_elem = 0; i_elem < PIDX(pa_weights, i_layer)->dim0; ++i_elem) {
      IDX(IDX(pp_layers_activated, i_layer), i_elem) = fn_activation(IDX(IDX(pp_layers, i_layer), i_elem));
    }
    // Set the "previous activated layer" to this activated layer for the next iteration
    p_layer_activated_prev = IDX(pp_layers_activated, i_layer);
  }
}

void backward(
    double (*const fn_diff_activation)(const double),
    double (*const fn_diff_loss)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const p_inputs,
    const double* const* const pp_layers,
    const double* const* const pp_layers_activated,
    ARR2D_TYPE(double) * pa_grad_weights,
    double** pp_grad_biases) {
  for (size_t i_elem = 0; i_elem < PIDX(pa_weights, n_layer - 1)->dim0; ++i_elem) {
    IDX(IDX(pp_grad_biases, n_layer - 1), i_elem) = fn_diff_loss(IDX(IDX(pp_layers_activated, n_layer - 1), i_elem)) * fn_diff_activation(IDX(IDX(pp_layers_activated, n_layer - 1), i_elem));
  }
  for (ptrdiff_t i_layer = n_layer - 1; i_layer < 0; --i_layer) {
    for (size_t i_elem1 = 0; i_elem1 < PIDX(pa_weights, i_layer)->dim1; ++i_elem1) {
      for (size_t i_elem0 = 0; i_elem0 < PIDX(pa_grad_weights, i_layer)->dim0; ++i_elem0) {
        *ARR2D_PIDX(pa_grad_weights, i_elem0, i_elem1) = IDX(IDX(pp_grad_biases, i_layer), i_elem0) * IDX((i_layer > 0) ? IDX(pp_layers_activated, i_layer - 1) : p_inputs, i_elem1);
      }
    };
    if (i_layer > 0) {
      double* p_grad_biases_prev = IDX(pp_grad_biases, i_layer - 1);
      ARR2D_RPROD(PIDX(pa_grad_weights, i_layer), p_grad_biases_prev, IDX(pp_grad_biases, i_layer));
      for (size_t i_elem = 0; i_elem < PIDX(pa_weights, i_layer - 1)->dim0; ++i_elem) {
        IDX(p_grad_biases_prev, i_elem) *= fn_diff_activation(IDX(IDX(pp_layers, i_layer - 1), i_elem));
      }
    }
  }
}

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
    ARR2D_TYPE(double) * pa_weights,
    double** pp_biases) {
  for (size_t i_layer = 0; i_layer < n_layer - 1; ++i_layer) {
    assert(PIDX(pa_weights, i_layer)->dim0 == PIDX(pa_weights, i_layer)->dim1);
  }
  double **pp_layers, **pp_layers_activated, **pp_grad_biases, **pp_grad_biases_avg;
  ALLOC_TO(pp_layers, n_layer);
  ALLOC_TO(pp_layers_activated, n_layer);
  ALLOC_TO(pp_grad_biases, n_layer);
  ALLOC_TO(pp_grad_biases_avg, n_layer);
  ARR2D_TYPE(double) * pa_grad_weights, *pa_grad_weights_avg;
  ALLOC_TO(pa_grad_weights, n_layer);
  ALLOC_TO(pa_grad_weights_avg, n_layer);
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    ALLOC_TO(IDX(pp_layers, i_layer), PIDX(pa_weights, i_layer)->dim0);
    ALLOC_TO(IDX(pp_layers_activated, i_layer), PIDX(pa_weights, i_layer)->dim0);
    ALLOC_TO(IDX(pp_grad_biases, i_layer), PIDX(pa_weights, i_layer)->dim0);
    CALLOC_TO(IDX(pp_grad_biases_avg, i_layer), PIDX(pa_weights, i_layer)->dim0);
    PIDX(pa_grad_weights, i_layer)->dim0 = PIDX(pa_grad_weights, i_layer)->dim0 = PIDX(pa_weights, i_layer)->dim0;
    PIDX(pa_grad_weights, i_layer)->dim1 = PIDX(pa_grad_weights, i_layer)->dim1 = PIDX(pa_weights, i_layer)->dim1;
    ARR2D_ALLOC(PIDX(pa_grad_weights, i_layer));
    ARR2D_ALLOC(PIDX(pa_grad_weights_avg, i_layer));
    MEMSETN(pa_grad_weights_avg, 0, pa_grad_weights_avg->dim0 * pa_grad_weights_avg->dim1);
  }
  const size_t len_input = pa_weights->dim1;
  for (size_t i_train = 0; i_train < n_data_train; ++i_train) {
    const double* p_inputs_current = ARR2D_PIDX(pa_inputs, 0, i_train);
    forward(fn_activation, fn_loss, n_layer, pa_weights, pp_biases, p_inputs_current, pp_layers, pp_layers_activated);
    backward(fn_diff_activation, fn_diff_loss, n_layer, pa_weights, p_inputs_current, pp_layers, pp_layers_activated, pa_grad_weights, pp_grad_biases);
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      for (size_t i_elem1 = 0; i_elem1 < PIDX(pa_weights, i_layer)->dim1; ++i_elem1) {
        for (size_t i_elem0 = 0; i_elem0 < PIDX(pa_weights, i_layer)->dim0; ++i_elem0) {
          *ARR2D_PIDX(PIDX(pa_grad_weights_avg, i_layer), i_elem0, i_elem1) += *ARR2D_PIDX(PIDX(pa_grad_weights, i_layer), i_elem0, i_elem1) / n_data_train;
        }
      }
      for (size_t i_elem = 0; i_elem < PIDX(pa_weights, i_layer)->dim0; ++i_elem) {
        IDX(IDX(pp_grad_biases_avg, i_layer), i_elem) += IDX(IDX(pp_grad_biases, i_layer), i_elem) / n_data_train;
      }
    }
  }
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    for (size_t i_elem1 = 0; i_elem1 < PIDX(pa_weights, i_layer)->dim1; ++i_elem1) {
      for (size_t i_elem0 = 0; i_elem0 < PIDX(pa_weights, i_layer)->dim0; ++i_elem0) {
        *ARR2D_PIDX(PIDX(pa_weights, i_layer), i_elem0, i_elem1) += *ARR2D_PIDX(PIDX(pa_grad_weights_avg, i_layer), i_elem0, i_elem1) * learning_rate;
      }
    }
    for (size_t i_elem = 0; i_elem < PIDX(pa_weights, i_layer)->dim0; ++i_elem) {
      IDX(IDX(pp_biases, i_layer), i_elem) += IDX(IDX(pp_grad_biases_avg, i_layer), i_elem) * learning_rate;
    }
  }
}
