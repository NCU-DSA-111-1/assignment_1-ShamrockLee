#include "libnn.h"

#include <assert.h>
#include <math.h>

#include <debug.h>

double logistic_shifted(const double x) {
  return (tanh((x - 0.5) * 0.5) + 1.) * 0.5;
}

double diff_logistic_shifted(const double x) {
  return logistic_shifted(x) * (1. - logistic_shifted(x));
}

double square_error(const double x, const double y) {
  return (x - y) * (x - y);
}

double diff_square_error(const double x, const double y) {
  return (x - y) + (x - y);
}

void init_param_weight(
    ARR2D_TYPE(double) * pa_weights,
    const size_t n_layer,
    const size_t* const p_dims,
    const bool zeroize) {
  DEBUG_PRINTF(1, "p_dims: %p\n", p_dims);
  for (size_t i_dim = 0; i_dim < n_layer + 1; ++i_dim) {
    DEBUG_PRINTF(1, "PIDX(p_dims, %zu): %p, IDX(p_dims, %zu): %zu\n", i_dim, PIDX(p_dims, i_dim), i_dim, IDX(p_dims, i_dim));
  }
  ALLOC_TO(pa_weights, n_layer);
  // pa_weights = (typeof(pa_weights))malloc(n_layer * sizeof(*pa_weights));
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    ARR2D_TYPE(double)* pa_weight_current = PIDX(pa_weights, i_layer);
    pa_weight_current->dim1 = IDX(p_dims, i_layer);
    pa_weight_current->dim0 = IDX(p_dims, i_layer + 1);
    DEBUG_PRINTF(1, "i_layer: %zu, pa_weight_current dim0: %zu, dim1: %zu\n", i_layer, pa_weight_current->dim0, pa_weight_current->dim1);
    ARR2D_ALLOC(pa_weight_current);
    if (zeroize) {
      ARR2D_REPEAT(pa_weight_current, 0.);
    }
  }
}

void free_param_weight(
    ARR2D_TYPE(double) * const pa_weights,
    const size_t n_layer) {
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer)
    ARR2D_FREE(PIDX(pa_weights, i_layer));
  free(pa_weights);
}

void init_param_bias(
    double** pp_biases,
    const size_t n_layer,
    const size_t* p_dims,
    const bool zeroize) {
  ALLOC_TO(pp_biases, n_layer);
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    ALLOC_TO(IDX(pp_biases, i_layer), IDX(p_dims, i_layer + 1));
    if (zeroize) {
      MEMSETN(IDX(pp_biases, i_layer), 0, IDX(p_dims, i_layer + 1));
    }
  }
}

void init_param_bias_skiplast(
    double** pp_biases,
    double* const p_output,
    const size_t n_layer,
    const size_t* p_dims,
    const bool zeroize) {
  ALLOC_TO(pp_biases, n_layer);
  for (size_t i_layer = 0; i_layer < n_layer < 1; ++i_layer) {
    ALLOC_TO(IDX(pp_biases, i_layer), IDX(p_dims, i_layer + 1));
    if (zeroize) {
      MEMSETN(IDX(pp_biases, i_layer), 0, IDX(p_dims, i_layer + 1));
    }
  }
  IDX(pp_biases, n_layer - 1) = p_output;
  if (zeroize) {
    MEMSETN(p_output, 0, IDX(p_dims, n_layer + 1));
  }
}

void free_param_bias(
    double** const pp_biases,
    const size_t n_layer) {
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer)
    free(IDX(pp_biases, i_layer));
  free(pp_biases);
}

void free_param_bias_skiplast(
    double** const pp_biases,
    const size_t n_layer) {
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    free(IDX(pp_biases, i_layer));
  }
  free(pp_biases);
}

void init_dims_from_weights(
    size_t* p_dims,
    const size_t n_layer,
    const ARR2D_TYPE(double) * pa_weights) {
  ALLOC_TO(p_dims, n_layer + 1);
  *p_dims = pa_weights->dim1;
  for (size_t i_layer = 0; i_layer < n_layer - 1; ++i_layer) {
    if (DEBUG_LEVEL > 0)
      assert(PIDX(pa_weights, i_layer)->dim0 == PIDX(pa_weights, i_layer)->dim1);
    IDX(p_dims, i_layer + 1) = PIDX(pa_weights, i_layer)->dim0;
  }
}

void forward(
    double (*const fn_activation)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double* const* const pp_layers,
    double* const* const pp_layers_activated) {
  // The "previous activated layer"
  // Set the "previous activated layer" to the inputs
  double* p_layer_activated_prev = *pp_layers_activated;
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    // Filling IDX(pp_layers, i_layer)
    // Multiply the weight and plus the bias
    ARR2D_LPROD_BIASED(PIDX(pa_weights, i_layer), IDX(pp_layers, i_layer), p_layer_activated_prev, IDX(pp_biases, i_layer));
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
    double (*const fn_diff_loss)(const double, const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const p_inputs,
    const double* const p_answers,
    const double* const* const pp_layers,
    const double* const* const pp_layers_activated,
    ARR2D_TYPE(double) * pa_grad_weights,
    double* const* const pp_grad_biases) {
  for (size_t i_elem = 0; i_elem < PIDX(pa_weights, n_layer - 1)->dim0; ++i_elem) {
    IDX(IDX(pp_grad_biases, n_layer - 1), i_elem) = fn_diff_loss(IDX(IDX(pp_layers_activated, n_layer - 1), i_elem), IDX(p_answers, i_elem)) * fn_diff_activation(IDX(IDX(pp_layers_activated, n_layer - 1), i_elem));
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
    double* const* const pp_biases) {
  size_t* p_dims;
  init_dims_from_weights(p_dims, n_layer, pa_weights);
  if (DEBUG_LEVEL > 0) {
    assert(pa_inputs->dim0 == PIDX(pa_weights, 0)->dim1);
    assert(pa_answers->dim0 == PIDX(pa_weights, n_layer - 1)->dim0);
  }
  DEBUG_PRINTF(1, "n_layer: %zu\n", n_layer);
  DEBUG_PRINTF(1, "shape of the index 0 a_weight: (%zu, %zu)\n", PIDX(pa_weights, 0)->dim0, PIDX(pa_weights, 0)->dim1);
  DEBUG_PRINTF(1, "shape of the index 1 a_weight: (%zu, %zu)\n", PIDX(pa_weights, 1)->dim0, PIDX(pa_weights, 1)->dim1);
  const size_t len_output = PIDX(pa_weights, n_layer - 1)->dim0;
  const size_t deno_loss = len_output * n_data_train;
  double **pp_layers, **pp_layers_activated, **pp_grad_biases, **pp_grad_biases_avg;
  init_param_bias(pp_layers, n_layer, p_dims, 0);
  init_param_bias(pp_layers_activated, n_layer, p_dims, 0);
  init_param_bias(pp_grad_biases, n_layer, p_dims, 0);
  init_param_bias(pp_grad_biases_avg, n_layer, p_dims, 1);
  ARR2D_TYPE(double) * pa_grad_weights, *pa_grad_weights_avg;
  init_param_weight(pa_grad_weights, n_layer, p_dims, 0);
  init_param_weight(pa_grad_weights_avg, n_layer, p_dims, 1);
  double loss_now = __DBL_MAX__, loss_prev = __DBL_MAX__, loss_delta_neg = __DBL_MAX__;
  size_t retrial = 0;
  do {
    loss_prev = loss_now;
    loss_now = 0;
    for (size_t i_train = 0; i_train < n_data_train; ++i_train) {
      const double* p_inputs_current = ARR2D_PIDX(pa_inputs, 0, i_train);
      forward(fn_activation, n_layer, pa_weights, pp_biases, p_inputs_current, pp_layers, pp_layers_activated);
      const double* const p_output = IDX(pp_layers_activated, n_layer - 1);
      for (size_t i_elem = 0; i_elem < len_output; ++i_elem) {
        loss_now += fn_loss(IDX(p_output, i_elem), *ARR2D_PIDX(pa_answers, i_elem, i_train)) / deno_loss;
      }
      backward(fn_diff_activation, fn_diff_loss, n_layer, pa_weights, p_inputs_current, ARR2D_PIDX(pa_answers, 0, i_train), pp_layers, pp_layers_activated, pa_grad_weights, pp_grad_biases);
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
        IDX(IDX(pp_biases, i_layer), i_elem) -= IDX(IDX(pp_grad_biases_avg, i_layer), i_elem) * learning_rate;
      }
    }
    loss_delta_neg = loss_prev - loss_now;
    fprintf(stderr, "loss_now, loss_delta_neg: %g, %g\n", loss_now, loss_delta_neg);
    if (loss_delta_neg <= 0) {
      if (++retrial >= max_retrial) {
        return 0;
      }
    } else {
      retrial = 0;
    }
  } while (loss_now > max_loss && loss_delta_neg > max_delta_loss);
  free_param_bias(pp_layers, n_layer);
  free_param_bias(pp_layers_activated, n_layer);
  free_param_bias(pp_grad_biases, n_layer);
  free_param_bias(pp_grad_biases_avg, n_layer);
  free_param_weight(pa_grad_weights, n_layer);
  free_param_weight(pa_grad_weights_avg, n_layer);
  free(p_dims);
  return 1;
}

void predict_raw(
    double (*const fn_activation)(const double),
    const size_t n_layer,
    const ARR2D_TYPE(double) * const pa_weights,
    const double* const* const pp_biases,
    const double* const p_inputs,
    double* const p_output) {
  size_t* p_dims;
  init_dims_from_weights(p_dims, n_layer, pa_weights);
  double **pp_layers, **pp_layers_activated;
  init_param_bias(pp_layers, n_layer, p_dims, 0);
  init_param_bias_skiplast(pp_layers_activated, p_output, n_layer, p_dims, 0);
  forward(fn_activation, n_layer, pa_weights, pp_biases, p_inputs, pp_layers, pp_layers_activated);
  free_param_bias(pp_layers, n_layer);
  free_param_bias_skiplast(pp_layers_activated, n_layer);
}
