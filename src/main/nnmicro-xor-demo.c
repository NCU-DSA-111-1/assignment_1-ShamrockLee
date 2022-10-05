#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arr.h"
#include "debug.h"
#include "idx.h"
#include "libnn.h"

/**
 * This project is a POC of the main project.
 * Many variables are hard-coded, including the number of input
 * and the size of hidden layer.
 *
 * The "main project", which offers
 * plenty of command-line parameters and model-saving features,
 * is too large to finish debugging within two weeks.
 * See nn-train-xor.c and the files it depends on for details.
 *
 * Above is the lesson learned after more than 80 hours of work.
 **/

// This one is always valid for xor.
static const size_t n_output = 1;

// These are hard to specify from the command line,
// so I just choose them for the user.
static double (*const fn_activation)(const double) = logistic_shifted;
static double (*const fn_diff_activation)(const double) = diff_logistic_shifted;
static double (*const fn_loss)(const double, const double) = square_error;
static double (*const fn_diff_loss)(const double, const double) = diff_square_error;

// These are hard-coded due to time constaint.
static const size_t n_input = 2;
static const size_t n_hidden = 1;
static const size_t n_layer = n_hidden + n_output;

static const double learning_rate = 0.001;
static const double max_loss = learning_rate * 30;
static const double max_delta_loss = learning_rate * 5;
static const size_t max_retrial = 10;

static const bool enable_rand_weight = true;
static const bool enable_rand_bias = true;
static const unsigned int rand_seed = 0;

int main(int argc, char** argv) {
  size_t* p_dims = NULL;
  // Hard-coded layer distribution
  ALLOC_TO(p_dims, n_layer + 1);
  p_dims[0] = n_input;
  p_dims[1] = n_input;
  p_dims[2] = n_output;

  // init_param_weight and init_param_biase is still segfaulting
  // so init manually.
  ARR2D_TYPE(double)* pa_weights = NULL;
  ARR2D_TYPE(double)* pa_grad_weights = NULL, *pa_grad_weights_avg = NULL;
  ALLOC_TO(pa_weights, n_layer);
  ALLOC_TO(pa_grad_weights, n_layer);
  ALLOC_TO(pa_grad_weights_avg, n_layer);
  pa_weights[0].dim1 = pa_grad_weights[0].dim1 = pa_grad_weights_avg[0].dim1 = n_input;
  pa_weights[1].dim1 = pa_grad_weights[1].dim1 = pa_grad_weights_avg[1].dim1 = pa_weights[0].dim0 = pa_grad_weights[0].dim0 = pa_grad_weights_avg[0].dim0 = n_input;
  pa_weights[1].dim0 = pa_grad_weights[1].dim0 = pa_grad_weights_avg[1].dim0 = n_output;
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    ARR2D_ALLOC(pa_weights + i_layer);
    ARR2D_ALLOC(pa_grad_weights + i_layer);
    ARR2D_ALLOC(pa_grad_weights_avg + i_layer);
    FILLN((pa_weights + i_layer)->p_data, -1., ARR2D_NELEM(pa_weights + i_layer));
    // FILLN((pa_weights + i_layer)->p_data, 1. / p_dims[i_layer], ARR2D_NELEM(pa_weights + i_layer));
    FILLN((pa_grad_weights + i_layer)->p_data, 0., ARR2D_NELEM(pa_weights + i_layer));
    FILLN((pa_grad_weights_avg + i_layer)->p_data, 0., ARR2D_NELEM(pa_weights + i_layer));
  }
  double** pp_biases = NULL;
  double **pp_layers = NULL, **pp_layers_activated = NULL, **pp_grad_biases = NULL, **pp_grad_biases_avg = NULL;
  ALLOC_TO(pp_biases, n_layer);
  ALLOC_TO(pp_layers, n_layer);
  ALLOC_TO(pp_layers_activated, n_layer);
  ALLOC_TO(pp_grad_biases, n_layer);
  ALLOC_TO(pp_grad_biases_avg, n_layer);
  for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
    CALLOC_TO(pp_biases[i_layer], p_dims[i_layer + 1]);
    CALLOC_TO(pp_layers[i_layer], p_dims[i_layer + 1]);
    CALLOC_TO(pp_layers_activated[i_layer], p_dims[i_layer + 1]);
    CALLOC_TO(pp_grad_biases[i_layer], p_dims[i_layer + 1]);
    CALLOC_TO(pp_grad_biases_avg[i_layer], p_dims[i_layer + 1]);
  }
  if (enable_rand_weight) {
    srand(rand_seed);
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      ARR2D_TYPE(double)* const pa_weight_now = PIDX(pa_weights, i_layer);
      const size_t n_elem_tot = pa_weight_now->dim0 * pa_weight_now->dim1;
      for (size_t i_elem = 0; i_elem < n_elem_tot; ++i_elem) {
        IDX(pa_weight_now->p_data, i_elem) += (rand() & 1) ? learning_rate : -learning_rate;
      }
    }
  }
  if (enable_rand_bias) {
    srand(rand_seed + 1);
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      for (size_t i_elem = 0; i_elem < IDX(p_dims, i_layer + 1); ++i_elem) {
        IDX(IDX(pp_biases, i_layer), i_elem) += (rand() & 1) ? learning_rate : -learning_rate;
      }
    }
  }

  // Prepare the train data
  // Simplify from nn-train-xor.c
  const size_t n_data_train = 1 << n_input;
  ARR2D_TYPE(double)
  a_inputs = {n_input, n_data_train, NULL};
  ARR2D_TYPE(double)
  a_answers = {n_output, n_data_train, NULL};
  ARR2D_ALLOC(&a_inputs);
  FILLN(a_inputs.p_data, 0., ARR2D_NELEM(&a_inputs));
  ARR2D_ALLOC(&a_answers);
  MEMSETN(a_answers.p_data, 0, a_answers.dim0 * a_answers.dim1);
  for (uint_fast64_t i_sample = 0; i_sample < n_data_train; ++i_sample) {
    bool bit_all = 0;
    for (size_t i_elem = 0; i_elem < n_input; ++i_elem) {
      const bool bit_now = (i_sample >> i_elem) & 0x1;
      *ARR2D_PIDX(&a_inputs, i_elem, (size_t)i_sample) = (double)(bit_now);
      bit_all ^= bit_now;
    };
    *ARR2D_PIDX(&a_answers, 0, (size_t)i_sample) = (double)(bit_all);
  }

  // Train
  // Simplify from the batch_train definition from libnn.c
  // but avoid calling init_param_*.
  {
    const size_t len_output = n_output;
    const size_t deno_loss = len_output * n_data_train;
    double loss_now = __DBL_MAX__, loss_prev = __DBL_MAX__, loss_delta_neg = __DBL_MAX__;
    size_t retrial = 0;
    do {
      const ARR2D_TYPE(double)* const pa_inputs = &a_inputs;
      const ARR2D_TYPE(double)* const pa_answers = &a_answers;
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
            *ARR2D_PIDX(PIDX(pa_weights, i_layer), i_elem0, i_elem1) -= *ARR2D_PIDX(PIDX(pa_grad_weights_avg, i_layer), i_elem0, i_elem1) * learning_rate;
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
          break;
        }
      } else {
        retrial = 0;
      }
    } while (loss_now > max_loss || loss_delta_neg > max_delta_loss);
  }

  // Predict
  // There's only two outputs, so just show the prediction.

  printf("Input\tRaw output\tOutput\tAnswer\n");
  for (size_t i_sample = 0; i_sample < n_data_train; ++i_sample) {
    const double* const p_inputs = ARR2D_PIDX(&a_inputs, 0, i_sample);
    const double const answer = *ARR2D_PIDX(&a_inputs, 0, i_sample);
    // Simplify from the predict_raw definition from libnn.c
    // but avoid calling init_param_*.
    forward(fn_activation, n_layer, pa_weights, pp_biases, p_inputs, pp_layers, pp_layers_activated);
    double output_raw = pp_layers_activated[n_layer - 1][0];
    printf("(%d, %d)\t%f\t%d\t%d\n", p_inputs[0] >= 0.5, p_inputs[1] >= 0.5, output_raw, output_raw >= 0.5, answer >= 0.5);
  }

  // Finalize
  // free_param_* should work.
  free_param_weight(pa_weights, n_hidden);
  free_param_weight(pa_grad_weights, n_hidden);
  free_param_weight(pa_grad_weights_avg, n_hidden);
  free_param_bias(pp_biases, n_layer);
  free_param_bias(pp_layers, n_layer);
  free_param_bias(pp_layers_activated, n_layer);
  free_param_bias(pp_grad_biases, n_layer);
  free_param_bias(pp_grad_biases_avg, n_layer);
  free(p_dims);
  return 0;
}
