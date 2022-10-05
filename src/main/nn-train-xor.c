#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h> // Unix standard file descriptor interface
// #include <unistd.h>

#include "arr.h"
#include "idx.h"
#include "libnn.h"
#include "libnn_io.h"
#include "skeeto_optparse.h"

static double (*const fn_activation_xor)(const double) = logistic_shifted;
static double (*const fn_diff_activation_xor)(const double) = diff_logistic_shifted;
static double (*const fn_loss_xor)(const double, const double) = square_error;
static double (*const fn_diff_loss_xor)(const double, const double) = diff_square_error;
static const size_t n_output = 1;

int batch_train_xor_to_file(
    FILE* const pf_out,
    const size_t n_layer,
    size_t* p_dims,
    const double learning_rate,
    const double max_loss,
    const double max_delta_loss,
    const size_t max_retrial,
    const bool enable_rand_weight,
    const bool enable_rand_bias,
    const unsigned int rand_seed) {
  DEBUG_PRINTF(1, "p_dims: %p\n", p_dims);
  for (size_t i_dim = 0; i_dim < n_layer + 1; ++i_dim) {
    DEBUG_PRINTF(1, "PIDX(p_dims, %zu): %p, IDX(p_dims, %zu): %zu\n", i_dim, PIDX(p_dims, i_dim), i_dim, IDX(p_dims, i_dim));
  }
  const size_t n_input = *p_dims;
  const size_t n_data_train = 1 << n_input;
  ARR2D_TYPE(double) a_inputs = {n_input, n_data_train, NULL};
  ARR2D_TYPE(double) a_answers = {n_output, n_data_train, NULL};
  ARR2D_ALLOC(&a_inputs);
  MEMSETN(a_inputs.p_data, 0, a_inputs.dim0 * a_inputs.dim1);
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
  ARR2D_TYPE(double) * pa_weights;
  double** pp_biases;
  srand(rand_seed);
  init_param_weight(pa_weights, n_layer, p_dims, 1);
  init_param_bias(pp_biases, n_layer, p_dims, 1);
  if (enable_rand_weight) {
    srand(rand_seed);
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      ARR2D_TYPE(double)* const pa_weight_now = PIDX(pa_weights, i_layer);
      const size_t n_elem_tot = pa_weight_now->dim0 * pa_weight_now->dim1;
      for (size_t i_elem = 0; i_elem < n_elem_tot; ++i_elem) {
        IDX(pa_weight_now->p_data, i_elem) = (rand() & 1) ? learning_rate : -learning_rate;
      }
    }
  }
  if (enable_rand_bias) {
    srand(rand_seed + 1);
    for (size_t i_layer = 0; i_layer < n_layer; ++i_layer) {
      for (size_t i_elem = 0; i_elem < IDX(p_dims, i_layer + 1); ++i_elem) {
        IDX(IDX(pp_biases, i_layer), i_elem) = (rand() & 1) ? learning_rate : -learning_rate;
      }
    }
  }
  const int ret = batch_train(fn_activation_xor, fn_diff_activation_xor, fn_loss_xor, fn_diff_loss_xor, n_layer, n_data_train, &a_inputs, &a_answers, learning_rate, max_loss, max_delta_loss, max_retrial, pa_weights, pp_biases);
  write_model_data(pf_out, n_layer, p_dims, pa_weights, pp_biases);
  free_param_bias(pp_biases, n_layer);
  free_param_weight(pa_weights, n_layer);
  return ret;
}

int main(int argc, char** argv) {
  size_t n_in = 2;
  size_t n_hidden = 1;
  const char* str_type = "rectangle";
  const char* str_shape = ":";
  const char* str_output = "-";
  double learning_rate = 0.0001;
  double max_loss = learning_rate * 30;
  double max_delta_loss = learning_rate * 5;
  size_t max_retrial = 10;
  bool enable_rand = false;
  unsigned int rand_seed = 0;
  
  struct optparse_long* longopts;
  // CALLOC_TO(&longopts, 100);  // 100 is chosen for a sufficiently large space to store the argument definitions
  longopts = calloc(100, sizeof(*longopts));
  {
    size_t idx_argdef = 0;
    IDX(longopts, idx_argdef++) = (struct optparse_long){"help", 'h', OPTPARSE_NONE};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"n-input", 'I', OPTPARSE_REQUIRED};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"n-hidden", 'H', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"type", 'T', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"custom-shape", 'S', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"output", 'o', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"learning-rate", 'R', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"max-loss", 'L', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"max-delta-loss", 'D', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"max-retrial", 't', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"enable-rand", 'r', OPTPARSE_NONE};
    IDX(longopts, idx_argdef++) = (struct optparse_long){"rand-seed", 's', OPTPARSE_OPTIONAL};
    IDX(longopts, idx_argdef++) = (struct optparse_long){NULL, '\0', OPTPARSE_NONE}; // NULL termination
  }
  struct optparse options;
  optparse_init(&options, argv);
  int option;
  while ((option = optparse_long(&options, longopts, NULL)) != -1) {
    switch (option) {
      case 'h':
        fprintf(stderr,
                "Description:\n"
                "nn-xor-train -I NUM_INPUT [-L NUM_HIDDEN_LAYER] [-T TYPE_HIDDEN_LAYER] [ -S SHAPE_HIDDEN_LAYER ] -o model_data.bin\n"
                "");
        fprintf(stderr,
                "  -h --help\tDisplay this help message and exit.\n");
        fprintf(stderr,
                "  -I --n-input\tThe number of inputs.\n");
        fprintf(stderr,
                "  -H --n-hidden\tThe number of hidden layers. Default to %zu.\n",
                n_hidden);
        fprintf(stderr,
                "  -T --type\tThe type of hidden layers. It should be either custom, triangle or rectangle. Default to %s.\n",
                str_type);
        fprintf(stderr,
                "  -S --custom-shape\tThe length of each hidden layers seperated by colons (:). \n"
                "\tRequired when -T custom and assume -T custom.\n"
                "\tExpect --n-hidden");
        fprintf(stderr,
                "  -o --output\tThe path to the file to store the trained data. -o - means the standard output. Default to %s.\n",
                str_output);
        fprintf(stderr,
                "  -R --learning-rate\tThe learning rate. Default to %F.\n",
                learning_rate);
        fprintf(stderr,
                "  -L --max-loss\tThe required upper bound of loss. Default to %F.\n",
                max_loss);
        fprintf(stderr,
                "  -D --max-delta-loss\tThe required upper bound of loss reduction. Default to %F.\n",
                max_delta_loss);
        fprintf(stderr,
                "  -t --max-retrial\tThe maximum number of retrial when the loss doesn't reduce. Default to %zu.\n",
                max_retrial);
        fprintf(stderr,
                "  -r --enable-rand\tAdd random fluctuation to the initial parameters.\n");
        fprintf(stderr,
                "  -s --rand-seed\tThe pseudorandom seed to use when --enable-rand. Default to %u.\n",
                rand_seed);
        return 0;
        break;
      case 'I':
        n_in = (size_t)atol(options.optarg);
        break;
      case 'H':
        n_hidden = (size_t)atol(options.optarg);
        break;
      case 'T':
        str_type = options.optarg;
        break;
      case 'S':
        str_shape = options.optarg;
        str_type = "custom";
        break;
      case 'o':
        str_output = options.optarg;
        break;
      case 'R':
        learning_rate = atof(options.optarg);
        break;
      case 'L':
        max_loss = atof(options.optarg);
        break;
      case 'D':
        max_delta_loss = atof(options.optarg);
        break;
      case 't':
        max_retrial = (size_t)atol(options.optarg);
        break;
      case 'r':
        enable_rand = true;
        break;
      case 's':
        rand_seed = (unsigned int)atol(options.optarg);
        break;
    }
  }
  optparse_final(&options);
  free(longopts);
  size_t* p_dims;
  if (n_in < 2) {
    fprintf(stderr, "error: Expect n_in >= 2; Got %zu.\n", n_in);
    return 1;
  }
  DEBUG_PRINTF(1, "str_type: %s\n", str_type);
  DEBUG_PRINTF(1, "n_hidden: %zu\n", n_hidden);
  p_dims = (size_t *)calloc(1000, sizeof(size_t));
  // ALLOC_TO(&p_dims, n_hidden + 2);
  *p_dims = n_in;
  IDX(p_dims, n_hidden + 1) = n_output;
  if (strcmp(str_type, "custom") == 0) {
    if (strcmp(str_shape, ":") == 0) {
      fprintf(stderr, "error: Expect --shape when --type is omitted or set to \"custom\".\n");
      return 1;
    } else if (!strlen(str_shape)) {
      if (n_hidden != 0) {
        fprintf(stderr, "error: Number of hidden layers by --n-hidden and --shape mismatched.\n");
        free(p_dims);
        return 1;
      }
    } else {
      size_t *dim_hidden_now = PIDX(p_dims, 1);
      const char* str_shape_remaining = str_shape;
      const char *iter_str_shape_prev_p1 = str_shape;
      char* str_dim_hidden_now = NULL;
      while (1) {
        const char* iter_str_shape = iter_str_shape_prev_p1;
        do ++iter_str_shape; while (*iter_str_shape != ':' && *iter_str_shape != '\0');
        if (iter_str_shape - iter_str_shape_prev_p1 <= 1) {
          fprintf(stderr, "error: Expect a positive integer between two colons.\n");
          free(p_dims);
          return 1;
        }
        str_dim_hidden_now = realloc(str_dim_hidden_now, (size_t)(iter_str_shape - iter_str_shape_prev_p1 + 1));
        memcpy(str_dim_hidden_now, iter_str_shape_prev_p1, (size_t)(iter_str_shape - iter_str_shape_prev_p1));
        *(str_dim_hidden_now + (iter_str_shape - iter_str_shape_prev_p1)) = '\0';
        if (dim_hidden_now - p_dims >= n_hidden + 1) {
          fprintf(stderr, "error: Number of hidden layers by --n-hidden and --shape mismatched.\n");
          free(str_dim_hidden_now);
          free(p_dims);
          return 1;
        }
        *dim_hidden_now = atol(str_dim_hidden_now);
        dim_hidden_now += sizeof(*dim_hidden_now);
        iter_str_shape_prev_p1 = iter_str_shape + 1;
        if (*iter_str_shape == '\0') {
          break;
        }
      }
      free(str_dim_hidden_now);
      if (dim_hidden_now - p_dims < n_hidden + 1) {
        fprintf(stderr, "error: Number of hidden layers by --n-hidden and --shape mismatched.\n");
        free(p_dims);
        return 1;
      }
    }
  } else if (strcmp(str_type, "triangle") == 0) {
    if (n_hidden > n_in) {
      fprintf(stderr, "error: n_hidden (%zu) must be smaller than n_in (%zu) with type triangle.\n", n_hidden, n_in);
      free(p_dims);
      return 1;
    }
    for (size_t i_layer = 0; i_layer < n_hidden; ++i_layer) {
      DEBUG_PRINTF(1, "Applying triangle, i_layer: %zu, n_in - i_layer: %zu\n", i_layer, n_in - i_layer);
      IDX(p_dims, i_layer + 1) = n_in - i_layer;
    };
  } else if (strcmp(str_type, "rectangle") == 0) {
    for (size_t i_layer = 0; i_layer < n_hidden; ++i_layer) {
      DEBUG_PRINTF(1, "Applying rectangle, i_layer: %zu, n_in: %zu\n", i_layer, n_in);
      IDX(p_dims, i_layer + 1) = n_in;
    }
  } else {
    fprintf(stderr, "error: Unknown hidden layer type %s\n", str_type);
    free(p_dims);
    return 1;
  }
  const size_t n_layer = n_hidden + 1;
  DEBUG_PRINTF(1, "p_dims: %p\n", p_dims);
  for (size_t i_dim = 0; i_dim < n_layer + 1; ++i_dim) {
    DEBUG_PRINTF(1, "PIDX(p_dims, %zu): %p, IDX(p_dims, %zu): %zu\n", i_dim, PIDX(p_dims, i_dim), i_dim, IDX(p_dims, i_dim));
  }
  FILE *pf_out = NULL;
  if (strcmp(str_output, "-") == 0) {
    pf_out = fdopen(STDOUT_FILENO, "wb");
  } else {
    pf_out = fopen(str_output, "wb");
    fseek(pf_out, 0, SEEK_SET);
  }
  DEBUG_PRINTF(1, "p_dims: %p\n", p_dims);
  for (size_t i_dim = 0; i_dim < n_layer + 1; ++i_dim) {
    DEBUG_PRINTF(1, "PIDX(p_dims, %zu): %p, IDX(p_dims, %zu): %zu\n", i_dim, PIDX(p_dims, i_dim), i_dim, IDX(p_dims, i_dim));
  }
  const bool is_success = batch_train_xor_to_file(pf_out, n_layer, p_dims, learning_rate, max_loss, max_delta_loss, max_retrial, enable_rand, enable_rand, rand_seed);
  fprintf(stderr, "Training %s.\n", is_success ? "success" : "failed");
  fflush(pf_out);
  fclose(pf_out);
  fprintf(stdout, "Test\n");
}
