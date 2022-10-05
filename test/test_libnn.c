#include "libnn.h"

#include <stdbool.h>

#include "debug.h"

int test_init_param_weight() {
  const size_t n_layer = 2;
  size_t *p_dims = NULL;
  ALLOC_TO(p_dims, n_layer + 1);
  IDX(p_dims, 0) = 2;
  IDX(p_dims, 1) = 2;
  IDX(p_dims, 2) = 1;
  ARR2D_TYPE(double) * pa_weights;
  init_param_weight(pa_weights, n_layer, p_dims, 1);
  RETURN_WHEN_FALSE(pa_weights[1].dim1 == 2,,"Assertion failed. a_weights[1].dim1 = %zu", pa_weights[1].dim1);
  free_param_weight(pa_weights, n_layer);
  free(p_dims);
  return 1;
}

int main(int argc, char** argv) {
  RETURN_WHEN_TRUE(!test_init_param_weight(),, "test_init_param_weight failed.\n");
  return 0;
}