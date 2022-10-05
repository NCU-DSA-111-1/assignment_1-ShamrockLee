#include <assert.h>
#include <stdio.h>

#include "debug.h"

#include "arr.h"
#include "idx.h"

int test_arr2d() {
  ARR2D_TYPE(double) a_foo = {3, 2, NULL};
  RETURN_WHEN_FALSE(a_foo.dim0 == 3,, "Assertion failed. Got a_foo.dim0: %zu\n", a_foo.dim0);
  RETURN_WHEN_FALSE(a_foo.dim1 == 2,, "Assertion failed. Got a_foo.dim1: %zu\n", a_foo.dim1);
  ARR2D_ALLOC(&a_foo);
  // 0.5 doesn't have round-off error
  *ARR2D_PIDX(&a_foo, 1, 1) = 0.5;
  RETURN_WHEN_FALSE(IDX(a_foo.p_data, 1 + 3 * 1) == 0.5,, "Test failed for PIDX\n");
  *ARR2D_PIDX(&a_foo, 0, 0) = 0.;
  *ARR2D_PIDX(&a_foo, 1, 0) = 1.;
  *ARR2D_PIDX(&a_foo, 2, 0) = 2.;
  *ARR2D_PIDX(&a_foo, 0, 1) = 3.;
  *ARR2D_PIDX(&a_foo, 1, 1) = 4.;
  *ARR2D_PIDX(&a_foo, 2, 1) = 5.;
  double* p_bar;
  ALLOC_TO(p_bar, 2);
  IDX(p_bar, 0) = 6.;
  IDX(p_bar, 1) = 7.;
  double* p_foo_bar;
  ALLOC_TO(p_foo_bar, 3);
  ARR2D_LPROD(&a_foo, p_foo_bar, p_bar);
  // integer products don't have round-off error
  // DEBUG_PRINTF(1, "IDX(p_foo_bar, 1): %F\n", IDX(p_foo_bar, 1));
  RETURN_WHEN_FALSE(IDX(p_foo_bar, 0) == 0. * 6. + 3. * 7.,,"Assertion failed. Got %F\n", IDX(p_foo_bar, 0));
  RETURN_WHEN_FALSE(IDX(p_foo_bar, 1) == 1. * 6. + 4. * 7.,,"Assertion failed. Got %F\n", IDX(p_foo_bar, 1));
  RETURN_WHEN_FALSE(IDX(p_foo_bar, 2) == 2. * 6. + 5. * 7.,,"Assertion failed. Got %F\n", IDX(p_foo_bar, 2));
  free(p_foo_bar);
  // free(p_bar);
  // ALLOC_TO(p_bar, 3);
  // IDX(p_bar, 0) = 6.;
  // IDX(p_bar, 1) = 7.;
  // IDX(p_bar, 2) = 8.;
  // double* p_bar_foo;
  // ALLOC_TO(p_bar_foo, 2);
  // ARR2D_RPROD(&a_foo, p_bar_foo, p_bar);
  // assert(IDX(p_bar_foo, 0) == 0. * 6. + 1. * 7. + 2. * 8.);
  // assert(IDX(p_bar_foo, 1) == 3. * 6. + 4. * 7. + 5. * 8.);
  // free(p_bar_foo);
  free(p_bar);
  ARR2D_FREE(&a_foo);
  return 1;
}

int test_arrnd() {
  ARRND_TYPE(double)
  a_foo;
  ARRND_TYPE(double)
  a_bar;
  ARRND_TYPE(double)* paBaz = (ARRND_TYPE(double)*)malloc(2 * sizeof(ARRND_TYPE(double)));
  free(paBaz);
  ARRND_ALLOC_FN(double)
  (&a_foo, 2, 3, 2);
  *ARRND_PIDX_FN(double)(&a_foo, 2, 1) = 0.5;
  DEBUG_PRINTF(1, "a_foo.p_ata + 5 * sizeof(double): %p\n", a_foo.p_data + 5 * sizeof(double));
  DEBUG_PRINTF(1, "&a_foo.p_data[5], %p, a_foo.p_data[5]: %g\n", &a_foo.p_data[5], a_foo.p_data[5]);
  DEBUG_PRINTF(1, "a_foo.p_data, %p, *a_foo.p_data: %g\n", a_foo.p_data, *a_foo.p_data);
  if (IDX(a_foo.p_data, 2 + 1 * 3) != 0.5) {
    DEBUG_PRINTF(-1, "ARRND_PIDX_FN doesn't index to the correct address (expect: %p, got: %p, from a_foo.p_data %p).",
                 PIDX(a_foo.p_data, 2 + 1 * 3), ARRND_PIDX_FN(double)(&a_foo, 2, 1), a_foo.p_data);
    return 0;
  }
  ARRND_FREE(&a_foo);
  return 1;
}

int main(int argc, char** argv) {
  RETURN_WHEN_TRUE(!test_arr2d(),, "test_arr2d failed.\n");
  assert((test_arrnd(), "test_arrnd failed.\n"));
  return 0;
}
