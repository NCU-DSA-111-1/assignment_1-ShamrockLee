#include <assert.h>
#include <stdio.h>

#include "debug.h"

#include "arr.h"
#include "idx.h"

int test_arr2d() {
  ARR2D_TYPE(double)
  a_foo = {3, 2, NULL};
  ARR2D_ALLOC(&a_foo);
  // 0.5 doesn't have round-off error
  *ARR2D_PIDX(&a_foo, 1, 1) = 0.5;
  assert(("Testing ARR2D_PIDX", IDX(a_foo.p_data, 1 + 3 * 1) == 0.5));
  *ARR2D_PIDX(&a_foo, 0, 0) = 0.;
  *ARR2D_PIDX(&a_foo, 1, 0) = 1.;
  *ARR2D_PIDX(&a_foo, 2, 0) = 2.;
  *ARR2D_PIDX(&a_foo, 0, 1) = 3.;
  *ARR2D_PIDX(&a_foo, 1, 1) = 4.;
  *ARR2D_PIDX(&a_foo, 2, 1) = 5.;
  double* p_bar;
  ALLOC_TO(&p_bar, 2);
  IDX(p_bar, 0) = 6.;
  IDX(p_bar, 1) = 7.;
  double* p_foo_bar;
  ALLOC_TO(&p_foo_bar, 3);
  ARR2D_LPROD(&a_foo, p_foo_bar, p_bar);
  // integer products don't have round-off error
  assert(IDX(p_foo_bar, 0) == 0. * 6. + 3. * 7.);
  assert(IDX(p_foo_bar, 1) == 1. * 6. + 4. * 7.);
  assert(IDX(p_foo_bar, 2) == 2. * 6. + 5. * 7.);
  free(p_foo_bar);
  free(p_bar);
  ALLOC_TO(&p_bar, 3);
  IDX(p_bar, 0) = 6.;
  IDX(p_bar, 1) = 7.;
  IDX(p_bar, 2) = 8.;
  double* p_bar_foo;
  ALLOC_TO(&p_bar_foo, 2);
  ARR2D_RPROD(&a_foo, p_bar_foo, p_bar);
  assert(IDX(p_bar_foo, 0) == 0. * 6. + 1. * 7. + 2. * 8.);
  assert(IDX(p_bar_foo, 1) == 3. * 6. + 4. * 7. + 5. * 8.);
  free(p_bar_foo);
  free(p_bar);
  ARR2D_FREE(&a_foo);
  return 0;
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
    return 1;
  }
  ARRND_FREE(&a_foo);
  return 0;
}

int main(int argc, char** argv) {
  assert(("test_arrnd", !test_arrnd()));
  return 0;
}
