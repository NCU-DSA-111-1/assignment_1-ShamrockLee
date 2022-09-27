#include <assert.h>
#include <stdio.h>

#include "debug.h"

#include "arr.h"
#include "idx.h"

int test_ndarr() {
  NDARR_TYPE(double)
  a_foo;
  NDARR_TYPE(double)
  a_bar;
  NDARR_TYPE(double)* paBaz = (NDARR_TYPE(double)*)malloc(2 * sizeof(NDARR_TYPE(double)));
  free(paBaz);
  NDARR_ALLOC_FN(double)
  (&a_foo, 2, 3, 2);
  *NDARR_PIDX_FN(double)(&a_foo, 2, 1) = 0.5;
  DEBUG_PRINTF(1, "a_foo.p_ata + 5 * sizeof(double): %p\n", a_foo.p_data + 5 * sizeof(double));
  DEBUG_PRINTF(1, "&a_foo.p_data[5], %p, a_foo.p_data[5]: %g\n", &a_foo.p_data[5], a_foo.p_data[5]);
  DEBUG_PRINTF(1, "a_foo.p_data, %p, *a_foo.p_data: %g\n", a_foo.p_data, *a_foo.p_data);
  if (IDX(a_foo.p_data, 2 + 1 * 3) != 0.5) {
    DEBUG_PRINTF(-1, "NDARR_PIDX_FN doesn't index to the correct address (expect: %p, got: %p, from a_foo.p_data %p).",
                 PIDX(a_foo.p_data, 2 + 1 * 3), NDARR_PIDX_FN(double)(&a_foo, 2, 1), a_foo.p_data);
    return 1;
  }
  NDARR_FREE(&a_foo);
  return 0;
}

int main(int argc, char** argv) {
  assert(("test_ndarr", !test_ndarr()));
  return 0;
}
