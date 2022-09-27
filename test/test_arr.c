#include <assert.h>
#include <stdio.h>

#include "debug.h"

#include "arr.h"
#include "idx.h"

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
