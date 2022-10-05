#include "idx.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t* foo;
  ALLOC_TO(foo, 3);
  *foo = 10;
  *PIDX(foo, 1) = 11;
  IDX(foo, 2) = 12;
  MEMSETN(foo, 0, 3);
  free(foo);
  double* bar;
  CALLOC_TO(bar, 4);
  IDX(bar, 0) = 20;
  IDX(bar, 1) = 21;
  IDX(bar, 2) = 22;
  IDX(bar, 3) = 23;
  FILLN(bar, 0., 4);
  double* baz = NULL;
  REALLOC_TO(baz, 12);
  MEMFILLN(baz, bar, 4, 3);
  // do for (size_t i = 0; i < 3; ++i)
  //   MEMCPYN(baz + sizeof(*baz) * (i * 4), bar, 4);
  // while (0);
  free(baz);
  free(bar);
}
