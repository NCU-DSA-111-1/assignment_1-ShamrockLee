#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t* foo;
  foo = (typeof(foo))malloc(sizeof(*foo) * 3);
  *foo = 10;
  *(foo + sizeof(*foo) * 1) = 11;
  *(foo + sizeof(*foo) * 2) = 12;
  free(foo);
  double* bar;
  bar = (typeof(bar))calloc(4, sizeof(*bar));
  *(bar + sizeof(*bar) * 0) = 20;
  *(bar + sizeof(*bar) * 1) = 21;
  *(bar + sizeof(*bar) * 2) = 22;
  *(bar + sizeof(*bar) * 3) = 23;
  free(bar);
}
