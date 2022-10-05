#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t* foo;
  foo = (typeof(foo))malloc(sizeof(*foo) * 3);
  *foo = 10;
  *(foo + 1) = 11;
  *(foo + 2) = 12;
  free(foo);
  double* bar;
  bar = (typeof(bar))calloc(4, sizeof(*bar));
  *(bar + 0) = 20;
  *(bar + 1) = 21;
  *(bar + 2) = 22;
  *(bar + 3) = 23;
  free(bar);
}
