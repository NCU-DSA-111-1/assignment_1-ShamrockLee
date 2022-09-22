#ifndef DEBUG_H
#define DEBUG_H

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#include <stdarg.h>
#include <stdio.h>

/**
 * This header provides helper macro implemented through a static function
 * to help printing debug messages prefixed with the file path and the line
 * where it is called.
 **/

static int __debug_printf(const int level, const char* fmt, ...) {
  if (level >= DEBUG_LEVEL) {
    va_list ap;
    va_start(ap, fmt);
    const int ret = vfprintf(stderr, fmt, ap);
    va_end(ap);
    return ret;
  }
  return 0;
}

#define DEBUG_PRINTF \
  fprintf(stderr, "%s:%d: debug: ", __FILE__, __LINE__); \
  __debug_printf

#endif  // DEBUG_H
