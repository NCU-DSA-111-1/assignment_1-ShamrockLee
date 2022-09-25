#ifndef DEBUG_H
#define DEBUG_H

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#include <stdarg.h>
#include <stdio.h>

/**
 * This header provides helper macro
 * implemented through ##__VA_ARGS__ macro supported by gcc, clang and xlc
 * to help printing debug messages prefixed with the file path and the line
 * where it is called.
 *
 * See https://stackoverflow.com/questions/37206118/va-args-not-swallowing-comma-when-zero-args-under-c99
 **/

#define DEBUG_PRINTF(LEVEL, FORMAT, ...) \
  ((LEVEL >= DEBUG_LEVEL) ? fprintf(stderr, "%s:%d: debug: " FORMAT, __FILE__, __LINE__, ##__VA_ARGS__) : 0)

#endif  // DEBUG_H
