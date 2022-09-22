#ifndef IDX_H

/**
 * This header provides helper macros to access the address / value of
 * a specific element following a pointer
 * by the "index" of the element.
 * The purpose is to increase readability and to reduce errors.
 **/

#define PIDX(PX, I) ((PX) + sizeof(*(PX)) * (I))
#define IDX(PX, I) (*PIDX(PX, I))

#define IDX_H
#endif
