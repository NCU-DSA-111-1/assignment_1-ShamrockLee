#ifndef IDX_H

/**
 * This header provides helper macros to access the address / value of
 * a specific element following a pointer
 * by the "index" of the element.
 * The purpose is to increase readability and to reduce errors.
 **/

#define PIDX(PX, I) ((PX) + sizeof(*(PX)) * (I))
#define IDX(PX, I) (*PIDX(PX, I))

#define PSIDX(PX, I) ((PX) + (ptrdiff_t)sizeof(*(PX)) * (ptrdiff_t)(I))
#define IDX(PX, I) (*PIDX(PX, I))

#define ALLOC_TO(PX, N) ((PX) = (typeof(PX))malloc(sizeof(*(PX)) * (N)))
#define CALLOC_TO(PX, N) ((PX) = (typeof(PX))calloc((N), sizeof(*(PX))))
#define MEMSETN(PX, VAL, N) (memset((PX), (VAL), sizeof(*(PX)) * (N)))
#define MEMCPYN(PDEST, PSRC, N) (memcpy((PDEST), (PSRC), sizeof(*(PDEST)) * (N)))

#define IDX_H
#endif
