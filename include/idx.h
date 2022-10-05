#ifndef IDX_H
#define IDX_H

/**
 * This header provides helper macros to access the address / value of
 * a specific element following a pointer
 * by the "index" of the element.
 * The purpose is to increase readability and to reduce errors.
 **/

#define PIDX(PX, I) ((PX) + (I))
#define IDX(PX, I) (*PIDX(PX, I))

#define PSIDX(PX, I) ((PX) + (ptrdiff_t)(I))
#define IDX(PX, I) (*PIDX(PX, I))

#define ALLOC_TO(PX, N) (PX) = (typeof(PX))malloc(sizeof(*(PX)) * (N))
#define CALLOC_TO(PX, N) (PX) = (typeof(PX))calloc((N), sizeof(*(PX)))
#define REALLOC_TO(PX, N) (PX) = (typeof(PX))realloc(PX, sizeof(*(PX)) * (N))
#define MEMSETN(PX, VAL, N) memset((PX), (VAL), sizeof(*(PX)) * (N))
#define MEMCPYN(PDEST, PSRC, N) memcpy((PDEST), (PSRC), sizeof(*(PDEST)) * (N))
#define FILLN(PDEST, VAL, N) do for (size_t _i_elem_filln = 0; _i_elem_filln < (N); ++_i_elem_filln) IDX(PDEST, _i_elem_filln) = (VAL); while (0)
#define MEMFILLN(PDEST, PSRC, M, N) do for (size_t _i_elem_memfilln = 0; _i_elem_memfilln < (N); ++_i_elem_memfilln) MEMCPYN(PIDX(PDEST, (M) * _i_elem_memfilln), PSRC, M); while (0)

#endif
