/*
    Telescoper of a bivariate rational function f = p/q by PseudoKrylovWidth1 (Algorithm 4)
    (2026-10-20), the counterpart of nmod_algeq_to_diffeq_width1 (algeqtodiffeq_width1.c), on
    the Hermite T of telescoper.c: phi1, phi2 and the width check (nmod_T_ctx_phi_T), then the
    generic core _nmod_T_ctx_width1_core with seed p. T has width 1 for generic q
    (tests/t-telescoper-phi-T.c), not for every q (e.g. BCCL2010's family (14) for d >= 3).
*/

#include <stdlib.h>
#include <time.h>
#include <flint/nmod_poly.h>
#include "nmod_poly_mat_extra.h"
#include "gfun.h"

/* f = p/q, PNT the r x 1 y-coefficients of p (deg_y p < r = deg_y q), QT those of q
   ((r+1) x 1, q square-free in y); n >= 2. Returns nz; the first nz columns of LT (n x n,
   initialized) are a minimal basis (zero shift) of the telescopers of order <= n-1. Throws if
   width(T) > 1 (deg phi1 != deg phi2), as nmod_algeq_to_diffeq_width1. */
slong nmod_telescoper_width1(nmod_poly_mat_t LT, const nmod_poly_mat_t PNT,
                             const nmod_poly_mat_t QT, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_telescoper_width1: n must be >= 2\n");

    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;

    /* same per-call seeding as nmod_algeq_to_diffeq_width1 (todo.md item 8) */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    nmod_poly_t Delta, phi1, phi2;
    nmod_poly_mat_t CT;
    nmod_telescoper_setup(Delta, CT, QT);

    nmod_T_ctx_struct T;
    nmod_telescoper_T_data_struct dat;
    nmod_telescoper_T_ctx_init(&T, &dat, QT, CT, Delta);

    /* width check: unlike AlgeqtoDiffeq (T(1) = 0, hence its r < 3 special case), the
       Hermite T has no zero column, so the 2 x 2 minor is meaningful from r = 2 on */
    nmod_poly_init(phi1, prime);
    nmod_poly_init(phi2, prime);
    if (r >= 2)
    {
        nmod_T_ctx_phi_T(phi1, phi2, &T, state);
        if (nmod_poly_degree(phi1) != nmod_poly_degree(phi2))
            flint_throw(FLINT_ERROR, "nmod_telescoper_width1: T does not have width <= 1 "
                        "(deg(phi1)=%wd, deg(phi2)=%wd)\n",
                        nmod_poly_degree(phi1), nmod_poly_degree(phi2));
    }
    else
        nmod_T_ctx_phi1(phi1, &T, state);

    slong nz = _nmod_T_ctx_width1_core(LT, PNT, (ulong) (n - 1), &T, phi1, state);

    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
    nmod_poly_clear(phi2);
    flint_rand_clear(state);
    return nz;
}
