/*
    Telescoper of a bivariate rational function f = p/q by PseudoKrylovLeftDesc
    (2026-10-20), the counterpart of nmod_algeq_to_diffeq_recursive
    (algeqtodiffeq_recursive.c), on the Hermite T of telescoper.c: description
    T = Q^{-1} P (nmod_T_ctx_left_description), recursion
    nmod_pseudo_Krylov_recursive with seed p, final nullspace of [D p | N].
*/

#include <stdlib.h>
#include <time.h>
#include <flint/nmod_poly.h>
#include "nmod_poly_mat_extra.h"
#include "nmod_poly_mat_forms.h"
#include "gfun.h"

/* f = p/q, PNT the r x 1 y-coefficients of p (deg_y p < r = deg_y q), QT those of q
   ((r+1) x 1, q square-free in y); n >= 2 the number of columns of the pseudo-Krylov
   matrix (telescopers of order <= m = n-1). Returns nz; the first nz columns of LT
   (n x n, initialized) are a minimal basis (zero shift) of the telescopers of order
   <= m, column j = (coefficients of d_x^0 .. d_x^m). Generic n = r+1: nz = 1. */
slong nmod_telescoper_recursive(nmod_poly_mat_t LT, const nmod_poly_mat_t PNT,
                                const nmod_poly_mat_t QT, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_telescoper_recursive: n must be >= 2\n");

    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;
    slong m = n - 1;

    /* same per-call seeding as nmod_algeq_to_diffeq_recursive (todo.md item 8) */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    nmod_poly_t Delta, phi1;
    nmod_poly_mat_t CT;
    nmod_telescoper_setup(Delta, CT, QT);

    nmod_T_ctx_struct T;
    nmod_telescoper_T_data_struct dat;
    nmod_telescoper_T_ctx_init(&T, &dat, QT, CT, Delta);

    nmod_poly_init(phi1, prime);
    nmod_T_ctx_phi1(phi1, &T, state);

    nmod_poly_mat_t Q, P, D, N, Qt, Pt, v, K;
    nmod_poly_mat_init(Q, r, r, prime);
    nmod_poly_mat_init(P, r, r, prime);
    nmod_T_ctx_left_description(Q, P, &T, phi1);

    nmod_poly_mat_init(D, r, r, prime);
    nmod_poly_mat_init(N, r, m, prime);
    nmod_poly_mat_init(Qt, r, r, prime);
    nmod_poly_mat_init(Pt, r, r, prime);
    nmod_pseudo_Krylov_recursive(D, N, Qt, Pt, Q, P, NULL, PNT, m);

    /* K = [D p | N]: D K_true = K, same kernel */
    nmod_poly_mat_init(v, r, 1, prime);
    nmod_poly_mat_mul(v, D, PNT);
    nmod_poly_mat_init(K, r, n, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set(nmod_poly_mat_entry(K, i, 0), nmod_poly_mat_entry(v, i, 0));
        for (slong j = 0; j < m; j++)
            nmod_poly_set(nmod_poly_mat_entry(K, i, j + 1), nmod_poly_mat_entry(N, i, j));
    }

    /* zero shift (not NULL, which means K's column degrees), as the other drivers */
    slong * pivind = flint_malloc(n * sizeof(slong));
    slong * shift = flint_calloc(n, sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, shift, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);
    flint_free(shift);

    nmod_poly_mat_clear(Q); nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(D); nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(Qt); nmod_poly_mat_clear(Pt);
    nmod_poly_mat_clear(v); nmod_poly_mat_clear(K);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
    flint_rand_clear(state);
    return nz;
}
