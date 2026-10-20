/*
    Telescoper of a bivariate rational function f = p/q by K + lifting (2026-10-20),
    the counterpart of nmod_algeq_to_diffeq_naive_lift (algeqtodiffeq_naive.c), on the
    Hermite T of telescoper.c. Degree of the result: G2026 Thm. 9 gives a minimal
    telescoper of order rho <= r and degree <= rho (d + 2rd).
*/

#include <stdlib.h>
#include <time.h>
#include <flint/nmod_poly.h>
#include "nmod_poly_mat_extra.h"
#include "gfun.h"

/* Telescoper by K + lifting (2026-10-20), the CTBivRatFun counterpart of
   nmod_algeq_to_diffeq_naive_lift. f = p/q, p given as the r x 1 matrix PNT of its
   y-coefficients (deg_y p < r = deg_y q), q as QT ((r+1) x 1). Since herm(f) = p and
   herm(d_x^j f) = theta^j(p) (pk_main.tex Sec. 3.5.1), L = sum_j eta_j d_x^j is a
   telescoper iff sum_j eta_j theta^j(p) = 0: K = [phi1^j theta^j(p)]_{j < n} (generic
   naive K on the Hermite context, seed p), column j padded by phi1^{(n-1)-j}, kernel
   vector by nmod_poly_mat_kernel_vector_lift. n must be r+1. Returns nz (1, or 0 if no
   kernel vector is found); LT (n x n, initialized) gets eta in column 0, zeros elsewhere;
   eta primitive and monic on its entry of highest degree. */
slong nmod_telescoper_naive_lift(nmod_poly_mat_t LT, const nmod_poly_mat_t PNT,
                                  const nmod_poly_mat_t QT, const slong n)
{
    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;

    if (n != r + 1)
        flint_throw(FLINT_DOMERR, "nmod_telescoper_naive_lift: n must be r+1 "
                    "(the lifting computes a one-dimensional kernel)\n");

    /* same per-call seeding as nmod_algeq_to_diffeq_naive_lift (todo.md item 8) */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    nmod_poly_t Delta, phi1, tpol;
    nmod_poly_mat_t CT, K, eta;
    nmod_telescoper_setup(Delta, CT, QT);

    nmod_T_ctx_struct T;
    nmod_telescoper_T_data_struct dat;
    nmod_telescoper_T_ctx_init(&T, &dat, QT, CT, Delta);

    nmod_poly_init(phi1, prime);
    nmod_T_ctx_phi1(phi1, &T, state);

    nmod_poly_mat_init(K, r, n, prime);
    nmod_T_ctx_pseudo_Krylov_naive(K, n, &T, PNT, phi1);

    /* column j represents phi1^j theta^j(p): times phi1^{(n-1)-j} */
    nmod_poly_init(tpol, prime);
    nmod_poly_one(tpol);
    for (slong j = n - 2; j >= 0; j--)
    {
        nmod_poly_mul(tpol, tpol, phi1);
        for (slong i = 0; i < r; i++)
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
    }

    nmod_poly_mat_init(eta, n, 1, prime);
    slong nz = nmod_poly_mat_kernel_vector_lift(eta, K, nmod_poly_mat_degree(K),
                                                NMOD_GFUN_SERIES_SOLVE_BASE, state) ? 1 : 0;

    nmod_poly_mat_zero(LT);
    if (nz)
        for (slong j = 0; j < n; j++)
            nmod_poly_swap(nmod_poly_mat_entry(LT, j, 0), nmod_poly_mat_entry(eta, j, 0));

    nmod_poly_mat_clear(K);
    nmod_poly_mat_clear(eta);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
    nmod_poly_clear(tpol);
    flint_rand_clear(state);
    return nz;
}
