/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <flint/nmod_poly.h>

#include "nmod_poly_mat_extra.h"

#include "gfun.h"

/*
    The symmetric product L1 (x) L2 (s = 2) via the Section-4/DAC1 family
    (algos.pdf Algorithm 6, nmod_pseudo_Krylov_recursive), after
    algeqtodiffeq_recursive.c and with the same three pieces:
      (a)+(b) an irreducible left description T = Q^{-1}P of this T,
      (c)     nmod_pseudo_Krylov_recursive on it (unchanged, general),
      (d)     prepend the seed rescaled by D and take the kernel (driver).
    Written for symprod rather than shared with algeqtodiffeq (per the user:
    rewrite now, factor later). Conventions as in symprod.c.

    PROPERNESS. Algorithm 6 requires an irreducible description with Q
    row-reduced and rdeg(Q) >= rdeg(P) row by row, i.e. T proper. This T is
    proper iff deg p_{i,j} <= deg l_i for all coefficients (true for generic
    operators, false e.g. when l_i has lower degree than some other
    coefficient). Per the user (2026-09-19) the code runs on any input
    anyway: for non-proper T the announced degree bounds and specifications
    may not hold -- to be worked on later.
*/


/** Irreducible left description T = Q^{-1}P of the symprod T (Q, P both
 *  R x R, R = r1*r2, initialized by the caller), as input to
 *  nmod_pseudo_Krylov_recursive. phi, m1, m2 from nmod_symprod_setup.
 *
 *  (a) phi*T as an explicit R x R matrix, column j = nmod_symprod_apply_T of
 *  the j-th unit vector.
 *  (b) Rows [Q | P] of a zero-shift approximant basis of B = [-phi*T ; phi*I]
 *  at order sigma, those of shifted degree <= target_degree -- the
 *  construction of nmod_algeqtodiffeq_T_left_description, with
 *
 *      target_degree = deg(phi) + NMOD_GFUN_NONPROPER_MARGIN
 *      sigma         = 2*target_degree + 1
 *
 *  (no division by r-1: that came from algeqtodiffeq's T having a zero first
 *  column, which this T does not have).
 *
 *  Why deg(phi), and why the result is then exact for proper T (not a guess,
 *  unlike the algeqtodiffeq target): [phi*I | phi*T] is itself a left
 *  description, of row degree deg(phi) when T is proper; a minimal one has
 *  row degrees no larger than those of any R independent descriptions, so
 *  its rows have degree <= deg(phi). And a row [Q_row | P_row] of degree
 *  <= target_degree leaves a residual Q_row*phi*T - phi*P_row of degree
 *  <= target_degree + deg(phi) < sigma, so vanishing mod x^sigma means
 *  vanishing exactly. NMOD_GFUN_NONPROPER_MARGIN (a properness margin, per
 *  the user: here it is about the degree of a description OF T) keeps that
 *  argument valid when T's excess degree at infinity is at most the margin.
 *
 *  Calibration, 20 generic instances + one example (2026-09-19): Q row-reduced
 *  and rdeg P <= rdeg Q in every row; deg det Q = r2*deg(l1) + r1*deg(l2)
 *  exactly, the McMillan degree predicted by the pole ranks (r2 at l1, r1 at
 *  l2); row degrees NOT balanced (max up to 4 where that sum over R is 1-3),
 *  and always <= deg(phi).
 *
 *  flint_throw's if fewer than R rows pass the filter.
 */
void nmod_symprod_T_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
                                     const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                     const nmod_poly_t phi, const nmod_poly_t m1,
                                     const nmod_poly_t m2)
{
    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);

    /* (a) phi*T, column by column */
    nmod_poly_mat_t Tmat, e, col;
    nmod_poly_mat_init(Tmat, R, R, prime);
    nmod_poly_mat_init(e, R, 1, prime);
    nmod_poly_mat_init(col, R, 1, prime);
    for (slong j = 0; j < R; j++)
    {
        nmod_poly_mat_zero(e);
        nmod_poly_one(nmod_poly_mat_entry(e, j, 0));
        nmod_symprod_apply_T(col, e, L1, L2, phi, m1, m2);
        for (slong i = 0; i < R; i++)
            nmod_poly_set(nmod_poly_mat_entry(Tmat, i, j), nmod_poly_mat_entry(col, i, 0));
    }

    /* (b) irreducible left description via a truncated approximant basis */
    slong target_degree = nmod_poly_degree(phi) + NMOD_GFUN_NONPROPER_MARGIN;
    slong sigma = (2 * R * target_degree + R - 1) / R + 1; /* ceil(2R*target_degree/R) + 1 */

    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, 2 * R, R, prime);
    for (slong i = 0; i < R; i++)
    {
        for (slong j = 0; j < R; j++)
        {
            nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(Tmat, i, j));
            nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
        }
        nmod_poly_set_trunc(nmod_poly_mat_entry(B, i + R, i), phi, sigma);
    }

    slong * shift = flint_calloc(2 * R, sizeof(slong));

    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, 2 * R, 2 * R, prime);
    nmod_poly_mat_pmbasis(ker, shift, B, sigma);

    slong nbrows = 0;
    for (slong i = 0; i < 2 * R && nbrows < R; i++)
    {
        if (shift[i] <= target_degree)
        {
            for (slong j = 0; j < R; j++)
            {
                nmod_poly_set(nmod_poly_mat_entry(Q, nbrows, j), nmod_poly_mat_entry(ker, i, j));
                nmod_poly_set(nmod_poly_mat_entry(P, nbrows, j), nmod_poly_mat_entry(ker, i, j + R));
            }
            nbrows++;
        }
    }
    if (nbrows < R)
        flint_throw(FLINT_ERROR, "nmod_symprod_T_left_description: no complete "
                    "description of degree at most %wd found (target_degree too small? "
                    "T may be far from proper)\n", target_degree);

    flint_free(shift);
    nmod_poly_mat_clear(Tmat);
    nmod_poly_mat_clear(e);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}


/** Description of the symprod pseudo-Krylov matrix: T's left description
 *  (nmod_symprod_T_left_description), then nmod_pseudo_Krylov_recursive with
 *  zero shift, Qt/Pt on local scratch -- as
 *  nmod_algeqtodiffeq_pseudo_krylov_description.
 *
 *  D (R x R) and N (R x m) initialized by the caller; a is R x 1.
 *  Output: [theta(a) ... theta^m(a)] = D^{-1}N, theta = d/dx + T.
 */
void nmod_symprod_pseudo_krylov_description(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                            const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                            const nmod_poly_t phi, const nmod_poly_t m1,
                                            const nmod_poly_t m2,
                                            const nmod_poly_mat_t a, const slong m)
{
    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);

    nmod_poly_mat_t Q, P;
    nmod_poly_mat_init(Q, R, R, prime);
    nmod_poly_mat_init(P, R, R, prime);
    nmod_symprod_T_left_description(Q, P, L1, L2, phi, m1, m2);

    nmod_poly_mat_t Qt, Pt;
    nmod_poly_mat_init(Qt, R, R, prime);
    nmod_poly_mat_init(Pt, R, R, prime);
    nmod_pseudo_Krylov_recursive(D, N, Qt, Pt, Q, P, NULL, a, m);

    nmod_poly_mat_clear(Q);
    nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(Qt);
    nmod_poly_mat_clear(Pt);
}


/** Symmetric product L1 (x) L2 via the recursive route: with
 *  [theta(e_1) ... theta^{n-1}(e_1)] = D^{-1}N, the matrix [D*e_1 | N] equals
 *  D times the Krylov matrix [e_1, theta(e_1), ...], so its kernel is the
 *  Krylov matrix's (D nonsingular) -- as nmod_algeq_to_diffeq_recursive, with
 *  seed e_1. Same output convention as nmod_symprod_naive: a kernel vector is
 *  the operator sum_l eta_l d^l. n >= 2 (n = r1*r2 + 1 generically).
 *
 *  Returns nz; LT (n x n, initialized by the caller) holds the solutions in
 *  its first nz columns. See the file header on properness.
 */
slong nmod_symprod_recursive(nmod_poly_mat_t LT, const nmod_poly_mat_t L1,
                             const nmod_poly_mat_t L2, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_symprod_recursive: n must be >= 2 "
                    "(n is the pseudo-Krylov matrix width -- use n = r1*r2 + 1)\n");

    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);
    slong m = n - 1;

    nmod_poly_t phi, m1, m2;
    nmod_poly_init(phi, prime);
    nmod_poly_init(m1, prime);
    nmod_poly_init(m2, prime);
    nmod_symprod_setup(phi, m1, m2, L1, L2);

    nmod_poly_mat_t a;
    nmod_poly_mat_init(a, R, 1, prime);
    nmod_poly_one(nmod_poly_mat_entry(a, 0, 0));

    nmod_poly_mat_t D, N;
    nmod_poly_mat_init(D, R, R, prime);
    nmod_poly_mat_init(N, R, m, prime);
    nmod_symprod_pseudo_krylov_description(D, N, L1, L2, phi, m1, m2, a, m);

    /* K = [D*e_1 | N] */
    nmod_poly_mat_t v, K;
    nmod_poly_mat_init(v, R, 1, prime);
    nmod_poly_mat_mul(v, D, a);
    nmod_poly_mat_init(K, R, n, prime);
    for (slong i = 0; i < R; i++)
    {
        nmod_poly_set(nmod_poly_mat_entry(K, i, 0), nmod_poly_mat_entry(v, i, 0));
        for (slong j = 0; j < m; j++)
            nmod_poly_set(nmod_poly_mat_entry(K, i, j + 1), nmod_poly_mat_entry(N, i, j));
    }

    slong * pivind = flint_malloc(n * sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, NULL, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    nmod_poly_mat_clear(a);
    nmod_poly_mat_clear(D);
    nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(K);
    nmod_poly_clear(phi);
    nmod_poly_clear(m1);
    nmod_poly_clear(m2);

    return nz;
}
