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
    The series route for the symmetric product L1 (x) L2 (s = 2), after
    algeqtodiffeq_series.c: build the pseudo-Krylov matrix K as truncated power
    series at x = 0, recover an exact left description D*K = N from an
    approximant basis, and take the kernel of N. Heuristic in the same sense as
    there: correct when the target degree bounds the description AND sigma is
    large enough relative to it. A too-small target fails loudly (too few rows);
    a too-small SIGMA used to fail SILENTLY -- spurious approximant rows of
    degree <= target were kept, giving a wrong operator (found 2026-09-19 on the
    user's p = 257 example and on balanced inputs of degree 6-12). Since then
    sigma accounts for the right description (nmod_symprod_series_parameters)
    and the description throws if more than R rows pass (ambiguity).

    Differences from the algeqtodiffeq version:
     - T(v) is applied by dividing only W's last row by l1 and last column by
       l2 (nmod_symprod_apply_T_series): r1 + r2 truncated divisions per step,
       rather than r1*r2 divisions by phi.
     - The Krylov columns are theta^k(e_1) themselves, never scaled by phi^k
       (the formulation verified equivalent for algeqtodiffeq on 2026-09-18).
     - The target degree, sigma and N are computed ONCE
       (nmod_symprod_series_parameters) and passed down, so the Krylov builder
       and the description cannot disagree on them.
     - The description is written here, not shared with algeqtodiffeq (per the
       user, 2026-09-19: rewrite now, factor later).

    Conventions (L_i layout, vector indexing) as in symprod.c.
*/


/** Target degree, approximant order and series precision for the series route.
 *
 *  target_degree = ceil(B / R) + NMOD_GFUN_DESCRIPTION_MARGIN, with R = r1*r2
 *  and B = (R - r1 - r2 + 2)(d1*r2 + d2*r1), d_i = deg L_i.
 *
 *  CALIBRATED FOR GENERIC INPUTS, r1, r2 >= 2 AND n = r1*r2 + 1 ONLY
 *  (2026-09-19). On 20
 *  generic instances (every coefficient of L_i of exact degree d_i, r_i up to
 *  3 x 4, d_i up to 3), the minimal left description of the exact Krylov
 *  matrix had balanced row degrees (min and max differ by at most 1) summing
 *  to exactly B, which was also exactly deg L -- so its max row degree was
 *  ceil(B/R) every time. B is Bostan's experimental degree bound (PhD thesis
 *  2003, Chap. 10, quoted in G2026 Sec. 5), itself a conjecture, not a
 *  theorem. The margin covers the gap; a too-small target makes
 *  nmod_symprod_series_left_description throw.
 *
 *  NOT valid as is when some r_i = 1: there B still bounds deg L, but the
 *  minimal description's row degrees sum to MORE than deg L (the maximal
 *  minors of N share a common factor), so ceil(B/R) underestimates them --
 *  observed +1 and +2 (row-degree sums 9 vs deg L = 7, 14 vs 11), absorbed by
 *  the margin. (r_i = 1 is anyway degenerate: the symmetric product is then a
 *  gauge transform of the other operator.) For comparison, the
 *  algeqtodiffeq analogue deg(phi) + margin would be too small here as soon as
 *  r1*r2 > 4, and d1*r2 + d2*r1 would be 1.4-2x larger than needed.
 *
 *  sigma = target_degree + (d1*r2 + d2*r1) + 1, and N = sigma + (n - 1) (each
 *  Krylov step costs one term of precision, through the derivative).
 *
 *  Why that sigma (changed 2026-09-19; it was ceil((R+n)*target/min(R,n)) + 1,
 *  about 2*target + 1, taken from the algeqtodiffeq series route). If
 *  K = N_R * E^{-1} is a right description whose columns have degree <= c, a
 *  candidate row [D_row | N_row] of degree <= target with D_row*K = N_row
 *  mod x^sigma gives D_row*N_R - N_row*E = 0 mod x^sigma, a POLYNOMIAL of
 *  degree <= target + c: so sigma > target + c makes every kept row exact, and
 *  no spurious row can pass the filter. The left degrees alone cannot give
 *  this. Measured on 21 instances: the minimal right description's column
 *  degrees sum to B like the left rows, but are unbalanced (e.g.
 *  0 0 12 13 25 26 38), and their max is EXACTLY d1*r2 + d2*r1 every time --
 *  observed, not proved. The old sigma was below target + d1*r2 + d2*r1 on the
 *  user's p = 257 example (d1 = 1, d2 = 12: 60 < 66) and on balanced inputs of
 *  moderate degree (r = 3, d = 10: 90 < 103; r = 2, d = 12; r = 4, d = 6), and
 *  returned a WRONG operator silently in all of them; it was only safe on small
 *  degrees, where the margin happened to exceed d1*r2 + d2*r1. The new sigma is
 *  larger than the old for large degrees, by about (1+c)/(2c) with
 *  c = (R - r1 - r2 + 2)/R (1.5x at r = 2, 1.4x at r = 3, 1.3x at r = 4) --
 *  the price of correctness, not overhead: the old value was wrong there.
 */
void nmod_symprod_series_parameters(slong * target_degree, slong * sigma, slong * N,
                                    const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                    const slong n)
{
    slong r1 = L1->r - 1;
    slong r2 = L2->r - 1;
    slong R = r1 * r2;
    slong d1 = nmod_poly_mat_degree(L1);
    slong d2 = nmod_poly_mat_degree(L2);

    /* (R - r1 - r2 + 2) = (r1-1)(r2-1) + 1 >= 1 */
    slong B = (R - r1 - r2 + 2) * (d1 * r2 + d2 * r1);
    *target_degree = (B + R - 1) / R + NMOD_GFUN_DESCRIPTION_MARGIN;

    *sigma = *target_degree + (d1 * r2 + d2 * r1) + 1;
    *N = *sigma + (n - 1);
}


/** Rv = T(V) mod x^N, for V a vector of truncated power series (V known mod
 *  x^N gives Rv known mod x^N: nothing divides by x).
 *
 *  Same map as nmod_symprod_apply_T, but with the two denominators handled
 *  separately rather than through phi: reading V as an r1 x r2 matrix W,
 *
 *      Rv[h][p] = W[h-1][p] + W[h][p-1] - p_{1,h}*u[p] - p_{2,p}*w[h]
 *      u[p] = W[r1-1][p] / l1,   w[h] = W[h][r2-1] / l2   (mod x^N)
 *
 *  The shift part has no denominator at all, so only W's last row and last
 *  column are divided: r1 + r2 truncated products by the inverse series
 *  il1 = 1/l1, il2 = 1/l2 (known mod x^N, computed once by the caller), plus
 *  2*r1*r2 products of a degree-d_i coefficient by a series.
 *
 *  Rv must not alias V.
 */
void nmod_symprod_apply_T_series(nmod_poly_mat_t Rv, const nmod_poly_mat_t V,
                                 const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                 const nmod_poly_t il1, const nmod_poly_t il2, const slong N)
{
    slong r1 = L1->r - 1;
    slong r2 = L2->r - 1;
    ulong prime = nmod_poly_mat_modulus(V);

    if (Rv == V)
        flint_throw(FLINT_DOMERR, "nmod_symprod_apply_T_series: Rv must not alias V\n");

    nmod_poly_mat_t u, w;
    nmod_poly_mat_init(u, r2, 1, prime);
    nmod_poly_mat_init(w, r1, 1, prime);

    for (slong p = 0; p < r2; p++)
        nmod_poly_mullow(nmod_poly_mat_entry(u, p, 0),
                         nmod_poly_mat_entry(V, (r1 - 1) * r2 + p, 0), il1, N);
    for (slong h = 0; h < r1; h++)
        nmod_poly_mullow(nmod_poly_mat_entry(w, h, 0),
                         nmod_poly_mat_entry(V, h * r2 + (r2 - 1), 0), il2, N);

    nmod_poly_t t;
    nmod_poly_init(t, prime);

    for (slong h = 0; h < r1; h++)
    {
        for (slong p = 0; p < r2; p++)
        {
            nmod_poly_struct * out = nmod_poly_mat_entry(Rv, h * r2 + p, 0);

            nmod_poly_zero(out);
            if (h > 0)
                nmod_poly_add(out, out, nmod_poly_mat_entry(V, (h - 1) * r2 + p, 0));
            if (p > 0)
                nmod_poly_add(out, out, nmod_poly_mat_entry(V, h * r2 + (p - 1), 0));

            nmod_poly_mullow(t, nmod_poly_mat_entry(L1, h, 0), nmod_poly_mat_entry(u, p, 0), N);
            nmod_poly_sub(out, out, t);

            nmod_poly_mullow(t, nmod_poly_mat_entry(L2, p, 0), nmod_poly_mat_entry(w, h, 0), N);
            nmod_poly_sub(out, out, t);

            nmod_poly_truncate(out, N);
        }
    }

    nmod_poly_clear(t);
    nmod_poly_mat_clear(u);
    nmod_poly_mat_clear(w);
}


/** Builds the (r1*r2) x n pseudo-Krylov matrix K as truncated power series at
 *  x = 0: column k is theta^k(e_1) itself (no phi^k scaling), by
 *
 *      K_{k+1} = T(K_k) + K_k'      mod x^N
 *
 *  with T applied by nmod_symprod_apply_T_series.
 *
 *  Precision: column 0 (e_1) is exact and column 1 = T(e_1) is valid mod x^N;
 *  after that each derivative costs one term, so column k is valid mod
 *  x^(N-(k-1)). Every column is truncated to the weakest guarantee, and that
 *  number is returned: every entry of K is correct mod x^prec, with
 *  prec = N - max(n-2, 0).
 *
 *  Needs l1(0) * l2(0) != 0, i.e. x = 0 a regular point, else the inverse
 *  series do not exist -- flint_throw's then (the standard remedy, expanding at
 *  a shifted point, is not done). K initialized by the caller, (r1*r2) x n.
 */
slong nmod_symprod_pseudo_Krylov_series(nmod_poly_mat_t K,
                                        const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                        const slong n, const slong N)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_symprod_pseudo_Krylov_series: n must be >= 1\n");

    slong r1 = L1->r - 1;
    slong r2 = L2->r - 1;
    slong R = r1 * r2;
    ulong prime = nmod_poly_mat_modulus(L1);

    const nmod_poly_struct * l1 = nmod_poly_mat_entry(L1, r1, 0);
    const nmod_poly_struct * l2 = nmod_poly_mat_entry(L2, r2, 0);
    if (nmod_poly_get_coeff_ui(l1, 0) == 0 || nmod_poly_get_coeff_ui(l2, 0) == 0)
        flint_throw(FLINT_ERROR, "nmod_symprod_pseudo_Krylov_series: l1(0)*l2(0) == 0, so "
                    "x = 0 is a singular point and the expansion around it does not "
                    "exist (needs a shifted expansion point)\n");

    nmod_poly_t il1, il2, t;
    nmod_poly_init(il1, prime);
    nmod_poly_init(il2, prime);
    nmod_poly_init(t, prime);
    nmod_poly_inv_series(il1, l1, N);
    nmod_poly_inv_series(il2, l2, N);

    nmod_poly_mat_t col, Tcol;
    nmod_poly_mat_init(col, R, 1, prime);
    nmod_poly_mat_init(Tcol, R, 1, prime);

    for (slong i = 0; i < R; i++)
        nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    nmod_poly_one(nmod_poly_mat_entry(K, 0, 0));

    for (slong k = 0; k < n - 1; k++)
    {
        for (slong i = 0; i < R; i++)
            nmod_poly_set(nmod_poly_mat_entry(col, i, 0), nmod_poly_mat_entry(K, i, k));

        nmod_symprod_apply_T_series(Tcol, col, L1, L2, il1, il2, N);

        for (slong i = 0; i < R; i++)
        {
            nmod_poly_derivative(t, nmod_poly_mat_entry(K, i, k));
            nmod_poly_add(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(Tcol, i, 0), t);
        }
    }

    slong prec = N - FLINT_MAX(n - 2, 0);
    for (slong i = 0; i < R; i++)
        for (slong j = 0; j < n; j++)
            nmod_poly_truncate(nmod_poly_mat_entry(K, i, j), prec);

    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(Tcol);
    nmod_poly_clear(il1);
    nmod_poly_clear(il2);
    nmod_poly_clear(t);

    return prec;
}


/** Left description D*K = N (D R x R, N R x n) of the truncated pseudo-Krylov
 *  matrix K, from an approximant basis -- rewritten from
 *  nmod_algeqtodiffeq_series_left_description, with target_degree and sigma
 *  passed in instead of derived from phi1 (see nmod_symprod_series_parameters).
 *
 *  A row [D_row | N_row] of the zero-shift approximant basis of B = [-K ; I_n]
 *  at order sigma satisfies N_row = D_row*K mod x^sigma. The rows of shifted
 *  degree <= target_degree are kept -- when the target is large enough these
 *  are the rows of an exact description of K. K must be known mod x^sigma.
 *
 *  N, D initialized by the caller. flint_throw's if the number of rows passing
 *  the filter is not exactly R:
 *   - fewer: target_degree too small;
 *   - MORE: sigma too small for this target -- some passing rows are spurious
 *     approximants, and which R of them to keep cannot be decided. Before this
 *     check (added 2026-09-19) the first R in basis order were kept, which
 *     silently produced a wrong description when a spurious row came first.
 *  Exactly R passing rows are the exact description when the target bounds
 *  its rows (then all genuine rows pass, and there are R of them).
 */
void nmod_symprod_series_left_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                                          const nmod_poly_mat_t K,
                                          const slong target_degree, const slong sigma)
{
    ulong prime = nmod_poly_mat_modulus(K);
    slong r = K->r;
    slong n = K->c;

    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, r + n, n, prime);
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < n; j++)
        {
            nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(K, i, j));
            nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
        }
    for (slong i = 0; i < n; i++)
        nmod_poly_one(nmod_poly_mat_entry(B, i + r, i));

    slong * shift = flint_calloc(r + n, sizeof(slong));

    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, r + n, r + n, prime);
    nmod_poly_mat_pmbasis(ker, shift, B, sigma);

    slong npass = 0;
    for (slong i = 0; i < r + n; i++)
        if (shift[i] <= target_degree)
            npass++;
    if (npass < r)
        flint_throw(FLINT_ERROR, "nmod_symprod_series_left_description: no complete "
                    "description of degree at most %wd found (target_degree too small? "
                    "It is calibrated for generic inputs and n = r1*r2 + 1 only)\n",
                    target_degree);
    if (npass > r)
        flint_throw(FLINT_ERROR, "nmod_symprod_series_left_description: %wd rows of "
                    "degree <= %wd for a description with %wd rows -- sigma = %wd is too "
                    "small for this target, some rows are spurious approximants\n",
                    npass, target_degree, r, sigma);

    slong nbrows = 0;
    for (slong i = 0; i < r + n; i++)
    {
        if (shift[i] <= target_degree)
        {
            for (slong j = 0; j < r; j++)
                nmod_poly_set(nmod_poly_mat_entry(D, nbrows, j), nmod_poly_mat_entry(ker, i, j));
            for (slong j = 0; j < n; j++)
                nmod_poly_set(nmod_poly_mat_entry(N, nbrows, j), nmod_poly_mat_entry(ker, i, j + r));
            nbrows++;
        }
    }

    flint_free(shift);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}


/** Symmetric product L1 (x) L2 via the series route: parameters, truncated
 *  Krylov matrix, left description D*K = N, kernel of N. Since D is
 *  nonsingular, K*eta = 0 iff N*eta = 0, so N's kernel gives the operator
 *  sum_l eta_l d^l directly, coefficients in increasing order -- same output
 *  convention as nmod_symprod_naive.
 *
 *  The target degree is calibrated for generic inputs and n = r1*r2 + 1 (see
 *  nmod_symprod_series_parameters); outside that it may throw. Needs
 *  l1(0)*l2(0) != 0. Returns nz; LT (n x n, initialized by the caller) holds
 *  the solutions in its first nz columns.
 */
slong nmod_symprod_series(nmod_poly_mat_t LT, const nmod_poly_mat_t L1,
                          const nmod_poly_mat_t L2, const slong n)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_symprod_series: n must be >= 1 (use "
                    "n = r1*r2 + 1 for the generic symmetric product)\n");

    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);

    slong target_degree, sigma, N;
    nmod_symprod_series_parameters(&target_degree, &sigma, &N, L1, L2, n);

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K, R, n, prime);
    nmod_symprod_pseudo_Krylov_series(K, L1, L2, n, N);

    nmod_poly_mat_t NN, DD;
    nmod_poly_mat_init(NN, R, n, prime);
    nmod_poly_mat_init(DD, R, R, prime);
    nmod_symprod_series_left_description(NN, DD, K, target_degree, sigma);

    slong * pivind = flint_malloc(n * sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, NULL, NN, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    nmod_poly_mat_clear(K);
    nmod_poly_mat_clear(NN);
    nmod_poly_mat_clear(DD);

    return nz;
}
