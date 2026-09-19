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
    The naive pseudo-Krylov route for the symmetric product L1 (x) L2 (s = 2),
    mirroring algeqtodiffeq_naive.c's pair: a fraction-free Krylov builder, then
    a driver that rescales it to an ordinary polynomial matrix and takes a
    kernel. Conventions (L_i layout, vector indexing, phi) as in symprod.c.

    Differences from the algeqtodiffeq version, all consequences of T being
    explicit here: phi*T is applied exactly by nmod_symprod_apply_T, so there
    is no evaluation bound D and no margin; and phi = lcm(l1, l2) comes from a
    gcd, so there is no random projection and no random state.
*/

/** Builds the (r1*r2) x n fraction-free pseudo-Krylov matrix K, column k
 *  (0-indexed) being phi^k * theta^k(e_1) for theta = d/dx + T, e_1 the
 *  coordinate vector of b_{0,0} = alpha1*alpha2. So column k holds, up to
 *  the factor phi^k, the coordinates of (alpha1*alpha2)^(k) on the basis
 *  b_{h,p} = alpha1^(h) alpha2^(p).
 *
 *  Recurrence (product rule, tracking K_k = phi^k theta^k(e_1) to stay
 *  fraction-free):
 *      K_{k+1} = phi*K_k' - k*phi'*K_k + phi*T(K_k)
 *  the last term being nmod_symprod_apply_T. Degrees grow by at most
 *  deg(phi) per step: every entry of phi*T has degree <= deg(phi).
 *
 *  phi, m1, m2 from nmod_symprod_setup. K initialized by the caller,
 *  (r1*r2) x n, same modulus. n >= 1: the base case writes column 0.
 */
void nmod_symprod_pseudo_Krylov_naive(nmod_poly_mat_t K, ulong n,
                                      const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                                      const nmod_poly_t phi, const nmod_poly_t m1,
                                      const nmod_poly_t m2)
{
    if (n == 0)
        flint_throw(FLINT_DOMERR, "nmod_symprod_pseudo_Krylov_naive: n must be >= 1 "
                    "(K needs at least one column, the base case e_1)\n");

    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);

    /* apply_T takes and returns stand-alone vectors, not columns of K. */
    nmod_poly_mat_t col, Tcol;
    nmod_poly_mat_init(col, R, 1, prime);
    nmod_poly_mat_init(Tcol, R, 1, prime);

    nmod_poly_t dphi, t;
    nmod_poly_init(dphi, prime);
    nmod_poly_init(t, prime);
    nmod_poly_derivative(dphi, phi);

    /* K_0 = e_1 */
    for (slong i = 0; i < R; i++)
        nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    nmod_poly_one(nmod_poly_mat_entry(K, 0, 0));

    for (slong k = 0; k < (slong) n - 1; k++)
    {
        for (slong i = 0; i < R; i++)
            nmod_poly_set(nmod_poly_mat_entry(col, i, 0), nmod_poly_mat_entry(K, i, k));

        nmod_symprod_apply_T(Tcol, col, L1, L2, phi, m1, m2);

        for (slong i = 0; i < R; i++)
        {
            nmod_poly_struct * out = nmod_poly_mat_entry(K, i, k + 1);
            const nmod_poly_struct * Kk = nmod_poly_mat_entry(K, i, k);

            nmod_poly_set(out, nmod_poly_mat_entry(Tcol, i, 0));

            nmod_poly_derivative(t, Kk);
            nmod_poly_mul(t, t, phi);
            nmod_poly_add(out, out, t);

            nmod_poly_mul(t, dphi, Kk);
            nmod_poly_scalar_mul_nmod(t, t, (ulong) k % prime);
            nmod_poly_sub(out, out, t);
        }
    }

    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(Tcol);
    nmod_poly_clear(dphi);
    nmod_poly_clear(t);
}


/** Symmetric product L1 (x) L2 via the naive pseudo-Krylov route: builds n
 *  columns of the pseudo-Krylov matrix for (theta, e_1) and a minimal kernel
 *  basis of it. A kernel vector eta gives sum_l eta_l (alpha1 alpha2)^(l) = 0,
 *  i.e. the operator sum_l eta_l d^l, coefficients in increasing order.
 *
 *  n is the Krylov matrix width, as in nmod_algeq_to_diffeq_naive: generically
 *  the symmetric product has order r1*r2, so n = r1*r2 + 1 gives one solution
 *  (nz = 1). Non-generic inputs can have lower order and then give nz > 1 at
 *  that n -- notably L1 = L2, whose iterates stay in the symmetric subspace.
 *  n must be >= 1.
 *
 *  Returns nz, the number of solutions; LT (n x n, initialized by the caller)
 *  holds them in its first nz columns.
 */
slong nmod_symprod_naive(nmod_poly_mat_t LT, const nmod_poly_mat_t L1,
                         const nmod_poly_mat_t L2, const slong n)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_symprod_naive: n must be >= 1 "
                    "(n is the pseudo-Krylov matrix width -- use n = r1*r2 + 1 "
                    "for the generic symmetric product)\n");

    ulong prime = nmod_poly_mat_modulus(L1);
    slong R = (L1->r - 1) * (L2->r - 1);

    nmod_poly_t phi, m1, m2;
    nmod_poly_init(phi, prime);
    nmod_poly_init(m1, prime);
    nmod_poly_init(m2, prime);
    nmod_symprod_setup(phi, m1, m2, L1, L2);

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K, R, n, prime);
    nmod_symprod_pseudo_Krylov_naive(K, n, L1, L2, phi, m1, m2);

    /* Column j represents phi^j * theta^j(e_1); multiply it by phi^{(n-1)-j}
     * so that every column carries the same phi^{n-1}. K then differs from
     * the true Krylov matrix by one scalar factor, so the kernel is the same. */
    nmod_poly_t tpol;
    nmod_poly_init(tpol, prime);
    nmod_poly_one(tpol);
    for (slong j = n - 2; j >= 0; j--)
    {
        nmod_poly_mul(tpol, tpol, phi);
        for (slong i = 0; i < R; i++)
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
    }

    /* NULL shift, as in nmod_algeq_to_diffeq_naive. Before changing it, check
     * K's column-degree spread (claude-pseudoKrylov CLAUDE.md: an explicit
     * zero shift only paid off where that spread was large). */
    slong * pivind = flint_malloc(n * sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, NULL, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    nmod_poly_mat_clear(K);
    nmod_poly_clear(tpol);
    nmod_poly_clear(phi);
    nmod_poly_clear(m1);
    nmod_poly_clear(m2);

    return nz;
}
