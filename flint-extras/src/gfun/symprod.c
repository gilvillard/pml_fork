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
    Symmetric product L = L1 (x) L2 of two linear differential operators, as a
    pseudo-Krylov problem (G2026.pdf Sec. 5, the s = 2 case).

    With alpha_i a generic solution of L_i, every derivative of alpha1*alpha2
    lies in the k(x)-span of b_{h,p} = alpha1^(h) * alpha2^(p), 0 <= h < r1,
    0 <= p < r2, and differentiation acts on coordinates there as
    theta = d/dx + T, with

        T = C1 (x) I_{r2} + I_{r1} (x) C2                        (G2026 eq. 12)

    C_i being the companion matrix of L_i: subdiagonal 1s, last column
    -(p_{i,0}, ..., p_{i,r_i-1})^t / p_{i,r_i}. L's coefficients are read off a
    solution of Problem 1 for (T, e_1).

    Conventions shared by every function in this file:
     - L_i is an (r_i+1) x 1 nmod_poly_mat, entry j being p_{i,j}; r_i >= 1 and
       the leading coefficient l_i = p_{i,r_i} is nonzero.
     - A vector on the basis b_{h,p} is an (r1*r2) x 1 nmod_poly_mat, index
       h*r2 + p (the paper's order b_{0,0}, ..., b_{0,r2-1}, b_{1,0}, ...).
     - Fraction-free, as elsewhere in gfun: T is never formed; phi*T(.) is
       computed instead, with phi = lcm(l1, l2).
*/


/** Setup for nmod_symprod_apply_T: phi = lcm(l1, l2), with the two cofactors
 *  m1 = phi/l1 = l2/g and m2 = phi/l2 = l1/g, where g = gcd(l1, l2).
 *
 *  Why the lcm and not simply l1*l2: both are valid common denominators and
 *  give the same kernel, but l1*l2 carries a spurious factor g^k into the k-th
 *  Krylov numerator. That is not a corner case -- L1 = L2 (symmetric square)
 *  gives g = l1, and phi = l1 instead of l1^2. Carrying the cofactors costs
 *  nothing in nmod_symprod_apply_T: when g = 1 its formula is the plain
 *  l1*l2 one.
 *
 *  For L1, L2 primitive (no common factor in all coefficients) and r1, r2 >= 2,
 *  this phi is exactly the least common denominator of T's entries, i.e. the
 *  phi1 of this T -- obtained deterministically, with no random projection.
 *
 *  phi, m1, m2 must be nmod_poly_init'd by the caller, same modulus as L1, L2.
 *  They depend on L1, L2 only: compute once, reuse across every apply.
 */
void nmod_symprod_setup(nmod_poly_t phi, nmod_poly_t m1, nmod_poly_t m2,
                        const nmod_poly_mat_t L1, const nmod_poly_mat_t L2)
{
    slong r1 = L1->r - 1;
    slong r2 = L2->r - 1;

    if (r1 < 1 || r2 < 1)
        flint_throw(FLINT_DOMERR, "nmod_symprod_setup: L1, L2 must have order >= 1\n");

    const nmod_poly_struct * l1 = nmod_poly_mat_entry(L1, r1, 0);
    const nmod_poly_struct * l2 = nmod_poly_mat_entry(L2, r2, 0);

    if (nmod_poly_is_zero(l1) || nmod_poly_is_zero(l2))
        flint_throw(FLINT_DOMERR, "nmod_symprod_setup: zero leading coefficient\n");

    nmod_poly_t g;
    nmod_poly_init(g, nmod_poly_mat_modulus(L1));
    nmod_poly_gcd(g, l1, l2);

    nmod_poly_divexact(m1, l2, g);
    nmod_poly_divexact(m2, l1, g);
    nmod_poly_mul(phi, l1, m1);

    nmod_poly_clear(g);
}


/** R = phi * T(V), for T = C1 (x) I + I (x) C2 -- without forming T.
 *
 *  Reading V as an r1 x r2 matrix W (W[h][p] = V[h*r2 + p]), T(V) is
 *  C1*W + W*C2^t: C1 shifts the h index up by one and rewrites the overflow
 *  alpha1^(r1) through L1; C2 does the same on the p index. Scaled by
 *  phi = l1*m1 = l2*m2 (see nmod_symprod_setup):
 *
 *      R[h][p] = phi*(W[h-1][p] + W[h][p-1]) - p_{1,h}*u[p] - p_{2,p}*w[h]
 *
 *      u[p] = m1 * W[r1-1][p]      (last row of W)
 *      w[h] = m2 * W[h][r2-1]      (last column of W)
 *
 *  out-of-range shift terms being absent. So the map is a shift-sum minus two
 *  rank-one terms. The cofactors are applied to the vector (r1 + r2
 *  multiplications) rather than to the coefficients, so that per entry the
 *  three multiplications are by phi, p_{1,h} and p_{2,p} -- of degree
 *  deg(phi), d1 and d2 -- instead of three of degree deg(phi).
 *
 *  R and V are (r1*r2) x 1, initialized by the caller, same modulus. R must
 *  NOT alias V: every entry of R reads W's last row and last column.
 */
void nmod_symprod_apply_T(nmod_poly_mat_t R, const nmod_poly_mat_t V,
                          const nmod_poly_mat_t L1, const nmod_poly_mat_t L2,
                          const nmod_poly_t phi, const nmod_poly_t m1, const nmod_poly_t m2)
{
    slong r1 = L1->r - 1;
    slong r2 = L2->r - 1;
    ulong prime = nmod_poly_mat_modulus(V);

    if (R == V)
        flint_throw(FLINT_DOMERR, "nmod_symprod_apply_T: R must not alias V\n");
    if (V->r != r1 * r2 || R->r != r1 * r2)
        flint_throw(FLINT_DOMERR, "nmod_symprod_apply_T: V and R must have r1*r2 rows\n");

    nmod_poly_mat_t u, w;
    nmod_poly_mat_init(u, r2, 1, prime);
    nmod_poly_mat_init(w, r1, 1, prime);

    for (slong p = 0; p < r2; p++)
        nmod_poly_mul(nmod_poly_mat_entry(u, p, 0), m1,
                      nmod_poly_mat_entry(V, (r1 - 1) * r2 + p, 0));
    for (slong h = 0; h < r1; h++)
        nmod_poly_mul(nmod_poly_mat_entry(w, h, 0), m2,
                      nmod_poly_mat_entry(V, h * r2 + (r2 - 1), 0));

    nmod_poly_t s, t;
    nmod_poly_init(s, prime);
    nmod_poly_init(t, prime);

    for (slong h = 0; h < r1; h++)
    {
        for (slong p = 0; p < r2; p++)
        {
            nmod_poly_struct * out = nmod_poly_mat_entry(R, h * r2 + p, 0);

            nmod_poly_zero(s);
            if (h > 0)
                nmod_poly_add(s, s, nmod_poly_mat_entry(V, (h - 1) * r2 + p, 0));
            if (p > 0)
                nmod_poly_add(s, s, nmod_poly_mat_entry(V, h * r2 + (p - 1), 0));
            nmod_poly_mul(out, phi, s);

            nmod_poly_mul(t, nmod_poly_mat_entry(L1, h, 0), nmod_poly_mat_entry(u, p, 0));
            nmod_poly_sub(out, out, t);

            nmod_poly_mul(t, nmod_poly_mat_entry(L2, p, 0), nmod_poly_mat_entry(w, h, 0));
            nmod_poly_sub(out, out, t);
        }
    }

    nmod_poly_clear(s);
    nmod_poly_clear(t);
    nmod_poly_mat_clear(u);
    nmod_poly_mat_clear(w);
}
