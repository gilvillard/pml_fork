/*
    Copyright (C) 2026 Vincent Neiger, Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <flint/nmod_mat.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_mat.h>
#include "nmod_poly_mat_forms.h"
#include "nmod_poly_mat_extra.h"  /* for prototypes */
#include "nmod_poly_mat_multiply.h"
#include "nmod_poly_mat_interpolant.h"

/* TODO currently specialized to ROW_LOWER (or at least ROW_stuff) */
int nmod_poly_mat_is_approximant_basis(const nmod_poly_mat_t appbas,
                                       const nmod_poly_mat_t pmat,
                                       slong order,
                                       const slong * shift,
                                       orientation_t orient)
{
    const slong rdim = pmat->r;
    const slong cdim = pmat->c;
    const ulong prime = pmat->modulus;

    nmod_poly_mat_t residual;
    nmod_poly_t pol;
    nmod_mat_t CP0;

    nmod_poly_init(pol, prime);
    nmod_poly_mat_init(residual, rdim, cdim, prime);
    nmod_mat_init(CP0, rdim, rdim+cdim, prime);

    int success = 1;

    /* check appbas is square with the right dimension */
    if (appbas->r != rdim || appbas->c != rdim)
    {
        printf("basis has wrong row dimension or column dimension\n");
        success = 0;
    }

    /* check appbas is shifted reduced */
    if (!nmod_poly_mat_is_ordered_weak_popov(appbas, shift, orient))
    {
        printf("basis is not shifted-weak Popov\n");
        success = 0;
    }

    /* compute residual, check rows of appbas are approximants */
    nmod_poly_mat_multiply(residual, appbas, pmat);

    for (slong i = 0; i < rdim; i++)
    {
        for (slong j = 0; j < cdim; j++)
        {
            nmod_poly_set_trunc(pol, nmod_poly_mat_entry(residual, i, j), order);
            if (!nmod_poly_is_zero(pol))
            {
                printf("entry %ld, %ld of residual has valuation less than the order\n",i,j);
                success = 0;
            }
        }
    }

    /* check generation: follows ideas from Algorithm 1 in Giorgi-Neiger, ISSAC 2018 */

    /* generation, test 1: check determinant of appbas is lambda * x**D      *
     * since ordered weak Popov, deg(det(appbas)) is sum of diagonal degrees *
     */
    slong D = 0;
    for (slong i = 0; i < rdim; i++)
        D += nmod_poly_degree(nmod_poly_mat_entry(appbas, i, i));
    nmod_poly_mat_det(pol, appbas);
    if (nmod_poly_degree(pol) != D)
    {
        printf("determinant is not lambda * x**(sum(diag-deg))");
        success = 0;
    }

    /* generation, test 2: check that [P(0)  C] has full rank *
     * where C = (appbas * pmat * X^{-order})  mod X          *
     * (coefficient "C" of degree order of the residual)      *
     **/
    for (slong i = 0; i < rdim; i++)
    {
        for (slong j = 0; j < rdim; j++)
        {
            ulong c = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(appbas, i, j), 0);
            nmod_mat_set_entry(CP0, i, j, c);
        }
        for (slong j = 0; j < cdim; j++)
        {
            ulong c = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(residual, i, j), order);
            nmod_mat_set_entry(CP0, i, rdim+j, c);
        }
    }

    slong rank = nmod_mat_rank(CP0);
    if (rank != rdim)
    {
        printf("generation test (step: [C P(0)] has full rank) failed");
        success = 0;
    }

    nmod_poly_clear(pol);
    nmod_poly_mat_clear(residual);
    nmod_mat_clear(CP0);

    return success;
}

/**  Mirrors is_approximant basis structure closely (same three checks: shape,
 * shift-reduced form, membership, generation), swapping the two pieces
 * that are genuinely order-truncation-specific for their points-based
 * counterparts. 
 *
 * - Membership: nmod_poly_mat_is_approximant_basis truncates the product
 *   appbas*pmat mod x^order and checks it is zero. Here, intbas is
 *   evaluated at each of the d points and multiplied by the corresponding
 *   evaluation E_k (E's own degree-k coefficient, per this header's
 *   "Conventions" -- not a truncation, a genuine pointwise product), and
 *   each of the d products must be zero.
 *
 * - Generation (does intbas generate the whole interpolant module, not
 *   just a valid sub-basis?): nmod_poly_mat_is_approximant_basis runs two
 *   checks here:
 *     (1) deg(det(appbas)) == sum of diagonal degrees of appbas. 
           Ported here verbatim below, purely as the same kind of
 *         implementation-bug safety net (an independent nmod_poly_mat_det
 *         call cross-checked against the diagonal-degree sum). 
 *     (2) a Monte Carlo full-rank test on [P(0) | C] (C being the
 *         order-truncated residual coefficient at degree order) -- this is
 *         the test that actually certifies generation. 
 *   For interpolants, check (2)'s role is instead filled directly, with no
 *   Monte Carlo needed and no genericity caveat: since the points are
 *   pairwise distinct, K[x]/(prod_k(x-pts_k)) splits via CRT into
 *   direct_sum_k K[x]/(x-pts_k) = direct_sum_k K (one field per point).
 *   The interpolant module I contains M(x)*K[x]^{1xm} unconditionally
 *   (M(x) = prod_k(x-pts_k) vanishes at every point, so M(x)*e_i is always
 *   an interpolant), and under the CRT splitting, I / (M(x)*K[x]^{1xm})
 *   corresponds exactly to direct_sum_k ker_left(E_k), of K-dimension
 *   sum_k (m - rank(E_k)). Combined with the standard facts
 *   dim_K(K[x]^{1xm}/I) = deg(det(any basis of I)) and
 *   dim_K(K[x]^{1xm}/(M(x)*K[x]^{1xm})) = m*d, a short exact sequence gives,
 *   deg(det(any basis of I)) = sum_k rank(E_k).
 *
 *   So "deg(det(intbas)) == sum_k rank(E_k)" is an exact formula for
 *   the same quantity check (2).
 */


/* TODO currently specialized to ROW_LOWER (or at least ROW_stuff), same
 * limitation as nmod_poly_mat_is_approximant_basis in this same file. */
int nmod_poly_mat_is_interpolant_basis(const nmod_poly_mat_t intbas,
                                       const nmod_poly_mat_t E,
                                       const ulong * pts,
                                       slong d,
                                       const slong * shift,
                                       orientation_t orient)
{
    const slong rdim = E->r;
    const slong cdim = E->c;
    const ulong prime = E->modulus;

    int success = 1;

    /* check intbas is square with the right dimension */
    if (intbas->r != rdim || intbas->c != rdim)
    {
        printf("basis has wrong row dimension or column dimension\n");
        success = 0;
    }

    /* check intbas is shifted reduced */
    if (!nmod_poly_mat_is_ordered_weak_popov(intbas, shift, orient))
    {
        printf("basis is not shifted-weak Popov\n");
        success = 0;
    }

    /* check rows of intbas are interpolants: intbas(pts[k]) * E_k == 0
     * for every one of the d points */
    nmod_mat_t Ik, Ek, prod;
    nmod_mat_init(Ik, rdim, rdim, prime);
    nmod_mat_init(Ek, rdim, cdim, prime);
    nmod_mat_init(prod, rdim, cdim, prime);

    for (slong k = 0; k < d; k++)
    {
        nmod_poly_mat_evaluate_nmod(Ik, intbas, pts[k]);
        for (slong i = 0; i < rdim; i++)
            for (slong j = 0; j < cdim; j++)
                nmod_mat_entry(Ek, i, j) = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(E, i, j), k);
        nmod_mat_mul(prod, Ik, Ek);
        if (!nmod_mat_is_zero(prod))
        {
            printf("intbas(pts[%ld]) * E_%ld is not zero\n", (long) k, (long) k);
            success = 0;
        }
    }

    /** check generation, using nmod_poly_mat_det, must equal D,
     * the sum of diagonal degrees of intbas. 
     * A cheap cross-check between two different code paths,
     * catching bugs in either nmod_poly_mat_is_ordered_weak_popov or this
     * diagonal-degree summation, kept purely as defense-in-depth.
     */
    slong D = 0;
    for (slong i = 0; i < rdim; i++)
        D += nmod_poly_degree(nmod_poly_mat_entry(intbas, i, i));

    nmod_poly_t det;
    nmod_poly_init(det, prime);
    nmod_poly_mat_det(det, intbas);
    if (nmod_poly_degree(det) != D)
    {
        printf("determinant degree (%ld) != sum of diagonal degrees (%ld)\n", (long) nmod_poly_degree(det), (long) D);
        success = 0;
    }
    nmod_poly_clear(det);

    /** check generation, test 2 (the actual generation/minimality test:      
     * D must equal sum_k rank(E_k). */
    slong target_D = 0;
    for (slong k = 0; k < d; k++)
    {
        for (slong i = 0; i < rdim; i++)
            for (slong j = 0; j < cdim; j++)
                nmod_mat_entry(Ek, i, j) = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(E, i, j), k);
        target_D += nmod_mat_rank(Ek);
    }

    if (D != target_D)
    {
        printf("sum of diagonal degrees (%ld) != sum of ranks of evaluation matrices (%ld)\n", (long) D, (long) target_D);
        success = 0;
    }

    nmod_mat_clear(Ik);
    nmod_mat_clear(Ek);
    nmod_mat_clear(prod);

    return success;
}



/* TODO currently does not check generation */
int nmod_poly_mat_is_kernel(const nmod_poly_mat_t ker,
                            const slong * shift,
                            const nmod_poly_mat_t mat,
                            poly_mat_form_t form,
                            orientation_t orient)
{
    if (orient == ROW_UPPER || orient == ROW_LOWER)
    {
        const slong rdim = mat->r;
        const slong cdim = mat->c;
        const ulong prime = mat->modulus;

        nmod_poly_mat_t residual;
        nmod_poly_mat_init(residual, rdim, cdim, prime);

        int success = 1;

        /* check kernel has the right column dimension */
        if (ker->c != rdim)
        {
            printf("basis has wrong column dimension\n");
            success = 0;
        }

        /* check rank */
        slong rk = nmod_poly_mat_rank(mat);
        if (ker->r != rdim - rk)
        {
            printf("number of rows does not equal nullity\n");
            success = 0;
        }

        /* check kernel has the right form */
        if (!nmod_poly_mat_is_form(ker, form, shift, orient))
        {
            printf("basis does not have the required form\n");
            success = 0;
        }

        /* compute residual, check rows of ker are in the kernel */
        nmod_poly_mat_multiply(residual, ker, mat);
        if (!nmod_poly_mat_is_zero(residual))
        {
            printf("not all rows are in the kernel\n");
            success = 0;
        }

        nmod_poly_mat_clear(residual);

        return success;
    }

    if (orient == COL_UPPER || orient == COL_LOWER)
    {
        const slong rdim = mat->r;
        const slong cdim = mat->c;
        const ulong prime = mat->modulus;

        nmod_poly_mat_t residual;
        nmod_poly_mat_init(residual, rdim, cdim, prime);

        int success = 1;

        /* check kernel has the right column dimension */
        if (ker->r != cdim)
        {
            printf("basis has wrong row dimension\n");
            success = 0;
        }

        /* check rank */
        slong rk = nmod_poly_mat_rank(mat);
        if (ker->c != cdim - rk)
        {
            printf("number of rows does not equal nullity\n");
            success = 0;
        }

        /* check kernel has the right form */
        if (!nmod_poly_mat_is_form(ker, form, shift, orient))
        {
            printf("basis does not have the required form\n");
            success = 0;
        }

        /* compute residual, check rows of ker are in the kernel */
        nmod_poly_mat_multiply(residual, mat, ker);
        if (!nmod_poly_mat_is_zero(residual))
        {
            printf("not all rows are in the kernel\n");
            success = 0;
        }

        nmod_poly_mat_clear(residual);

        return success;
    }

    flint_throw(FLINT_ERROR, "Exception (nmod_poly_mat_is_kernel). Requested orientation not implemented.");
}
