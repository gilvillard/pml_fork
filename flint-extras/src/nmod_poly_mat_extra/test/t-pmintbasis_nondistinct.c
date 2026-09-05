/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

/** Formalizes the exploration of the non-distinct points case.
 *
 * Of nmod_poly_mat_is_interpolant_basis's (verification.c) three checks,
 * only one depends on distinctness:
 *   1. shape (square) -- property of `intbas` alone, untouched.
 *   2. shift-ordered-weak-Popov form -- also a property of `intbas` alone
 *      (nmod_poly_mat_is_ordered_weak_popov, reused verbatim).
 *   3. membership: intbas(pts[k])*E_k == 0 for every k -- evaluate-and-
 *      check per k, independent of whether some k's share a point value.
 *      Untouched.
 *   4. generation: deg(det(intbas)) == sum_k rank(E_k). This is the one
 *      that changes. That formula is derived from K[x]/(prod_k(x-pts_k))
 *      splitting via CRT into direct_sum_k K[x]/(x-pts_k) -- valid only
 *      when the pts_k are pairwise distinct (coprime linear factors). A
 *      repeated point v makes (x-v) appear more than once in the product;
 *      repeated linear factors are not coprime, so that direct-sum
 *      decomposition does not hold, and summing rank(E_k) over every k
 *      independently overcounts whenever two E_k's sharing a point v have
 *      overlapping column space.
 *
 * Generalized formula TO CHECK.
 * Group the k's by their common point value v; for each distinct v,
 * horizontally concatenate (stack as columns) the E_k's sharing that v
 * into one m x (n*mult(v)) matrix, and take that matrix's rank -- not the
 * sum of the individual rank(E_k)'s, which can overcount when those E_k's
 * are not independent. Then
 *     deg(det(intbas)) == sum_{distinct v} rank(stacked E's at v).
 * This collapses to the ordinary sum_k rank(E_k) exactly when every point
 * is distinct (each group has multiplicity 1), so it is a strict
 * generalization of the real verifier's own generation check.
 * 
 * Checks, per trial:
 *   (1) nmod_mat_poly_mintbasis (M-level, converted via
 *       nmod_poly_mat_set_from_mat_poly) and nmod_poly_mat_pmintbasis
 *       (PM-level, divide-and-conquer) agree bit-for-bit. 
 *       This checks that agreement survives non-distinct points too.
 *   (2) the PM-level output satisfies the generalized defining property
 *       above, w.r.t. the original (pre-call) shift -- shift is mutated in
 *       place by the construction calls, see t-verification.c's own
 *       header comment for the real bug this caught there.
 *
 * `d` ranges both below and above PMINTBASIS_THRES (32), so pmintbasis's
 * base case and its genuine divide-and-conquer recursion are both
 * exercised under non-distinct points -- including the case where the
 * point list is split in half and a repeated point's occurrences land on
 * opposite sides of that split.
 */

#include <flint/test_helpers.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_mat.h>
#include <flint/ulong_extras.h>

#include "nmod_poly_mat_interpolant.h"
#include "nmod_poly_mat_forms.h"
#include "testing_collection.h"
#include "interpolant_test_utils.h"

/* sum over distinct point values of the rank of the horizontally-stacked
   E's sharing that value -- see this file's own header comment for the
   full derivation. Collapses to sum_k rank(E_k) when every point is
   distinct (every group has multiplicity 1). */
static slong sum_rank_stacked_per_distinct_point(const nmod_mat_struct * E,
                                                 const ulong * pts, slong d,
                                                 slong m, slong n, ulong prime)
{
    slong target = 0;
    for (slong k = 0; k < d; k++)
    {
        int first = 1;
        for (slong j = 0; j < k; j++)
            if (pts[j] == pts[k]) { first = 0; break; }
        if (!first)
            continue;

        slong mult = 0;
        for (slong j = k; j < d; j++)
            if (pts[j] == pts[k])
                mult++;

        nmod_mat_t stacked;
        nmod_mat_init(stacked, m, n * mult, prime);
        slong col = 0;
        for (slong j = k; j < d; j++)
            if (pts[j] == pts[k])
            {
                for (slong i = 0; i < m; i++)
                    for (slong c = 0; c < n; c++)
                        nmod_mat_entry(stacked, i, col + c) = nmod_mat_entry(E + j, i, c);
                col += n;
            }
        target += nmod_mat_rank(stacked);
        nmod_mat_clear(stacked);
    }
    return target;
}

/* Generalized verifier, mirroring nmod_poly_mat_is_interpolant_basis's own
   structure (verification.c) with the generation check's formula replaced
   as described in this file's header comment. Prints the same kind of
   diagnostic lines on failure. */
static int is_interpolant_basis_nondistinct(const nmod_poly_mat_t intbas,
                                            const ulong * pts,
                                            const nmod_mat_struct * E,
                                            slong d,
                                            const slong * shift,
                                            orientation_t orient)
{
    slong m = intbas->r;
    slong n = (d > 0) ? E[0].c : 0;
    ulong prime = intbas->modulus;
    int success = 1;

    if (intbas->r != intbas->c)
    {
        printf("basis has wrong row dimension or column dimension\n");
        success = 0;
    }

    if (!nmod_poly_mat_is_ordered_weak_popov(intbas, shift, orient))
    {
        printf("basis is not shifted-weak Popov\n");
        success = 0;
    }

    nmod_mat_t Ik, prod;
    nmod_mat_init(Ik, m, m, prime);
    nmod_mat_init(prod, m, n, prime);
    for (slong k = 0; k < d; k++)
    {
        nmod_poly_mat_evaluate_nmod(Ik, intbas, pts[k]);
        nmod_mat_mul(prod, Ik, E + k);
        if (!nmod_mat_is_zero(prod))
        {
            printf("intbas(pts[%ld]) * E_%ld is not zero\n", (long) k, (long) k);
            success = 0;
        }
    }
    nmod_mat_clear(Ik);
    nmod_mat_clear(prod);

    slong D = 0;
    for (slong i = 0; i < m; i++)
        D += nmod_poly_degree(nmod_poly_mat_entry(intbas, i, i));

    nmod_poly_t det;
    nmod_poly_init(det, prime);
    nmod_poly_mat_det(det, intbas);
    if (nmod_poly_degree(det) != D)
    {
        printf("determinant degree (%ld) != sum of diagonal degrees (%ld)\n",
               (long) nmod_poly_degree(det), (long) D);
        success = 0;
    }
    nmod_poly_clear(det);

    slong target = sum_rank_stacked_per_distinct_point(E, pts, d, m, n, prime);
    if (D != target)
    {
        printf("sum of diagonal degrees (%ld) != sum of ranks of stacked evaluation matrices per distinct point (%ld)\n",
               (long) D, (long) target);
        success = 0;
    }

    return success;
}

/* d points drawn from a pool of `pool_size` distinct values (with
   replacement) -- pool_size < d forces repeats; pool_size == d forces
   every point distinct (the ordinary case), the two ends of the same
   spectrum this file's main loop ranges over. */
static ulong * make_nondistinct_pts(slong d, slong pool_size, ulong prime, flint_rand_t state)
{
    ulong * pool = (ulong *) flint_malloc(pool_size * sizeof(ulong));
    for (slong i = 0; i < pool_size; i++)
    {
        int distinct;
        do
        {
            pool[i] = n_randint(state, prime);
            distinct = 1;
            for (slong j = 0; j < i; j++)
                if (pool[j] == pool[i])
                    distinct = 0;
        } while (!distinct);
    }

    ulong * pts = (ulong *) flint_malloc(FLINT_MAX(d, 1) * sizeof(ulong));
    for (slong k = 0; k < d; k++)
        pts[k] = pool[n_randint(state, pool_size)];

    flint_free(pool);
    return pts;
}

static int core_test(slong m, ulong prime, const nmod_mat_struct * E,
                     const ulong * pts, slong d, const slong * shift0)
{
    slong * sh_m = flint_malloc(m * sizeof(slong));
    slong * sh_pm = flint_malloc(m * sizeof(slong));
    for (slong i = 0; i < m; i++)
        sh_m[i] = sh_pm[i] = shift0[i];

    nmod_mat_poly_t intbasmp;
    nmod_mat_poly_init(intbasmp, m, m, prime);
    nmod_mat_poly_mintbasis(intbasmp, sh_m, pts, E, d);
    nmod_poly_mat_t intbas_m;
    nmod_poly_mat_init(intbas_m, m, m, prime);
    nmod_poly_mat_set_from_mat_poly(intbas_m, intbasmp);
    nmod_mat_poly_clear(intbasmp);

    nmod_poly_mat_t intbas_pm;
    nmod_poly_mat_init(intbas_pm, m, m, prime);
    nmod_poly_mat_pmintbasis(intbas_pm, sh_pm, pts, E, d);

    int res = 1;

    /* (1) M-level and PM-level agree bit-for-bit, even under non-distinct
       points (see header comment) */
    if (!nmod_poly_mat_equal(intbas_m, intbas_pm))
        res = 0;
    for (slong i = 0; i < m; i++)
        if (sh_m[i] != sh_pm[i])
            res = 0;

    /* (2) genuine minimal interpolant basis for the generalized defining
       property, w.r.t. the original shift0 */
    if (res && !is_interpolant_basis_nondistinct(intbas_pm, pts, E, d, shift0, ROW_LOWER))
        res = 0;

    nmod_poly_mat_clear(intbas_m);
    nmod_poly_mat_clear(intbas_pm);
    flint_free(sh_m);
    flint_free(sh_pm);

    return res;
}

TEST_FUNCTION_START(nmod_poly_mat_pmintbasis_nondistinct, state)
{
    int i, result;

    /* explicit maximally-degenerate case: every one of the d points is the
       same single value -- the extreme end of the pool_size spectrum
       (pool_size = 1), not left to chance in the randomized loop below.
       i == 0 forces d = 0 explicitly (the single point repeated zero
       times, i.e. no points at all -- a boundary of this same case, not
       left to a random draw either). i == 1 forces the repeated point
       value itself to be 0 explicitly (X - 0 recentering degenerates to
       plain X, the same degree-bump mbasis's own order-truncation update
       uses -- worth exercising directly rather than leaving to a random
       draw that only occasionally lands on it). Small primes are
       exercised here too (nbits down to 2), unlike the main loop below. */
    for (i = 0; i < 15; i++)
    {
        slong n = 1 + n_randint(state, 10);
        slong m = 1 + n_randint(state, 10);
        slong d = (i == 0) ? 0 : 1 + n_randint(state, 60);

        ulong nbits = 2 + n_randint(state, 60);
        ulong prime = n_randprime(state, nbits, 1);

        ulong single_pt = (i == 1) ? 0 : n_randint(state, prime);
        ulong * pts = flint_malloc(FLINT_MAX(d, 1) * sizeof(ulong));
        for (slong k = 0; k < d; k++)
            pts[k] = single_pt;

        nmod_poly_mat_t E_poly;
        nmod_poly_mat_init(E_poly, m, n, prime);
        gen_E(E_poly, d, state);
        nmod_mat_struct * E = poly_mat_to_array(E_poly, d);
        nmod_poly_mat_clear(E_poly);

        slong * shift0 = flint_malloc(m * sizeof(slong));
        gen_shift(shift0, m, d, state);

        result = core_test(m, prime, E, pts, d, shift0);

        if (!result)
            TEST_FUNCTION_FAIL("all-same-point case: m = %wd, n = %wd, d = %wd, prime = %wu\n",
                               m, n, d, prime);

        free_array(E, d);
        flint_free(pts);
        flint_free(shift0);
    }

    /* main randomized loop: pool_size ranges from 1 (all points identical)
       up to d (every point distinct, the ordinary case) -- this test's
       whole range subsumes the classical distinct-points test as one
       endpoint, not a separate case. d ranges both below and above
       PMINTBASIS_THRES (32), so pmintbasis's base case and its genuine
       recursion are both exercised, including repeats straddling the D&C
       split. */
    for (i = 0; i < 100 * flint_test_multiplier(); i++)
    {
        slong n = 1 + n_randint(state, 16);
        slong m = 1 + n_randint(state, 15);
        slong d = 1 + n_randint(state, 150);
        slong pool_size = 1 + n_randint(state, d);

        ulong bound = (ulong) pool_size;
        ulong nbits = FLINT_BIT_COUNT(bound) + 1 + n_randint(state, 50);
        ulong prime;
        do { prime = n_randprime(state, nbits, 1); } while (prime <= bound);

        ulong * pts = make_nondistinct_pts(d, pool_size, prime, state);

        nmod_poly_mat_t E_poly;
        nmod_poly_mat_init(E_poly, m, n, prime);
        gen_E(E_poly, d, state);
        nmod_mat_struct * E = poly_mat_to_array(E_poly, d);
        nmod_poly_mat_clear(E_poly);

        slong * shift0 = flint_malloc(m * sizeof(slong));
        gen_shift(shift0, m, d, state);

        result = core_test(m, prime, E, pts, d, shift0);

        if (!result)
            TEST_FUNCTION_FAIL("m = %wd, n = %wd, d = %wd, pool_size = %wd, prime = %wu\n",
                               m, n, d, pool_size, prime);

        free_array(E, d);
        flint_free(pts);
        flint_free(shift0);
    }

    TEST_FUNCTION_END(state);
}
