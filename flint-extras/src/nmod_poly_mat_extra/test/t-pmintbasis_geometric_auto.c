/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

/* Quick test for nmod_poly_mat_pmintbasis_geometric_auto
 * (interpolant_basis_geometric_auto.c): a thin wrapper around
 * nmod_poly_mat_pmintbasis_geometric that finds `r` itself (via
 * nmod_find_root) instead of taking it as a parameter. Since it delegates
 * to the exact same underlying algorithm, this doesn't re-derive
 * correctness from scratch -- it just checks the wrapper wires things up
 * right: the output is a genuine minimal interpolant basis for the points
 * it actually used (nmod_poly_mat_is_interpolant_basis, verification.c),
 * and it agrees bit-for-bit with the general-points nmod_poly_mat_pmintbasis
 * on those same points (matching t-pmintbasis_geometric.c's own such check
 * against the r-explicit version). Also covers the d=0 edge case, which
 * the wrapper must never throw on regardless of how small the modulus is
 * (see interpolant_basis_geometric_auto.c's own doc comment).
 */

#include <flint/test_helpers.h>
#include <flint/nmod_poly.h>
#include <flint/ulong_extras.h>

#include "nmod_poly_mat_interpolant.h"


static int core_test_pmintbasis_geometric_auto(slong m, slong n, slong d, ulong p, flint_rand_t state)
{
    nmod_poly_mat_t E;
    nmod_poly_mat_init(E, m, n, p);
    nmod_poly_mat_rand(E, state, d);

    slong * shift0 = flint_malloc(m * sizeof(slong));
    for (slong i = 0; i < m; i++)
        shift0[i] = (n_randint(state, 4) == 0) ? n_randint(state, 20) : 0;
    slong * shift_auto = flint_malloc(m * sizeof(slong));
    slong * shift_gen = flint_malloc(m * sizeof(slong));
    for (slong i = 0; i < m; i++)
        shift_auto[i] = shift_gen[i] = shift0[i];

    ulong * pts = flint_malloc(FLINT_MAX(d, 1) * sizeof(ulong));

    nmod_poly_mat_t intbas_auto, intbas_gen;
    nmod_poly_mat_init(intbas_auto, m, m, p);
    nmod_poly_mat_init(intbas_gen, m, m, p);

    nmod_poly_mat_pmintbasis_geometric_auto(intbas_auto, shift_auto, E, d, pts);
    nmod_poly_mat_pmintbasis(intbas_gen, shift_gen, E, pts, d);

    int res = nmod_poly_mat_equal(intbas_auto, intbas_gen);
    for (slong i = 0; i < m; i++)
        if (shift_auto[i] != shift_gen[i])
            res = 0;

    if (res && !nmod_poly_mat_is_interpolant_basis(intbas_auto, E, pts, d, shift0, ROW_LOWER))
        res = 0;

    nmod_poly_mat_clear(intbas_auto);
    nmod_poly_mat_clear(intbas_gen);
    nmod_poly_mat_clear(E);
    flint_free(pts);
    flint_free(shift0);
    flint_free(shift_auto);
    flint_free(shift_gen);

    return res;
}

TEST_FUNCTION_START(nmod_poly_mat_pmintbasis_geometric_auto, state)
{
    int i, result;

    /* explicit d=0 regression case, including very small primes -- the
       wrapper must never throw here (it skips root-finding entirely when
       d=0, see interpolant_basis_geometric_auto.c) */
    {
        ulong small_primes[] = {2, 3, 5, 7, 11};
        for (i = 0; i < 20; i++)
        {
            slong n = 1 + n_randint(state, 10);
            slong m = n + n_randint(state, 10);
            ulong prime = small_primes[n_randint(state, 5)];

            nmod_poly_mat_t E, out;
            nmod_poly_mat_init(E, m, n, prime);
            nmod_poly_mat_init(out, m, m, prime);
            slong * shift = flint_calloc(m, sizeof(slong));

            nmod_poly_mat_pmintbasis_geometric_auto(out, shift, E, 0, NULL);
            if (!nmod_poly_mat_is_one(out))
                TEST_FUNCTION_FAIL("d=0 case: m = %wd, n = %wd, prime = %wu\n", m, n, prime);

            nmod_poly_mat_clear(E);
            nmod_poly_mat_clear(out);
            flint_free(shift);
        }
    }

    for (i = 0; i < 100 * flint_test_multiplier(); i++)
    {
        slong n = 1 + n_randint(state, 16);
        slong m = 1 + n_randint(state, 16);
        slong d = 1 + n_randint(state, 150);

        /* same dynamic-floor rationale as t-pmintbasis_geometric.c's own
           prime search: the floor must guarantee some prime of that bit
           length exceeds bound = 2d+1, or the retry loop never
           terminates. */
        ulong bound = (ulong) (2 * d + 1);
        ulong nbits = FLINT_BIT_COUNT(bound) + 1 + n_randint(state, 50);
        ulong p;
        do { p = n_randprime(state, nbits, 1); } while (p <= bound);

        result = core_test_pmintbasis_geometric_auto(m, n, d, p, state);

        if (!result)
            TEST_FUNCTION_FAIL("m = %wd, n = %wd, d = %wd, p = %wu\n", m, n, d, p);
    }

    TEST_FUNCTION_END(state);
}
