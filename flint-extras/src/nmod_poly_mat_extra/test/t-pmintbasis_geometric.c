/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

/** Targets nmod_poly_mat_pmintbasis_geometric. Checks it against the
 * general-points nmod_poly_mat_pmintbasis, called with the SAME points
 * (the ones the geometric version actually used, returned via its own
 * `pts` output parameter): since both make the same D&C splits and share
 * the same mintbasis base case, only the residual step's method of
 * computing the same mathematical quantity differs, so the two should
 * agree bit-for-bit, not just up to the defining property -- a strong
 * test, since geometric points are valid general points too. Also
 * re-checks the defining property directly on the geometric version's
 * own output, as an independent cross-check, now via nmod_poly_mat_is_
 * interpolant_basis (verification.c, this same staging area). 
 *
 * Shift and E diversity, and the small-primes stress block, mirror
 * t-pmintbasis.c's own (see its header comment for the rationale) --
 * ported here with one adjustment specific to geometric points:
 *   - the small-primes block's `d` range is capped by `p > 2d+1` (this
 *     algorithm's own geometric-points requirement, not just "d distinct
 *     points < p" as in the general-points file), so it reaches much
 *     smaller d at a given prime, and primes 2/3 are skipped entirely
 *     (they only admit d=0, already covered by ordinary d=0 draws in the
 *     main loop's own d range).
 */

#include <flint/test_helpers.h>
#include <flint/nmod_poly.h>
#include <flint/ulong_extras.h>

#include "nmod_poly_mat_interpolant.h"
#include "testing_collection.h"

/* One of testing_collection.h's 8 shift forms, chosen by index (matching
   t-pmbasis.c's own one_test_pmbasis, which tries all 8 on a single input;
   here we pick one at random per trial instead */ 
static void gen_shift(slong * shift, slong m, slong d, flint_rand_t state)
{
    switch (n_randint(state, 8))
    {
        case 0: _test_collection_shift_uniform(shift, m); break;
        case 1: _test_collection_shift_increasing(shift, m); break;
        case 2: _test_collection_shift_decreasing(shift, m); break;
        case 3: _test_collection_shift_shuffle(shift, m, state); break;
        case 4: _test_collection_shift_hermite(shift, m, d); break;
        case 5: _test_collection_shift_rhermite(shift, m, d); break;
        case 6: _test_collection_shift_plateau(shift, m, d); break;
        default: _test_collection_shift_rplateau(shift, m, d); break;
    }
}

/* One of testing_collection.h's matrix forms, chosen by index (same
   rationale as gen_shift above: pick one per trial rather than looping over
   all of them). _test_collection_mat_rkdef degrades to the zero matrix on
   its own when m <= 1 or n <= 1 (its own convention), so no guard is needed
   here for that case. */
static void gen_E(nmod_poly_mat_t E, slong d, flint_rand_t state)
{
    switch (n_randint(state, 7))
    {
        case 0: _test_collection_mat_zero(E); break;
        case 1: _test_collection_mat_uniform(E, d, state); break;
        case 2: _test_collection_mat_unbalanced_rdeg(E, d, state); break;
        case 3: _test_collection_mat_unbalanced_cdeg(E, d, state); break;
        case 4: _test_collection_mat_test(E, d, state); break;
        case 5: _test_collection_mat_sparse(E, d, state); break;
        default: _test_collection_mat_rkdef(E, d, state); break;
    }
}

static int core_test(slong m, slong n, slong d, ulong p, flint_rand_t state)
{
    /* r must have sufficient multiplicative order: none of the first d
       powers of r^2 should be 1 -- matching FLINT's own test convention
       (nmod_poly/test/t-evaluate_geometric_nmod_vec_fast.c); caller
       guarantees p > 2d+1 and p prime. */
    ulong r = n_primitive_root_prime(p);

    nmod_poly_mat_t E;
    nmod_poly_mat_init(E, m, n, p);
    gen_E(E, d, state);

    slong * shift0 = flint_malloc(m * sizeof(slong));
    gen_shift(shift0, m, d, state);
    slong * shift_geo = flint_malloc(m * sizeof(slong));
    slong * shift_gen = flint_malloc(m * sizeof(slong));
    for (slong i = 0; i < m; i++)
        shift_geo[i] = shift_gen[i] = shift0[i];

    ulong * pts = flint_malloc(FLINT_MAX(d, 1) * sizeof(ulong));

    nmod_poly_mat_t intbas_geo, intbas_gen;
    nmod_poly_mat_init(intbas_geo, m, m, p);
    nmod_poly_mat_init(intbas_gen, m, m, p);

    nmod_poly_mat_pmintbasis_geometric(intbas_geo, shift_geo, E, r, d, pts);
    nmod_poly_mat_pmintbasis(intbas_gen, shift_gen, E, pts, d);

    int res = nmod_poly_mat_equal(intbas_geo, intbas_gen);
    for (slong i = 0; i < m; i++)
        if (shift_geo[i] != shift_gen[i])
            res = 0;

    /* defining property, directly on the geometric version's own output */
    if (res && !nmod_poly_mat_is_interpolant_basis(intbas_geo, E, pts, d, shift0, ROW_LOWER))
        res = 0;

    nmod_poly_mat_clear(intbas_geo);
    nmod_poly_mat_clear(intbas_gen);
    nmod_poly_mat_clear(E);
    flint_free(pts);
    flint_free(shift0);
    flint_free(shift_geo);
    flint_free(shift_gen);

    return res;
}

TEST_FUNCTION_START(nmod_poly_mat_pmintbasis_geometric, state)
{
    int i, result;

    for (i = 0; i < 100 * flint_test_multiplier(); i++)
    {
        slong n = 1 + n_randint(state, 16);
        slong m = n + n_randint(state, 16);
        slong d = 1 + n_randint(state, 150);

        /* nbits' floor must guarantee some prime of that bit length
           exceeds bound = 2d+1 (this algorithm's own requirement), or the
           retry loop below never terminates -- see t-pmintbasis.c's own
           version of this same fix for the full rationale. */
        ulong bound = (ulong) (2 * d + 1);
        ulong nbits = FLINT_BIT_COUNT(bound) + 1 + n_randint(state, 50);
        ulong p;
        do { p = n_randprime(state, nbits, 1); } while (p <= bound);

        result = core_test(m, n, d, p, state);

        if (!result)
            TEST_FUNCTION_FAIL("m = %wd, n = %wd, d = %wd, p = %wu\n", m, n, d, p);
    }

    /* explicit small-prime stress coverage, see header comment above */
    {
        ulong small_primes[] = {5, 7, 11, 13};
        for (slong sp = 0; sp < (slong)(sizeof(small_primes) / sizeof(small_primes[0])); sp++)
        {
            ulong p = small_primes[sp];
            slong max_d = (slong) ((p - 2) / 2); /* largest d with p > 2d+1 */

            for (i = 0; i < 20 * flint_test_multiplier(); i++)
            {
                slong n = 1 + n_randint(state, 4);
                slong m = n + n_randint(state, 4);
                slong d = n_randint(state, max_d + 1); /* 0 <= d <= max_d */

                result = core_test(m, n, d, p, state);

                if (!result)
                    TEST_FUNCTION_FAIL("small-prime case: p = %wu, m = %wd, n = %wd, d = %wd\n",
                                       p, m, n, d);
            }
        }
    }

    TEST_FUNCTION_END(state);
}
