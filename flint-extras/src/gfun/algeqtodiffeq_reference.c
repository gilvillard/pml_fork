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
    nmod_pseudo_Krylov_naive_delta (2026-09-16, renamed/relocated from
    nmod_pseudo_Krylov_for_kernel, gfun.c) is not one of the four algorithm
    families (naive/width1/series/section4, see CLAUDE.md) -- it's kept in
    its own file because of what it's genuinely independent OF: it builds
    the same fraction-free pseudo-Krylov numerator matrix K as the naive
    family, but by going through the older, un-cleaned nmod_pseudo_Krylov
    (Delta-scaled throughout, not rescaled to phi1) rather than
    nmod_pseudo_Krylov_naive's own phi1-optimized code path. That makes it
    genuinely independent of both nmod_pseudo_Krylov_naive AND the width-1
    family's own construction, which is exactly why tests/t-algeq-to-diffeq-
    naive.c and tests/t-algeq-to-diffeq-width1.c (claude-pseudoKrylov) use
    it as their trusted, unrelated reference for the defining-property
    check on both drivers, rather than reusing either family's own code
    path to check itself.

    Still an active dependency beyond testing: gfun.c's own
    CRT_pseudo_Krylov_for_kernel (part of the not-yet-started CRT/fmpz
    reconstruction track, see claude-pseudoKrylov/todo.md item 4) calls it
    directly. Deliberately NOT touched further than this relocation/rename
    -- rewriting it to share code with the families it validates would
    defeat its purpose as an independent check; a fuller cleanup pass (if
    ever wanted) belongs with the CRT track's own turn, not here.
*/

/** Builds the r x n fraction-free pseudo-Krylov matrix K for a = y (the
 *  monomial basis vector (0,1,0,...,0)^t), by the Delta-scaled route: sets
 *  up its own Delta/CT (nmod_biv_resultant_geometric,
 *  nmod_biv_mulmod_geometric) and calls the old nmod_pseudo_Krylov with
 *  phi1 forced equal to Delta -- i.e. deliberately bypassing the phi1
 *  optimization nmod_pseudo_Krylov_naive relies on, staying at Delta-scale
 *  throughout. Column k (0-indexed) is theta^k(y) as an ordinary
 *  polynomial (already rescaled back from nmod_pseudo_Krylov's own
 *  varying-power-of-phi1 internal representation, see the loop below) --
 *  unlike nmod_pseudo_Krylov_naive's K, this one's columns are directly
 *  comparable without an extra phi1-power correction.
 *
 *  Precondition: n >= 1 (not checked here, matches the draft this was
 *  relocated from -- see nmod_pseudo_Krylov_naive's own n=0 precondition
 *  note for why that matters elsewhere; this function's own n=0 safety
 *  hasn't been separately audited since it's not meant for direct new use,
 *  only as the existing dependents above rely on it).
 */
void nmod_pseudo_Krylov_naive_delta(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t PT)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);

    nmod_poly_t Delta;
    nmod_poly_init(Delta, prime);
    nmod_poly_mat_t iPyT;
    nmod_poly_mat_init(iPyT, r, 1, prime);
    nmod_biv_resultant_geometric(Delta, iPyT, PT);

    nmod_poly_mat_t PxT;
    nmod_poly_mat_init(PxT, r + 1, 1, prime);
    for (slong i = 0; i < r + 1; i++)
    {
        nmod_poly_derivative(nmod_poly_mat_entry(PxT, i, 0), nmod_poly_mat_entry(PT, i, 0));
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(PxT, i, 0), nmod_poly_mat_entry(PxT, i, 0), prime - 1);
    }

    slong D = (2 * r - 1) * d - 1;
    nmod_poly_mat_t CT;
    nmod_poly_mat_init(CT, r, 1, prime);
    nmod_biv_mulmod_geometric(CT, PxT, iPyT, PT, D);

    /* phi1 forced equal to Delta -- see the file header comment above for
     * why this is the whole point of this function. */
    nmod_poly_t phi1;
    nmod_poly_init(phi1, prime);
    nmod_poly_set(phi1, Delta);

    nmod_pseudo_Krylov(K, n, CT, PT, phi1, Delta);

    /* nmod_pseudo_Krylov's own column j represents phi1^j * theta^j(a):
     * rescale back to an ordinary polynomial matrix, column j needing an
     * extra phi1^{(n-1)-j} to match column n-1's normalization (same
     * rescaling nmod_algeq_to_diffeq_naive applies to its own K). */
    nmod_poly_t tpol;
    nmod_poly_init(tpol, prime);
    nmod_poly_one(tpol);
    for (slong j = (slong) n - 2; j >= 0; j--)
    {
        nmod_poly_mul(tpol, tpol, phi1);
        for (slong i = 0; i < r; i++)
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
    }

    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(PxT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
    nmod_poly_clear(tpol);
}
