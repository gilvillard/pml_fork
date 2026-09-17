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

    nmod_pseudo_Krylov itself moved here too (2026-09-17, per the user),
    grouped with its own project-owned caller above -- prompted by the user
    noticing its debug flint_printf ("K: n-th column: ... sec.") firing on
    every test run through nmod_pseudo_Krylov_naive_delta. Relocation +
    that one print dropped only (same treatment nmod_pseudo_Krylov_iterative
    got during its own relocation); no other behavior change -- still
    called from gfun.c's own nmod_algeq_to_diffeq (kept alive for the
    external ../work-algeqtodiffeq project) via the unchanged declaration
    in gfun.h. Per the user (2026-09-17): "We will come back to prints and
    traces when we will experiment" -- so this is a one-off removal
    prompted by this specific relocation, not the start of a general
    debug-print sweep across the rest of gfun.c.
*/

/** Computation of the numerators of the pseudo-Krylov matrix, an r x n
 *  polynomial matrix, fraction-free (uses phi1 as computed by the caller,
 *  non-monic since directly related to the resultant, itself non-monic).
 *  Relocated verbatim from gfun.c (2026-09-17, see the file header comment
 *  above) except for one dropped debug flint_printf inside the main loop
 *  -- no other change.
 */
void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT,
                         const nmod_poly_mat_t PT, const nmod_poly_t phi1, const nmod_poly_t Delta)
{
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);
    ulong prime = nmod_poly_mat_modulus(PT);

    nmod_poly_mat_t tempN; /* for calling ffT */
    nmod_poly_mat_init(tempN, r, 1, prime);

    nmod_poly_mat_t temp;
    nmod_poly_mat_init(temp, r, 1, prime);

    nmod_poly_t p1, p2;
    nmod_poly_init(p1, prime);
    nmod_poly_init(p2, prime);

    nmod_poly_t dphi1;
    nmod_poly_init(dphi1, prime);
    nmod_poly_derivative(dphi1, phi1);

    slong deg_phi1 = nmod_poly_degree(phi1);

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_div(g, Delta, phi1);

    /* First column: y. */
    for (slong i = 0; i < r; i++)
        nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(K, 1, 0), 0, 1);

    /* Main loop, for the n-1 new columns of K. */
    slong D = 0;
    for (slong k = 0; k < (slong) n - 1; k++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(tempN, i, 0), nmod_poly_mat_entry(K, i, k));

        /* We add the degree (2r-2)d + (d-1) = (2r-1)d-1 for M^* and Y
         * (temporarily) when we apply T, (2r-1)d-1 is ok (bounds the
         * x-degree of the resultant, btw) and then, afterwards, we will
         * recover the degree of phi1 by simplification -- TO CHECK. */
        D = k * deg_phi1 + (2 * r - 1) * d - 1;

        nmod_apply_T(temp, tempN, CT, PT, D);

        for (slong i = 0; i < r; i++)
        {
            nmod_poly_div(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(temp, i, 0), g);

            nmod_poly_derivative(p1, nmod_poly_mat_entry(K, i, k));
            nmod_poly_mul(p1, p1, phi1);

            nmod_poly_mul(p2, dphi1, nmod_poly_mat_entry(K, i, k));
            nmod_poly_scalar_mul_nmod(p2, p2, k);

            nmod_poly_add(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(K, i, k + 1), p1);
            nmod_poly_sub(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(K, i, k + 1), p2);
        }
    }

    nmod_poly_mat_clear(temp);
    nmod_poly_mat_clear(tempN);
    nmod_poly_clear(g);
    nmod_poly_clear(dphi1);
    nmod_poly_clear(p1);
    nmod_poly_clear(p2);
}


/** Builds the r x n fraction-free pseudo-Krylov matrix K for a = y (the
 *  monomial basis vector (0,1,0,...,0)^t), by the Delta-scaled route: sets
 *  up its own Delta/CT (nmod_biv_resultant_geometric,
 *  nmod_biv_mulmod_geometric) and calls the old nmod_pseudo_Krylov with
 *  phi1 forced equal to Delta -- i.e. deliberately bypassing the phi1
 *  optimization nmod_pseudo_Krylov_naive relies on, staying at Delta-scale
 *  throughout.
 *
 *  Output convention (CORRECTED 2026-09-17, see the removed-loop comment in
 *  the body): column k (0-indexed) is the fraction-free NUMERATOR
 *  Delta^k * theta^k(y), with the denominator Delta^k left IMPLICIT --
 *  exactly the same convention as nmod_pseudo_Krylov_naive's own K (with
 *  Delta in place of phi1), and this module's convention generally.
 *  theta^k(y) itself is a genuine RATIONAL function for k >= 1, so there is
 *  no "ordinary polynomial" form of it for this function to return -- the
 *  doc used to claim there was, and a loop here used to force every column
 *  to one common implicit denominator Delta^{n-1} in an attempt at it,
 *  which also made the output depend on n. Both are gone.
 *
 *  A caller that needs a genuinely polynomial matrix (e.g. to take a kernel,
 *  or to check a relation sum_k eta_k*theta^k(y)=0 against a solution) must
 *  account for those per-column denominators itself: either rescale column k
 *  by Delta^{(n-1)-k} to reach one common denominator Delta^{n-1} (what
 *  nmod_algeq_to_diffeq_naive does to its own K before its kernel call), or
 *  equivalently weight eta_k by Delta^{(n-1)-k} instead (what this project's
 *  tests do -- cheaper, and it leaves K's meaning untouched).
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

    /* No rescale loop here: nmod_pseudo_Krylov's own column j IS the
     * answer, in this module's fraction-free convention -- column j is the
     * NUMERATOR Delta^j * theta^j(a), with the denominator Delta^j left
     * implicit (per the user, 2026-09-17: "to not introduce denominators we
     * only compute the numerators, and denominators are implicitly the
     * powers of either delta or phi1"). theta^j(a) itself is a genuine
     * RATIONAL function for j >= 1, so there is no "ordinary polynomial"
     * form of it to rescale back to -- which is exactly why the loop that
     * used to be here should not be (the user, same day: "the last loop in
     * nmod_pseudo_Krylov_naive_delta should not be there"). What it
     * actually did was force every column to one COMMON implicit
     * denominator Delta^{n-1} (multiplying column j by Delta^{(n-1)-j}) --
     * a legitimate operation in itself, but one that belongs to a CALLER
     * preparing a kernel computation, not baked in here, exactly as
     * nmod_pseudo_Krylov_naive leaves it to nmod_algeq_to_diffeq_naive. It
     * also made this function's output silently depend on n. */

    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(PxT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
}
