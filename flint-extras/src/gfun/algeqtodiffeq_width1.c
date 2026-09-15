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
    The "width-1" family (algos.pdf Sec. 3: a rank-one decomposition of T,
    exploiting width(T) <= 1, then a triangular description of the
    pseudo-Krylov matrix built from it -- Algorithm 3/DescriptionFromRank1,
    Algorithm 4/PseudoKrylovWidth1). Naming note (2026-09-15, per the
    user): "width", not "rank" -- algos.pdf's own vocabulary is width(T),
    a determinantal-denominator notion (Sec. 3.1), genuinely different
    from matrix rank; the draft's own "_rank1"/"Rank1" naming
    (Description_From_Rank_1, nmod_algeq_to_diffeq_new,
    pm_algeq2diffeq_rank1, mod/Algeq2diffeqRank1) conflates the two and
    will be corrected to "width1"-flavored names as each is cleaned up in
    turn, starting here with find_uv (which doesn't have "rank" in its own
    name, nothing to rename yet).

    find_uv is being cleaned first (2026-09-15), the family's own
    counterpart to nmod_phi1: both compute something from T via a single
    random projection, both rely on width(T) <= 1 for correctness. Not yet
    moved here: Description_From_Rank_1, nmod_algeq_to_diffeq_new -- next,
    one at a time, per the user.
*/


/** Rank-one decomposition of Delta*T (algos.pdf Lemma 3.3, Sec. 3.2):
 *  writing B = Delta*T, finds u, v such that B = u*v^T / e mod Delta for
 *  some scalar e, via one random constant projection on each side (z, w),
 *  then returns U = u, V = v^T/e already normalized (i.e. B = U*V mod
 *  Delta, no leftover e for the caller to divide out) -- everything here
 *  worked in the phi1-scaled setting throughout (see below), so really
 *  U*V = (phi1-scaled T's matrix) mod phi1, not mod Delta; the caller-
 *  facing Delta/phi1/CT convention is otherwise unchanged from the draft.
 *
 *  Why one projection on each side suffices, and where this needs width
 *  <= 1: same argument as nmod_phi1 (see that function's own doc) --
 *  width(T) <= 1 means B = u_0 v_0^T mod Delta for some fixed u_0, v_0,
 *  so random z, w generically recover a scalar multiple of that same
 *  rank-one pair. Does not extend to width > 1.
 *
 *  Inputs: CT, PT, Delta as elsewhere in this module (CT is Delta-scaled,
 *  i.e. as built by nmod_algeqtodiffeq_setup, NOT pre-rescaled by
 *  nmod_algeqtodiffeq_rescale_by_phi1); phi1 already computed by the
 *  caller (e.g. via nmod_phi1). Outputs U (r x 1), V (1 x r), already
 *  nmod_poly_mat_init'd by the caller with the same modulus/shape.
 *
 *  `state` must be initialized by the caller -- same convention and same
 *  reasoning as nmod_phi1 (this function used to reseed a fresh,
 *  function-local flint_rand_t from srand(time(NULL)) on every call,
 *  correlating draws with any other such function called in the same
 *  wall-clock second).
 *
 *  Internal optimization (2026-09-15, found while cleaning this up): T's
 *  r-1 nonzero columns (column 0 is identically zero -- theta(1) = 0,
 *  since d/dy(1) = 0 -- skipped, not computed) are built one at a time via
 *  nmod_apply_T. The draft called nmod_apply_T at the full Delta-based
 *  degree bound for every column, then divided each result by
 *  Delta/phi1 -- the same "big bound, divide after" pattern
 *  nmod_pseudo_Krylov (not nmod_pseudo_Krylov_naive) used. Here, CT is
 *  instead rescaled to phi1-scale ONCE, on a local working copy (CT
 *  itself, the parameter, is const and unchanged), via
 *  nmod_algeqtodiffeq_rescale_CT_by_phi1, so all r-1 nmod_apply_T calls
 *  run at the smaller phi1-based bound directly -- mirrors exactly the
 *  win nmod_pseudo_Krylov_naive gets over nmod_pseudo_Krylov.
 */
void find_uv(nmod_poly_mat_t U, nmod_poly_mat_t V, const nmod_poly_t phi1,
             const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
             const nmod_poly_t Delta, flint_rand_t state)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);
    slong D = nmod_gfun_delta_T_degree_bound(r, d);

    /* Local, phi1-scaled working copy of CT -- see the optimization note
     * above. CT itself (the parameter) is not modified. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    /* Random constant projections z (column) and w (row) -- see
     * nmod_phi1's doc for the same n_randbits-over-the-word-size
     * reasoning. */
    nmod_poly_mat_t z, w;
    nmod_poly_mat_init(z, r, 1, prime);
    nmod_poly_mat_init(w, 1, r, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(z, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(w, 0, i), 0, n_randbits(state, FLINT_BITS - 2));
    }

    /* T's matrix (phi1-scaled), column by column; column 0 stays zero. */
    nmod_poly_mat_t Tmat, Yk, col;
    nmod_poly_mat_init(Tmat, r, r, prime);
    nmod_poly_mat_init(Yk, r, 1, prime);
    nmod_poly_mat_init(col, r, 1, prime);

    for (slong i = 0; i < r; i++)
        nmod_poly_zero(nmod_poly_mat_entry(Tmat, i, 0));

    for (slong j = 1; j < r; j++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_zero(nmod_poly_mat_entry(Yk, i, 0));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Yk, j, 0), 0, 1);

        nmod_apply_T(col, Yk, CT_phi1, PT, D);

        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(Tmat, i, j), nmod_poly_mat_entry(col, i, 0));
    }

    /* U = T*z, V = w*T (Lemma 3.3's u, v^T). */
    nmod_poly_mat_multiply(U, Tmat, z);
    nmod_poly_mat_multiply(V, w, Tmat);

    /* e = w*T*z = w*U; normalize V := V * e^{-1} mod phi1, so U*V already
     * equals T mod phi1 with no leftover e for the caller. */
    nmod_poly_mat_t e;
    nmod_poly_mat_init(e, 1, 1, prime);
    nmod_poly_mat_multiply(e, w, U);

    nmod_poly_t einv;
    nmod_poly_init(einv, prime);
    nmod_poly_set(einv, nmod_poly_mat_entry(e, 0, 0));
    nmod_poly_invmod(einv, einv, phi1);

    for (slong i = 0; i < r; i++)
    {
        nmod_poly_mul(nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(V, 0, i), einv);
        nmod_poly_rem(nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(V, 0, i), phi1);
    }

    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_mat_clear(z);
    nmod_poly_mat_clear(w);
    nmod_poly_mat_clear(Tmat);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(e);
    nmod_poly_clear(einv);
}
