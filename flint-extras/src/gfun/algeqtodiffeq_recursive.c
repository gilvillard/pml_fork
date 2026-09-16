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
    Algeqtodiffeq via the Section-4/DAC1 family (pseudo_krylov_recursive.c),
    the "second step" the user referred to when that general algorithm was
    built: use nmod_pseudo_Krylov_recursive as an alternative to
    nmod_apply_T's specialised T-application, by first building an explicit
    irreducible left description (Q,P) of algeqtodiffeq's own T (T=Q^{-1}P),
    then feeding it into the general algorithm.

    Draft correspondence: nmod_algeq_to_diffeq_last_phi1 (gfun.c, not
    otherwise touched) does this in four conceptually distinct pieces,
    discussed explicitly with the user (2026-09-16):
      (a) build T's explicit r x r matrix (phi1-scaled),
      (b) compute an irreducible left description (Q,P) of it,
      (c) nmod_pseudo_Krylov_recursive itself (already done, its own file),
      (d) prepend the seed a (rescaled by D) and take the final kernel.
    (a)+(b) are implemented here as one function (nmod_algeqtodiffeq_
    left_description) -- independently testable/verifiable on its own
    (T=Q^{-1}P), and NOT folded together with (c): nmod_pseudo_Krylov_
    recursive is already its own clean, separately-tested general
    primitive, called directly by whatever driver needs (c)+(d) (not yet
    written). (d) is comparatively mechanical and, per the user's own
    established pattern (nmod_algeq_to_diffeq_naive/_width1 both do their
    own "finishing touches" directly in the top-level driver rather than
    as a separate function), belongs in that driver, not here.
*/

/** Computes an irreducible left description (Q,P) of algeqtodiffeq's own
 *  pseudo-linear map's rational part T (T = Q^{-1}P), suitable as direct
 *  input to nmod_pseudo_Krylov_recursive (pseudo_krylov_recursive.c).
 *  Q, P must already be nmod_poly_mat_init'd by the caller, r x r
 *  (r = (PT->r)-1). CT, PT, Delta as elsewhere in this module (CT is
 *  Delta-scaled); phi1 already computed by the caller.
 *
 *  (a) Builds T's explicit r x r matrix (phi1-scaled: T_matrix = phi1*T)
 *  via one nmod_apply_T call per standard basis column (column 0 stays
 *  zero: theta(1)=0) -- literally the SAME construction find_uv
 *  (algeqtodiffeq_width1.c) already computes internally as its own Tmat.
 *  Currently duplicated rather than shared -- factoring this out into a
 *  common helper (algeqtodiffeq.c) is flagged as a TODO
 *  (claude-pseudoKrylov/todo.md), deliberately not done now (per the
 *  user, 2026-09-16), since it's the second use of the exact same
 *  construction and the right shared shape is clearer with two call
 *  sites in hand than it was with one.
 *
 *  (b) Computes (Q,P) via a MINIMAL APPROXIMANT BASIS
 *  (nmod_poly_mat_pmbasis) of B=[-T_matrix; phi1*I_r] at a truncation
 *  order sigma, rather than an exact kernel computation of the same B
 *  (the draft's own simpler, unused `via_kernel==1` alternative) -- per
 *  the user's explicit choice, 2026-09-16, matching the draft's own
 *  already-active default. This needs a GUESS at Q,P's own degree
 *  (`target_degree`) to choose sigma large enough and to filter the
 *  approximant basis's rows for the genuine (not truncation-artifact)
 *  ones (shift[i] <= target_degree).
 *
 *  Why `(r-1)`, not `r`, in target_degree's formula (kept as an inline
 *  comment at the computation itself, from the draft -- the user
 *  deliberately preserved it: this is exactly the part of the "difficult
 *  question" below that IS actually known, not arbitrary): T's own
 *  matrix always has a zero first column (theta(1)=0, column 0 of Tmat
 *  above is never touched), so only r-1 of T's r columns can carry any
 *  real degree at all. Spreading phi1's degree evenly over those r-1
 *  "live" columns, rather than over all r, is the (generic-case)
 *  reasoning behind dividing by (r-1) specifically.
 *
 *  **MAIN TODO, flagged explicitly by the user (2026-09-16), not resolved
 *  here**: even granting the `(r-1)` structural fact above, `target_degree`
 *  as a WHOLE is still a heuristic guess, not a principled bound --
 *  "even distribution over the r-1 live columns" is itself a genericity
 *  assumption, and the guess can be wrong (too small) for non-generic or
 *  non-proper inputs, either aborting outright (handled below via
 *  flint_throw) or, worse, silently admitting the wrong r rows if sigma
 *  wasn't large enough to reveal genuine degree-<=target_degree rows
 *  correctly. The draft's own literal `+30000` was an oversized,
 *  undocumented magic-number guess at the MARGIN on top of that estimate;
 *  replaced here with the already-established NMOD_GFUN_NONPROPER_MARGIN
 *  (gfun.h) purely for consistency with the same placeholder pattern used
 *  elsewhere in this module (the naive family's own per-column bound) --
 *  NOT because 8 is actually the right margin for this genuinely
 *  different use: here the margin sits INSIDE the division by (r-1),
 *  unlike the naive family's own additive-after-the-fact use of the same
 *  constant, so the identical numeric value plays a structurally
 *  different role here. The right, principled way to compute this
 *  description (a real degree bound, or a verify-and-retry scheme bumping
 *  sigma/target_degree on failure) is a MAIN open item -- see
 *  claude-pseudoKrylov/todo.md.
 *
 *  flint_throw's if no valid r-row description is found at this
 *  target_degree (upgraded from the draft's own silent `return 0`, for
 *  consistency with how this project already treats other construction
 *  failures, e.g. width(T)>1 in the width-1 family).
 */
void nmod_algeqtodiffeq_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
                                          const nmod_poly_t phi1,
                                          const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                          const nmod_poly_t Delta)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    /* (a) T's explicit r x r matrix, phi1-scaled. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    slong d = nmod_poly_mat_degree(PT);
    slong Dbound = nmod_gfun_delta_T_degree_bound(r, d) + nmod_poly_degree(phi1);

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

        nmod_apply_T(col, Yk, CT_phi1, PT, Dbound);

        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(Tmat, i, j), nmod_poly_mat_entry(col, i, 0));
    }

    /* (b) Irreducible left description via a truncated approximant basis. */
    slong deg_phi1 = nmod_poly_degree(phi1);
    // Counting on column less on the right (zero column in T)
    slong target_degree = (deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN + r - 2) / (r - 1); /* ceil */
    slong sigma = (2 * r * target_degree + r - 1) / r + 1; /* ceil(2r*target_degree/r) + 1 */

    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, 2 * r, r, prime);
    for (slong i = 0; i < r; i++)
    {
        for (slong j = 0; j < r; j++)
        {
            nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(Tmat, i, j));
            nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
        }
        nmod_poly_set_trunc(nmod_poly_mat_entry(B, i + r, i), phi1, sigma);
    }

    slong * shift = flint_malloc(2 * r * sizeof(slong));
    for (slong i = 0; i < 2 * r; i++)
        shift[i] = 0;

    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, 2 * r, 2 * r, prime);
    nmod_poly_mat_pmbasis(ker, shift, B, sigma);

    slong nbrows = 0;
    for (slong i = 0; i < 2 * r && nbrows < r; i++)
    {
        if (shift[i] <= target_degree)
        {
            for (slong j = 0; j < r; j++)
            {
                nmod_poly_set(nmod_poly_mat_entry(Q, nbrows, j), nmod_poly_mat_entry(ker, i, j));
                nmod_poly_set(nmod_poly_mat_entry(P, nbrows, j), nmod_poly_mat_entry(ker, i, j + r));
            }
            nbrows++;
        }
    }
    if (nbrows < r)
        flint_throw(FLINT_ERROR, "nmod_algeqtodiffeq_left_description: no complete "
                    "description of degree at most %wd found (target_degree too small?)\n",
                    target_degree);

    flint_free(shift);
    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_mat_clear(Tmat);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}
