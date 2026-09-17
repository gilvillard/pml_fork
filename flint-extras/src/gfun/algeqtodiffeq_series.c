/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <time.h>

#include <flint/nmod_poly.h>

#include "nmod_poly_mat_extra.h"

#include "gfun.h"

/*
    The "Series" (heuristic) family, algeqtodiffeq's fourth pseudo-Krylov
    approach (draft nmod_algeq_to_diffeq_series_phi1, gfun.c, not otherwise
    touched). Unlike the naive/width1/Section-4 families, this one has no
    numbered algorithm in algos.pdf or G2026.pdf to cross-check against --
    both papers only cite matrix Hermite-Padé approximants ([2]
    Beckermann-Labahn) as background, not as a specific algorithm this
    matches. Genuinely a heuristic (per the user and CLAUDE.md): instead of
    the other families' exact fraction-free tracking (growing degree
    forever), this keeps the pseudo-Krylov matrix K as a power series
    TRUNCATED to a fixed working precision throughout, then recovers an
    exact rational description from that truncation via a matrix
    Hermite-Padé-style approximant basis (a truncated approximant basis via
    nmod_poly_mat_pmbasis, same core tool and same B=[-H;I]-then-filter-
    by-shift construction as nmod_algeqtodiffeq_T_left_description, applied
    here to K instead of T's own matrix). PML also has a generic version of
    this pattern (nmod_poly_mat_left_description, nmod_poly_mat_description.c)
    -- deliberately NOT used here, per the user (2026-09-17): "I don't want
    to rely on nmod_poly_mat_left_description for the moment... it is not
    stable at all (we will consider it later)". If the truncation precision
    guess is too small, the reconstruction can silently produce a rational
    function consistent with the truncation but not the true answer -- that
    risk is exactly what makes this family unproven, unlike the other three.

    Discussed in detail with the user (2026-09-17) before implementing:
      (a) build a truncated power-series expansion of the pseudo-Krylov
          matrix K (nmod_pseudo_Krylov_series below),
      (b) recover an exact left description (D,N) of K from that
          truncation, directly via nmod_poly_mat_pmbasis
          (nmod_algeqtodiffeq_series_left_description below) -- NOT via
          PML's own nmod_poly_mat_left_description (see above).
    Starting with the LEFT description only (per the user): the draft's own
    `right_des=1` (the active default there) embeds the r x n K into a
    square n x n system via a random projection, which is a genuine
    efficiency win for n << r (deferred here as an important todo, see
    gfun.h/todo.md) but has a real bug in the draft as written -- see
    todo.md for the analysis. The draft's own hardcoded sigma override
    (`ceil(2*target_degree+1)`, tied to that right-description work) is not
    carried forward here; the left-description route uses the "principled"
    sigma formula that was already in the draft (just dead code there,
    overwritten before ever being used).

    Naming: nmod_algeq_to_diffeq_series is already taken (still used by
    Maple's pm_algeq2diffeq_series, confirmed by grep, pointing at the
    non-_phi1 draft in gfun.c) -- nmod_algeq_to_diffeq_series_phi1 itself
    (the draft cleaned up here) has NO Maple binding, so this is a fresh
    name, not a repointing. Named nmod_algeq_to_diffeq_series_left rather
    than plain "_series", since a nmod_algeq_to_diffeq_series_right sibling
    (the n << r efficiency route above) is an explicit future todo, meant
    to coexist with this one, not replace it.
*/

/** Builds the r x n pseudo-Krylov matrix K, column k (0-indexed)
 *  representing theta^k(a) for a = y (the monomial basis vector
 *  (0,1,0,...,0)^t), each entry truncated to a fixed power-series
 *  precision N (not exact, unlike nmod_pseudo_Krylov_naive) -- see the
 *  file header comment above for why this is genuinely heuristic.
 *
 *  Precision bookkeeping (per the user, 2026-09-17): two different kinds
 *  of bound are in play here, not one.
 *   - N is a target NUMBER OF TERMS (power-series truncation order): large
 *     enough that nmod_algeqtodiffeq_series_left_description's own
 *     approximant-basis reconstruction (below) has enough precision in K
 *     to recover a degree-<=target_degree description
 *     (nmod_poly_mat_left_description's own documented requirement,
 *     "at least (m+n)*delta/min(n,m)+1" -- sigma below matches that
 *     formula exactly, computed the same way here as it will be
 *     recomputed internally there), PLUS n-1 extra terms since each of
 *     the n-1 loop steps below differentiates the previous column,
 *     consuming one term of "clean" precision per step.
 *   - D is a NUMBER OF POINTS for nmod_apply_T's own geometric
 *     evaluation-interpolation machinery (an unrelated kind of bound --
 *     see that function's own doc) needed to compute phi1*T(.) correctly
 *     up to x^N; replaces the draft's own bare "+200" margin with the
 *     already-established NMOD_GFUN_NONPROPER_MARGIN, matching the same
 *     placeholder pattern used throughout this module.
 *
 *  target_degree = deg(phi1) + NMOD_GFUN_NONPROPER_MARGIN: unlike
 *  nmod_algeqtodiffeq_T_left_description's own target_degree (which
 *  divides by (r-1), a structural fact about T's own r x r matrix having
 *  a zero first column), this bounds D's degree for the description of
 *  the FULL K directly, so no such division applies -- a flat additive
 *  margin on deg(phi1) is the natural analogue instead. Like every other
 *  target_degree/margin in this module, this is a heuristic guess, not a
 *  derived bound -- MAIN TODO (per the user): expose target_degree as a
 *  caller-supplied parameter later, rather than only ever deriving it
 *  internally (claude-pseudoKrylov/todo.md).
 *
 *  CT must already be phi1-scaled (nmod_algeqtodiffeq_rescale_by_phi1),
 *  matching nmod_pseudo_Krylov_naive's own convention exactly -- this
 *  function needs no Delta at all, for the same reason that one doesn't.
 */
void nmod_pseudo_Krylov_series(nmod_poly_mat_t K, const nmod_poly_t phi1,
                                const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                const slong n)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_pseudo_Krylov_series: n must be >= 1 "
                    "(K needs at least one column, the base case a=y)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong deg_phi1 = nmod_poly_degree(phi1);

    slong target_degree = deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN;
    slong minrn = FLINT_MIN(r, n);
    slong sigma = ((r + n) * target_degree + minrn - 1) / minrn + 1; /* ceil((r+n)*target_degree/min(r,n)) + 1 */
    slong N = sigma + (n - 1);
    slong D = N + deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN;

    nmod_poly_t phi1k, iphi1, iphi1k;
    nmod_poly_init(phi1k, prime);
    nmod_poly_init(iphi1, prime);
    nmod_poly_init(iphi1k, prime);
    nmod_poly_inv_series(iphi1, phi1, N);
    nmod_poly_one(phi1k);
    nmod_poly_one(iphi1k);

    nmod_poly_mat_t numer, temp;
    nmod_poly_mat_init(numer, r, 1, prime);
    nmod_poly_mat_init(temp, r, 1, prime);

    nmod_poly_t tpol;
    nmod_poly_init(tpol, prime);

    /* K_0 = numer_0 = a = y, i.e. column (0,1,0,...,0)^t. */
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
        nmod_poly_zero(nmod_poly_mat_entry(numer, i, 0));
    }
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(K, 1, 0), 0, 1);
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(numer, 1, 0), 0, 1);

    /* Invariant entering iteration k: numer = phi1^k * theta^k(a) mod x^N,
     * K[:,k] = theta^k(a) mod x^N. */
    for (slong k = 0; k < n - 1; k++)
    {
        nmod_poly_mullow(phi1k, phi1k, phi1, N);
        nmod_poly_mullow(iphi1k, iphi1k, iphi1, N);

        /* temp = phi1*T(numer) = phi1^{k+1} * T(theta^k(a)) (CT is
         * phi1-scaled), truncated at the geometric evaluation-
         * interpolation bound D -- not yet at power-series precision N. */
        nmod_apply_T(temp, numer, CT, PT, D);
        nmod_poly_mat_swap(numer, temp);

        for (slong i = 0; i < r; i++)
        {
            /* Cancel the phi1^{k+1} factor via the truncated inverse
             * series (the heuristic step: an exact division would need
             * phi1^{k+1} to actually divide the numerator, which the
             * fraction-free families rely on and this one does not
             * track) -- gives T(theta^k(a)) mod x^N. */
            nmod_poly_truncate(nmod_poly_mat_entry(numer, i, 0), N);
            nmod_poly_mullow(nmod_poly_mat_entry(numer, i, 0), nmod_poly_mat_entry(numer, i, 0), iphi1k, N);

            /* theta^{k+1}(a) = d/dx(theta^k(a)) + T(theta^k(a)) mod x^N. */
            nmod_poly_derivative(tpol, nmod_poly_mat_entry(K, i, k));
            nmod_poly_add_series(nmod_poly_mat_entry(numer, i, 0), nmod_poly_mat_entry(numer, i, 0), tpol, N);

            nmod_poly_set(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(numer, i, 0));

            /* Re-scale numer back to phi1^{k+1}*theta^{k+1}(a) mod x^N,
             * ready for the next iteration's nmod_apply_T call. */
            nmod_poly_mullow(nmod_poly_mat_entry(numer, i, 0), nmod_poly_mat_entry(numer, i, 0), phi1k, N);
        }
    }

    nmod_poly_mat_clear(numer);
    nmod_poly_mat_clear(temp);
    nmod_poly_clear(phi1k);
    nmod_poly_clear(iphi1);
    nmod_poly_clear(iphi1k);
    nmod_poly_clear(tpol);
}


/** Computes an irreducible left description (D,N) of the (truncated)
 *  pseudo-Krylov matrix K (D*K=N, D r x r, N r x n, r=K->r, n=K->c) --
 *  same convention as nmod_algeqtodiffeq_T_left_description (D on the
 *  left, inverted: K = D^{-1}N), and the SAME construction technique
 *  (a truncated approximant basis via nmod_poly_mat_pmbasis, filtering
 *  rows by shift <= target_degree) applied to K directly instead of to
 *  T's own r x r matrix -- direct pmbasis, NOT PML's generic
 *  nmod_poly_mat_left_description, per the user (2026-09-17): "I don't
 *  want to rely on nmod_poly_mat_left_description for the moment ... it
 *  is not stable at all (we will consider it later)".
 *
 *  target_degree/sigma computed internally from phi1 (same MAIN TODO as
 *  nmod_pseudo_Krylov_series above: exposing target_degree as a
 *  caller-supplied parameter is flagged for later, not done here) --
 *  matches nmod_pseudo_Krylov_series's own formula exactly, so K's own
 *  precision (computed there from the same phi1/r/n) is always enough:
 *  no separate precision bookkeeping needed between phase (a) and (b).
 *
 *  Derivation (unlike T's own B=[-Tmat;phi1*I_r], no extra phi1 scaling
 *  is needed here, since K itself -- not phi1*K -- is the quantity being
 *  described): a row [D_row(len r) | N_row(len n)] of the approximant
 *  basis of B=[-K;I_n] ((r+n) x n) at order sigma satisfies
 *  D_row*(-K) + N_row*I ~= 0 mod x^sigma, i.e. N_row ~= D_row*K --
 *  exactly D*K=N, row by row, with plain +1's on the identity block
 *  (not phi1, since there is no separate scaling factor to cancel here).
 *
 *  N, D must already be nmod_poly_mat_init'd by the caller (N r x n,
 *  D r x r). flint_throw's if no complete (r-row) description is found
 *  at this target_degree, matching this module's existing convention for
 *  construction failures (e.g. nmod_algeqtodiffeq_T_left_description).
 */
void nmod_algeqtodiffeq_series_left_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                                                 const nmod_poly_mat_t K, const nmod_poly_t phi1)
{
    ulong prime = nmod_poly_mat_modulus(K);
    slong r = K->r;
    slong n = K->c;

    slong deg_phi1 = nmod_poly_degree(phi1);
    slong target_degree = deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN;
    slong minrn = FLINT_MIN(r, n);
    slong sigma = ((r + n) * target_degree + minrn - 1) / minrn + 1; /* ceil((r+n)*target_degree/min(r,n)) + 1 */

    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, r + n, n, prime);
    for (slong i = 0; i < r; i++)
    {
        for (slong j = 0; j < n; j++)
        {
            nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(K, i, j));
            nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
        }
    }
    for (slong i = 0; i < n; i++)
        nmod_poly_one(nmod_poly_mat_entry(B, i + r, i));

    slong * shift = flint_malloc((r + n) * sizeof(slong));
    for (slong i = 0; i < r + n; i++)
        shift[i] = 0;

    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, r + n, r + n, prime);
    nmod_poly_mat_pmbasis(ker, shift, B, sigma);

    slong nbrows = 0;
    for (slong i = 0; i < r + n && nbrows < r; i++)
    {
        if (shift[i] <= target_degree)
        {
            for (slong j = 0; j < r; j++)
                nmod_poly_set(nmod_poly_mat_entry(D, nbrows, j), nmod_poly_mat_entry(ker, i, j));
            for (slong j = 0; j < n; j++)
                nmod_poly_set(nmod_poly_mat_entry(N, nbrows, j), nmod_poly_mat_entry(ker, i, j + r));
            nbrows++;
        }
    }
    if (nbrows < r)
        flint_throw(FLINT_ERROR, "nmod_algeqtodiffeq_series_left_description: no complete "
                    "description of degree at most %wd found (target_degree too small?)\n",
                    target_degree);

    flint_free(shift);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}


/** Cockle's algorithm (G2026.pdf Sec. 7) via the Series/Padé family's own
 *  left-description route: builds the truncated pseudo-Krylov matrix K
 *  (nmod_pseudo_Krylov_series above), recovers an exact left description
 *  (D,N) of it (nmod_algeqtodiffeq_series_left_description above), then
 *  takes the kernel of N directly.
 *
 *  Why no rescale by D is needed here, unlike the draft's own
 *  right-description branch (gfun.c, not otherwise touched): D*K=N (left
 *  description) means K*eta=0 iff D*K*eta=0 iff N*eta=0 for any eta, since
 *  D is a fixed nonsingular left multiplier -- so a plain kernel of N
 *  already gives exactly the eta with sum eta_i*theta^i(a)=0, no further
 *  transform needed. (The draft's right-description branch has H=N*D^{-1}
 *  instead, where a kernel vector w of N corresponds to eta=D*w, not eta
 *  itself -- a genuinely different situation, not carried over here.)
 *
 *  Same n/seed=y convention as nmod_algeq_to_diffeq_naive/_width1/
 *  _recursive (n = total pseudo-Krylov matrix width, n >= 1).
 */
slong nmod_algeq_to_diffeq_series_left(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_series_left: n must be >= 1 "
                    "(n is the pseudo-Krylov matrix width itself)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_t Delta;
    nmod_poly_mat_t iPyT, CT;
    nmod_algeqtodiffeq_setup(Delta, iPyT, CT, PT);

    /* TODO(cleanup): matches the rest of this module's not-yet-fixed
     * per-call reseeding, see claude-pseudoKrylov/todo.md item 7/8. */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    nmod_poly_t phi1;
    nmod_poly_init(phi1, prime);
    nmod_algeqtodiffeq_rescale_by_phi1(phi1, CT, PT, Delta, state);
    flint_rand_clear(state);

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K, r, n, prime);
    nmod_pseudo_Krylov_series(K, phi1, CT, PT, n);

    nmod_poly_mat_t NN, DD;
    nmod_poly_mat_init(NN, r, n, prime);
    nmod_poly_mat_init(DD, r, r, prime);
    nmod_algeqtodiffeq_series_left_description(NN, DD, K, phi1);

    slong * pivind = flint_malloc(n * sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, NULL, NN, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    nmod_poly_mat_clear(K);
    nmod_poly_mat_clear(NN);
    nmod_poly_mat_clear(DD);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);

    return nz;
}
