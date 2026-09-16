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
    random projection, both rely on width(T) <= 1 for correctness.

    nmod_width1_description (2026-09-16, renamed from
    Description_From_Rank_1, algos.pdf's "DescriptionFromRank1", Algorithm
    3 / Proposition 3.2, Sec. 3.3) is next.

    nmod_pseudo_Krylov_width1 / nmod_algeq_to_diffeq_width1 (2026-09-16,
    Algorithm 4/PseudoKrylovWidth1, replacing the draft's
    nmod_algeq_to_diffeq_new) complete the family: two real bugs were
    found and fixed relative to the draft (wrong seed for the second
    nmod_width1_description call; the block matrix P missing its
    Algorithm-4-required augmentation, which silently forced eta_0=0) --
    see nmod_pseudo_Krylov_width1's own doc and claude-pseudoKrylov/todo.md.
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


/** One step of Proposition 3.2's recurrence (phi1 in place of Delta,
 *  sigma = id): given N_prev (r x 1) and the rank-one pair (U,V), computes
 *
 *      beta_next = -(v^T . N_prev) mod phi1
 *      N_next    = (phi1 * theta(N_prev) + beta_next * u) / phi1
 *
 *  i.e. N_next = theta(N_prev) - b*u/phi1 with b = -beta_next (theta =
 *  d/dx + T), matching Proposition 3.2's N_{i+1} = theta(N_i) - b_{i+1}
 *  u/Delta. Factored out (2026-09-16) so nmod_width1_description's own
 *  loop and Lemma 3.7's one-off F_1 = theta(a) - b_1*u/Delta computation
 *  (nmod_pseudo_Krylov_width1, same file) share one formula instead of two
 *  independently-typed copies of it.
 *
 *  CT_phi1 must already be phi1-scaled (nmod_algeqtodiffeq_rescale_CT_by_phi1);
 *  Dbound is the nmod_apply_T truncation bound (see callers for how it's
 *  derived). N_next and beta_next must already be initialized by the
 *  caller; N_prev may alias N_next's storage is NOT supported (this
 *  function reads N_prev entries after starting to write N_next in the
 *  same loop, so pass distinct matrices).
 */
static void
nmod_width1_recurrence_step(nmod_poly_mat_t N_next, nmod_poly_t beta_next,
                             const nmod_poly_mat_t N_prev,
                             const nmod_poly_mat_t U, const nmod_poly_mat_t V,
                             const nmod_poly_t phi1, const nmod_poly_mat_t CT_phi1,
                             const nmod_poly_mat_t PT, slong Dbound)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_t tpol;
    nmod_poly_init(tpol, prime);

    nmod_poly_mul(beta_next, nmod_poly_mat_entry(V, 0, 0), nmod_poly_mat_entry(N_prev, 0, 0));
    for (slong i = 1; i < r; i++)
    {
        nmod_poly_mul(tpol, nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(N_prev, i, 0));
        nmod_poly_add(beta_next, beta_next, tpol);
    }
    nmod_poly_rem(beta_next, beta_next, phi1);
    nmod_poly_neg(beta_next, beta_next);

    nmod_poly_mat_t temp;
    nmod_poly_mat_init(temp, r, 1, prime);
    nmod_apply_T(temp, N_prev, CT_phi1, PT, Dbound);

    for (slong i = 0; i < r; i++)
    {
        nmod_poly_derivative(tpol, nmod_poly_mat_entry(N_prev, i, 0));
        nmod_poly_mul(tpol, tpol, phi1);
        nmod_poly_add(tpol, tpol, nmod_poly_mat_entry(temp, i, 0));

        nmod_poly_mul(nmod_poly_mat_entry(N_next, i, 0), beta_next, nmod_poly_mat_entry(U, i, 0));
        nmod_poly_add(nmod_poly_mat_entry(N_next, i, 0), nmod_poly_mat_entry(N_next, i, 0), tpol);
        nmod_poly_div(nmod_poly_mat_entry(N_next, i, 0), nmod_poly_mat_entry(N_next, i, 0), phi1);
    }

    nmod_poly_mat_clear(temp);
    nmod_poly_clear(tpol);
}


/** Triangular description of a pseudo-Krylov matrix built from a rank-one
 *  pair (algos.pdf Algorithm 3 "DescriptionFromRank1", Proposition 3.2,
 *  Sec. 3.3). Given u, v with Delta*T = u*v^T mod Delta (phi1-scaled here,
 *  as everywhere else in this family: U*V mod phi1 == T mod phi1, see
 *  find_uv), a seed alpha/N_ini defining q = N_ini/alpha, and letting
 *  theta = d/dx + T (differential case, sigma = id -- this project's
 *  scope, see CLAUDE.md), this computes N (r x n, deg < deg(phi1)) and D
 *  (n x n, upper triangular, deg < deg(phi1)) such that, writing
 *  q_i = theta^(i-1)(q), the columns of N satisfy
 *
 *      sum_{j=1}^{i} D[j,i] * q_j = N_i    for all 1 <= i <= n.          (12)
 *
 *  I.e. [q_1 ... q_n] = N * D^{-1} -- a compressed, denominator-bounded
 *  description of the (a priori unbounded-denominator) pseudo-Krylov
 *  sequence q, q, theta(q), theta^2(q), ...
 *
 *  Two calling conventions, both used by nmod_pseudo_Krylov_width1
 *  (Algorithm 4, same file):
 *   - alpha = phi1, N_ini = U: literally Proposition 3.2 (q = u/phi1),
 *     giving the (N,D) description of Q itself.
 *   - alpha = -v^T.sigma(a) mod phi1, N_ini = theta(a) - alpha's-negation
 *     * u/phi1 (i.e. Lemma 3.7's beta, f): builds the (F,A) pair of
 *     Proposition 3.3 for a general seed vector a (Sec. 3.4) -- computing
 *     alpha/N_ini this way (one extra step of the SAME recurrence, seeded
 *     at a) is what Lemma 3.7 actually specifies; nmod_pseudo_Krylov_width1
 *     computes them via nmod_width1_recurrence_step (this file) before
 *     calling this function a second time. The draft this was cleaned up
 *     from instead passed alpha=0, N_ini=a directly -- a real bug, not
 *     just a simplification (see claude-pseudoKrylov/todo.md).
 *
 *  Preconditions: NN (r x n) and DD (n x n) must already be
 *  nmod_poly_mat_init'd by the caller, with DD's entries starting at zero
 *  (its own zero fill from nmod_poly_mat_init) -- only DD's upper-
 *  triangular part (row <= column) is ever written, matching D's
 *  triangular structure. U (r x 1), V (1 x r) as returned by find_uv. CT,
 *  PT, Delta as elsewhere (CT is Delta-scaled).
 *
 *  Internal optimization (2026-09-16, mirrors find_uv's own): the draft
 *  called nmod_apply_T at the Delta-based bound for every column, then
 *  divided each of the r result entries by g = Delta/phi1 (CT is exactly
 *  divisible by g throughout, same fact find_uv relies on) -- n-1 columns
 *  x r divisions. Here CT is rescaled to phi1-scale ONCE, on a local
 *  working copy, via nmod_algeqtodiffeq_rescale_CT_by_phi1, so
 *  nmod_apply_T already returns phi1*T(.) directly at every column. The
 *  bound D itself is unchanged (same reasoning as find_uv: generically
 *  deg(phi1) == deg(Delta) for this width-1 family, so there is no smaller
 *  bound to fall back to in the regime this family targets) -- purely
 *  dropping the repeated post-division.
 */
void nmod_width1_description(nmod_poly_mat_t NN, nmod_poly_mat_t DD, const ulong n,
                              const nmod_poly_mat_t U, const nmod_poly_mat_t V,
                              const nmod_poly_t phi1, const nmod_poly_mat_t CT,
                              const nmod_poly_mat_t PT, const nmod_poly_t alpha,
                              const nmod_poly_mat_t N_ini, const nmod_poly_t Delta)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);
    slong deg_phi1 = nmod_poly_degree(phi1);

    /* Local, phi1-scaled working copy of CT -- see the optimization note
     * above. CT itself (the parameter) is not modified. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    /* Bound for phi1*T(N_{j-1}) via nmod_apply_T -- not re-derived here,
     * same "every D bound in this module is a to-check estimate" caveat
     * as elsewhere (claude-pseudoKrylov/todo.md); numerically identical to
     * the draft's own (2*r-1)*d + deg(phi1) - 1. */
    slong D = nmod_gfun_delta_T_degree_bound(r, d) + deg_phi1;

    /* Column 0 (paper's N_1, D's beta_{1,1}): the base case, given
     * directly rather than computed. */
    nmod_poly_set(nmod_poly_mat_entry(DD, 0, 0), alpha);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(NN, i, 0), nmod_poly_mat_entry(N_ini, i, 0));

    nmod_poly_mat_t N_prev, N_next;
    nmod_poly_mat_init(N_prev, r, 1, prime);
    nmod_poly_mat_init(N_next, r, 1, prime);

    nmod_poly_t b, tpol;
    nmod_poly_init(b, prime);
    nmod_poly_init(tpol, prime);

    for (slong j = 1; j < (slong) n; j++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(N_prev, i, 0), nmod_poly_mat_entry(NN, i, j - 1));

        /* b = beta_{.,j+1}'s "-b_{j+1}" term (Proposition 3.2), N_next =
         * N_j -- see nmod_width1_recurrence_step's own doc for the exact
         * formula (CT_phi1 already phi1-scaled, so no post-division by
         * Delta/phi1 needed here, unlike the draft). */
        nmod_width1_recurrence_step(N_next, b, N_prev, U, V, phi1, CT_phi1, PT, D);

        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(NN, i, j), nmod_poly_mat_entry(N_next, i, 0));

        /* D's column j (paper's beta_{.,j+1}): beta_{1,j+1} = b_{j+1} (as
         * negated above) + d/dx(beta_{1,j}); beta_{i+1,j+1} =
         * beta_{i,j} + d/dx(beta_{i+1,j}) for 1 <= i <= j (sigma = id). */
        nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, 0, j - 1));
        nmod_poly_add(nmod_poly_mat_entry(DD, 0, j), b, tpol);

        for (slong i = 1; i <= j; i++)
        {
            nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, i, j - 1));
            nmod_poly_add(nmod_poly_mat_entry(DD, i, j), nmod_poly_mat_entry(DD, i - 1, j - 1), tpol);
        }
    }

    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_mat_clear(N_prev);
    nmod_poly_mat_clear(N_next);
    nmod_poly_clear(b);
    nmod_poly_clear(tpol);
}


/** Algorithm 4 (PseudoKrylovWidth1, algos.pdf Sec. 3.4): given theta =
 *  d/dx + T (differential case, sigma = id) via CT/PT/Delta, a seed vector
 *  a (r x 1), and m >= 1, computes a minimal basis Y (of size (m+1) x
 *  (m+1), first `nz` columns meaningful -- same "extra columns undefined"
 *  convention as nmod_poly_mat_kernel's other callers in this project) of
 *  the k[x]-module of solutions eta = (eta_0,...,eta_m) to the pseudo-
 *  Krylov system sum_{i=0}^{m} eta_i * theta^i(a) = 0. flint_throw's if T
 *  does not have width <= 1 (Algorithm 4's own FAIL case, step 8) --
 *  detected via nmod_phi_T's deg(phi1) == deg(phi2) check, the same proxy
 *  find_uv's own test uses (see claude-pseudoKrylov/todo.md item 14: the
 *  user's decision, 2026-09-16, was flint_throw here rather than a
 *  sentinel return or falling back to the naive family).
 *
 *  Two real bugs found and fixed relative to the draft this replaces
 *  (nmod_algeq_to_diffeq_new, gfun.c, not otherwise touched -- see
 *  claude-pseudoKrylov/todo.md for the full analysis):
 *
 *   1. The second nmod_width1_description call (for Proposition 3.3's
 *      (F,A)) needs alpha = -v^T.sigma(a) mod phi1 and N_ini =
 *      theta(a) - (-alpha)*u/phi1 (Lemma 3.7's beta, f = F_1 -- one step
 *      of the SAME Prop. 3.2 recurrence, seeded at a instead of u). The
 *      draft passed alpha=0, N_ini=a directly (the raw seed, unadvanced)
 *      -- a different computation, not merely an unoptimized one.
 *
 *   2. The block matrix P (eq. 19) must be built from the AUGMENTED
 *      Dbar=Diag(1,D), Nbar=[0 N], Abar=Diag(1,A), Fbar=[a F] -- one
 *      dimension larger than D,N,A,F themselves. This is what lets the
 *      nullspace computation capture the eta_0 (coefficient of a itself)
 *      term. The draft built P directly from the *unaugmented* [[D,A],
 *      [N,F]]; working through Lemma 3.8's own derivation on that smaller
 *      matrix shows its nullspace instead gives relations of the
 *      restricted form eta_1*theta(a) + ... + eta_m*theta^m(a) = 0 --
 *      structurally forcing eta_0 = 0, a degenerate special case rather
 *      than a general solution to Problem 1/2.
 */
slong nmod_pseudo_Krylov_width1(nmod_poly_mat_t Y, const nmod_poly_mat_t a, const ulong m,
                                 const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                 const nmod_poly_t Delta, flint_rand_t state)
{
    if (m < 1)
        flint_throw(FLINT_DOMERR, "nmod_pseudo_Krylov_width1: m must be >= 1\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);

    /* Phase 1: rank-one decomposition (u,v), width check. r < 3 is
     * special-cased: T is then r x r with column 0 forced to zero
     * (theta(1)=0), i.e. at most 2x2 with a zero column, trivially width
     * <= 1 as a plain matrix -- nmod_phi_T's own phi2 needs r >= 3 to be
     * meaningful (see its doc), so deg(phi1)==deg(phi2) would spuriously
     * fail here (found 2026-09-16, reproduced by tests/t-algeq-to-diffeq-
     * width1.c's own r=2 trials: deg(phi1)=9, deg(phi2)=0). Same
     * special-casing already used by the test-side filters in
     * t-find-uv.c/t-width1-description.c; folded into the library
     * function itself here since it's a genuine structural fact about
     * algeqtodiffeq's T, not just a test-generation convenience. */
    nmod_poly_t phi1, phi2;
    nmod_poly_init(phi1, prime);
    nmod_poly_init(phi2, prime);

    if (r < 3)
    {
        nmod_phi1(phi1, CT, PT, Delta, state);
    }
    else
    {
        nmod_phi_T(phi1, phi2, CT, PT, Delta, state);
        if (nmod_poly_degree(phi1) != nmod_poly_degree(phi2))
            flint_throw(FLINT_ERROR, "nmod_pseudo_Krylov_width1: T does not have width <= 1 "
                        "(deg(phi1)=%wd, deg(phi2)=%wd)\n",
                        nmod_poly_degree(phi1), nmod_poly_degree(phi2));
    }

    nmod_poly_mat_t U, V;
    nmod_poly_mat_init(U, r, 1, prime);
    nmod_poly_mat_init(V, 1, r, prime);
    find_uv(U, V, phi1, CT, PT, Delta, state);

    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    slong D = nmod_gfun_delta_T_degree_bound(r, d) + nmod_poly_degree(phi1);

    /* Phase 2: realisations (N,D) of Q (Proposition 3.2) and (F,A) of a
     * (Proposition 3.3, via Lemma 3.7's beta/f). */
    nmod_poly_mat_t N, D_mat;
    nmod_poly_mat_init(N, r, m, prime);
    nmod_poly_mat_init(D_mat, m, m, prime);
    nmod_width1_description(N, D_mat, m, U, V, phi1, CT, PT, phi1, U, Delta);

    nmod_poly_t beta;
    nmod_poly_init(beta, prime);
    nmod_poly_mat_t f;
    nmod_poly_mat_init(f, r, 1, prime);
    nmod_width1_recurrence_step(f, beta, a, U, V, phi1, CT_phi1, PT, D);

    nmod_poly_mat_t F, A;
    nmod_poly_mat_init(F, r, m, prime);
    nmod_poly_mat_init(A, m, m, prime);
    nmod_width1_description(F, A, m, U, V, phi1, CT, PT, beta, f, Delta);

    /* Phase 3: augmented block matrix P (eq. 19):
     *   Dbar=Diag(1,D), Nbar=[0 N], Abar=Diag(1,A), Fbar=[a F]
     *   P = [[Dbar, Abar], [Nbar, Fbar]], size (r + n1) x 2*n1, n1 = m+1.
     * P is freshly initialized (zero), so only the nonzero entries below
     * need to be set explicitly. */
    slong n1 = m + 1;
    nmod_poly_mat_t P;
    nmod_poly_mat_init(P, r + n1, 2 * n1, prime);

    nmod_poly_one(nmod_poly_mat_entry(P, 0, 0));   /* Dbar[0][0] = 1 */
    nmod_poly_one(nmod_poly_mat_entry(P, 0, n1));  /* Abar[0][0] = 1 */
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, n1), nmod_poly_mat_entry(a, i, 0)); /* Fbar[:,0] = a */
    /* Nbar[:,0] stays zero (P's own fresh zero-init). */

    for (slong i = 0; i < (slong) m; i++)
        for (slong j = 0; j < (slong) m; j++)
        {
            nmod_poly_set(nmod_poly_mat_entry(P, 1 + i, 1 + j), nmod_poly_mat_entry(D_mat, i, j));
            nmod_poly_set(nmod_poly_mat_entry(P, 1 + i, n1 + 1 + j), nmod_poly_mat_entry(A, i, j));
        }
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < (slong) m; j++)
        {
            nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, 1 + j), nmod_poly_mat_entry(N, i, j));
            nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, n1 + 1 + j), nmod_poly_mat_entry(F, i, j));
        }

    /* Nullspace of P: W = [Z^T Y^T]^T (Lemma 3.8), Y = W's last n1 rows. */
    slong *pivind = flint_malloc(2 * n1 * sizeof(slong));
    nmod_poly_mat_t W;
    nmod_poly_mat_init(W, 2 * n1, 2 * n1, prime);
    slong nz = nmod_poly_mat_kernel(W, pivind, NULL, P, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    for (slong i = 0; i < n1; i++)
        for (slong j = 0; j < n1; j++)
            nmod_poly_set(nmod_poly_mat_entry(Y, i, j), nmod_poly_mat_entry(W, n1 + i, j));

    nmod_poly_mat_clear(U);
    nmod_poly_mat_clear(V);
    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(D_mat);
    nmod_poly_mat_clear(f);
    nmod_poly_mat_clear(F);
    nmod_poly_mat_clear(A);
    nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(W);
    nmod_poly_clear(phi1);
    nmod_poly_clear(phi2);
    nmod_poly_clear(beta);

    return nz;
}


/** Cockle's algorithm (G2026.pdf Sec. 7) via the width-1 pseudo-Krylov
 *  family (Algorithm 4): computes a minimal basis of solutions
 *  eta_0,...,eta_{n-1} to sum eta_i * theta^i(y) = 0. `n` is the actual
 *  pseudo-Krylov matrix width (same convention as
 *  nmod_algeq_to_diffeq_naive: n total powers theta^0(y)..theta^{n-1}(y)),
 *  so Algorithm 4's own m = n-1 (must be >= 1, hence n >= 2 here).
 *
 *  Returns nz, the number of solutions found; LT is an n x n polynomial
 *  matrix whose first nz columns are the solutions (LT[i][j] = eta_i of
 *  the j-th solution). flint_throw's if T does not have width <= 1 --
 *  see nmod_pseudo_Krylov_width1's own doc.
 */
slong nmod_algeq_to_diffeq_width1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_width1: n must be >= 2 "
                    "(n is the pseudo-Krylov matrix width itself; Algorithm 4's "
                    "own m = n-1 must be >= 1)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_t Delta;
    nmod_poly_mat_t iPyT, CT;
    nmod_algeqtodiffeq_setup(Delta, iPyT, CT, PT);

    nmod_poly_mat_t a;
    nmod_poly_mat_init(a, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_zero(nmod_poly_mat_entry(a, i, 0));
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(a, 1, 0), 0, 1);

    /* TODO(cleanup): matches the rest of this module's not-yet-fixed
     * per-call reseeding, see claude-pseudoKrylov/todo.md item 7/8. */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    slong nz = nmod_pseudo_Krylov_width1(LT, a, n - 1, CT, PT, Delta, state);

    flint_rand_clear(state);
    nmod_poly_mat_clear(a);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);

    return nz;
}
