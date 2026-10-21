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

    _nmod_algeq_to_diffeq_width1 / nmod_algeq_to_diffeq_width1 (2026-09-16,
    Algorithm 4/PseudoKrylovWidth1, replacing the draft's
    nmod_algeq_to_diffeq_new) complete the family: two real bugs were
    found and fixed relative to the draft (wrong seed for the second
    nmod_width1_description call; the block matrix P missing its
    Algorithm-4-required augmentation, which silently forced eta_0=0) --
    see _nmod_algeq_to_diffeq_width1's own doc and claude-pseudoKrylov/todo.md.

    Renamed from nmod_pseudo_Krylov_width1 (2026-09-17, per the user): this
    function is algeqtodiffeq-specific end to end (takes CT/PT/Delta,
    computes through the final kernel), unlike the general
    nmod_pseudo_Krylov_recursive/_iterative family (which stops at a
    description (N,D)) -- the leading underscore marks it as
    nmod_algeq_to_diffeq_width1's own lower-level core (general seed a/m,
    no setup), not a "pseudo_Krylov" building block.
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

    /* Local, phi1-scaled working copy of CT -- see the optimization note
     * above. CT itself (the parameter) is not modified. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    /* 2026-10-20: the computation itself is the generic nmod_T_ctx_find_uv (T_ctx.c),
     * on an AlgeqtoDiffeq context with scale = phi1; same draws, same result (column 0
     * of phi1 T, zero since T(1) = 0, is now computed rather than set). Its bound for
     * apply, (2r-1)d - 1 + margin, is this function's former
     * nmod_gfun_delta_T_degree_bound(r, d) + margin. Margin for the same reason as
     * nmod_width1_description's D below: the properness-derived bound is exactly the
     * degree phi1*T(e_j) reaches here (the operands are constant basis vectors), so it
     * carries no cushion for an input where T is not exactly proper -- and
     * nmod_apply_T aliases silently rather than raising. Unlike the description
     * loop's bound, this one's headroom has not actually been measured; added
     * 2026-09-18 on the same reasoning, per the user. */
    nmod_T_ctx_struct T;
    nmod_algeqtodiffeq_T_data_struct dat;
    nmod_algeqtodiffeq_T_ctx_init(&T, &dat, CT_phi1, PT, phi1);
    nmod_T_ctx_find_uv(U, V, phi1, &T, state);

    nmod_poly_mat_clear(CT_phi1);
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
 *  (_nmod_algeq_to_diffeq_width1, same file) share one formula instead of two
 *  independently-typed copies of it.
 *
 *  CT_phi1 must already be phi1-scaled (nmod_algeqtodiffeq_rescale_CT_by_phi1);
 *  Dbound is the nmod_apply_T truncation bound (see callers for how it's
 *  derived). N_next and beta_next must already be initialized by the
 *  caller; N_prev may alias N_next's storage is NOT supported (this
 *  function reads N_prev entries after starting to write N_next in the
 *  same loop, so pass distinct matrices).
 */
/* (2026-10-20: the recurrence step itself is now the generic
 * _nmod_T_ctx_width1_recurrence_step, T_ctx.c / gfun.h; the doc above still describes it.) */


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
 *  Two calling conventions, both used by _nmod_algeq_to_diffeq_width1
 *  (Algorithm 4, same file):
 *   - alpha = phi1, N_ini = U: literally Proposition 3.2 (q = u/phi1),
 *     giving the (N,D) description of Q itself.
 *   - alpha = -v^T.sigma(a) mod phi1, N_ini = theta(a) - alpha's-negation
 *     * u/phi1 (i.e. Lemma 3.7's beta, f): builds the (F,A) pair of
 *     Proposition 3.3 for a general seed vector a (Sec. 3.4) -- computing
 *     alpha/N_ini this way (one extra step of the SAME recurrence, seeded
 *     at a) is what Lemma 3.7 actually specifies; _nmod_algeq_to_diffeq_width1
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

    /* Local, phi1-scaled working copy of CT -- see the optimization note
     * above. CT itself (the parameter) is not modified. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    /* 2026-10-20: the computation itself is the generic nmod_T_ctx_width1_description
     * (T_ctx.c) on an AlgeqtoDiffeq context with scale = phi1; same result. Its bound
     * for apply, (2r-1)d - 1 + deg(phi1) + margin, is this function's former
     * nmod_gfun_delta_T_degree_bound(r, d) + deg(phi1) + margin: the properness-derived
     * part, (2*r-1)*d - 1 + deg(phi1) (= 2*deg(phi1) - 1 generically), is exactly
     * algos.pdf's own ceiling for this quantity (Prop. 3.2's proof:
     * deg(Delta*theta(N_i)) < 2*delta) -- and it is TIGHT, not conservative: measured
     * slack is a constant 3 coefficients at r,d = 8,5 / 12,8 / 20,12
     * (claude-pseudoKrylov/experiments/bench-applyT-breakdown.c, 2026-09-18), i.e.
     * 98-99.7% of D is actually used, more so as sizes grow. Since nmod_apply_T's
     * geometric evaluation has NO bounds checking -- exceeding D aliases silently into
     * a wrong answer rather than raising (see NMOD_GFUN_NONPROPER_MARGIN's own doc in
     * gfun.h) -- those 3 coefficients are no cushion at all for an input where T is
     * not exactly proper, which is what the margin covers. */
    nmod_T_ctx_struct T;
    nmod_algeqtodiffeq_T_data_struct dat;
    nmod_algeqtodiffeq_T_ctx_init(&T, &dat, CT_phi1, PT, phi1);
    nmod_T_ctx_width1_description(NN, DD, n, U, V, phi1, &T, alpha, N_ini);

    nmod_poly_mat_clear(CT_phi1);
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
slong _nmod_algeq_to_diffeq_width1(nmod_poly_mat_t Y, const nmod_poly_mat_t a, const ulong m,
                                 const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                 const nmod_poly_t Delta, flint_rand_t state)
{
    if (m < 1)
        flint_throw(FLINT_DOMERR, "_nmod_algeq_to_diffeq_width1: m must be >= 1\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

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
            flint_throw(FLINT_ERROR, "_nmod_algeq_to_diffeq_width1: T does not have width <= 1 "
                        "(deg(phi1)=%wd, deg(phi2)=%wd)\n",
                        nmod_poly_degree(phi1), nmod_poly_degree(phi2));
    }

    /* 2026-10-20: the rest (rank-one pair, the two width-1 descriptions with Lemma 3.7's
     * step for the seed, the block matrix P of eq. (19) and its nullspace) is the generic
     * _nmod_T_ctx_width1_core (T_ctx.c), on an AlgeqtoDiffeq context with scale = phi1;
     * same draws, bounds and result. */
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    nmod_T_ctx_struct T;
    nmod_algeqtodiffeq_T_data_struct dat;
    nmod_algeqtodiffeq_T_ctx_init(&T, &dat, CT_phi1, PT, phi1);
    slong nz = _nmod_T_ctx_width1_core(Y, a, m, &T, phi1, state);

    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_clear(phi1);
    nmod_poly_clear(phi2);

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
 *  see _nmod_algeq_to_diffeq_width1's own doc.
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

    slong nz = _nmod_algeq_to_diffeq_width1(LT, a, n - 1, CT, PT, Delta, state);

    flint_rand_clear(state);
    nmod_poly_mat_clear(a);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);

    return nz;
}


/* The block T_b of AlgeqtoDiffeq as a T context (2026-10-21): since T(1) = 0, the first
 * column of T is zero, so T_b = T without its first row and column is the map induced on the
 * quotient by k(x).1. apply(R, A, D, data) for A ((r-1) x 1): rows 1..r-1 of
 * scale * T((0, A)) with the full context's apply; same scale and bound. */
typedef struct
{
    const nmod_T_ctx_struct * full;
} _nmod_algeqtodiffeq_Tb_data_struct;

static void
_nmod_algeqtodiffeq_Tb_apply(nmod_poly_mat_t R, const nmod_poly_mat_t A, slong D, const void * data)
{
    const _nmod_algeqtodiffeq_Tb_data_struct * dat = (const _nmod_algeqtodiffeq_Tb_data_struct *) data;
    slong r = dat->full->r;
    ulong prime = nmod_poly_mat_modulus(A);
    nmod_poly_mat_t Af, Rf;
    nmod_poly_mat_init(Af, r, 1, prime);
    nmod_poly_mat_init(Rf, r, 1, prime);
    for (slong i = 1; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(Af, i, 0), nmod_poly_mat_entry(A, i - 1, 0));
    dat->full->apply(Rf, Af, D, dat->full->data);
    for (slong i = 1; i < r; i++)
        nmod_poly_swap(nmod_poly_mat_entry(R, i - 1, 0), nmod_poly_mat_entry(Rf, i, 0));
    nmod_poly_mat_clear(Af);
    nmod_poly_mat_clear(Rf);
}

/** Inhomogeneous operators by PseudoKrylovWidth1 on the block (2026-10-21): the same
 *  output as nmod_algeq_to_diffeq_inhomogeneous (a minimal basis, zero shift, of the
 *  operators L of order <= n-1 with L(y) in k(x)), computed by _nmod_T_ctx_width1_core on
 *  theta_b = d/dx + T_b, T_b = T without its first row and column (dimension r-1), seed
 *  a_b = e_0 (the image of y). phi1(T) and C rescaled to it as in
 *  _nmod_algeq_to_diffeq_width1; phi1(T_b) (and phi2 for the width check, r-1 >= 2) on the
 *  block context. Throws if T_b does not have width <= 1. Requires n >= 2, r >= 2.
 */
slong nmod_algeq_to_diffeq_inhomogeneous_width1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_inhomogeneous_width1: n must be >= 2\n");
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong k = r - 1;
    if (r < 2)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_inhomogeneous_width1: needs r >= 2\n");

    nmod_poly_t Delta;
    nmod_poly_mat_t iPyT, CT;
    nmod_algeqtodiffeq_setup(Delta, iPyT, CT, PT);

    /* same per-call seeding as nmod_algeq_to_diffeq_width1 */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    /* full context with scale phi1(T), as _nmod_algeq_to_diffeq_width1 */
    nmod_poly_t phi1, phi1b, phi2b;
    nmod_poly_init(phi1, prime);
    nmod_poly_init(phi1b, prime);
    nmod_poly_init(phi2b, prime);
    nmod_phi1(phi1, CT, PT, Delta, state);
    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);
    nmod_T_ctx_struct T;
    nmod_algeqtodiffeq_T_data_struct dat;
    nmod_algeqtodiffeq_T_ctx_init(&T, &dat, CT_phi1, PT, phi1);

    /* block context */
    _nmod_algeqtodiffeq_Tb_data_struct datb;
    datb.full = &T;
    nmod_T_ctx_struct Tb;
    Tb.r = k;
    Tb.apply = _nmod_algeqtodiffeq_Tb_apply;
    Tb.data = &datb;
    Tb.scale = T.scale;
    Tb.deg_bound = T.deg_bound;

    if (k >= 2)
    {
        nmod_T_ctx_phi_T(phi1b, phi2b, &Tb, state);
        if (nmod_poly_degree(phi1b) != nmod_poly_degree(phi2b))
            flint_throw(FLINT_ERROR, "nmod_algeq_to_diffeq_inhomogeneous_width1: T_b does not have "
                        "width <= 1 (deg(phi1)=%wd, deg(phi2)=%wd)\n",
                        nmod_poly_degree(phi1b), nmod_poly_degree(phi2b));
    }
    else
        nmod_T_ctx_phi1(phi1b, &Tb, state);

    nmod_poly_mat_t ab;
    nmod_poly_mat_init(ab, k, 1, prime);
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(ab, 0, 0), 0, 1);
    slong nz = _nmod_T_ctx_width1_core(LT, ab, (ulong) (n - 1), &Tb, phi1b, state);

    nmod_poly_mat_clear(ab);
    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_clear(phi1);
    nmod_poly_clear(phi1b);
    nmod_poly_clear(phi2b);
    flint_rand_clear(state);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    return nz;
}
