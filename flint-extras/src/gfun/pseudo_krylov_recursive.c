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
#include "nmod_poly_mat_forms.h"

#include "gfun.h"

/*
    Section 4.2 of algos.pdf, "Arbitrary pseudo-linear map" -- Algorithm 6,
    "PseudoKrylovDAC1" in the paper's own vocabulary ("DAC" = Divide And
    Conquer, confirmed by Algorithm 5's own name "PseudoKrylovDAC2", the
    shift-case-only fast-doubling sibling in Sec. 4.1, not implemented
    here). Differential case only for now (per the user, 2026-09-16) --
    the shift case is a straightforward extension (Algorithm 6 branches on
    it explicitly at Step 5) but deliberately deferred.

    Unlike the width-1 family (algeqtodiffeq_width1.c), this solves the
    GENERAL pseudo-Krylov description problem for an arbitrary pseudo-
    linear map theta = d/dx + T*sigma given directly via an irreducible
    left description T = Q^{-1}P (Q, P genuine n x n polynomial matrices,
    no width assumption at all) -- not algeqtodiffeq's own CT/PT/Delta/phi1
    convention. That's why this lives in its own file, outside the
    algeqtodiffeq_<family>.c naming pattern: wiring this up as a second,
    alternative construction for algeqtodiffeq itself (replacing
    nmod_apply_T's specialised T-application with a genuine general Q,P
    description) is explicitly a separate, later step, not done here.

    Draft correspondence (gfun.c, not otherwise touched): _rec_pseudo_krylov/
    rec_pseudo_krylov implement the same algorithm, but fold Algorithm 6's
    own explicit "advance (Q,P) by one step" computation (Steps 4-6) into
    the k==1 base case instead of keeping it in the m>1 recursive branch.
    Discussed with the user (2026-09-16): the two approaches are
    computationally equivalent (same O(m) total nullspace calls, same
    growing shifts) -- implemented here literally matching Algorithm 6's
    own structure instead, since this is fresh code (not a clean-up-in-
    place), for direct traceability against Lemmas 4.2-4.4 and to avoid a
    minor wasted computation the folded version has at the very last leaf.

    Naming: kept general (no "DAC1" in the function name itself, per the
    user) -- nmod_pseudo_Krylov_recursive, matching this project's existing
    nmod_pseudo_Krylov_<family> convention (naive, width1, naive_delta).
*/

/** Algorithm 6 (PseudoKrylovDAC1, algos.pdf Sec. 4.2.1), differential case.
 *
 *  Input: Q, P in k[x]^{n x n} such that T = Q^{-1}P is an irreducible
 *  description with s-reduced denominator Q and rdeg_s(Q) >= rdeg_s(P);
 *  s a shift (length n, may be NULL for the all-zero shift); a in k[x]^n;
 *  m >= 1.
 *
 *  Output: D, Qt, Pt in k[x]^{n x n}, N in k[x]^{n x m} such that, writing
 *  theta = d/dx + T (sigma = id, differential case), the pseudo-Krylov
 *  matrix K = [theta(a) ... theta^m(a)] in k(x)^{n x m} satisfies K = D^{-1}N
 *  with D s-reduced, and Qt^{-1}Pt is an irreducible description of the
 *  "advanced" pseudo-linear map theta_m (Lemma 4.3's commutation rule,
 *  iterated m times from theta_1 = theta) -- Qt, Pt are only useful to a
 *  caller that wants to continue the sequence further (e.g. call this
 *  function again with (Qt, Pt, rdeg_s(D), theta_m's own seed, more
 *  columns)); they play no role in K itself.
 *
 *  D, N, Qt, Pt must already be nmod_poly_mat_init'd by the caller with
 *  shapes D: n x n, N: n x m, Qt: n x n, Pt: n x n. D is guaranteed
 *  nonsingular (Lemma 4.2's proof: it is a product of the Q_i's, each
 *  nonsingular by the algorithm's own invariant); its correct s-reduced
 *  row degree can be recovered by the caller via
 *  nmod_poly_mat_row_degree(rdeg, D, s) if needed for a further call.
 *
 *  Base case (m=1): theta(a) = d/dx(a) + T(a) = Q^{-1}(Q*d/dx(a) + P*a),
 *  so D=Q, N = Q*d/dx(a) + P*a already gives the (trivial) description;
 *  theta_1 = theta itself, so Qt=Q, Pt=P unchanged (matches Algorithm 6's
 *  own literal Step 1 exactly -- no "advance" happens here).
 *
 *  Recursive case (m>1): splits into h = ceil(m/2) and m-h (Steps 2-3,
 *  7-8), with the "advance" (Steps 4-6) computed explicitly in between --
 *  see Lemma 4.2/4.3 and the inline comments below for the exact formulas.
 *
 *  Shared core of nmod_pseudo_Krylov_recursive and
 *  nmod_pseudo_Krylov_recursive_renorm (below): when renorm is nonzero, at
 *  every internal "advance" node (Step 6) whose raw (Qbar,Pbar) is no
 *  longer minimal in the ABSOLUTE (zero-shift) sense -- i.e.
 *  sum(rdeg_0(Qbar)) > deg_det_Q -- both are replaced in place by a
 *  zero-shift weak Popov form (nmod_poly_mat_weak_popov_iter, the SAME
 *  unimodular row operations applied to both, so Qbar^{-1}Pbar and hence
 *  every output are completely unaffected). deg_det_Q is deg(det(Q)) of
 *  the ORIGINAL top-level Q: det(Qbar) = det(Q) up to a unit at every node
 *  (Lemma 4.2: each advance is a gauge transform of the same module, so
 *  the determinant -- a module invariant -- never changes), which is
 *  exactly why this is the right absolute-minimality threshold at every
 *  depth, not just the top one. See nmod_pseudo_Krylov_recursive_renorm's
 *  own doc for what this buys and its one open caveat.
 */
static void
_nmod_pseudo_Krylov_recursive_core(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                    nmod_poly_mat_t Qt, nmod_poly_mat_t Pt,
                                    const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                    const slong * s, const nmod_poly_mat_t a,
                                    const slong m, int renorm, slong deg_det_Q)
{
    ulong prime = nmod_poly_mat_modulus(Q);
    slong n = Q->r;

    if (m == 1)
    {
        nmod_poly_mat_set(D, Q);
        nmod_poly_mat_set(Qt, Q);
        nmod_poly_mat_set(Pt, P);

        nmod_poly_mat_t da, Qda, Pa;
        nmod_poly_mat_init(da, n, 1, prime);
        nmod_poly_mat_init(Qda, n, 1, prime);
        nmod_poly_mat_init(Pa, n, 1, prime);

        for (slong i = 0; i < n; i++)
            nmod_poly_derivative(nmod_poly_mat_entry(da, i, 0), nmod_poly_mat_entry(a, i, 0));

        nmod_poly_mat_mul(Qda, Q, da);
        nmod_poly_mat_mul(Pa, P, a);
        for (slong i = 0; i < n; i++)
            nmod_poly_add(nmod_poly_mat_entry(N, i, 0), nmod_poly_mat_entry(Qda, i, 0), nmod_poly_mat_entry(Pa, i, 0));

        nmod_poly_mat_clear(da);
        nmod_poly_mat_clear(Qda);
        nmod_poly_mat_clear(Pa);
        return;
    }

    slong h = (m + 1) / 2;   /* ceil(m/2) */
    slong h2 = m - h;

    /* Step 3: first recursive call, covering theta(a),...,theta^h(a). */
    nmod_poly_mat_t D1, N1, Q1, P1;
    nmod_poly_mat_init(D1, n, n, prime);
    nmod_poly_mat_init(N1, n, h, prime);
    nmod_poly_mat_init(Q1, n, n, prime);
    nmod_poly_mat_init(P1, n, n, prime);
    _nmod_pseudo_Krylov_recursive_core(D1, N1, Q1, P1, Q, P, s, a, h, renorm, deg_det_Q);

    /* Step 4: t = rdeg_s(D1) -- D1 describes K_h, distinct from (Q1,P1)
     * which describe theta_h itself (they coincide only when h=1). */
    slong * t = flint_malloc(n * sizeof(slong));
    nmod_poly_mat_row_degree(t, D1, s);

    /* Step 5 (differential case): W = [[P1 - d/dx(Q1)], [Q1]], a 2n x n
     * matrix (Lemma 4.2 with A = P1 - d/dx(Q1), B = Q1, so that
     * T_{h+1} = AB^{-1}). */
    nmod_poly_mat_t W;
    nmod_poly_mat_init(W, 2 * n, n, prime);
    nmod_poly_t dQ1ij;
    nmod_poly_init(dQ1ij, prime);
    for (slong i = 0; i < n; i++)
    {
        for (slong j = 0; j < n; j++)
        {
            nmod_poly_derivative(dQ1ij, nmod_poly_mat_entry(Q1, i, j));
            nmod_poly_sub(nmod_poly_mat_entry(W, i, j), nmod_poly_mat_entry(P1, i, j), dQ1ij);
            nmod_poly_set(nmod_poly_mat_entry(W, i + n, j), nmod_poly_mat_entry(Q1, i, j));
        }
    }
    nmod_poly_clear(dQ1ij);

    /* Step 6: [Qbar -Pbar] = ShiftedMinimalNullspaceBasis(W, (t,t)) -- a
     * left kernel basis of W with shift t duplicated over both n-blocks
     * (nmod_poly_mat_kernel's own shift parameter has length = nrows(W) =
     * 2n, matching this exactly). Lemma 4.2 guarantees nullity = n
     * exactly (not just generically) given Algorithm 6's own invariants,
     * so ker is n x 2n; Qbar is the raw first block, Pbar is the NEGATED
     * second block (Lemma 4.2: -Qbar^{-1}Pbar_raw is the irreducible
     * description of T_{h+1} = AB^{-1}, i.e. T_{h+1} = Qbar^{-1}(-Pbar_raw)). */
    slong * shift2n = flint_malloc(2 * n * sizeof(slong));
    for (slong i = 0; i < n; i++)
    {
        shift2n[i] = t[i];
        shift2n[i + n] = t[i];
    }

    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, 2 * n, 2 * n, prime);  /* worst-case size, only first `nullity` rows are meaningful */
    slong nullity = nmod_poly_mat_kernel(ker, NULL, shift2n, W, ORD_WEAK_POPOV, ROW_UPPER);
    if (nullity != n)
        flint_throw(FLINT_ERROR, "nmod_pseudo_Krylov_recursive: expected nullity %wd "
                    "(Lemma 4.2), got %wd\n", n, nullity);

    nmod_poly_mat_t Qbar, Pbar;
    nmod_poly_mat_init(Qbar, n, n, prime);
    nmod_poly_mat_init(Pbar, n, n, prime);
    for (slong i = 0; i < n; i++)
    {
        for (slong j = 0; j < n; j++)
        {
            nmod_poly_set(nmod_poly_mat_entry(Qbar, i, j), nmod_poly_mat_entry(ker, i, j));
            nmod_poly_neg(nmod_poly_mat_entry(Pbar, i, j), nmod_poly_mat_entry(ker, i, j + n));
        }
    }

    /* renorm (2026-09-28, tested in claude-pseudoKrylov/experiments/
     * bench-recursive-degrees-renorm.c): if Qbar has stopped being
     * absolute-minimal, bring it (and Pbar, with the same row operations)
     * back to a zero-shift weak Popov form before threading them into the
     * rest of the recursion. Conditional on the trigger firing -- an
     * UNCONDITIONAL renormalization at every node was tested too and found
     * to break t-reducedness of the accumulated D (see this function's own
     * doc and nmod_pseudo_Krylov_recursive_renorm's below). */
    if (renorm && deg_det_Q >= 0)
    {
        slong * rdeg0 = flint_malloc(n * sizeof(slong));
        nmod_poly_mat_row_degree(rdeg0, Qbar, NULL);
        slong sum_rdeg = 0;
        for (slong i = 0; i < n; i++)
            sum_rdeg += rdeg0[i];
        if (sum_rdeg > deg_det_Q)
        {
            slong * pivind = flint_malloc((n + 1) * sizeof(slong));
            nmod_poly_mat_weak_popov_iter(Qbar, NULL, Pbar, pivind, NULL, ROW_UPPER);
            flint_free(pivind);
        }
        flint_free(rdeg0);
    }

    /* Step 7: second recursive call, seeded at u = (N1)_{*,h} (the last
     * column of N1, i.e. D1*theta^h(a) -- Lemma 4.4), covering
     * theta_{h+1}(u),...,theta_{h+1}^{m-h}(u), which equals
     * D1*theta^{h+1}(a),...,D1*theta^m(a). */
    nmod_poly_mat_t v;
    nmod_poly_mat_init(v, n, 1, prime);
    for (slong i = 0; i < n; i++)
        nmod_poly_set(nmod_poly_mat_entry(v, i, 0), nmod_poly_mat_entry(N1, i, h - 1));

    nmod_poly_mat_t D2, N2;
    nmod_poly_mat_init(D2, n, n, prime);
    nmod_poly_mat_init(N2, n, h2, prime);
    _nmod_pseudo_Krylov_recursive_core(D2, N2, Qt, Pt, Qbar, Pbar, t, v, h2, renorm, deg_det_Q);

    /* Step 8: D = D2*D1, N = [D2*N1 | N2]; (Qt,Pt) already set by the
     * second recursive call above (Algorithm 6's own Q^(2),P^(2)). */
    nmod_poly_mat_mul(D, D2, D1);

    nmod_poly_mat_t D2N1;
    nmod_poly_mat_init(D2N1, n, h, prime);
    nmod_poly_mat_mul(D2N1, D2, N1);
    for (slong i = 0; i < n; i++)
    {
        for (slong j = 0; j < h; j++)
            nmod_poly_set(nmod_poly_mat_entry(N, i, j), nmod_poly_mat_entry(D2N1, i, j));
        for (slong j = 0; j < h2; j++)
            nmod_poly_set(nmod_poly_mat_entry(N, i, h + j), nmod_poly_mat_entry(N2, i, j));
    }

    flint_free(t);
    flint_free(shift2n);
    nmod_poly_mat_clear(D1);
    nmod_poly_mat_clear(N1);
    nmod_poly_mat_clear(Q1);
    nmod_poly_mat_clear(P1);
    nmod_poly_mat_clear(W);
    nmod_poly_mat_clear(ker);
    nmod_poly_mat_clear(Qbar);
    nmod_poly_mat_clear(Pbar);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(D2);
    nmod_poly_mat_clear(N2);
    nmod_poly_mat_clear(D2N1);
}

void nmod_pseudo_Krylov_recursive(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                   nmod_poly_mat_t Qt, nmod_poly_mat_t Pt,
                                   const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                   const slong * s, const nmod_poly_mat_t a,
                                   const slong m)
{
    _nmod_pseudo_Krylov_recursive_core(D, N, Qt, Pt, Q, P, s, a, m, 0, -1);
}

/** Same algorithm as nmod_pseudo_Krylov_recursive above (identical D, N as
 *  a rational description: K = D^{-1}N is unchanged, since every extra
 *  step here is a unimodular row transform of an already-irreducible
 *  description -- see the shared core's own doc), except that the raw
 *  per-node degree growth measured past m=r for algeqtodiffeq-shaped
 *  inputs (claude-pseudoKrylov/CLAUDE.md, "Degree growth past m=r, and why
 *  it's mostly cosmetic") is cleaned up along the way: whenever an
 *  internal "advance" node's (Qbar,Pbar) is no longer minimal in the
 *  absolute (zero-shift) sense, it is replaced in place by an equivalent
 *  zero-shift-reduced pair before continuing the recursion.
 *
 *  Tested (bench-recursive-degrees-renorm.c, 2026-09-22): with this
 *  conditional trigger, the per-node degree stays flat at deg(Q) for
 *  every node, at every size tried, and an evaluation-based A/B check
 *  against the non-renormalizing version never found a mismatch. An
 *  UNCONDITIONAL version (renormalizing at every node regardless of the
 *  trigger) was also tried and is NOT what this function does: it breaks
 *  t-reducedness of the accumulated product and was observed to leave the
 *  final D unreduced -- see the same session notes ("harmless" holds only
 *  because the renormalization is CONDITIONAL).
 *
 *  CAVEAT, not yet closed: this conditional trigger has only been
 *  observed empirically (never proven) to leave Qbar still t-reduced for
 *  the shift t it is about to be fed into the next recursive call as --
 *  checked exhaustively on algeqtodiffeq-shaped (Q,P) instances, and on
 *  the same random (Q,P,a,m) generator used by
 *  tests/t-pseudo-krylov-recursive.c's own "genuinely arbitrary Q,P"
 *  checks (see tests/t-pseudo-krylov-recursive-renorm.c), but not proven
 *  in general. Does NOT change deg(D) (D is a product of many always-
 *  same-determinant-degree factors, so its degree is unaffected by how
 *  minimal any one factor is on its own) -- this only cleans up
 *  intermediate bookkeeping, it is not a complexity improvement.
 *
 *  One consequence worth knowing for a caller that chains further calls
 *  via Qt, Pt: unlike nmod_pseudo_Krylov_recursive's own Qt, Pt (t-reduced
 *  for the shift t = rdeg_s(D) of the node that produced them), here they
 *  may instead be zero-shift-reduced, if the very last advance node
 *  happened to trigger the renormalization.
 */
void nmod_pseudo_Krylov_recursive_renorm(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                          nmod_poly_mat_t Qt, nmod_poly_mat_t Pt,
                                          const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                          const slong * s, const nmod_poly_mat_t a,
                                          const slong m)
{
    nmod_poly_t detQ;
    nmod_poly_init(detQ, nmod_poly_mat_modulus(Q));
    nmod_poly_mat_det(detQ, Q);
    slong deg_det_Q = nmod_poly_degree(detQ);
    nmod_poly_clear(detQ);

    _nmod_pseudo_Krylov_recursive_core(D, N, Qt, Pt, Q, P, s, a, m, 1, deg_det_Q);
}


/** Same recurrence as nmod_pseudo_Krylov_recursive (Lemma 4.3/4.4's
 *  "advance (Q,P) by one step" formula), but via a plain O(m) sequential
 *  loop instead of divide-and-conquer -- relocated/cleaned up (2026-09-16)
 *  from the draft's iterative_pseudo_krylov (gfun.c), which had zero live
 *  callers anywhere and a leftover debug file-write inside its main loop
 *  (removed here, not carried forward). A natural independent reference
 *  for testing nmod_pseudo_Krylov_recursive: same per-step formula but no
 *  recursive splitting/recombination, so it exercises the DAC
 *  recombination logic independently -- though it shares the low-level
 *  "advance" formula with nmod_pseudo_Krylov_recursive, so it would not,
 *  by itself, catch a bug in that shared formula.
 *
 *  Signature adjusted (2026-09-16) to match nmod_pseudo_Krylov_recursive's
 *  own Algorithm-6-literal convention directly, for direct comparability:
 *  computes D, N such that K = [theta(a) ... theta^m(a)] = D^{-1}N (D n x n,
 *  N n x m) -- the draft's own version instead prepended "a" itself as an
 *  extra leading column and never returned any D at all (that variable was
 *  computed once, at the very start, then never touched again -- its own
 *  accumulation, `D := Q*D`, was commented out in the draft; a genuine gap
 *  now filled in properly here, not merely uncommented, since the draft's
 *  single commented line accumulated in the wrong place relative to the
 *  loop's own column-rescale timing).
 *
 *  Uses a fixed zero shift throughout for the "advance" kernel calls,
 *  unlike nmod_pseudo_Krylov_recursive's own precise (t,t) shift -- fine
 *  for a correctness-only reference, just not necessarily degree-optimal.
 */
void nmod_pseudo_Krylov_iterative(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                   const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                   const nmod_poly_mat_t a, const slong m)
{
    ulong prime = nmod_poly_mat_modulus(Q);
    slong n = Q->r;

    nmod_poly_mat_t Qc, Pc;
    nmod_poly_mat_init(Qc, n, n, prime);
    nmod_poly_mat_init(Pc, n, n, prime);
    nmod_poly_mat_set(Qc, Q);
    nmod_poly_mat_set(Pc, P);

    nmod_poly_mat_t Dacc;
    nmod_poly_mat_init(Dacc, n, n, prime);
    nmod_poly_mat_one(Dacc);

    /* Working array of m+1 columns: index 0 holds "a" itself (theta^0(a)),
     * matching Lemma 4.4's own indexing; only columns 1..m are copied into
     * the output N at the end. */
    nmod_poly_mat_t W;
    nmod_poly_mat_init(W, n, m + 1, prime);
    for (slong i = 0; i < n; i++)
        nmod_poly_set(nmod_poly_mat_entry(W, i, 0), nmod_poly_mat_entry(a, i, 0));

    nmod_poly_mat_t v, dv, w, Qdv;
    nmod_poly_mat_init(v, n, 1, prime);
    nmod_poly_mat_init(dv, n, 1, prime);
    nmod_poly_mat_init(w, n, 1, prime);
    nmod_poly_mat_init(Qdv, n, 1, prime);

    nmod_poly_mat_t B, ker;
    nmod_poly_mat_init(B, 2 * n, n, prime);
    nmod_poly_mat_init(ker, 2 * n, 2 * n, prime);  /* worst-case size, only first `nullity` rows are meaningful */
    nmod_poly_t dQij;
    nmod_poly_init(dQij, prime);

    for (slong k = 1; k <= m; k++)
    {
        for (slong i = 0; i < n; i++)
            nmod_poly_set(nmod_poly_mat_entry(v, i, 0), nmod_poly_mat_entry(W, i, k - 1));

        /* w = Qc*dv/dx + Pc*v = Qc*theta_k(v). */
        for (slong i = 0; i < n; i++)
            nmod_poly_derivative(nmod_poly_mat_entry(dv, i, 0), nmod_poly_mat_entry(v, i, 0));
        nmod_poly_mat_mul(Qdv, Qc, dv);
        nmod_poly_mat_mul(w, Pc, v);
        nmod_poly_mat_add(w, w, Qdv);

        /* Accumulate D and rescale columns 0..k-1 of W by Qc (the current
         * step's own denominator), before advancing Qc itself. */
        nmod_poly_mat_mul(Dacc, Qc, Dacc);

        nmod_poly_mat_t Wk, TWk;
        nmod_poly_mat_window_init(Wk, W, 0, 0, n, k);
        nmod_poly_mat_init(TWk, n, k, prime);
        nmod_poly_mat_mul(TWk, Qc, Wk);
        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < k; j++)
                nmod_poly_set(nmod_poly_mat_entry(W, i, j), nmod_poly_mat_entry(TWk, i, j));
        nmod_poly_mat_window_clear(Wk);
        nmod_poly_mat_clear(TWk);

        for (slong i = 0; i < n; i++)
            nmod_poly_set(nmod_poly_mat_entry(W, i, k), nmod_poly_mat_entry(w, i, 0));

        /* Advance (Qc,Pc) by one step -- same formula and sign convention
         * as nmod_pseudo_Krylov_recursive's own Steps 5-6, unshifted. */
        for (slong i = 0; i < n; i++)
        {
            for (slong j = 0; j < n; j++)
            {
                nmod_poly_derivative(dQij, nmod_poly_mat_entry(Qc, i, j));
                nmod_poly_sub(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(Pc, i, j), dQij);
                nmod_poly_set(nmod_poly_mat_entry(B, i + n, j), nmod_poly_mat_entry(Qc, i, j));
            }
        }
        slong nullity = nmod_poly_mat_kernel(ker, NULL, NULL, B, ORD_WEAK_POPOV, ROW_UPPER);
        if (nullity != n)
            flint_throw(FLINT_ERROR, "nmod_pseudo_Krylov_iterative: expected nullity %wd, "
                        "got %wd\n", n, nullity);
        for (slong i = 0; i < n; i++)
        {
            for (slong j = 0; j < n; j++)
            {
                nmod_poly_set(nmod_poly_mat_entry(Qc, i, j), nmod_poly_mat_entry(ker, i, j));
                nmod_poly_neg(nmod_poly_mat_entry(Pc, i, j), nmod_poly_mat_entry(ker, i, j + n));
            }
        }
    }

    nmod_poly_mat_set(D, Dacc);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < m; j++)
            nmod_poly_set(nmod_poly_mat_entry(N, i, j), nmod_poly_mat_entry(W, i, j + 1));

    nmod_poly_clear(dQij);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
    nmod_poly_mat_clear(Qc);
    nmod_poly_mat_clear(Pc);
    nmod_poly_mat_clear(Dacc);
    nmod_poly_mat_clear(W);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(dv);
    nmod_poly_mat_clear(w);
    nmod_poly_mat_clear(Qdv);
}
