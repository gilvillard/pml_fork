/*
    Copyright (C) 2026 Gilles Villard 

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#ifndef GFUN_H
#define GFUN_H

#include <flint/fmpz_poly_mat.h>
#include <flint/nmod_mpoly.h>

#include "nmod_poly_mat_extra.h"


/** Bound on the x-degree of Delta*T(v) for v of x-degree 0, where
 *  Delta = resultant(P, Py) and P has (x-degree, y-degree) = (d, r).
 *  Also a valid bound wherever gfun.c computes "C = -Px (Py)^(-1) times the
 *  resultant" itself (same shape of bound, M^* and Y/Px contributions).
 *  Centralizes what used to be the literal "(2*r-1)*d - 1" typed out
 *  separately at each call site (nmod_algeq_to_diffeq, find_uv, nmod_phi1,
 *  nmod_phi_T) -- those call sites are being migrated to it one at a time,
 *  not all at once (see claude-pseudoKrylov/todo.md).
 */
static inline slong nmod_gfun_delta_T_degree_bound(slong r, slong d)
{
    return (2*r - 1)*d - 1;
}


/** Additive margin added on top of a degree bound that assumes T is
 *  exactly proper (finite limit at infinity) -- for inputs where that
 *  assumption doesn't quite hold, the "properness-assuming" bound alone
 *  can be too small. NOT a derived value: a fixed margin the user has
 *  found sufficient in practice, promoted here (2026-09-15) from a bare
 *  "+200" literal so it's named, documented, and adjustable from one
 *  place instead of copy-pasted.
 *
 *  Important limitation, not fixed by this constant: the underlying
 *  geometric evaluation-interpolation (nmod_apply_T /
 *  nmod_biv_mulmod_geometric) has no bounds checking of its own -- if the
 *  true degree exceeds the bound (margin included), the result is not an
 *  error, it's a silently *wrong* (aliased) polynomial. A fixed margin,
 *  however generous, is a guess about every future input, not a
 *  guarantee. The real fix (claude-pseudoKrylov/todo.md, "principled D
 *  margin" future work) is a verify-and-retry scheme: spend one extra
 *  evaluation point beyond the bound, check the interpolated result
 *  against a direct evaluation there, and bump D and redo if they
 *  disagree -- not implemented yet. Use this constant to remove the
 *  literal, not as evidence the underlying risk is handled.
 */
#define NMOD_GFUN_NONPROPER_MARGIN 8


/** Additive margin on the target degree for a description of the PSEUDO-KRYLOV
 *  construction as a whole -- currently K, in algeqtodiffeq_series.c. Split out
 *  2026-09-18, per the user: it had been sharing NMOD_GFUN_NONPROPER_MARGIN's
 *  name and value while covering something else entirely.
 *
 *  The distinction is NOT "degree margin vs evaluation-points margin" -- that
 *  was the first, wrong reading of it, corrected by the user the same day.
 *  NMOD_GFUN_NONPROPER_MARGIN belongs in degree bounds too:
 *  nmod_algeqtodiffeq_T_left_description (algeqtodiffeq_recursive.c) uses it
 *  for the target degree of a description of T, and that IS a properness
 *  question -- how far T's own degree may exceed what the properness-assuming
 *  bound predicts. It should keep using NMOD_GFUN_NONPROPER_MARGIN.
 *
 *  What THIS constant covers is different in kind: the degree of a description
 *  of the whole pseudo-Krylov machine -- K built through n applications of
 *  theta, with power-series truncation and an inverse series along the way --
 *  whose degree uncertainty does not reduce to T's properness.
 *
 *  The failure modes also differ, which is a useful check on which constant you
 *  are reaching for:
 *   - too small a NONPROPER margin in an evaluation bound => nmod_apply_T
 *     aliases and silently returns a WRONG polynomial (see its doc above);
 *   - too small a margin on a description's target degree => no row passes the
 *     shift[i] <= target_degree filter and the caller flint_throw's "no
 *     description of degree at most ... found" -- loud, not silent.
 *
 *  Same heuristic status as its sibling: a value that works in practice, not a
 *  derived bound. MAIN TODO (per the user) is to expose target_degree as a
 *  caller-supplied parameter rather than always deriving it internally --
 *  see claude-pseudoKrylov/todo.md.
 *
 *  LOAD-BEARING: nmod_pseudo_Krylov_series and
 *  nmod_algeqtodiffeq_series_left_description (algeqtodiffeq_series.c) must
 *  compute target_degree/sigma by the SAME formula -- that identity is exactly
 *  what guarantees K's own precision suffices for the description later built
 *  from it, with no precision bookkeeping needed between the two phases.
 *  Change one, change the other.
 */
#define NMOD_GFUN_DESCRIPTION_MARGIN 8


/**
 *    Assume that the degree r in the second variable, say y, is the row dimension - 1 of PT
 */

void mat_to_xy(nmod_mpoly_t P, nmod_mpoly_ctx_t ctx, const nmod_poly_mat_t PT);


/**  Resultant of P and the derivative Py, and the inverse of Py mod P times the resultant
 *     as poly_mat
 *    deg_y P = r, hence PT has r+1 rows
 *
 *    Implementation in resultant.c (moved out of gfun.c 2026-09, mechanical
 *    relocation only -- this computation is self-contained, not specific to
 *    the rest of gfun.c). See claude-pseudoKrylov/todo.md item 11 for a real,
 *    unaddressed correctness gap (an unguarded "bad evaluation point" hazard)
 *    found while reading it, not yet fixed.
 */

void nmod_biv_resultant_geometric(nmod_poly_t Delta, nmod_poly_mat_t  iPyT, const nmod_poly_mat_t PT);


/** Geometric bivariate multiplication A*B mod P, with respect to y
 *    the geometric progression is initialized outside
 *
 *   The memainder is known - in advance - to be a polynomial
 *    the geometric progression is driven, in particular, by its x-degree
 *
 *   To see: aliasing ?
 *
 *   Implementation in algeqtodiffeq.c, grouped there with nmod_apply_T and
 *   nmod_phi1 (2026-09, "for the moment" -- see that file's own header
 *   comment for why: every current caller of this function is
 *   algeqtodiffeq-specific, even though its own signature is generic).
 */

void nmod_biv_mulmod_geometric(nmod_poly_mat_t  RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t BT, \
                                const nmod_poly_mat_t PT,  const ulong D);



/** Linear transformation T for algeqtodiffeq
 *   CT is C = -Px (Py)^(-1) that has been precomputed
 *
 *   The result is known - in advance - to be a polynomial of x-degree at most D
 *    the geometric progression is driven by D
 *
 *   To see: aliasing?
 *
 *   Implementation in algeqtodiffeq.c.
 *
 */

void nmod_apply_T(nmod_poly_mat_t  RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const ulong D);


/**  Randomized computation of phi1 (and, via nmod_phi_T, phi2)
 *   -----------------------------------------------------------
 *
 *    using one (resp. two) random constant combination(s) of the columns of
 *    T; see algeqtodiffeq.c for the full derivation (both moved/cleaned up
 *    2026-09), the width-1 dependency, and the caller-supplied `state`
 *    convention. deg(phi1)==deg(phi2) is the width(T)<=1 check used by
 *    _nmod_algeq_to_diffeq_width1 (algeqtodiffeq_width1.c, flint_throw's on
 *    failure) -- see nmod_phi_T's own doc in algeqtodiffeq.c and
 *    claude-pseudoKrylov/todo.md.
 *
 *    r is assumed >= 3 for phi2 (nmod_phi_T only) ?
 *
 */

void nmod_phi1(nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta, \
                     flint_rand_t state);

void nmod_phi_T(nmod_poly_t  phi1, nmod_poly_t  phi2, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta, \
                     flint_rand_t state);


/** Common algeqtodiffeq driver setup, factored out 2026-09 (see
 *  algeqtodiffeq.c for the full doc on each): nmod_algeqtodiffeq_setup
 *  builds Delta/iPyT/CT (Delta-scaled) from P alone;
 *  nmod_algeqtodiffeq_rescale_CT_by_phi1 rescales CT in place given an
 *  already-known phi1; nmod_algeqtodiffeq_rescale_by_phi1 additionally
 *  computes phi1 itself (nmod_phi1) first, for a caller that doesn't
 *  already have it.
 */

void nmod_algeqtodiffeq_setup(nmod_poly_t Delta, nmod_poly_mat_t iPyT,
                               nmod_poly_mat_t CT, const nmod_poly_mat_t PT);

void nmod_algeqtodiffeq_rescale_CT_by_phi1(nmod_poly_mat_t CT,
                                            const nmod_poly_mat_t PT,
                                            const nmod_poly_t Delta,
                                            const nmod_poly_t phi1);

void nmod_algeqtodiffeq_rescale_by_phi1(nmod_poly_t phi1, nmod_poly_mat_t CT,
                                         const nmod_poly_mat_t PT,
                                         const nmod_poly_t Delta,
                                         flint_rand_t state);


/** Rank-one decomposition (algos.pdf Lemma 3.3, Sec. 3.2) of T (width <= 1)
 *  -- the width-1 family's own counterpart to nmod_phi1 (both a
 *  random-projection Monte Carlo construction, both exploit the same
 *  width-1 structure). Implementation in algeqtodiffeq_width1.c. Cleaned
 *  up 2026-09-15: takes an explicit flint_rand_t (see nmod_phi1's own doc
 *  for why); internally rescales its own local working copy of CT via
 *  nmod_algeqtodiffeq_rescale_CT_by_phi1 once, rather than the
 *  per-column-then-divide-by-Delta/phi1 pattern the draft used (same
 *  Delta-scale-vs-phi1-scale distinction as nmod_pseudo_Krylov vs.
 *  nmod_pseudo_Krylov_naive) -- CT/Delta/phi1 themselves are unchanged in
 *  meaning and are still passed in Delta-scaled, exactly as before.
 */
void find_uv(nmod_poly_mat_t U, nmod_poly_mat_t V, const nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta, \
                     flint_rand_t state);

/** Triangular description of a pseudo-Krylov matrix built from a rank-one
 *  pair (algos.pdf Algorithm 3 "DescriptionFromRank1", Proposition 3.2,
 *  Sec. 3.3) -- renamed from Description_From_Rank_1 (2026-09-16, "width1"
 *  per the user's naming correction, see find_uv's doc above). Cleaned up:
 *  same optimization as find_uv (CT rescaled to phi1-scale once instead of
 *  a post-division by Delta/phi1 at every column). Full derivation and
 *  calling-convention notes in algeqtodiffeq_width1.c, where it now lives.
 */

void nmod_width1_description(nmod_poly_mat_t NN, nmod_poly_mat_t DD, const ulong n,
                              const nmod_poly_mat_t U, const nmod_poly_mat_t V,
                              const nmod_poly_t phi1, const nmod_poly_mat_t CT,
                              const nmod_poly_mat_t PT, const nmod_poly_t alpha,
                              const nmod_poly_mat_t N_ini, const nmod_poly_t Delta);

/** Algorithm 4 (PseudoKrylovWidth1, algos.pdf Sec. 3.4) -- the width-1
 *  family's own core computation, replacing nmod_algeq_to_diffeq_new
 *  (gfun.c, superseded, kept only for reference -- see
 *  claude-pseudoKrylov/todo.md for the two real bugs found and fixed here,
 *  not just a cleanup-in-place). flint_throw's on width(T) > 1 (per the
 *  user's decision, 2026-09-16). Leading underscore (2026-09-17, per the
 *  user): this is NOT a generic "pseudo_Krylov" building block like
 *  nmod_pseudo_Krylov_recursive/_iterative (those stop at a description
 *  (N,D)); it takes algeqtodiffeq's own CT/PT/Delta directly and runs the
 *  computation all the way through the final kernel/nullspace step,
 *  returning the solutions Y themselves -- i.e. it's the same kind of
 *  function as nmod_algeq_to_diffeq_width1 below (full algeqtodiffeq
 *  solve), just with a more general interface (arbitrary seed a and m, no
 *  setup step) instead of the fixed-seed Cockle wrapper. The underscore
 *  marks exactly that relationship, matching FLINT's own convention:
 *  nmod_algeq_to_diffeq_width1 (the convenience wrapper) calls this
 *  function internally. Full doc in algeqtodiffeq_width1.c, where it lives
 *  alongside find_uv and nmod_width1_description.
 */
slong _nmod_algeq_to_diffeq_width1(nmod_poly_mat_t Y, const nmod_poly_mat_t a, const ulong m,
                                    const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                    const nmod_poly_t Delta, flint_rand_t state);

/** Cockle's algorithm (G2026.pdf Sec. 7) via _nmod_algeq_to_diffeq_width1
 *  above, seeding a = y -- same n convention as nmod_algeq_to_diffeq_naive
 *  (n = total pseudo-Krylov width, n >= 2 here since Algorithm 4's own
 *  m = n-1 must be >= 1). See algeqtodiffeq_width1.c for the full doc.
 */
slong nmod_algeq_to_diffeq_width1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/**  Computation of the numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix
 *
 *    fraction-free approach
 *   uses phi1 as computed previously, non monic since directly related to the resultant (non monic either)
 *
 *    nmod_pseudo_Krylov (this one, without a CT already rescaled by phi1)
 *    is superseded -- see claude-pseudoKrylov/todo.md, "the naive family":
 *    nmod_pseudo_Krylov_naive (algeqtodiffeq_naive.c) is the renamed,
 *    cleaned-up former nmod_pseudo_Krylov_phi1 (removed 2026-09-16, had
 *    zero live callers anywhere -- see claude-pseudoKrylov/todo.md).
 *    This declaration stays because gfun.c's own nmod_algeq_to_diffeq
 *    (non-phi1 driver, itself still called live from the sibling
 *    ../work-algeqtodiffeq project) still calls this one and hasn't been
 *    touched; also called by nmod_pseudo_Krylov_naive_delta below.
 */

void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);


/** The naive pseudo-Krylov family (algeqtodiffeq_naive.c): renamed,
 *  cleaned-up versions of the former nmod_pseudo_Krylov_phi1 /
 *  nmod_algeq_to_diffeq_phi1 (both removed 2026-09-16, see
 *  claude-pseudoKrylov/todo.md). Uses nmod_algeqtodiffeq_setup +
 *  nmod_algeqtodiffeq_rescale_by_phi1 instead of duplicating that
 *  preamble; nmod_pseudo_Krylov_naive drops the `Delta` parameter (dead in
 *  the *_phi1 original -- computed into an unused `g` and never
 *  referenced, since CT is already phi1-scaled by the time this is
 *  called).
 */

void nmod_pseudo_Krylov_naive(nmod_poly_mat_t K, ulong n, const nmod_poly_mat_t CT,
                               const nmod_poly_mat_t PT, const nmod_poly_t phi1);

slong nmod_algeq_to_diffeq_naive(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/** Independent, Delta-scaled reference construction of the r x n
 *  fraction-free pseudo-Krylov matrix for a=y -- renamed/relocated
 *  (2026-09-16, per the user) from nmod_pseudo_Krylov_for_kernel, now in
 *  its own file (algeqtodiffeq_reference.c) precisely because it's NOT
 *  one of the four algorithm families: it goes through the old
 *  nmod_pseudo_Krylov with phi1 forced equal to Delta, deliberately
 *  bypassing the phi1 optimization the naive/width-1 families rely on.
 *  That independence is what makes it the trusted reference in
 *  tests/t-algeq-to-diffeq-naive.c and tests/t-algeq-to-diffeq-width1.c
 *  (claude-pseudoKrylov) -- see that file's own header comment for the
 *  full rationale, including why it's deliberately NOT given the families'
 *  own cleanup treatment. Still used by gfun.c's own
 *  CRT_pseudo_Krylov_for_kernel (CRT/fmpz track, not yet started).
 */
void nmod_pseudo_Krylov_naive_delta(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t PT);



/**  algeqtodiffeq
 *
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 *
 *    nmod_algeq_to_diffeq_phi1 (superseded by nmod_algeq_to_diffeq_naive,
 *    algeqtodiffeq_naive.c) was removed 2026-09-16 -- had zero live
 *    callers anywhere (fork-pml, mapml, tests, or the sibling
 *    ../work-algeqtodiffeq project), unlike nmod_algeq_to_diffeq
 *    (non-phi1) below, which stays untouched, not ported forward, because
 *    ../work-algeqtodiffeq's own work.c/algeqtodiffeq.c still call it live
 *    (see claude-pseudoKrylov/todo.md).
 *
 */

slong nmod_algeq_to_diffeq(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/**  algeqtodiffeq series and descirption 
 * 
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 *    
 */ 

slong nmod_algeq_to_diffeq_series(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

slong nmod_algeq_to_diffeq_series_phi1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/* nmod_algeq_to_diffeq_new (gfun.c) is superseded by
 * nmod_algeq_to_diffeq_width1 above (algeqtodiffeq_width1.c) -- two real
 * bugs found relative to Algorithm 4 (see that function's own doc and
 * claude-pseudoKrylov/todo.md), not just an unoptimized version of the
 * same computation. Declaration kept only because gfun.c's own body
 * hasn't been removed yet. */
slong nmod_algeq_to_diffeq_new(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/** Algorithm 6 (algos.pdf Sec. 4.2.1, "PseudoKrylovDAC1" in the paper's own
 *  vocabulary -- "DAC" = Divide And Conquer, see Algorithm 5's own name
 *  "PseudoKrylovDAC2"), differential case only for now. Solves the
 *  GENERAL pseudo-Krylov description problem for an arbitrary theta =
 *  d/dx + T*sigma given directly via T = Q^{-1}P (no width assumption,
 *  unlike the width-1 family) -- lives in its own file
 *  (pseudo_krylov_recursive.c), not algeqtodiffeq-specific. Full doc
 *  there. Corresponds to the draft's _rec_pseudo_krylov/rec_pseudo_krylov
 *  (gfun.c, not otherwise touched) -- reimplemented from scratch matching
 *  Algorithm 6's own structure literally, per the user's choice
 *  (2026-09-16): the draft folds the "advance" step into its k==1 base
 *  case rather than keeping it in the m>1 recursive branch, which is
 *  computationally equivalent but less directly traceable against the
 *  paper.
 */
void nmod_pseudo_Krylov_recursive(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                   nmod_poly_mat_t Qt, nmod_poly_mat_t Pt,
                                   const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                   const slong * s, const nmod_poly_mat_t a,
                                   const slong m);

/** Same recurrence as nmod_pseudo_Krylov_recursive above, via a plain
 *  O(m) sequential loop instead of divide-and-conquer -- a natural
 *  independent reference for testing it (same per-step formula, no
 *  recursive splitting). Renamed/relocated (2026-09-16, per the user)
 *  from the draft's iterative_pseudo_krylov (gfun.c, zero live callers
 *  anywhere); signature adjusted to match nmod_pseudo_Krylov_recursive's
 *  own (D,N) convention directly (the draft prepended "a" as an extra
 *  leading column and never properly returned D). Full doc in
 *  pseudo_krylov_recursive.c.
 */
void nmod_pseudo_Krylov_iterative(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                   const nmod_poly_mat_t Q, const nmod_poly_mat_t P,
                                   const nmod_poly_mat_t a, const slong m);

/** Computes an irreducible left description (Q,P) of algeqtodiffeq's own
 *  T (T=Q^{-1}P), suitable as direct input to nmod_pseudo_Krylov_recursive
 *  above -- the "second step" (algeqtodiffeq wiring for the Section-4/DAC1
 *  family), draft correspondence nmod_algeq_to_diffeq_last_phi1's pieces
 *  (a)+(b) (gfun.c, not otherwise touched). Via a truncated approximant
 *  basis (PM-Basis), per the user's choice, 2026-09-16 -- see
 *  algeqtodiffeq_recursive.c for the full doc, including a MAIN open todo
 *  on the heuristic degree bound this relies on
 *  (claude-pseudoKrylov/todo.md).
 */
void nmod_algeqtodiffeq_T_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
                                          const nmod_poly_t phi1,
                                          const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                          const nmod_poly_t Delta);

/** Description of algeqtodiffeq's own pseudo-Krylov matrix K =
 *  [theta(a) ... theta^m(a)] = D^{-1}N (D, N as in nmod_pseudo_Krylov_recursive),
 *  for theta = d/dx + T (T algeqtodiffeq's own, via CT/PT/Delta). Pieces
 *  (a)+(b)+(c) of the "second step" (draft nmod_algeq_to_diffeq_last_phi1):
 *  builds an irreducible left description of T via
 *  nmod_algeqtodiffeq_T_left_description above, then feeds it directly to
 *  nmod_pseudo_Krylov_recursive (zero shift, Qt/Pt discarded -- no caller
 *  needs to extend the sequence further yet). D, N must already be
 *  nmod_poly_mat_init'd by the caller, r x r and r x m (r = (PT->r)-1).
 *  phi1 already computed by the caller (e.g. via nmod_phi1). See
 *  algeqtodiffeq_recursive.c for the full doc.
 */
void nmod_algeqtodiffeq_pseudo_krylov_description(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                                   const nmod_poly_t phi1,
                                                   const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                                   const nmod_poly_t Delta,
                                                   const nmod_poly_mat_t a, const slong m);

/** Cockle's algorithm (G2026.pdf Sec. 7) via the Section-4/DAC1 family
 *  (Algorithm 6): an alternative to nmod_algeq_to_diffeq_width1 that needs
 *  no width <= 1 assumption on T. Seeds a = y, gets (D,N) from
 *  nmod_algeqtodiffeq_pseudo_krylov_description above with m = n-1, then
 *  piece (d): rescales the seed itself by D (v = D*a, matching the other
 *  columns' common left factor) and takes a plain column kernel of
 *  [v | N] directly -- valid since D is a fixed nonsingular left
 *  multiplier, so [v|N]*eta = 0 iff [a, theta(a), ..., theta^{n-1}(a)]*eta
 *  = 0 (no division by D ever needed). Same n convention as
 *  nmod_algeq_to_diffeq_naive/_width1 (n = total pseudo-Krylov matrix
 *  width, n >= 2). See algeqtodiffeq_recursive.c for the full doc.
 */
slong nmod_algeq_to_diffeq_recursive(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

/** The "Series" (heuristic) family, algeqtodiffeq's fourth pseudo-Krylov
 *  approach (draft nmod_algeq_to_diffeq_series_phi1, gfun.c, not otherwise
 *  touched) -- no numbered algorithm in algos.pdf/G2026.pdf to cross-check
 *  against, genuinely heuristic (see algeqtodiffeq_series.c's own header
 *  comment): builds the pseudo-Krylov matrix K only as a power series
 *  TRUNCATED to a fixed precision (unlike the other three families' exact
 *  fraction-free tracking), then recovers an exact rational description
 *  from that truncation via a matrix Hermite-Padé-style approximant basis.
 *  If the truncation precision guess is too small, this can silently
 *  produce a wrong answer consistent with the truncation -- that risk,
 *  not present in the other families, is what makes this one unproven.
 *
 *  MAIN TODO (per the user, 2026-09-17): target_degree (hence the
 *  precision N/sigma derived from it) is currently computed internally
 *  from deg(phi1) alone, matching this module's existing heuristic-margin
 *  pattern -- exposing it as a caller-supplied parameter is flagged as a
 *  likely future need, not done here (claude-pseudoKrylov/todo.md).
 *
 *  Convention note: unlike every other pseudo-Krylov builder here (which
 *  return fraction-free NUMERATORS, column j carrying an implicit
 *  denominator phi1^j or Delta^j), this one returns the honest truncated
 *  TAYLOR SERIES of theta^j(a) itself -- the phi1^j is divided out as a
 *  power series internally, which is why this route needs phi1(0) != 0
 *  (checked; flint_throw's otherwise, since x=0 is then a pole).
 *
 *  Returns N, the truncation order actually used (every entry is correct
 *  mod x^N).
 */
slong nmod_pseudo_Krylov_series(nmod_poly_mat_t K, const nmod_poly_t phi1,
                                 const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                 const slong n);

/** Computes an irreducible left description (D,N) of the (truncated)
 *  pseudo-Krylov matrix K (D*K=N) directly via nmod_poly_mat_pmbasis --
 *  the SAME construction technique as nmod_algeqtodiffeq_T_left_description
 *  (a truncated approximant basis, filtering rows by shift<=target_degree),
 *  applied to K instead of T's own matrix. Deliberately NOT built on top of
 *  PML's generic nmod_poly_mat_left_description (nmod_poly_mat_description.c)
 *  -- per the user, 2026-09-17: "I don't want to rely on
 *  nmod_poly_mat_left_description for the moment ... it is not stable at
 *  all (we will consider it later)". target_degree/sigma computed
 *  internally from phi1, matching nmod_pseudo_Krylov_series's own formula
 *  exactly. See algeqtodiffeq_series.c for the full doc.
 */
void nmod_algeqtodiffeq_series_left_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                                                 const nmod_poly_mat_t K, const nmod_poly_t phi1);

/** Cockle's algorithm (G2026.pdf Sec. 7) via the Series/Padé family's own
 *  LEFT-description route only, for now -- a nmod_algeq_to_diffeq_series_right
 *  sibling (via a random n x r projection reducing the r x n K to a
 *  square n x n system before the description step) is an explicit future
 *  todo, important for efficiency when n << r (claude-pseudoKrylov/todo.md)
 *  -- meant to coexist with this one, not replace it. nmod_algeq_to_diffeq_series
 *  itself is already taken (still used by Maple's pm_algeq2diffeq_series,
 *  pointing at the non-_phi1 draft in gfun.c) -- this is a fresh name, not
 *  a repointing, since nmod_algeq_to_diffeq_series_phi1 (the draft cleaned
 *  up here) has no Maple binding at all. Same n/seed=y convention as
 *  nmod_algeq_to_diffeq_naive/_width1/_recursive. See algeqtodiffeq_series.c
 *  for the full doc.
 */
slong nmod_algeq_to_diffeq_series_left(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

slong nmod_algeq_to_diffeq_last(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

slong nmod_algeq_to_diffeq_last_phi1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

void _rec_pseudo_krylov(nmod_poly_mat_t D1, nmod_poly_mat_t N1, \
                        nmod_poly_mat_t P1, nmod_poly_mat_t Q1, \
                        const nmod_poly_mat_t P, const nmod_poly_mat_t Q,\
                                 const nmod_poly_mat_t a, const slong k, \
                                 const slong * rdeg);

void rec_pseudo_krylov(nmod_poly_mat_t N, const nmod_poly_mat_t iP, const nmod_poly_mat_t iQ,\
                                 const nmod_poly_mat_t a, const slong n);

/** CRT for Krylov polynomial matrix ready for kernel
 * 
 *  return M 
 *   and found 
 * 
 */

void CRT_pseudo_Krylov_for_kernel(fmpz_poly_mat_t int_residues, slong * degs, const ulong n, const slong L, const nn_ptr primes,\
                                  fmpz_poly_mat_t  PZT);


void CRT_poly_mat_combine(fmpz_poly_mat_t int_residues, slong * degs,\
                            const fmpz_poly_mat_t int_residues_1, const fmpz_t P_1,\
                            const fmpz_poly_mat_t int_residues_2, const fmpz_t P_2);


void fmpz_poly_mat_print_pretty(const fmpz_poly_mat_t mat, const char * var);

void fmpz_poly_mat_fprint_pretty(FILE *file, const fmpz_poly_mat_t mat, const char * var);

void nmod_poly_mat_fprint_pretty(FILE *file, const nmod_poly_mat_t mat, const char * var);



// One column
void  fmpz_to_nmod_poly_mat(nmod_poly_mat_t PT, const fmpz_poly_mat_t PZT);

// One column
void  nmod_to_fmpz_poly_mat(fmpz_poly_mat_t PZT, const nmod_poly_mat_t PT);


#endif // GFUN_H

