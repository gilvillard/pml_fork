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
void nmod_algeqtodiffeq_T_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
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
        flint_throw(FLINT_ERROR, "nmod_algeqtodiffeq_T_left_description: no complete "
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


/** Description of algeqtodiffeq's own pseudo-Krylov matrix, piece (c) of
 *  the "second step" wired onto pieces (a)+(b) above: builds an
 *  irreducible left description of T via nmod_algeqtodiffeq_T_left_description,
 *  then feeds it directly to the general nmod_pseudo_Krylov_recursive
 *  (pseudo_krylov_recursive.c) -- zero shift (no reason yet to favor any
 *  row of the description), Qt/Pt discarded on a local scratch pair (no
 *  caller of this function needs to extend the sequence further; they're
 *  only useful to nmod_pseudo_Krylov_recursive's own recursive calls, and
 *  it manages those internally).
 *
 *  D, N must already be nmod_poly_mat_init'd by the caller, r x r and
 *  r x m (r = (PT->r)-1). phi1, CT, PT, Delta as elsewhere in this module.
 *  Output: theta = d/dx + T satisfies [theta(a) ... theta^m(a)] = D^{-1}N.
 */
void nmod_algeqtodiffeq_pseudo_krylov_description(nmod_poly_mat_t D, nmod_poly_mat_t N,
                                                   const nmod_poly_t phi1,
                                                   const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                                   const nmod_poly_t Delta,
                                                   const nmod_poly_mat_t a, const slong m)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_mat_t Q, P;
    nmod_poly_mat_init(Q, r, r, prime);
    nmod_poly_mat_init(P, r, r, prime);
    nmod_algeqtodiffeq_T_left_description(Q, P, phi1, CT, PT, Delta);

    nmod_poly_mat_t Qt, Pt;
    nmod_poly_mat_init(Qt, r, r, prime);
    nmod_poly_mat_init(Pt, r, r, prime);
    nmod_pseudo_Krylov_recursive(D, N, Qt, Pt, Q, P, NULL, a, m);

    nmod_poly_mat_clear(Q);
    nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(Qt);
    nmod_poly_mat_clear(Pt);
}


/** Cockle's algorithm (G2026.pdf Sec. 7) via the Section-4/DAC1 family
 *  (Algorithm 6) -- piece (d) on top of nmod_algeqtodiffeq_pseudo_krylov_description
 *  above, an alternative to nmod_algeq_to_diffeq_width1 that needs no
 *  width <= 1 assumption on T (Section 4's whole point, per CLAUDE.md).
 *
 *  Draft correspondence (rec_pseudo_krylov/nmod_algeq_to_diffeq_last_phi1,
 *  gfun.c, not otherwise touched): given D, N with [theta(a) ... theta^{n-1}(a)]
 *  = D^{-1}N (m = n-1 columns), the draft rescales the seed itself by the
 *  SAME D (v = D*a) and prepends it, giving a combined matrix
 *  [v | N] = D*[a, theta(a), ..., theta^{n-1}(a)] all sharing one common
 *  left factor D. Since D is a fixed nonsingular matrix (Lemma 4.2), a
 *  plain column kernel of [v|N] already gives exactly the eta with
 *  sum eta_i*theta^i(a) = 0 -- no division by D is ever needed:
 *  D*K*eta = 0 iff K*eta = 0. Reimplemented fresh here (not a clean-up of
 *  the draft, matching how nmod_pseudo_Krylov_recursive itself was done),
 *  same convention throughout this project's Cockle drivers
 *  (nmod_algeq_to_diffeq_naive/_width1): n is the total pseudo-Krylov
 *  matrix width (n >= 2), seeding a = y.
 *
 *  Returns nz, the number of solutions found; LT is an n x n polynomial
 *  matrix whose first nz columns are the solutions (LT[i][j] = eta_i of
 *  the j-th solution) -- same convention as nmod_algeq_to_diffeq_width1.
 */
slong nmod_algeq_to_diffeq_recursive(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_recursive: n must be >= 2 "
                    "(n is the pseudo-Krylov matrix width itself)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong m = n - 1;

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
    nmod_phi1(phi1, CT, PT, Delta, state);

    nmod_poly_mat_t a;
    nmod_poly_mat_init(a, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_zero(nmod_poly_mat_entry(a, i, 0));
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(a, 1, 0), 0, 1);

    nmod_poly_mat_t D, N;
    nmod_poly_mat_init(D, r, r, prime);
    nmod_poly_mat_init(N, r, m, prime);
    nmod_algeqtodiffeq_pseudo_krylov_description(D, N, phi1, CT, PT, Delta, a, m);

    /* (d): v = D*a, K = [v | N], plain column kernel. */
    nmod_poly_mat_t v;
    nmod_poly_mat_init(v, r, 1, prime);
    nmod_poly_mat_mul(v, D, a);

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K, r, n, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set(nmod_poly_mat_entry(K, i, 0), nmod_poly_mat_entry(v, i, 0));
        for (slong j = 0; j < m; j++)
            nmod_poly_set(nmod_poly_mat_entry(K, i, j + 1), nmod_poly_mat_entry(N, i, j));
    }

    /* Explicit zero shift, not NULL (which means K's column degrees): for
     * nz > 1 this returns the zero-shift minimal basis of the solutions,
     * the same one as the other drivers. shift is in/out, length ncols(K). */
    slong * pivind = flint_malloc(n * sizeof(slong));
    slong * shift = flint_calloc(n, sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, shift, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);
    flint_free(shift);

    flint_rand_clear(state);
    nmod_poly_mat_clear(a);
    nmod_poly_mat_clear(D);
    nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(K);
    nmod_poly_clear(phi1);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);

    return nz;
}


/** Irreducible left description T_b = Q^{-1}P of the quotient of
 *  algeqtodiffeq's module by k(x).1: T_b is T without its first row and
 *  column (T's first column is zero since theta(1) = 0). Q, P must be
 *  initialized (r-1) x (r-1), r = (PT->r)-1 >= 2; phi1, CT, PT, Delta as in
 *  nmod_algeqtodiffeq_T_left_description, whose construction this follows
 *  (pmbasis of [-phi1*T_b; phi1*I] at order sigma, rows of shift <= target),
 *  with phi1's degree spread over the r-1 rows of T_b. Throws unless exactly
 *  r-1 rows pass the filter: more would make the choice ambiguous.
 */
void nmod_algeqtodiffeq_Tb_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
                                           const nmod_poly_t phi1,
                                           const nmod_poly_mat_t CT, const nmod_poly_mat_t PT,
                                           const nmod_poly_t Delta)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong k = r - 1;

    if (k < 1)
        flint_throw(FLINT_DOMERR, "nmod_algeqtodiffeq_Tb_left_description: needs r >= 2\n");

    nmod_poly_mat_t CT_phi1;
    nmod_poly_mat_init(CT_phi1, r, 1, prime);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(CT_phi1, i, 0), nmod_poly_mat_entry(CT, i, 0));
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT_phi1, PT, Delta, phi1);

    slong d = nmod_poly_mat_degree(PT);
    slong Dbound = nmod_gfun_delta_T_degree_bound(r, d) + nmod_poly_degree(phi1);

    /* Tb = (phi1*T)[1:,1:], column j from phi1*T(y^j) */
    nmod_poly_mat_t Tb, Yk, col;
    nmod_poly_mat_init(Tb, k, k, prime);
    nmod_poly_mat_init(Yk, r, 1, prime);
    nmod_poly_mat_init(col, r, 1, prime);
    for (slong j = 1; j < r; j++)
    {
        nmod_poly_mat_zero(Yk);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Yk, j, 0), 0, 1);
        nmod_apply_T(col, Yk, CT_phi1, PT, Dbound);
        for (slong i = 1; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(Tb, i - 1, j - 1), nmod_poly_mat_entry(col, i, 0));
    }

    slong deg_phi1 = nmod_poly_degree(phi1);
    slong target_degree = (deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN + k - 1) / k; /* ceil */
    slong sigma = 2 * target_degree + 1;

    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, 2 * k, k, prime);
    for (slong i = 0; i < k; i++)
    {
        for (slong j = 0; j < k; j++)
        {
            nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(Tb, i, j));
            nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
        }
        nmod_poly_set_trunc(nmod_poly_mat_entry(B, i + k, i), phi1, sigma);
    }

    slong * shift = flint_calloc(2 * k, sizeof(slong));
    nmod_poly_mat_t ker;
    nmod_poly_mat_init(ker, 2 * k, 2 * k, prime);
    nmod_poly_mat_pmbasis(ker, shift, B, sigma);

    slong nbrows = 0;
    for (slong i = 0; i < 2 * k; i++)
    {
        if (shift[i] <= target_degree)
        {
            if (nbrows == k)
                flint_throw(FLINT_ERROR, "nmod_algeqtodiffeq_Tb_left_description: more than "
                            "r-1 rows of degree at most %wd (sigma too small?)\n", target_degree);
            for (slong j = 0; j < k; j++)
            {
                nmod_poly_set(nmod_poly_mat_entry(Q, nbrows, j), nmod_poly_mat_entry(ker, i, j));
                nmod_poly_set(nmod_poly_mat_entry(P, nbrows, j), nmod_poly_mat_entry(ker, i, j + k));
            }
            nbrows++;
        }
    }
    if (nbrows < k)
        flint_throw(FLINT_ERROR, "nmod_algeqtodiffeq_Tb_left_description: no complete "
                    "description of degree at most %wd found (target_degree too small?)\n",
                    target_degree);

    flint_free(shift);
    nmod_poly_mat_clear(CT_phi1);
    nmod_poly_mat_clear(Tb);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}


/** Cockle's algorithm via the recursive algorithm on the quotient by the
 *  trace (pk_main.tex Sec. 6, "Modified recursive algorithm"). With
 *  K = [a, theta(a), ..., theta^m(a)], a = y, m = n-1: deleting the first
 *  row of K gives the pseudo-Krylov matrix K_b of theta_b = d/dx + T_b with
 *  seed a_b = e_0, and tau = (Tr y^i)_i has tau[0] = r, so K eta = 0 iff
 *  K_b eta = 0 and tau K eta = 0. Hence the kernel of
 *  [D_b a_b | N_b ; p_r^{m+1} tau K], with [theta_b(a_b) ...] = D_b^{-1}N_b
 *  from nmod_pseudo_Krylov_recursive in dimension r-1, and
 *  tau K = ((Tr y)^{(j)})_j, Tr y = -p_{r-1}/p_r. Same output convention as
 *  nmod_algeq_to_diffeq_recursive. Requires n >= 2, r >= 2, r != 0 in k.
 */
slong nmod_algeq_to_diffeq_recursive_trace(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong m = n - 1;
    slong k = r - 1;

    if (n < 2)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_recursive_trace: n must be >= 2\n");
    if (r < 2 || (ulong) r % prime == 0)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_recursive_trace: needs r >= 2 "
                    "and r invertible mod p\n");

    nmod_poly_t Delta;
    nmod_poly_mat_t iPyT, CT;
    nmod_algeqtodiffeq_setup(Delta, iPyT, CT, PT);

    /* same per-call seeding as nmod_algeq_to_diffeq_recursive */
    flint_rand_t state;
    flint_rand_init(state);
    srand((unsigned int) clock());
    flint_rand_set_seed(state, rand(), rand());

    nmod_poly_t phi1;
    nmod_poly_init(phi1, prime);
    nmod_phi1(phi1, CT, PT, Delta, state);

    nmod_poly_mat_t Q, P;
    nmod_poly_mat_init(Q, k, k, prime);
    nmod_poly_mat_init(P, k, k, prime);
    nmod_algeqtodiffeq_Tb_left_description(Q, P, phi1, CT, PT, Delta);

    nmod_poly_mat_t a, D, N, Qt, Pt;
    nmod_poly_mat_init(a, k, 1, prime);
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(a, 0, 0), 0, 1);
    nmod_poly_mat_init(D, k, k, prime);
    nmod_poly_mat_init(N, k, m, prime);
    nmod_poly_mat_init(Qt, k, k, prime);
    nmod_poly_mat_init(Pt, k, k, prime);
    nmod_pseudo_Krylov_recursive(D, N, Qt, Pt, Q, P, NULL, a, m);

    nmod_poly_mat_t v;
    nmod_poly_mat_init(v, k, 1, prime);
    nmod_poly_mat_mul(v, D, a);

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K, r, n, prime);
    for (slong i = 0; i < k; i++)
    {
        nmod_poly_set(nmod_poly_mat_entry(K, i, 0), nmod_poly_mat_entry(v, i, 0));
        for (slong j = 0; j < m; j++)
            nmod_poly_set(nmod_poly_mat_entry(K, i, j + 1), nmod_poly_mat_entry(N, i, j));
    }

    /* trace row: entry j = g_j p_r^{m-j}, where (Tr y)^{(j)} = g_j / p_r^{j+1},
     * g_0 = -p_{r-1}, g_{j+1} = g_j' p_r - (j+1) g_j p_r' */
    const nmod_poly_struct * pr = nmod_poly_mat_entry(PT, r, 0);
    nmod_poly_t g, dg, dpr, t, prpow;
    nmod_poly_init(g, prime);
    nmod_poly_init(dg, prime);
    nmod_poly_init(dpr, prime);
    nmod_poly_init(t, prime);
    nmod_poly_init(prpow, prime);
    nmod_poly_derivative(dpr, pr);
    nmod_poly_neg(g, nmod_poly_mat_entry(PT, r - 1, 0));
    for (slong j = 0; j <= m; j++)
    {
        nmod_poly_set(nmod_poly_mat_entry(K, k, j), g);
        nmod_poly_derivative(dg, g);
        nmod_poly_mul(dg, dg, pr);
        nmod_poly_mul(t, g, dpr);
        nmod_poly_scalar_mul_nmod(t, t, (ulong) (j + 1) % prime);
        nmod_poly_sub(g, dg, t);
    }
    nmod_poly_one(prpow);
    for (slong j = m; j >= 0; j--)
    {
        nmod_poly_mul(nmod_poly_mat_entry(K, k, j), nmod_poly_mat_entry(K, k, j), prpow);
        nmod_poly_mul(prpow, prpow, pr);
    }

    slong * pivind = flint_malloc(n * sizeof(slong));
    slong * shift = flint_calloc(n, sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, shift, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);
    flint_free(shift);

    nmod_poly_clear(g);
    nmod_poly_clear(dg);
    nmod_poly_clear(dpr);
    nmod_poly_clear(t);
    nmod_poly_clear(prpow);
    flint_rand_clear(state);
    nmod_poly_mat_clear(Q);
    nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(a);
    nmod_poly_mat_clear(D);
    nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(Qt);
    nmod_poly_mat_clear(Pt);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(K);
    nmod_poly_clear(phi1);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);

    return nz;
}
