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
    The "naive" pseudo-Krylov family (algos.pdf's own vocabulary via
    ../work-algeqtodiffeq/timings.md's "Naive pseudo Krylov" column):
    fraction-free, generic in the pseudo-linear map theta (nothing here
    assumes T has width <= 1, unlike the "new description" family) --
    build phi1*theta^k(a) directly, column by column, knowing the true
    denominators are powers of phi1 rather than the larger Delta.

    2026-09 cleanup: renamed from nmod_pseudo_Krylov_phi1 /
    nmod_algeq_to_diffeq_phi1 (see gfun.h for the old, now-superseded
    declarations and why they're still there), and rewritten to use the
    factored nmod_algeqtodiffeq_setup / nmod_algeqtodiffeq_rescale_by_phi1
    (algeqtodiffeq.c) instead of duplicating that preamble. The sibling
    without phi1 (nmod_pseudo_Krylov / nmod_algeq_to_diffeq, still in
    gfun.c) is not ported forward -- per the user, it's superseded by this
    one and can be forgotten.

    Algorithm and every numeric bound preserved exactly from the *_phi1
    originals -- this is a relocation-and-rename plus removal of dead code
    and debug instrumentation, not a re-derivation. See the "+200" comment
    below for one open question flagged by the user, deliberately not
    resolved here.
*/

/** Builds the r x n fraction-free pseudo-Krylov matrix K, whose column k
 *  (0-indexed) represents phi1^k * theta^k(a) for a = (0,1,0,...,0)^t
 *  (i.e. a = y in the monomial basis (1,y,...,y^{r-1})), given CT already
 *  rescaled to be phi1-scaled (nmod_algeqtodiffeq_rescale_by_phi1) -- so
 *  that nmod_apply_T(., ., CT, PT, D) computes phi1*T(.), not Delta*T(.).
 *
 *  Recurrence (product rule for theta = d/dx + T, tracking phi1^k * K_k
 *  rather than K_k itself to stay fraction-free): with K_k = phi1^k *
 *  theta^k(a),
 *    K_{k+1} = phi1 * K_k' - k*phi1' * K_k + phi1 * T(K_k)
 *  where the last term is exactly what nmod_apply_T(., K_k, CT, PT, D)
 *  returns given the phi1-scaled CT.
 *
 *  Does not use `Delta`: the *_phi1-suffixed original this was renamed
 *  from took it too, but only to compute an unused g = Delta/phi1 (dead
 *  code, presumably left over from adapting the non-phi1 sibling, which
 *  does use such a g at every iteration since its CT stays Delta-scaled)
 *  -- dropped here.
 */
void nmod_pseudo_Krylov_naive(nmod_poly_mat_t K, ulong n, const nmod_poly_mat_t CT,
                               const nmod_poly_mat_t PT, const nmod_poly_t phi1)
{
    /* n=0 would mean K has zero columns, but the base case below writes
     * unconditionally into column 0 -- found as a real, reproducible
     * segfault (2026-09-15, standalone repro, not guessed) before this
     * check existed. n=1..r don't crash (they just generically return no
     * relation, which is a legitimate answer, not an error) -- only n=0
     * is actually unsafe. */
    if (n == 0)
        flint_throw(FLINT_DOMERR, "nmod_pseudo_Krylov_naive: n must be >= 1 "
                    "(K needs at least one column, the base case a=y)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_mat_t tempN, temp;
    nmod_poly_mat_init(tempN, r, 1, prime);
    nmod_poly_mat_init(temp, r, 1, prime);

    nmod_poly_t p1, p2;
    nmod_poly_init(p1, prime);
    nmod_poly_init(p2, prime);

    nmod_poly_t dphi1;
    nmod_poly_init(dphi1, prime);
    nmod_poly_derivative(dphi1, phi1);

    slong deg_phi1 = nmod_poly_degree(phi1);

    /* K_0 = a = y, i.e. column (0,1,0,...,0)^t in the monomial basis */
    for (slong i = 0; i < r; i++)
        nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(K, 1, 0), 0, 1);

    for (slong k = 0; k < (slong) n - 1; k++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(tempN, i, 0), nmod_poly_mat_entry(K, i, k));

        /* D bounds the x-degree of phi1*T(K_k) (the derivative terms
         * below don't go through nmod_apply_T, so don't need to be
         * covered by D). k*deg(phi1) + deg(phi1) is the "properness"
         * bound one would derive assuming T is an exactly proper rational
         * map; NMOD_GFUN_NONPROPER_MARGIN on top covers inputs where T(x)
         * is not exactly proper -- see that constant's own doc (gfun.h)
         * for why it's a guess, not a guarantee, and what the real fix
         * looks like. Per the user (2026-09-15): every D bound in this
         * module is currently uncertain in this same way; this one is
         * merely named now, not re-derived. */
        slong D = k * deg_phi1 + deg_phi1 + NMOD_GFUN_NONPROPER_MARGIN;

        nmod_apply_T(temp, tempN, CT, PT, D);

        for (slong i = 0; i < r; i++)
        {
            nmod_poly_set(nmod_poly_mat_entry(K, i, k + 1), nmod_poly_mat_entry(temp, i, 0));

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
    nmod_poly_clear(dphi1);
    nmod_poly_clear(p1);
    nmod_poly_clear(p2);
}


/** Cockle's algorithm (G2026.pdf Sec. 7) via the naive pseudo-Krylov
 *  family: computes n columns of the pseudo-Krylov matrix for (theta, y)
 *  and a minimal kernel basis of the result. `n` is the actual pseudo-
 *  Krylov matrix width, not "extra columns beyond r" -- a relation is
 *  guaranteed to be found for n >= r+1 (the generic case: use n=r+1 for
 *  one solution), but one may already be found for smaller n on
 *  non-generic P (a lower-order relation); larger n beyond r+1 trades
 *  construction cost for a lower-degree solution (see algos.pdf Sec.
 *  3.5-style degree/order tradeoff -- claude-pseudoKrylov/todo.md item 6,
 *  not yet checked against this code). n must be >= 1 (see
 *  nmod_pseudo_Krylov_naive's own doc for why n=0 is invalid, not just
 *  degenerate).
 *
 *  Returns nz, the number of solutions found (0 if none); LT is an n x n
 *  polynomial matrix whose first nz columns are the solutions.
 */
slong nmod_algeq_to_diffeq_naive(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n)
{
    if (n < 1)
        flint_throw(FLINT_DOMERR, "nmod_algeq_to_diffeq_naive: n must be >= 1 "
                    "(n is the pseudo-Krylov matrix width itself, not extra "
                    "columns beyond r -- use n=r+1 for one solution)\n");

    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_t Delta;
    nmod_poly_mat_t iPyT, CT;
    nmod_algeqtodiffeq_setup(Delta, iPyT, CT, PT);

    /* TODO(cleanup): matches the rest of this module's not-yet-fixed
     * per-call reseeding, see claude-pseudoKrylov/todo.md item 8. */
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
    nmod_pseudo_Krylov_naive(K, n, CT, PT, phi1);

    /* Back to an honest polynomial matrix: K's column j currently
     * represents phi1^j * theta^j(a), so column j needs an extra
     * phi1^{(n-1)-j} to match column n-1's normalization before the
     * kernel computation below (which needs an ordinary, not
     * fraction-free-relative-to-varying-powers, polynomial matrix). */
    nmod_poly_t tpol;
    nmod_poly_init(tpol, prime);
    nmod_poly_one(tpol);
    for (slong j = n - 2; j >= 0; j--)
    {
        nmod_poly_mul(tpol, tpol, phi1);
        for (slong i = 0; i < r; i++)
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
    }

    /* NOTE: the *_phi1 original this was renamed from computed a `pivind`
     * via nmod_poly_mat_column_degree(pivind, K, shift) here, with a
     * zero-initialized `shift`, before the kernel call below -- but
     * nmod_poly_mat_kernel's own `pivind` is documented output-only (it
     * doesn't read the value passed in), so that precomputation's result
     * was always just overwritten, and `shift` itself was never used
     * (NULL is passed for shift to the kernel call, not that computed
     * array) -- both dropped here as dead code. Preserved as found: NULL
     * for shift (uniform shift), not the discarded computed one -- worth
     * asking whether that was deliberate or itself a leftover, not
     * decided here. */
    slong * pivind = flint_malloc(n * sizeof(slong));
    slong nz = nmod_poly_mat_kernel(LT, pivind, NULL, K, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);

    nmod_poly_mat_clear(K);
    nmod_poly_mat_clear(iPyT);
    nmod_poly_mat_clear(CT);
    nmod_poly_clear(Delta);
    nmod_poly_clear(phi1);
    nmod_poly_clear(tpol);

    return nz;
}
