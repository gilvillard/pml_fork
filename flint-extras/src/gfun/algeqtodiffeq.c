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

#include "nmod_extra.h" // for nmod_find_root
#include "nmod_poly_mat_extra.h"

#include "gfun.h"

/*
    File organization note (2026-09, "for the moment" -- may change):
    grouped here, in one file named after the problem they jointly serve, are
    the pieces that are specific to algeqtodiffeq's T -- the k(x)-linear map
    a -> -Py(x,a)^{-1} Px(x,a) d/dy(a) mod P (Cockle's algorithm, G2026.pdf
    Sec. 7) -- as opposed to nmod_biv_resultant_geometric (resultant.c),
    which computes something self-contained and meaningful on its own (not
    algeqtodiffeq-specific in what it computes, only in how it happens to be
    used here), and so was given its own file instead.

    nmod_biv_mulmod_geometric's own signature is generic (bivariate A*B mod
    P), and nothing about its *implementation* is algeqtodiffeq-specific --
    but every current caller of it is (nmod_apply_T, directly; nmod_phi1,
    through nmod_apply_T), so for now it lives alongside them rather than in
    its own file. If it ever grows a caller outside this problem, it should
    move out (mirroring why nmod_biv_resultant_geometric moved out to
    resultant.c) rather than pulling an unrelated caller in here.

    Naming convention (2026-09-15, cleaned up after the user caught the
    inconsistency): throughout this file (and resultant.c), a bare `r`
    always means deg_y(P) = PT->r - 1, matching both papers' own notation
    -- never a row count. Where a row count is genuinely what's needed
    (nmod_biv_mulmod_geometric, which cares about A/B/P's own coefficient
    counts, not "the degree" as a concept), name it distinctly, e.g.
    `rP`/`ra`/`rb` -- never a bare `r`. This file's own
    nmod_biv_mulmod_geometric used to violate this (a bare `r` meant
    PT->r, not PT->r - 1, while every other function here already used the
    `r = deg_y(P)` convention) -- fixed; keep the convention when adding
    more functions to this file or a sibling algeqtodiffeq_<family>.c.
*/


/** Geometric bivariate multiplication A*B mod P, with respect to y
 *    the geometric progression is initialized outside
 *
 *   The memainder is known - in advance - to be a polynomial of x-degree at most D
 *    the geometric progression is driven by D
 *
 *   To see: aliasing?
 */

void nmod_biv_mulmod_geometric(nmod_poly_mat_t  RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t BT, \
                                const nmod_poly_mat_t PT,  const ulong D)
{

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    nmod_t mod;
    nmod_geometric_progression_t F;

    ulong L;
    ulong w;

    L = D+1;
    nmod_init(&mod, prime);

    w = nmod_find_root(2*L, mod);
    nmod_geometric_progression_init(F, w, L, mod);

    int i,j;

    /* rP/ra/rb are row COUNTS (of PT/AT/BT), not degrees -- unlike every
     * other function in this module, where a bare `r` always means
     * deg_y(P) = PT->r - 1. Kept distinctly named (rather than a bare `r`
     * for rP) specifically to avoid that collision; see this file's own
     * header comment for the convention. */
    slong rP = PT->r;
    slong ra = AT->r;
    slong rb = BT->r;


    slong rmax = FLINT_MAX(rP, FLINT_MAX(ra, rb));

    nn_ptr val[rmax+1];

    for (j=0; j<rmax+1; j++)
    {
        val[j] = _nmod_vec_init(L);
    }

    /**
     * Evaluations
     *  rather make a loop on the vals ?
     */

    // ========  P
    for (j=0; j<rP; j++)
    {
        _nmod_poly_evaluate_geometric_nmod_vec_fast_precomp(val[j], (nmod_poly_mat_entry(PT, j, 0))->coeffs,\
                                                            (nmod_poly_mat_entry(PT, j, 0))->length, F, L, mod);
    }

    // Reconstruction of the L polynomials in y
    nmod_poly_mat_t  evalP;
    nmod_poly_mat_init(evalP,L,1,prime);

    for (i=0; i<L; i++)
    {
        for (j=0; j<rP; j++)
        {
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(evalP, i, 0), j, val[j][i]);
        }
    }


    // ========  A
    for (j=0; j<ra; j++)
    {
        _nmod_poly_evaluate_geometric_nmod_vec_fast_precomp(val[j], (nmod_poly_mat_entry(AT, j, 0))->coeffs,\
                                                            (nmod_poly_mat_entry(AT, j, 0))->length, F, L, mod);
    }

    // Reconstruction of the L polynomials in y
    nmod_poly_mat_t  evalA;
    nmod_poly_mat_init(evalA,L,1,prime);

    for (i=0; i<L; i++)
    {
        for (j=0; j<ra; j++)
        {
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(evalA, i, 0), j, val[j][i]);
        }
    }

    // ========  B
    for (j=0; j<rb; j++)
    {
        _nmod_poly_evaluate_geometric_nmod_vec_fast_precomp(val[j], (nmod_poly_mat_entry(BT, j, 0))->coeffs,\
                                                            (nmod_poly_mat_entry(BT, j, 0))->length, F, L, mod);
    }

    // Reconstruction of the L polynomials in y
    nmod_poly_mat_t  evalB;
    nmod_poly_mat_init(evalB,L,1,prime);

    for (i=0; i<L; i++)
    {
        for (j=0; j<rb; j++)
        {
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(evalB, i, 0), j, val[j][i]);
        }
    }


    /**
     *  Loop on the L values for the L resulting polynomials
     */

    // Evaluations of the product modulo

    nmod_poly_mat_t  evalR;
    nmod_poly_mat_init(evalR,L,1,prime);


    for (i=0; i<L; i++)
    {

        nmod_poly_mulmod(nmod_poly_mat_entry(evalR, i, 0), \
                            nmod_poly_mat_entry(evalA, i, 0), \
                            nmod_poly_mat_entry(evalB, i, 0), \
                            nmod_poly_mat_entry(evalP, i, 0));
    }


    nn_ptr tvals;
    tvals = _nmod_vec_init(L);

    for (j=0; j<rP-1; j++)
    {
        // vals for the coeff i in x
        for (i=0; i<L; i++)
        {
            tvals[i] = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(evalR, i, 0),j);
        }

        nmod_poly_interpolate_geometric_nmod_vec_fast_precomp(nmod_poly_mat_entry(RT, j, 0), tvals, F, L);
    }


    for (j=0; j<rmax+1; j++)
    {
         _nmod_vec_clear(val[j]);
    }
    _nmod_vec_clear(tvals);
    nmod_geometric_progression_clear(F);
    nmod_poly_mat_clear(evalA);
    nmod_poly_mat_clear(evalB);
    nmod_poly_mat_clear(evalP);
    nmod_poly_mat_clear(evalR);
}



/** Linear transformation T for algeqtodiffeq
 *   CT is C = -Px (Py)^(-1) that has been precomputed
 *
 *   The result is known - in advance - to be a polynomial of x-degree at most D
 *    the geometric progression is driven by D
 *
 *   To see: aliasing?
 *
 */

void nmod_apply_T(nmod_poly_mat_t  RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const ulong D)
{
    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong ra = AT->r;

    nmod_poly_mat_t  DAT;
    nmod_poly_mat_init(DAT,ra,1,prime);

     // Diff A
    for (int i=0; i<ra-1; i++)
    {
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(DAT, i, 0),nmod_poly_mat_entry(AT, i+1, 0),i+1);
    }
    nmod_poly_zero(nmod_poly_mat_entry(DAT, ra-1, 0));

    nmod_biv_mulmod_geometric(RT, DAT, CT, PT, D);

    nmod_poly_mat_clear(DAT);
}


/** Randomized computation of phi1 = phi1(T), the true (monic-free) common
 *  denominator of T's entries, for the algeqtodiffeq setting: T is the
 *  k(x)-linear map on A = k(x)[y]_{<r} above, and Delta (generically the
 *  resultant of P, Py) is an a priori, possibly larger, denominator with
 *  phi1 | Delta.
 *
 *  Inputs:
 *   - CT: the precomputed C = -Px * (Delta * Py^{-1} mod P), such that
 *     nmod_apply_T(., a, CT, PT, D) returns Delta*T(a) as a polynomial for
 *     any a and any D at least the bound below (see nmod_apply_T);
 *   - PT: P as an (r+1) x 1 matrix of x-polynomial y-coefficients;
 *   - Delta: the a priori denominator.
 *  Output: phi1, into an nmod_poly_t already init'd by the caller with the
 *  same modulus as PT. Not made monic (matches the rest of gfun.c).
 *
 *  Method (one random projection): writing B = Delta*T, phi1 is by
 *  definition Delta / gcd(B, Delta) (the gcd of Delta together with every
 *  entry of B). Rather than build all of B, this computes it from a single
 *  random constant vector z: phi1 = Delta / gcd(Delta*T(z), Delta).
 *
 *  Why one projection suffices -- and exactly where this relies on T having
 *  width <= 1 (algos.pdf, Sec. 3.1-3.2): when width(T) <= 1,
 *  B = u v^T mod Delta for some u, v (Lemma 3.3 of algos.pdf), so *every*
 *  column of B is already a scalar multiple of u mod Delta. Then for random
 *  z, gcd(Bz, Delta) = gcd((v^T z) u, Delta) = gcd(u, Delta) = gcd(B, Delta)
 *  generically in z (Lemma 3.4). This argument uses the rank-one structure
 *  of B mod Delta; it does not extend to width > 1, where a single random
 *  column no longer certifies the whole matrix's gcd with Delta. A
 *  general-T version (dropping the width-1 assumption) is future work, not
 *  attempted here -- see claude-pseudoKrylov/todo.md.
 *
 *  Monte Carlo, no verification: for an unlucky z the returned phi1 would
 *  be a proper divisor of the true one. The failure probability is bounded
 *  by a degree/field-size argument (algos.pdf Lemma 3.4's
 *  Phi(z) = Res(v^T z, Delta) construction) but is not computed or checked
 *  here; no retry is attempted. Separately -- and not a hazard this
 *  function's own logic can fix -- nmod_apply_T above depends on
 *  nmod_biv_mulmod_geometric's own geometric evaluation/interpolation,
 *  which has an unguarded "bad point" hazard of its own; see
 *  claude-pseudoKrylov/todo.md item 11.
 *
 *  `state` must be initialized by the caller (standard FLINT/PML
 *  convention). This function does not reseed it -- draws are exactly the
 *  next values `state` produces. Earlier versions of this routine reseeded
 *  a fresh, function-local flint_rand_t from srand(time(NULL)) on every
 *  call; since time() has 1-second resolution, two calls to *any* such
 *  function (this one, or the sibling nmod_phi_T / find_uv, which use the
 *  same pattern) within the same wall-clock second drew identical or
 *  correlated randomness -- silently threatening the genericity this
 *  routine's correctness relies on. Taking `state` as a parameter removes
 *  that hazard and matches how randomness is threaded everywhere else in
 *  FLINT/PML (e.g. nmod_poly_randtest).
 */
void nmod_phi1(nmod_poly_t phi1, const nmod_poly_mat_t CT,
               const nmod_poly_mat_t PT, const nmod_poly_t Delta,
               flint_rand_t state)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);
    slong D = nmod_gfun_delta_T_degree_bound(r, d);

    nmod_poly_mat_t randz, Bz;
    nmod_poly_mat_init(randz, r, 1, prime);
    nmod_poly_mat_init(Bz, r, 1, prime);

    /* Random constant projection z. n_randbits over (near-)the full machine
     * word, rather than e.g. nmod_poly_randtest, so that an all-zero draw
     * (which would make z the zero vector, an unusable projection) is
     * astronomically unlikely rather than a realistic corner case -- not a
     * proof of nonzero-ness, matches the draft's own informal reasoning. */
    for (slong i = 0; i < r; i++)
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randz, i, 0), 0,
                                n_randbits(state, FLINT_BITS - 2));

    /* Bz = Delta * T(z), fraction-free (see nmod_apply_T). */
    nmod_apply_T(Bz, randz, CT, PT, D);

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_gcd_hgcd(g, nmod_poly_mat_entry(Bz, 0, 0), Delta);
    for (slong i = 1; i < r; i++)
        nmod_poly_gcd_hgcd(g, g, nmod_poly_mat_entry(Bz, i, 0));

    nmod_poly_div(phi1, Delta, g);

    nmod_poly_clear(g);
    nmod_poly_mat_clear(randz);
    nmod_poly_mat_clear(Bz);
}


/** Common first-stage setup for every algeqtodiffeq driver (2026-09,
 *  factored out at the user's request once it was confirmed identical
 *  across drivers read so far -- naive, and by inspection also
 *  description/series/last, though those haven't had their own cleanup
 *  pass yet to confirm they end up calling this exact function).
 *
 *  From P alone: builds Delta = resultant(P,Py) and iPyT (via
 *  nmod_biv_resultant_geometric), and CT = the Delta-scaled rational part
 *  of T (via nmod_biv_mulmod_geometric on PxT = -Px and iPyT).
 *
 *  Does not return r = deg_y(P) or d = deg_x(P) (an earlier version of
 *  this function did, via output parameters -- dropped 2026-09-15): both
 *  are one-line, O(1)-ish reads directly off PT (`(PT->r)-1` and
 *  nmod_poly_mat_degree(PT)), so every caller/callee that needs them
 *  already gets them that way anyway (e.g. nmod_pseudo_Krylov_naive
 *  recomputes r itself rather than taking it as a parameter) -- there was
 *  no actual saving in threading them through here, only an unused `d`
 *  output in the one caller that existed at the time.
 *
 *  Unlike most functions in this file, Delta/iPyT/CT are initialized
 *  *inside* this function (not by the caller) -- deliberately, since the
 *  whole point is to save the caller the boilerplate of computing r just
 *  to size iPyT/CT before it can call anything (that part of the
 *  motivation still applies to Delta/iPyT/CT specifically, which really
 *  do need a real computation, unlike r/d). The caller still owns them
 *  and must nmod_poly_clear(Delta)/nmod_poly_mat_clear(iPyT)/(CT)
 *  afterward.
 */
void nmod_algeqtodiffeq_setup(nmod_poly_t Delta, nmod_poly_mat_t iPyT,
                               nmod_poly_mat_t CT, const nmod_poly_mat_t PT)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);

    nmod_poly_init(Delta, prime);
    nmod_poly_mat_init(iPyT, r, 1, prime);
    nmod_biv_resultant_geometric(Delta, iPyT, PT);

    nmod_poly_mat_t PxT;
    nmod_poly_mat_init(PxT, r + 1, 1, prime);
    for (slong i = 0; i <= r; i++)
    {
        /* PxT := -Px (the sign is folded in here, once, rather than at
         * every caller) */
        nmod_poly_derivative(nmod_poly_mat_entry(PxT, i, 0), nmod_poly_mat_entry(PT, i, 0));
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(PxT, i, 0), nmod_poly_mat_entry(PxT, i, 0), prime - 1);
    }

    slong D = nmod_gfun_delta_T_degree_bound(r, d);

    nmod_poly_mat_init(CT, r, 1, prime);
    nmod_biv_mulmod_geometric(CT, PxT, iPyT, PT, D);

    nmod_poly_mat_clear(PxT);
}


/** Rescales CT in place given an ALREADY KNOWN phi1: CT := CT / (Delta/phi1),
 *  exact because T has width <= 1 (same argument as nmod_phi1 itself --
 *  every entry of Delta*T, not just the single random column nmod_phi1
 *  inspects, shares the same Delta/phi1 factor). After this call,
 *  nmod_apply_T(., a, CT, PT, D) computes phi1*T(a), not Delta*T(a) --
 *  callers must size D accordingly (smaller than
 *  nmod_gfun_delta_T_degree_bound would give, since phi1 can be a proper
 *  divisor of Delta).
 *
 *  Split out (2026-09-15) from nmod_algeqtodiffeq_rescale_by_phi1 below,
 *  which computes phi1 itself via nmod_phi1 -- for a caller that already
 *  has phi1 in hand (e.g. find_uv, algeqtodiffeq_width1.c, which receives
 *  it as a parameter rather than computing it), calling the full
 *  rescale_by_phi1 would redundantly recompute it via another random
 *  projection.
 */
void nmod_algeqtodiffeq_rescale_CT_by_phi1(nmod_poly_mat_t CT,
                                            const nmod_poly_mat_t PT,
                                            const nmod_poly_t Delta,
                                            const nmod_poly_t phi1)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_div(g, Delta, phi1);

    for (slong i = 0; i < r; i++)
        nmod_poly_div(nmod_poly_mat_entry(CT, i, 0), nmod_poly_mat_entry(CT, i, 0), g);

    nmod_poly_clear(g);
}


/** Second-stage setup, for drivers that want a phi1-scaled CT rather than
 *  a Delta-scaled one (2026-09, factored out alongside
 *  nmod_algeqtodiffeq_setup -- see that function's doc). Computes phi1
 *  (nmod_phi1) and rescales CT in place via
 *  nmod_algeqtodiffeq_rescale_CT_by_phi1 above -- see that function's doc
 *  for the rescale itself.
 *
 *  `phi1` must be nmod_poly_init'd by the caller (matches nmod_phi1's own
 *  convention) -- unlike nmod_algeqtodiffeq_setup's outputs, phi1 is
 *  typically still needed by the caller afterward (e.g. to rescale a
 *  fraction-free construction's output back to an honest polynomial
 *  matrix), not just setup scratch to be cleared.
 */
void nmod_algeqtodiffeq_rescale_by_phi1(nmod_poly_t phi1, nmod_poly_mat_t CT,
                                         const nmod_poly_mat_t PT,
                                         const nmod_poly_t Delta,
                                         flint_rand_t state)
{
    nmod_phi1(phi1, CT, PT, Delta, state);
    nmod_algeqtodiffeq_rescale_CT_by_phi1(CT, PT, Delta, phi1);
}


/** Randomized computation of phi1 AND phi2 (algos.pdf Sec. 3.1's
 *  determinantal-denominator sequence phi_1 | phi_2 | ...): phi1 exactly
 *  as nmod_phi1 (one random column of Delta*T, gcd with Delta); phi2 the
 *  same idea one determinantal level up -- a random 2x2 "minor" of
 *  Delta*T, sampled via two random columns (two applications of
 *  nmod_apply_T) and two random row combinations, gcd'd with Delta the
 *  same way. Moved here 2026-09-15, next to nmod_phi1 (its natural
 *  sibling), takes an explicit flint_rand_t for the same reason -- not
 *  otherwise cleaned up (the algorithm/structure is unchanged from the
 *  draft; only nmod_phi1 itself has had a full pass so far).
 *
 *  Why this checks width(T) <= 1: phi_1 | phi_2 always (Sec. 3.1), with
 *  equality iff width(T) <= 1 (the width is the first index where the
 *  determinantal-denominator sequence stops growing). So `deg(phi1) ==
 *  deg(phi2)` (comparing degrees, as nmod_pseudo_Krylov_width1
 *  (algeqtodiffeq_width1.c) does with this function's output) is a
 *  direct, meaningful width-1 check -- confirmed empirically 2026-09-15
 *  while testing find_uv: for a P where this comparison disagrees,
 *  find_uv's rank-one reconstruction was verified (via a direct
 *  2x2-minors check on an independently built reference matrix) to fail
 *  100% of the time, deterministically, exactly as expected for a genuine
 *  width > 1 instance -- not a bug in find_uv, a violated precondition.
 *  Per the user: width <= 1 is only a generic property of algeqtodiffeq's
 *  T, never guaranteed for a given P; nmod_pseudo_Krylov_width1
 *  flint_throw's on failure (2026-09-16, per the user's decision) --
 *  the old draft (nmod_algeq_to_diffeq_new, superseded) only printed a
 *  message and continued regardless. See claude-pseudoKrylov/todo.md.
 *
 *  r is assumed >= 3 for phi2 to be meaningful (a 2x2 minor needs at
 *  least 2 independent row/column directions distinct from whatever phi1
 *  already used) -- not checked here, matches the draft.
 */
void nmod_phi_T(nmod_poly_t phi1, nmod_poly_t phi2, const nmod_poly_mat_t CT,
                const nmod_poly_mat_t PT, const nmod_poly_t Delta,
                flint_rand_t state)
{
    ulong prime = nmod_poly_mat_modulus(PT);
    slong r = (PT->r) - 1;
    slong d = nmod_poly_mat_degree(PT);
    slong D = nmod_gfun_delta_T_degree_bound(r, d);

    nmod_poly_mat_t randT1, randT2;
    nmod_poly_mat_init(randT1, r, 1, prime);
    nmod_poly_mat_init(randT2, r, 1, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
    }

    nmod_poly_mat_t colT1, colT2;
    nmod_poly_mat_init(colT1, r, 1, prime);
    nmod_poly_mat_init(colT2, r, 1, prime);
    nmod_apply_T(colT1, randT1, CT, PT, D);
    nmod_apply_T(colT2, randT2, CT, PT, D);

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_gcd_hgcd(g, nmod_poly_mat_entry(colT1, 0, 0), Delta);
    for (slong i = 1; i < r; i++)
        nmod_poly_gcd_hgcd(g, g, nmod_poly_mat_entry(colT1, i, 0));
    nmod_poly_div(phi1, Delta, g);

    /* phi2: a random 2x2 minor of Delta*T, via two random row
     * combinations (randU) applied to the same two random columns above. */
    nmod_poly_mat_t randU;
    nmod_poly_mat_init(randU, 2, r, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 0, i), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 1, i), 0, n_randtest(state) % prime);
    }

    nmod_poly_mat_t P1, P2;
    nmod_poly_mat_init(P1, 2, 1, prime);
    nmod_poly_mat_init(P2, 2, 1, prime);
    nmod_poly_mat_mul(P1, randU, colT1);
    nmod_poly_mat_mul(P2, randU, colT2);

    nmod_poly_t tp1, tp2;
    nmod_poly_init(tp1, prime);
    nmod_poly_init(tp2, prime);
    nmod_poly_mul(tp1, nmod_poly_mat_entry(P1, 0, 0), nmod_poly_mat_entry(P2, 1, 0));
    nmod_poly_mul(tp2, nmod_poly_mat_entry(P1, 1, 0), nmod_poly_mat_entry(P2, 0, 0));
    nmod_poly_sub(tp1, tp1, tp2);

    nmod_poly_div(tp1, tp1, Delta);
    nmod_poly_gcd_hgcd(g, tp1, Delta);
    nmod_poly_div(phi2, Delta, g);

    nmod_poly_clear(g);
    nmod_poly_clear(tp1);
    nmod_poly_clear(tp2);
    nmod_poly_mat_clear(randT1);
    nmod_poly_mat_clear(randT2);
    nmod_poly_mat_clear(colT1);
    nmod_poly_mat_clear(colT2);
    nmod_poly_mat_clear(randU);
    nmod_poly_mat_clear(P1);
    nmod_poly_mat_clear(P2);
}
