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
 *    T; see algeqtodiffeq.c for the full derivation, the width-1 dependency,
 *    and the caller-supplied `state` convention (nmod_phi1 was cleaned up on
 *    2026-09; nmod_phi_T -- which also computes phi2 -- has not been yet,
 *    and keeps its own internal, unseeded flint_rand_t for now).
 *
 *    r is assumed >= 3 for phi2 (nmod_phi_T only) ?
 *
 */

void nmod_phi1(nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta, \
                     flint_rand_t state);

void nmod_phi_T(nmod_poly_t  phi1, nmod_poly_t  phi2, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta);


/** Common algeqtodiffeq driver setup, factored out 2026-09 (see
 *  algeqtodiffeq.c for the full doc on both): nmod_algeqtodiffeq_setup
 *  builds Delta/iPyT/CT (Delta-scaled) from P alone;
 *  nmod_algeqtodiffeq_rescale_by_phi1 optionally rescales CT to be
 *  phi1-scaled instead, for drivers that want that (computes phi1 as a
 *  side effect, since a caller wanting the rescale needs phi1 anyway).
 */

void nmod_algeqtodiffeq_setup(nmod_poly_t Delta, nmod_poly_mat_t iPyT,
                               nmod_poly_mat_t CT, const nmod_poly_mat_t PT);

void nmod_algeqtodiffeq_rescale_by_phi1(nmod_poly_t phi1, nmod_poly_mat_t CT,
                                         const nmod_poly_mat_t PT,
                                         const nmod_poly_t Delta,
                                         flint_rand_t state);


void find_uv(nmod_poly_mat_t U, nmod_poly_mat_t V, const nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta);

/**  Includes simplication to have phi1 at denominator 
 */

void Description_From_Rank_1(nmod_poly_mat_t NN, nmod_poly_mat_t DD, const ulong n,\
                             const nmod_poly_mat_t U, const nmod_poly_mat_t V,\
                             const nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                             const nmod_poly_mat_t PT, \
                             const nmod_poly_t beta, const nmod_poly_mat_t iN, const nmod_poly_t Delta);


/**  Computation of the numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix
 *
 *    fraction-free approach
 *   uses phi1 as computed previously, non monic since directly related to the resultant (non monic either)
 *
 *    nmod_pseudo_Krylov (this one, without a CT already rescaled by phi1)
 *    is superseded -- see claude-pseudoKrylov/todo.md, "the naive family":
 *    nmod_pseudo_Krylov_naive (algeqtodiffeq_naive.c) is the renamed,
 *    cleaned-up former nmod_pseudo_Krylov_phi1, meant to replace this one
 *    and nmod_pseudo_Krylov_phi1 both. This declaration and
 *    nmod_pseudo_Krylov_phi1's stay only because gfun.c's own
 *    nmod_algeq_to_diffeq (non-phi1 driver) still calls this one and
 *    hasn't been touched.
 */

void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);

void nmod_pseudo_Krylov_phi1(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);


/** The naive pseudo-Krylov family (algeqtodiffeq_naive.c): renamed,
 *  cleaned-up versions of nmod_pseudo_Krylov_phi1 /
 *  nmod_algeq_to_diffeq_phi1 above -- the *_phi1 names are being retired
 *  (see claude-pseudoKrylov/todo.md). Uses nmod_algeqtodiffeq_setup +
 *  nmod_algeqtodiffeq_rescale_by_phi1 instead of duplicating that
 *  preamble; nmod_pseudo_Krylov_naive drops the `Delta` parameter (dead in
 *  the *_phi1 original -- computed into an unused `g` and never
 *  referenced, since CT is already phi1-scaled by the time this is
 *  called).
 */

void nmod_pseudo_Krylov_naive(nmod_poly_mat_t K, ulong n, const nmod_poly_mat_t CT,
                               const nmod_poly_mat_t PT, const nmod_poly_t phi1);

slong nmod_algeq_to_diffeq_naive(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/**  Computation of the appropriate matrix for kernel solution 
 * 
 *    i.e. numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix 
 *     with columns multiplied by an appropriate multiple of Delta (not phi1) 
 *     for the moment  
 *   
 */ 

void nmod_pseudo_Krylov_for_kernel(nmod_poly_mat_t LT, const ulong n, const nmod_poly_mat_t PT);



/**  algeqtodiffeq
 *
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 *
 *    nmod_algeq_to_diffeq_phi1 is superseded by nmod_algeq_to_diffeq_naive
 *    above (algeqtodiffeq_naive.c) -- kept only because nothing has been
 *    repointed at the new name yet. nmod_algeq_to_diffeq (non-phi1) stays
 *    untouched, not ported forward (see claude-pseudoKrylov/todo.md).
 *
 */

slong nmod_algeq_to_diffeq(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

slong nmod_algeq_to_diffeq_phi1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


/**  algeqtodiffeq series and descirption 
 * 
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 *    
 */ 

slong nmod_algeq_to_diffeq_series(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);

slong nmod_algeq_to_diffeq_series_phi1(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


slong nmod_algeq_to_diffeq_new(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong n);


void iterative_pseudo_krylov(nmod_poly_mat_t N, const nmod_poly_mat_t iP, const nmod_poly_mat_t iQ,\
                                 const nmod_poly_mat_t a, const slong n);

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

