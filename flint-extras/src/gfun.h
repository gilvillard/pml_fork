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


/** 
 *    Assume that the degree r in the second variable, say y, is the row dimension - 1 of PT
 */

void mat_to_xy(nmod_mpoly_t P, nmod_mpoly_ctx_t ctx, const nmod_poly_mat_t PT);


/**  Resultant of P and the derivative Py, and the inverse of Py mod P times the resultant 
 *     as poly_mat
 *    deg_y P = r, hence PT has r+1 rows 
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
 */

void nmod_apply_T(nmod_poly_mat_t  RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const ulong D);


/**  Randomized Computation of phi_1 and phi_2
 *   -----------------------------------------
 * 
 *    using two random constant combinations of the column of T 
 * 
 *    r is assumed >= 3 for phi2 ? 
 * 
 */

void nmod_phi1(nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta);

void nmod_phi_T(nmod_poly_t  phi1, nmod_poly_t  phi2, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta);


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
 */ 

void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);

void nmod_pseudo_Krylov_phi1(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);


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

