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

void nmod_phi_T(nmod_poly_t  phi1, nmod_poly_t  phi2, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta);


/**  Computation of the numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix 
 * 
 *    fraction-free approach 
 *   uses phi1 as computed previously, non monic since directly related to the resultant (non monic either) 
 */ 

void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta);


/**  algeqtodiffeq 
 * 
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 *    
 */ 

slong nmod_algeq_to_diffeq(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong k);



#endif // GFUN_H
