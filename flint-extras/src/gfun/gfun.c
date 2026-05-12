/*
    Copyright (C) 2025 Gilles Villard, Vincent Neiger
    Copyright (C) 2026 Gilles Villard, Vincent Neiger

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/


#include <flint/fmpz_poly.h>
#include <flint/fmpq.h>
#include <flint/nmod_mpoly.h>

#include "nmod_extra.h" // for nmod_find_root
#include "nmod_poly_mat_extra.h"

#include "gfun.h"


/** 
 *    Assume that the degree r in the second variable, say y, is the row dimension - 1 of PT
 */

void mat_to_xy(nmod_mpoly_t P, nmod_mpoly_ctx_t ctx, const nmod_poly_mat_t PT)
{

    int i,j;

    slong r;
    r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    ulong exp[2];

    // Loop on y 
    for (i=0; i<r+1; i++)
    {
        exp[1]=i; // y
        for (j=0; j<d+1; j++)
        {
            exp[0]=j; // x 
            nmod_mpoly_set_coeff_ui_ui(P, nmod_poly_get_coeff_ui(nmod_poly_mat_entry(PT, i, 0),j), exp, ctx);
        }
    }
}


/**  Resultant of P and the derivative Py, and the inverse of Py mod P times the resultant 
 *     as poly_mat
 *    deg_y P = r, hence PT has r+1 rows 
 */

void nmod_biv_resultant_geometric(nmod_poly_t Delta, nmod_poly_mat_t  iPyT, const nmod_poly_mat_t PT)
{

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    nmod_t mod;
    nmod_geometric_progression_t F;

    ulong L;
    ulong w;

    slong r;
    r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    // Bound on the degree of the resultant 
    L = (2*r-1)*d+1;  
    nmod_init(&mod, prime);

    w = nmod_find_root(2*L, mod);
    nmod_geometric_progression_init(F, w, L, mod);


    int i,j;

    nn_ptr val[r+1];

    for (j=0; j<r+1; j++) 
    {
        val[j] = _nmod_vec_init(L);
    }

    /** 
     * Evaluation loop on the r+1 coeffs j in y
     *   generates a univariate polynomial evalP[i] in y 
     * 
     *  rather make a loop on the vals ? 
     */

    for (j=0; j<r+1; j++) 
    {
        _nmod_poly_evaluate_geometric_nmod_vec_fast_precomp(val[j], (nmod_poly_mat_entry(PT, j, 0))->coeffs,\
                                                            (nmod_poly_mat_entry(PT, j, 0))->length, F, L, mod);
    }

    // Reconstruction of the L polynomials in y 
    //   having segmentation fault with nmod_poly_t evalP[L]; (to see, ?)

    nmod_poly_mat_t  evalP;
    nmod_poly_mat_init(evalP,L,1,prime);

    for (i=0; i<L; i++) 
    {
        for (j=0; j<r+1; j++) 
        {
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(evalP, i, 0), j, val[j][i]);
        }
    }

    /**
     *  Loop on the L values for the L resultants and inverses 
     */

    nn_ptr evalR;
    evalR = _nmod_vec_init(L);

    // Evaluations of the inverse mod P times the resultant, polynomials i in y  

    nmod_poly_mat_t  evaliPy;
    nmod_poly_mat_init(evaliPy,L,1,prime);

    nmod_poly_t evalPy;
    nmod_poly_init(evalPy,prime);

    for (i=0; i<L; i++) 
    {
        nmod_poly_derivative(evalPy, nmod_poly_mat_entry(evalP, i, 0));

        evalR[i] = nmod_poly_resultant(nmod_poly_mat_entry(evalP, i, 0),evalPy);
        
        nmod_poly_invmod(nmod_poly_mat_entry(evaliPy, i, 0),\
                        evalPy,\
                        nmod_poly_mat_entry(evalP, i, 0)); // Todo together with the resultant

        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(evaliPy, i, 0),\
                                    nmod_poly_mat_entry(evaliPy, i, 0), evalR[i]);
    }


    // Interpolation: the resultant 

    nmod_poly_interpolate_geometric_nmod_vec_fast_precomp(Delta, evalR, F, L);


    // Interpolation: the inverse of Py times the resultant 

    nn_ptr tvals;
    tvals = _nmod_vec_init(L);

    for (j=0; j<r; j++)
    {
        // vals for the coeff i in x 
        for (i=0; i<L; i++) 
        {
            tvals[i] = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(evaliPy, i, 0),j); 
        }

        nmod_poly_interpolate_geometric_nmod_vec_fast_precomp(nmod_poly_mat_entry(iPyT, j, 0), tvals, F, L);
    }


}    




/** Geometric bivariate multiplication A*B mod P, with respect to y 
 *    the geometric progression is initialized outside
 * 
 *   The memainder is known - in advance - to be a polynomial of x-degree D
 *    the geometric progression is driven by D
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

    slong r = PT->r;
    slong ra = AT->r;
    slong rb = BT->r;


    slong rr = FLINT_MAX(r, FLINT_MAX(ra, rb));

    nn_ptr val[rr+1];

    for (j=0; j<rr+1; j++) 
    {
        val[j] = _nmod_vec_init(L);
    }

    /** 
     * Evaluations  
     *  rather make a loop on the vals ? 
     */

    // ========  P 
    for (j=0; j<r; j++) 
    {
        _nmod_poly_evaluate_geometric_nmod_vec_fast_precomp(val[j], (nmod_poly_mat_entry(PT, j, 0))->coeffs,\
                                                            (nmod_poly_mat_entry(PT, j, 0))->length, F, L, mod);
    }

    // Reconstruction of the L polynomials in y 
    nmod_poly_mat_t  evalP;
    nmod_poly_mat_init(evalP,L,1,prime);

    for (i=0; i<L; i++) 
    {
        for (j=0; j<r; j++) 
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

    for (j=0; j<r-1; j++)
    {
        // vals for the coeff i in x 
        for (i=0; i<L; i++) 
        {
            tvals[i] = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(evalR, i, 0),j); 
        }

        nmod_poly_interpolate_geometric_nmod_vec_fast_precomp(nmod_poly_mat_entry(RT, j, 0), tvals, F, L);
    }
}










