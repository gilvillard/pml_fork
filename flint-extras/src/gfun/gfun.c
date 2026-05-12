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

#include <stdlib.h>
#include <time.h>

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

    // Bound on the degree of the resultant + 1
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

}


/**  Randomized Computation of phi_1 and phi_2
 *   -----------------------------------------
 * 
 *    Warning: not made monic 
 * 
 *    The row dimension of PT must be the y-degree of P, exactly 
 * 
 *    using two random constant combinations of the column of T 
 *    (r >= 2 for phi2) 
 * 
 *    To see: first colmun zero is a particular case ?
 * 
 */

void nmod_phi_T(nmod_poly_t  phi1, nmod_poly_t  phi2, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta)
{

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    /** Bound on the output degree 
     *   using constant random column projections  
     */

    slong D = (2*r-1)*d -1; // M^* and Y  (2r-2)d + (d-1)  

    flint_rand_t state;
    flint_rand_init(state);
    srand(time(NULL));
    flint_rand_set_seed(state, rand(), rand());


    nmod_poly_mat_t randT1, randT2;
    nmod_poly_mat_init(randT1,r,1,prime);
    nmod_poly_mat_init(randT2,r,1,prime);

    // Better than randtest matrix to be sure to have nonzero entries / Check nonzero ? 
    for (int i=0; i<r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randtest(state) % prime); 
    }

   
    nmod_poly_mat_t  colT1,colT2;
    nmod_poly_mat_init(colT1,r,1,prime);
    nmod_poly_mat_init(colT2,r,1,prime);


    nmod_apply_T(colT1, randT1, CT, PT, D); 
    nmod_apply_T(colT2, randT2, CT, PT, D); 

    nmod_poly_t g;
    nmod_poly_init(g,prime); // re-used below 

    nmod_poly_gcd_hgcd(g, nmod_poly_mat_entry(colT1, 0, 0), Delta);

    for (int i=1; i<r; i++)
    {
        nmod_poly_gcd_hgcd(g, g, nmod_poly_mat_entry(colT1, i, 0));
    }

    nmod_poly_div(phi1,Delta,g);


    /** Random row projections for phi2
     *   use matrices for potential generalizations 
     */    

    nmod_poly_mat_t randU;
    nmod_poly_mat_init(randU,2,r,prime);
    // Check nonzero ? 
    for (int i=0; i<r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 0, i), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 1, i), 0, n_randtest(state) % prime); 
    }

    nmod_poly_mat_t P1,P2;
    nmod_poly_mat_init(P1,2,1,prime);
    nmod_poly_mat_init(P2,2,1,prime);

    nmod_poly_mat_mul(P1,randU,colT1);
    nmod_poly_mat_mul(P2,randU,colT2);

    nmod_poly_t tp1,tp2;
    nmod_poly_init(tp1,prime); 
    nmod_poly_init(tp2,prime); 

    nmod_poly_mul(tp1,nmod_poly_mat_entry(P1, 0, 0),nmod_poly_mat_entry(P2, 1, 0));
    nmod_poly_mul(tp2,nmod_poly_mat_entry(P1, 1, 0),nmod_poly_mat_entry(P2, 0, 0));
    nmod_poly_sub(tp1,tp1,tp2);

    nmod_poly_div(tp1,tp1,Delta);
    nmod_poly_gcd_hgcd(g, tp1, Delta);
    nmod_poly_div(phi2,Delta,g);   
}




/**  Computation of the numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix 
 * 
 *    fraction-free approach 
 *   uses phi1 as computed previously, non monic since directly related to the resultant (non monic either) 
 */ 

// !!! g   pas K --> n 

void nmod_pseudo_Krylov(mod_poly_mat_t K, const ulong n, onst nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1)
{


    int i,k;

    slong r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);


    nmod_poly_mat_t  tempN; // for calling ffT 
    nmod_poly_mat_init(tempN,r,1,prime);

    nmod_poly_mat_t  temp; 
    nmod_poly_mat_init(temp,r,1,prime);


    nmod_poly_t p1,p2;
    nmod_poly_init(p1,prime);
    nmod_poly_init(p2,prime);

    nmod_poly_t dphi;
    nmod_poly_init(dphi,prime);
    nmod_poly_derivative(dphi,phi1);

    slong deg_phi;
    deg_phi=nmod_poly_degree(phi1);

    /** First column y 
     */
    for (i=0; i<r; i++)
    {
         nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    }
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(K, 1, 0), 0, 1);


    /** Main loop, for the n-1 new colmuns of K
     *  ---------------------------------------
     */

    slong D=0;

    for (int k=0; k<n-1; k++)
    {


        for (i=0; i<r; i++)
        {
            nmod_poly_set(nmod_poly_mat_entry(tempN, i, 0),nmod_poly_mat_entry(K, i, k));
        }


        /** TO CHECK
         *  ********
         * 
         *  we add the degree (2r-2)d + (d-1) = (2r-1)d -1 for M^* and Y  (temporarily) 
         *    when we apply T, thus (2r-1)d is ok and bounds the x-degree of the resultant 
         *  and then, afterwards, we will recover the degree of phi1 
         * 
         */

        D = k*deg_phi + (2*r-1)*d -1; 

        nmod_apply_T(temp, tempN, CT, PT, D);

    

     for (i=0; i<r; i++)
     {

        // flint_printf("\n temp i: %d\n",i);
        // nmod_poly_print_pretty(nmod_poly_mat_entry(temp, i, 0),"x");
        // flint_printf("\n ");


        nmod_poly_div(nmod_poly_mat_entry(NA, i, k+1),nmod_poly_mat_entry(temp, i, 0),g);

        // flint_printf("\n i: %d\n",i);
        // nmod_poly_print_pretty(nmod_poly_mat_entry(NA, i, k+1),"x");
        // flint_printf("\n ");

        nmod_poly_derivative(p1,nmod_poly_mat_entry(NA, i, k));
        nmod_poly_mul(p1,p1,phi);


        nmod_poly_mul(p2,dphi,nmod_poly_mat_entry(NA, i, k));
        nmod_poly_scalar_mul_nmod(p2,p2,k);

        nmod_poly_add(nmod_poly_mat_entry(NA, i, k+1),nmod_poly_mat_entry(NA, i, k+1),p1);

        nmod_poly_sub(nmod_poly_mat_entry(NA, i, k+1),nmod_poly_mat_entry(NA, i, k+1),p2);
    }

    t += (double)(clock()-tt) / CLOCKS_PER_SEC;

    flint_printf("\n Loop %d   %.3f sec.\n", k,t);


    }      


    t=0.0;
    tt=clock();

    for (i=0; i<r; i++)
    {
        flint_printf("\n %ld ", nmod_poly_degree(nmod_poly_mat_entry(NA, i, K-1)));
    } 









