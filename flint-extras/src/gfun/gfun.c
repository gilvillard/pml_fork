/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <flint/fmpz_poly.h>
#include <flint/fmpq.h>
#include <flint/nmod_mpoly.h>
#include <flint/fmpz_poly_mat.h>

#include "nmod_extra.h" // for nmod_find_root
#include "nmod_poly_mat_extra.h"

#include "nmod_poly_mat_description.h"

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
        //nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randbits(state,FLINT_BITS-2));

        //nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randtest(state) % prime); 
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randbits(state,FLINT_BITS-2));

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



/**  Includes simplication to have phi1 at denominator 
 */

void find_uv(nmod_poly_mat_t U, nmod_poly_mat_t V, const nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                     const nmod_poly_mat_t PT, const nmod_poly_t Delta)
{

    int i,j;

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    /** Bound on the output degree 
     *   using constant random column projections  
     */

    slong D = (2*r-1)*d -1; // M^* and Px  (2r-2)d + (d-1)  

    flint_rand_t state;
    flint_rand_init(state);
    srand(time(NULL));
    flint_rand_set_seed(state, rand(), rand());


    nmod_poly_mat_t Z, W;
    nmod_poly_mat_init(Z,r,1,prime);
    nmod_poly_mat_init(W,1,r,prime);

    // Better than randtest matrix to be sure to have nonzero entries / Check nonzero ? 
    for (i=0; i<r; i++)
    {
        //nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Z, i, 0), 0, n_randbits(state,FLINT_BITS-2));

        //nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randtest(state) % prime); 
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(W, 0, i), 0, n_randbits(state,FLINT_BITS-2));

    }


    nmod_poly_t g;
    nmod_poly_init(g,prime);
    nmod_poly_div(g,Delta,phi1);


    /** Computation of T 
     *  ----------------
     *     could be done via applyT for U 
     *      but the transpose for V ? 
     */

    nmod_poly_mat_t Yk, T, temp;

    nmod_poly_mat_init(Yk,r,1,prime);
    nmod_poly_mat_init(temp,r,1,prime);

    nmod_poly_mat_init(T,r,r,prime);


    for (i=0; i<r; i++)
    {
        nmod_poly_zero(nmod_poly_mat_entry(T, i, 0)); 
    }

    for (j=1; j<r; j++)
    {
        for (i=0; i<r; i++)
        {
            nmod_poly_zero(nmod_poly_mat_entry(Yk, i, 0));
        }
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Yk, j, 0), 0, 1);

        nmod_apply_T(temp, Yk, CT, PT, D); 

        for (i=0; i<r; i++)
        {
            nmod_poly_div(nmod_poly_mat_entry(T, i, j), nmod_poly_mat_entry(temp, i, 0), g);
        }
    }

    /**  Computation of U = T.Z
     *   ----------------------
     */

    nmod_poly_mat_multiply(U,T,Z);


    /**  Computation of V = W.T
     *   ----------------------
     */

    nmod_poly_mat_multiply(V,W,T);


    nmod_poly_mat_t e;
    nmod_poly_mat_init(e,1,1,prime);
    nmod_poly_mat_multiply(e,W,U);

    nmod_poly_t epol;
    nmod_poly_init(epol,prime);
    nmod_poly_set(epol,nmod_poly_mat_entry(e, 0, 0));

    nmod_poly_invmod(epol, epol, phi1); 


    for (i=0; i<r; i++)
    {
        nmod_poly_mul(nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(V, 0, i), epol); 
        nmod_poly_rem(nmod_poly_mat_entry(V, 0, i),nmod_poly_mat_entry(V, 0, i),phi1);

    }



    char namef[500];

    FILE* file;

    sprintf(namef,"res.txt");

    file = fopen(namef, "w");

    flint_fprintf(file,"BB:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,T,"x");

    flint_fprintf(file,");\n");

    flint_fprintf(file,"U:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,U,"x");

    flint_fprintf(file,");\n");

    flint_fprintf(file,"V:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,V,"x");

    flint_fprintf(file,");\n");

    flint_fprintf(file,"ee:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,e,"x");

    flint_fprintf(file,");\n");

    fclose(file);

    nmod_poly_mat_clear(W);
    nmod_poly_mat_clear(Z);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(temp);
    nmod_poly_mat_clear(T);
    nmod_poly_mat_clear(e);

    nmod_poly_clear(g);
    nmod_poly_clear(epol);


}



/**  Includes simplication to have phi1 at denominator 
 */

void Description_From_Rank_1(nmod_poly_mat_t NN, nmod_poly_mat_t DD, const ulong n,\
                             const nmod_poly_mat_t U, const nmod_poly_mat_t V,\
                             const nmod_poly_t  phi1, const nmod_poly_mat_t CT, \
                             const nmod_poly_mat_t PT, const nmod_poly_t Delta)
{

    int i,j;

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    nmod_poly_set(nmod_poly_mat_entry(DD, 0, 0), phi1);

    nmod_poly_t g;
    nmod_poly_init(g,prime);
    nmod_poly_div(g,Delta,phi1);


    
    slong D; 

    nmod_poly_mat_t temp; // for calling ffT 
    nmod_poly_mat_init(temp,r,1,prime);

    nmod_poly_mat_t  tempN; // for calling ffT 
    nmod_poly_mat_init(tempN,r,1,prime);


    for (i=0; i<r; i++)
    {
        nmod_poly_set(nmod_poly_mat_entry(NN, i, 0), nmod_poly_mat_entry(U, i, 0));
    }


    nmod_poly_t b;
    nmod_poly_init(b,prime);

    nmod_poly_t tpol;
    nmod_poly_init(tpol,prime);

    /**  M^*  (2r-2)d
     *   NN  < deg phi1
     */

    slong deg_phi1;
    deg_phi1 = nmod_poly_degree(phi1);


    D = (2*r-1)*d + deg_phi1 -1; // To check and better tune 

    // We start with the second columns, hence C index j 

    for (j=1; j<n; j++)
    {
    
        nmod_poly_mul(b, nmod_poly_mat_entry(V, 0, 0), nmod_poly_mat_entry(NN, 0, j-1));

        for (i=1; i<r; i++)
        {
            nmod_poly_mul(tpol, nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(NN, i, j-1));
            nmod_poly_add(b,b,tpol);
        }

        nmod_poly_rem(b, b, phi1);


        //flint_printf("\n ----------------------------  \n");
        //nmod_poly_print_pretty(b,"x");

        nmod_poly_neg(b,b);


        for (i=0; i<r; i++)
        {   
            nmod_poly_set(nmod_poly_mat_entry(tempN, i, 0), nmod_poly_mat_entry(NN, i, j-1));
        }

        nmod_apply_T(temp, tempN, CT, PT, D);


        //flint_printf("\n\n temp \n\n");

        //nmod_poly_mat_print_pretty(temp,"x");

        //flint_printf("\n\n");


        for (i=0; i<r; i++)
        {   
            nmod_poly_div(nmod_poly_mat_entry(temp, i, 0), nmod_poly_mat_entry(temp, i, 0), g);

            nmod_poly_derivative(tpol, nmod_poly_mat_entry(NN, i, j-1));
            nmod_poly_mul(tpol, tpol, phi1);
            nmod_poly_add(tpol, tpol, nmod_poly_mat_entry(temp, i, 0));

            nmod_poly_mul(nmod_poly_mat_entry(NN, i, j), b, nmod_poly_mat_entry(U, i, 0));
            nmod_poly_add(nmod_poly_mat_entry(NN, i, j), nmod_poly_mat_entry(NN, i, j), tpol);

            nmod_poly_div(nmod_poly_mat_entry(NN, i, j), nmod_poly_mat_entry(NN, i, j), phi1);

        }

        nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, 0, j-1));

        nmod_poly_add(nmod_poly_mat_entry(DD, 0, j),b,tpol);

        for (i=1; i<=j; i++)
        {

            nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, i, j-1));

            nmod_poly_add(nmod_poly_mat_entry(DD, i, j), nmod_poly_mat_entry(DD, i-1, j-1), tpol);

        }

    }


    flint_printf("\n");

    //nmod_poly_mat_print_pretty(NN,"x");

    //nmod_poly_mat_print_pretty(DD,"x");

    char namef[500];

    FILE* file;

    sprintf(namef,"desc.txt");

    file = fopen(namef, "w");

    flint_fprintf(file,"newN:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,NN,"x");

    flint_fprintf(file,");\n");

    flint_fprintf(file,"newD:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,DD,"x");

    flint_fprintf(file,");\n");

    fclose(file);

    nmod_poly_mat_clear(temp);
    nmod_poly_mat_clear(tempN);
        

    nmod_poly_clear(g);
    nmod_poly_clear(b);
    nmod_poly_clear(tpol);


}




/**  Computation of the numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix 
 * 
 *     central procedure 
 * 
 *    fraction-free approach 
 *   uses phi1 as computed previously, non monic since directly related to the resultant (non monic either) 
 */ 

void nmod_pseudo_Krylov(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t CT, \
                        const nmod_poly_mat_t PT, const nmod_poly_t  phi1, const nmod_poly_t  Delta)
{
    int i;

    slong r = (PT->r)-1;

    slong d;
    d = nmod_poly_mat_degree(PT);

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);


    nmod_poly_mat_t  tempN; // for calling ffT 
    nmod_poly_mat_init(tempN,r,1,prime);

    nmod_poly_mat_t  temp; 
    nmod_poly_mat_init(temp,r,1,prime);


    nmod_poly_t p1,p2;
    nmod_poly_init(p1,prime);
    nmod_poly_init(p2,prime);

    nmod_poly_t dphi1;
    nmod_poly_init(dphi1,prime);
    nmod_poly_derivative(dphi1,phi1);

    slong deg_phi1;
    deg_phi1=nmod_poly_degree(phi1);

    nmod_poly_t g;
    nmod_poly_init(g,prime);
    nmod_poly_div(g,Delta,phi1);

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
         *    when we apply T, (2r-1)d-1 is ok (bounds the x-degree of the resultant, btw)
         *  and then, afterwards, we will recover the degree of phi1 by simplification
         * 
         */

        D = k*deg_phi1 + (2*r-1)*d -1; 

        nmod_apply_T(temp, tempN, CT, PT, D);

        for (i=0; i<r; i++)
        {

            nmod_poly_div(nmod_poly_mat_entry(K, i, k+1), nmod_poly_mat_entry(temp, i, 0),g);

            nmod_poly_derivative(p1,nmod_poly_mat_entry(K, i, k));
            nmod_poly_mul(p1,p1,phi1);


            nmod_poly_mul(p2,dphi1,nmod_poly_mat_entry(K, i, k));
            nmod_poly_scalar_mul_nmod(p2,p2,k);

            nmod_poly_add(nmod_poly_mat_entry(K, i, k+1),nmod_poly_mat_entry(K, i, k+1),p1);

            nmod_poly_sub(nmod_poly_mat_entry(K, i, k+1),nmod_poly_mat_entry(K, i, k+1),p2);
        }

    } // main loop on the columns of K 
}





/**  Computation of the appropriate matrix for kernel solution 
 * 
 *    i.e. numerators of the pseudo-Krylov matrix
 *     an r x n polynomial matrix 
 *     with columns multiplied by an appropriate multiple of Delta (not phi1) 
 *     for the moment  
 *   
 */ 

void nmod_pseudo_Krylov_for_kernel(nmod_poly_mat_t K, const ulong n, const nmod_poly_mat_t PT) 
{


    int i,j;

    /**
     *  Resultant and inverse of Py
     *  ---------------------------
     */

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d=nmod_poly_mat_degree(PT);


    nmod_poly_t Delta; 
    nmod_poly_init(Delta,prime);

    nmod_poly_mat_t iPyT;
    nmod_poly_mat_init(iPyT,r,1,prime);

    nmod_biv_resultant_geometric(Delta, iPyT, PT);


    /** 
     *  Starting for the pseudo-Krylov matrix 
     *  -------------------------------------
     */

    nmod_poly_mat_t PxT;
    nmod_poly_mat_init(PxT,r+1,1,prime);


    // negation sign included 
    for (i=0; i<r+1; i++)
    {
        nmod_poly_derivative(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PT, i, 0));
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PxT, i, 0),prime-1);
    }


    /** Precomputation of C = -Px (Py)^(-1)
     *  -----------------------------------
     */

    ulong D;

    D=(2*r-1)*d-1;   // M^* and Y  (2r-2)d + (d-1)

    nmod_poly_mat_t  CT;
    nmod_poly_mat_init(CT,r,1,prime);

    nmod_biv_mulmod_geometric(CT, PxT, iPyT, PT, D); 


    /**  Randomized Computation of phi_1 and phi_2 (non monic)
     *   -----------------------------------------------------
     */

    nmod_poly_t phi1,phi2;
    nmod_poly_init(phi1,prime);
    nmod_poly_init(phi2,prime);


    nmod_phi_T(phi1, phi2, CT, PT, Delta);

    //flint_printf("\n deg phi1: %ld\n",nmod_poly_degree(phi1));

    /** 
     *   !!!! Version not using phi1
     *   ---------------------------
     */

    nmod_poly_set(phi1,Delta);

     /**  Computation of the numerators of K w.r.t. phi1
     *   -----------------------------------------------
     */

    
    nmod_pseudo_Krylov(K, n, CT, PT, phi1, Delta);
    

 
    /** Back to the trivial polynomial matrix 
     *  -------------------------------------
     */

    nmod_poly_t tpol;
    nmod_poly_init(tpol,prime);
    nmod_poly_zero(tpol);
    nmod_poly_set_coeff_ui(tpol,0,1);

    for (j=n-2; j>=0; j--)
    {
        nmod_poly_mul(tpol,tpol,phi1); 

        for (i=0; i<r; i++)
        {
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
        } 
    }


}


/**  algeqtodiffeq 
 * 
 *   Fraction-free pseudo-Krylov matrix: full computation w.r.t. phi1
 * 
 *      at least k solutions, i.e. n = r+k
 *   
 *   returns nz, the number of solutions found 
 * 
 *   output: LT, (r+k) x (r+k) polynomial matrix whose nz first columns give 
 *      the solutions
 * 
 * 
 *   !!!! Check using phi1 or not (simply Delta)
 *   ===========================================
 *    
 */ 

slong nmod_algeq_to_diffeq(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong k) 
{

    

    int i,j;

    /**
     *  Resultant and inverse of Py
     *  ---------------------------
     */

    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d=nmod_poly_mat_degree(PT);


    nmod_poly_t Delta; 
    nmod_poly_init(Delta,prime);

    nmod_poly_mat_t iPyT;
    nmod_poly_mat_init(iPyT,r,1,prime);

    nmod_biv_resultant_geometric(Delta, iPyT, PT);


    /** 
     *  Starting for the pseudo-Krylov matrix 
     *  -------------------------------------
     */

    nmod_poly_mat_t PxT;
    nmod_poly_mat_init(PxT,r+1,1,prime);


    // negation sign included 
    for (i=0; i<r+1; i++)
    {
        nmod_poly_derivative(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PT, i, 0));
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PxT, i, 0),prime-1);
    }


    /** Precomputation of C = -Px (Py)^(-1)
     *  -----------------------------------
     */

    ulong D;

    D=(2*r-1)*d-1;   // M^* and Y  (2r-2)d + (d-1)

    nmod_poly_mat_t  CT;
    nmod_poly_mat_init(CT,r,1,prime);

    nmod_biv_mulmod_geometric(CT, PxT, iPyT, PT, D); 


    /**  Randomized Computation of phi_1 and phi_2 (non monic)
     *   -----------------------------------------------------
     */

    nmod_poly_t phi1,phi2;
    nmod_poly_init(phi1,prime);
    nmod_poly_init(phi2,prime);


    nmod_phi_T(phi1, phi2, CT, PT, Delta);

    flint_printf("\n deg phi1: %ld\n",nmod_poly_degree(phi1));

    /** 
     *   !!!! Version not using phi1
     *   ---------------------------
     */

    nmod_poly_set(phi1,Delta);

     /**  Computation of the numerators of K w.r.t. phi1
     *   -----------------------------------------------
     */

    slong n;
    n=r+k;

    nmod_poly_mat_t  K;
    nmod_poly_mat_init(K,r,n,prime);

    
    nmod_pseudo_Krylov(K, n, CT, PT, phi1, Delta);
    

 
    /** Back to the trivial polynomial matrix 
     *  -------------------------------------
     */

    nmod_poly_t tpol;
    nmod_poly_init(tpol,prime);
    nmod_poly_zero(tpol);
    nmod_poly_set_coeff_ui(tpol,0,1);

    for (j=n-2; j>=0; j--)
    {
        nmod_poly_mul(tpol,tpol,phi1); 

        for (i=0; i<r; i++)
        {
            nmod_poly_mul(nmod_poly_mat_entry(K, i, j), nmod_poly_mat_entry(K, i, j), tpol);
        } 
    }

    slong nz=0;

    slong pivind[n];
    slong shift[n];
    for (j=0; j<n; j++)
    {
        shift[j]=0;
    }


    nmod_poly_mat_column_degree(pivind, K, shift);


    flint_printf("\n %{slong*}\n", pivind, n);



    double t=0.0;
    clock_t tt;
    tt=clock();
    nz=nmod_poly_mat_kernel(LT, pivind, shift, K, ORD_WEAK_POPOV, COL_UPPER);
    t += (double)(clock()-tt) / CLOCKS_PER_SEC;
    flint_printf("\n Kernel naive: %.3f sec.\n", t);

    return nz; 

}





/**  algeqtodiffeq 
 * 
 *   Series expansion and description udo-Krylov matrix: full computation w.r.t. phi1
 * 
 *      at least k solutions, i.e. n = r+k
 *   
 *   returns nz, the number of solutions found 
 * 
 *   output: LT, (r+k) x (r+k) polynomial matrix whose nz first columns give 
 *      the solutions
 * 
 * 
 *   !!!! Check using phi1 or not (simply Delta)
 *   ===========================================
 *    
 */ 

slong nmod_algeq_to_diffeq_series(nmod_poly_mat_t LT, const nmod_poly_mat_t PT, const slong k) 
{

    

    int i;


    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;


    /**
     *  Resultant and inverse of Py
     *  ---------------------------
     */

    
    nmod_poly_t Delta; 
    nmod_poly_init(Delta,prime);

    nmod_poly_t Deltak; 
    nmod_poly_init(Deltak,prime);

    nmod_poly_t iDelta; 
    nmod_poly_init(iDelta,prime);

    nmod_poly_t iDeltak; 
    nmod_poly_init(iDeltak,prime);

    nmod_poly_t tpol; 
    nmod_poly_init(tpol,prime);

    nmod_poly_mat_t iPyT;
    nmod_poly_mat_init(iPyT,r,1,prime);


    slong n=r+k;

    nmod_poly_mat_t K;
    nmod_poly_mat_init(K,r,n,prime);

    nmod_poly_mat_t numer;
    nmod_poly_mat_init(numer,r,1,prime);

    nmod_poly_mat_t temp;
    nmod_poly_mat_init(temp,r,1,prime);


    /** First column y 
     */
    for (i=0; i<r; i++)
    {
         nmod_poly_zero(nmod_poly_mat_entry(K, i, 0));
    }
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(K, 1, 0), 0, 1);

    for (i=0; i<r; i++)
    {
         nmod_poly_zero(nmod_poly_mat_entry(numer, i, 0));
    }
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(numer, 1, 0), 0, 1);


    slong d=nmod_poly_mat_degree(PT);

    nmod_biv_resultant_geometric(Delta, iPyT, PT);

    // Target truncation order for the descripion 
    slong sigma;
    sigma = ceil((r+n)*nmod_poly_degree(Delta)/r +1);

    slong N;
    N = sigma + (n-1); // Including the order for the derivation 

    
    nmod_poly_inv_series(iDelta, Delta, N);

    nmod_poly_set_coeff_ui(Deltak, 0, 1);

    nmod_poly_set_coeff_ui(iDeltak, 0, 1);


    /** 
     *  Starting for the pseudo-Krylov matrix 
     *  -------------------------------------
     */

    nmod_poly_mat_t PxT;
    nmod_poly_mat_init(PxT,r+1,1,prime);


    // negation sign included 
    for (i=0; i<r+1; i++)
    {
        nmod_poly_derivative(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PT, i, 0));
        nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(PxT, i, 0),nmod_poly_mat_entry(PxT, i, 0),prime-1);
    }


    /** Precomputation of C = -Px (Py)^(-1)
     *  -----------------------------------
     */

    ulong D;

    D=(2*r-1)*d-1;   // M^* and Y  (2r-2)d + (d-1)

    nmod_poly_mat_t  CT;
    nmod_poly_mat_init(CT,r,1,prime);

    nmod_biv_mulmod_geometric(CT, PxT, iPyT, PT, D); 

    // Starting from the second column, hence C index k+1 

    D=N+(2*r-2)*d;   // M^* is (2r-2)d  

    for (int k=0; k<n-1; k++)
    {
        nmod_poly_mullow(Deltak, Deltak, Delta, N);

        nmod_poly_mullow(iDeltak, iDeltak, iDelta, N);

        // Because aliasing not sure 
        nmod_apply_T(temp, numer, CT, PT, D);
        nmod_poly_mat_swap(numer,temp);

        for (i=0; i<r; i++)
        {

            nmod_poly_truncate(nmod_poly_mat_entry(numer, i, 0), N); // useful ? 

            nmod_poly_mullow(nmod_poly_mat_entry(numer, i, 0), nmod_poly_mat_entry(numer, i, 0), iDeltak, N);

            nmod_poly_derivative(tpol,nmod_poly_mat_entry(K, i, k));

            nmod_poly_add_series(nmod_poly_mat_entry(numer, i, 0),nmod_poly_mat_entry(numer, i, 0),tpol,N);

            nmod_poly_set(nmod_poly_mat_entry(K, i, k+1),nmod_poly_mat_entry(numer, i, 0));

            nmod_poly_mullow(nmod_poly_mat_entry(numer, i, 0), nmod_poly_mat_entry(numer, i, 0), Deltak, N);

        }
    }

    
    char namef[500];

    FILE* file;

    sprintf(namef,"res.txt");

    file = fopen(namef, "w");

    flint_fprintf(file,"W:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,K,"x");

    flint_fprintf(file,");\n");

    fclose(file);


    /**  Search for a description 
     *   ------------------------
     */


    nmod_poly_mat_t  NN;
    nmod_poly_mat_init(NN,r,n,prime);

    nmod_poly_mat_t  DD;
    nmod_poly_mat_init(DD,n,n,prime);

    nmod_poly_mat_right_description(NN, DD, K, nmod_poly_degree(Delta));

    sprintf(namef,"desc.txt");

    file = fopen(namef, "w");

    flint_fprintf(file,"NN:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,NN,"x");

    flint_fprintf(file,");\n");

    flint_fprintf(file,"\n");

    flint_fprintf(file,"DD:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,DD,"x");

    flint_fprintf(file,");\n");

    fclose(file);


    slong nz=0;

    slong pivind[n];
    slong shift[n];

    for (i=0; i<n; i++)
    {
        shift[i]=0;
    }


    nmod_poly_mat_column_degree(pivind, NN, shift);

    flint_printf("\n %{slong*}\n", pivind, n);


    double t=0.0;
    clock_t tt;
    tt=clock();
    nz=nmod_poly_mat_kernel(LT, pivind, shift, NN, ORD_WEAK_POPOV, COL_UPPER);
    t += (double)(clock()-tt) / CLOCKS_PER_SEC;
    flint_printf("\n Kernel new: %.3f sec.\n", t);

    nmod_poly_mat_multiply(LT,DD,LT);


    sprintf(namef,"sol.txt");

    file = fopen(namef, "w");

    flint_fprintf(file,"sol:=Matrix(");

    nmod_poly_mat_fprint_pretty(file,LT,"x");

    flint_fprintf(file,");\n");

   
    return nz; 

}

/*------------------------------------------------------------*/
/*------------------------------------------------------------*/
/* PRETTY PRINTING THE MATRIX                                 */
/*------------------------------------------------------------*/
/*------------------------------------------------------------*/



void fmpz_poly_mat_print_pretty(const fmpz_poly_mat_t mat, const char * var)
{
    slong rdim = mat->r, cdim = mat->c;

    flint_printf("<%wd x %wd matrix over Z[%s]>\n", mat->r, mat->c, var);
    flint_printf("[");
    for (slong i = 0; i < rdim; i++)
    {
        flint_printf("[");
        for (slong j = 0; j < cdim; j++)
        {
            fmpz_poly_print_pretty(fmpz_poly_mat_entry(mat, i, j), var);
            if (j+1 < cdim)
                flint_printf(", ");
        }
        if (i != rdim -1)
            flint_printf("],\n");
        else
            flint_printf("]");
    }
    flint_printf("]\n");
}



void fmpz_poly_mat_fprint_pretty(FILE *file, const fmpz_poly_mat_t mat, const char * var)
{
    slong rdim = mat->r, cdim = mat->c;

    flint_fprintf(file,"[");
    for (slong i = 0; i < rdim; i++)
    {
        flint_fprintf(file,"[");
        for (slong j = 0; j < cdim; j++)
        {
            fmpz_poly_fprint_pretty(file,fmpz_poly_mat_entry(mat, i, j), var);
            if (j+1 < cdim)
                flint_fprintf(file,", ");
        }
        if (i != rdim -1)
            flint_fprintf(file,"],\n");
        else
            flint_fprintf(file,"]");
    }
    flint_fprintf(file,"]");
}

void nmod_poly_mat_fprint_pretty(FILE *file, const nmod_poly_mat_t mat, const char * var)
{
    slong rdim = mat->r, cdim = mat->c;

    flint_fprintf(file,"[");
    for (slong i = 0; i < rdim; i++)
    {
        flint_fprintf(file,"[");
        for (slong j = 0; j < cdim; j++)
        {
            nmod_poly_fprint_pretty(file,nmod_poly_mat_entry(mat, i, j), var);
            if (j+1 < cdim)
                flint_fprintf(file,", ");
        }
        if (i != rdim -1)
            flint_fprintf(file,"],\n");
        else
            flint_fprintf(file,"]");
    }
    flint_fprintf(file,"]");
}

// One column
void  fmpz_to_nmod_poly_mat(nmod_poly_mat_t PT, const fmpz_poly_mat_t PZT)
{
    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    fmpz_poly_t tpolZ;
    fmpz_poly_init(tpolZ);

    fmpz_t tZ;
    fmpz_init(tZ);

    nmod_t q; 
    nmod_init(&q, prime);

    slong d;

    for (int i=0; i<r+1; i++)
    {
        nmod_poly_zero(nmod_poly_mat_entry(PT, i, 0));

        fmpz_poly_set(tpolZ,fmpz_poly_mat_entry(PZT, i, 0));
        d = fmpz_poly_degree(tpolZ);

        for (int j=0; j<=d; j++)
        {
            fmpz_poly_get_coeff_fmpz(tZ,tpolZ,j);
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(PT, i, 0), j,\
                fmpz_get_nmod(tZ,q));
        }
    }
}

// One column
void  nmod_to_fmpz_poly_mat(fmpz_poly_mat_t PZT, const nmod_poly_mat_t PT)
{
    ulong prime;
    prime = nmod_poly_mat_modulus(PT);

    slong r = (PT->r)-1;

    slong d;

    nmod_poly_t tpol;
    nmod_poly_init(tpol,prime);

    for (int i=0; i<r+1; i++)
    {
        fmpz_poly_zero(fmpz_poly_mat_entry(PZT, i, 0));

        nmod_poly_set(tpol,nmod_poly_mat_entry(PT, i, 0));

        d = nmod_poly_degree(tpol);

        for (int j=0; j<=d; j++)
        {
            fmpz_poly_set_coeff_ui(fmpz_poly_mat_entry(PZT, i, 0), j,\
                nmod_poly_get_coeff_ui(tpol,j));
        }
    }
}



/** CRT for Krylov polynomial matrix ready for kernel
 * 
 *  return M 
 *   and found 
 * 
 *   fixed number of primes 
 * 
 */

void CRT_pseudo_Krylov_for_kernel(fmpz_poly_mat_t int_residues, slong * degs, const ulong n, const slong L, const nn_ptr primes,\
                                    fmpz_poly_mat_t  PZT)
{


    slong r = (PZT->r)-1;
   
    nmod_poly_mat_t PT;

    nmod_poly_mat_t Kmod[L];

    nn_ptr residues;
    residues = _nmod_vec_init(L);

    fmpz_comb_t comb;
    fmpz_comb_temp_t ctemp;

    fmpz_t M;
    fmpz_init(M);


    double t=0.0;
    double t2=0.0;
    double t3=0.0;
    clock_t tt,tt2,tt3;

    tt=clock();
       
    for (int k=0; k < L; k++) 
    {
        nmod_poly_mat_init(Kmod[k], r, n, primes[k]);

        nmod_poly_mat_init(PT,r+1,1,primes[k]);

        fmpz_to_nmod_poly_mat(PT,PZT);

        nmod_pseudo_Krylov_for_kernel(Kmod[k], n, PT); 
    }
    
    t += (double)(clock()-tt) / CLOCKS_PER_SEC;

    flint_printf("\n Residues: %.3f sec.\n", t);



    for (int ri=0; ri < r; ri++)
    {
        for (int cj=0; cj < n; cj++)
            {
            degs[ri*n+cj] = nmod_poly_degree(nmod_poly_mat_entry(Kmod[0], ri, cj));
            }
    }

    tt2=clock();

    fmpz_comb_init(comb,primes,L);
    fmpz_comb_temp_init(ctemp,comb);

    for (int ri=0; ri < r; ri++)
    {
        for (int cj=0; cj < n; cj++)
        {
            for (int l=0; l<degs[ri*n+cj]+1; l++)
            {
                
                for (int i=0; i<L; i++)
                {
                residues[i] = nmod_poly_get_coeff_ui(nmod_poly_mat_entry(Kmod[i],ri,cj),l);
                }

                tt3=clock();

                fmpz_multi_CRT_ui(M, residues, comb, ctemp, 1);

                t3 += (double)(clock()-tt3) / CLOCKS_PER_SEC; 
                
                fmpz_poly_set_coeff_fmpz(fmpz_poly_mat_entry(int_residues,ri,cj),l,M);

            } // loop on the coefficients l 
        } // loop on the columns cj 
    } // loop on the rows ri 
        

    t2 += (double)(clock()-tt2) / CLOCKS_PER_SEC;    

    flint_printf("\n CRT tot: %.3f sec.\n", t2);
    flint_printf("\n CRT: %.3f sec.\n", t3);

    //flint_printf("\n %{slong*}\n", degs, n*r);

    _nmod_vec_clear(residues);
    fmpz_clear(M);
    for (int k=0; k < L; k++) 
    {
        nmod_poly_mat_clear(Kmod[k]);
    }
    nmod_poly_mat_clear(PT);
    fmpz_comb_clear(comb);
    fmpz_comb_temp_clear(ctemp);
}



// void CRT_poly_mat_combine(fmpz_poly_mat_t int_residues, slong * degs,\
//                             const fmpz_poly_mat_t int_residues_1, const fmpz_t P_1,\
//                             const fmpz_poly_mat_t int_residues_2, const fmpz_t P_2)
// {


//     slong r = (int_residues_1 -> r);

//     slong n = (int_residues_1 -> c);


    
//     fmpz_t M;
//     fmpz_init(M);

//     fmpz_t M_1;
//     fmpz_init(M_1);
    
//     fmpz_t M_2;
//     fmpz_init(M_2);


//     for (int ri=0; ri < r; ri++)
//     {
//         for (int cj=0; cj < n; cj++)
//         {
//                 for (int l=0; l<degs[ri*n+cj]+1; l++)
//                 {

//                     fmpz_poly_get_coeff_fmpz(M_1, fmpz_poly_mat_entry(int_residues_1,ri,cj), l);
//                     fmpz_poly_get_coeff_fmpz(M_2, fmpz_poly_mat_entry(int_residues_2,ri,cj), l);

//                     fmpz_CRT(M, M_1, P_1, M_2, P_2, 1);

//                     fmpz_poly_set_coeff_fmpz(fmpz_poly_mat_entry(int_residues,ri,cj), l, M);
                    

//                 } // loop on the coefficients l 
//         } // loop on the columns cj 
//     } // loop on the rows ri 
    

//     fmpz_clear(M);
//     fmpz_clear(M_1);
//     fmpz_clear(M_2);

// }


void CRT_poly_mat_combine(fmpz_poly_mat_t int_residues, slong * degs,\
                            const fmpz_poly_mat_t int_residues_1, const fmpz_t P_1,\
                            const fmpz_poly_mat_t int_residues_2, const fmpz_t P_2)
{


    slong r = (int_residues_1 -> r);

    slong n = (int_residues_1 -> c);


    fmpz_t M;
    fmpz_init(M);

    fmpz_t M_1;
    fmpz_init(M_1);
    
    fmpz_t M_2;
    fmpz_init(M_2);

    fmpz_t g;
    fmpz_init(g);

    fmpz_t s;
    fmpz_init(s);

    fmpz_t t;
    fmpz_init(t);

    fmpz_xgcd(g, s, t, P_2, P_1); 

    fmpz_t c1;
    fmpz_init(c1);

    fmpz_t c2;
    fmpz_init(c2);

    fmpz_t P;
    fmpz_init(P);

    fmpz_mul(P,P_1,P_2);

    fmpz_t sP2;
    fmpz_init(sP2);

    fmpz_mul(sP2,s,P_2);

    fmpz_t tP1;
    fmpz_init(tP1);

    fmpz_mul(tP1,t,P_1);


    for (int ri=0; ri < r; ri++)
    {
        for (int cj=0; cj < n; cj++)
        {
                for (int l=0; l<degs[ri*n+cj]+1; l++)
                {

                    fmpz_poly_get_coeff_fmpz(M_1, fmpz_poly_mat_entry(int_residues_1,ri,cj), l);
                    fmpz_poly_get_coeff_fmpz(M_2, fmpz_poly_mat_entry(int_residues_2,ri,cj), l);


                    fmpz_mul(c1,M_1,sP2);
                    fmpz_mul(c2,M_2,tP1);

                    // fmpz_mul(c1,M_1,s);
                    // fmpz_smod(c1,c1,P_1);

                    // fmpz_mul(c2,M_2,t);
                    // fmpz_smod(c2,c2,P_2);

                    // fmpz_mul(c1,c1,P_2);
                    // fmpz_mul(c2,c2,P_1);

                    fmpz_add(M,c1,c2);
                    fmpz_smod(M,M,P);

    
                    //fmpz_CRT(M, M_1, P_1, M_2, P_2, 1);

                    fmpz_poly_set_coeff_fmpz(fmpz_poly_mat_entry(int_residues,ri,cj), l, M);
                    

                } // loop on the coefficients l 
        } // loop on the columns cj 
    } // loop on the rows ri 
    

    fmpz_clear(M);
    fmpz_clear(M_1);
    fmpz_clear(M_2);

}












