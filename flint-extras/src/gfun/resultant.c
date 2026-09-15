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
    Mechanical relocation only (2026-09), not a cleanup pass: moved out of gfun.c
    into its own file because the question it computes (resultant of a bivariate
    P and its own y-derivative Py, via geometric evaluation-interpolation) is
    self-contained and independent of the rest of gfun.c's algeqtodiffeq-specific
    machinery -- worth being able to look at, test, and eventually fix in
    isolation. This file's location under src/gfun/ is itself considered
    temporary: this function may belong in a more general PML module once it
    gets its own real cleanup pass (it isn't algeqtodiffeq-specific in what it
    computes, only in how gfun.c happens to use its output).

    A real, unaddressed correctness gap was found while reading this function
    (not fixed here -- see claude-pseudoKrylov/todo.md item 11): it evaluates P
    at exactly L = deg(Delta)+1 geometric-progression points with zero spare
    points, and at each one inverts Py(beta,y) modulo P(beta,y)
    (nmod_poly_invmod below). If beta happens to be an actual root of Delta
    (where gcd(P(beta,.),Py(beta,.)) != 1) or a root of P's leading
    y-coefficient (a degree drop, which breaks the evaluation-commutes-with-
    resultant identity this method relies on), this either fails inside
    invmod or silently interpolates a wrong Delta/iPyT -- reproducibly for
    that (P, prime) pair, since the progression is deterministic. Also: the
    return value of nmod_find_root (0 signals failure) is never checked here.
*/

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


    nmod_geometric_progression_clear(F);
    _nmod_vec_clear(evalR);
    nmod_poly_mat_clear(evalP);
    nmod_poly_mat_clear(evaliPy);
    nmod_poly_clear(evalPy);
}
