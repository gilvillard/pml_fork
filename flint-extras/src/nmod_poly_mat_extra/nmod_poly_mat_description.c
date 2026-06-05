/*
    Copyright (C) 2025 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <math.h>

#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_mat.h>
#include <flint/perm.h>

#include "nmod_poly_mat_extra.h"
#include "nmod_poly_mat_forms.h"
#include "nmod_poly_mat_kernel.h"
#include "nmod_poly_mat_multiply.h"
#include "nmod_poly_mat_utils.h"

#include "nmod_poly_mat_description.h"

/**
 *  Left description computation for H(x) n x m in K(x) (power series)
 *    with target degree delta
 * 
 *  requires enough precision in input: i.e. at least (m+n)*delta/m +1
 * 
 *  returns nbrows and a partial (or full) description when 0 < nbrows <= n rows, 
 *    or zero if no candidates 
 * 
 *   N  n x m and and D n x n  are initialized outside 
 *    hence the result is part of N and D  
 * 
 */

slong nmod_poly_mat_left_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                            const nmod_poly_mat_t H, 
                            slong delta)

{
    slong i,j;

    slong n = H->r;
    slong m = H->c;

    slong sigma = ceil((double) (m+n)*delta/FLINT_MIN(m,n) +1); 

    nmod_poly_mat_t M;
    nmod_poly_mat_init(M, n+m, m, H->modulus);

    nmod_poly_mat_set(M, H); // GV, correct even the dimension of M is larger ? 

    nmod_poly_t mone;
    nmod_poly_init(mone, H->modulus);
    nmod_poly_set_coeff_ui(mone, 0, H->modulus -1); // 1

    for (i = 0; i < m; i++)
        nmod_poly_set(nmod_poly_mat_entry(M, n+i, i), mone);


    nmod_poly_mat_t B;
    nmod_poly_mat_init(B, n+m, n+m, H->modulus);

    /** Appropriate shift */

    slong shift[n+m];

    for (i = 0; i < n+m; i++) 
        shift[i]=0; 

    /** Shifted approximant computation */

    nmod_poly_mat_pmbasis(B, shift, M, sigma); 


    slong rows[n+m];
    slong nbrows=0;

    for (i = 0; i < n+m; i++) 
    {
        if (shift[i] <= delta) 
        {
            rows[nbrows]=i;
            nbrows +=1;           
        }
    }

    if (nbrows != n) 
    {
        flint_printf("\nA complete description of degree at most %{slong} probably doesn't exist\n", delta);
    }

    if (nbrows == 0) 
        return 0; 
    else 
    {
        for (i = 0; i < nbrows; i++)
        {
            for (j = 0; j < m; j++)
                nmod_poly_set(nmod_poly_mat_entry(N, i, j), nmod_poly_mat_entry(B, rows[i], n+j));
        }


        for (i = 0; i < nbrows; i++)
        {
            for (j = 0; j < n; j++)
                nmod_poly_set(nmod_poly_mat_entry(D, i, j), nmod_poly_mat_entry(B, rows[i], j));
        }
    }

    nmod_poly_mat_clear(M); 
    nmod_poly_mat_clear(B);

    nmod_poly_clear(mone); 

    return nbrows;

}


/**
 *  Right description computation for H(x) m x n in K(x) (power series)
 *    with target degree delta
 * 
 *  requires enough precision in input: ie at least (m+n)*delta/m +1
 * 
 *  returns nbcols and a partial (or full) description when 0 < nbcols <= n rows, 
 *    or zero if no candidates 
 * 
 *    N m x n and and D n x n  are initialized outside  
 *       hence the result is part of N and D
 * 
 */


slong nmod_poly_mat_right_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                                        const nmod_poly_mat_t H, slong delta)

{
    slong nbcols;

    nmod_poly_mat_t NT;
    nmod_poly_mat_init(NT, H->c, H->r, H->modulus);

    nmod_poly_mat_t DT;
    nmod_poly_mat_init(DT, H->c, H->c, H->modulus);

    nmod_poly_mat_t HT;
    nmod_poly_mat_init(HT, H->c, H->r, H->modulus);


    nmod_poly_mat_transpose(HT,H);

    nbcols = nmod_poly_mat_left_description(NT,DT,HT,delta);

    if (nbcols == 0) 
        return nbcols;

    nmod_poly_mat_transpose(N,NT);
    nmod_poly_mat_transpose(D,DT);
    
    nmod_poly_mat_clear(NT); 
    nmod_poly_mat_clear(DT);
    nmod_poly_mat_clear(HT);

    return nbcols;

}

