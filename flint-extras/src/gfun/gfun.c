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

