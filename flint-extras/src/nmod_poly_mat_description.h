/*
    Copyright (C) 2025, 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#ifndef NMOD_POLY_MAT_DESCRIPTION_H
#define NMOD_POLY_MAT_DESCRIPTION_H

#include <flint/nmod_poly_mat.h>

#include "nmod_poly_mat_forms.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  Left description computation for H(x) n x m in K(x) (power series)
 *    with target degree delta
 * 
 *  requires enough precision in input: ie at least (m+n)*delta/m +1
 * 
 *  returns nbrows and a partial (or full) description when 0 < nbrows <= n rows, 
 *    or zero if no candidates 
 * 
 *   N  n x m and and D n x n  are initialized outside 
 *    hence the result is part of N and D  
 * 
 */

slong nmod_poly_mat_left_description(nmod_poly_mat_t N, nmod_poly_mat_t D,
                            const nmod_poly_mat_t H, slong delta);


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
                            const nmod_poly_mat_t H, slong delta);




#ifdef __cplusplus
}
#endif

#endif


/* -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
