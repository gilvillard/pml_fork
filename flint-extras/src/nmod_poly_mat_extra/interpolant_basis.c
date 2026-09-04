/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/


#include "nmod_poly_mat_multiply.h"
#include "nmod_poly_mat_interpolant.h"
#include "nmod_mat_poly.h"
#include "nmod_poly_mat_utils.h"

void nmod_poly_mat_mintbasis(nmod_poly_mat_t intbas,
                             slong * shift,
                             const nmod_poly_mat_t E,
                             const ulong * pts,
                             slong d)
{
    /** nmod_mat_poly_mintbasis takes E as a plain, flat array of `d`
       already-initialized nmod_mat_t's (nmod_mat_struct *), just a 
       container for `d`*/
    nmod_mat_struct * Earr = (nmod_mat_struct *) flint_malloc(d * sizeof(nmod_mat_struct));
    for (slong k = 0; k < d; k++)
    {
        nmod_mat_init(Earr + k, E->r, E->c, E->modulus);
        nmod_poly_mat_get_coeff_mat(Earr + k, E, k);
    }
    nmod_mat_poly_t intbasmp;
    nmod_mat_poly_init(intbasmp, E->r, E->r, E->modulus);
    nmod_mat_poly_mintbasis(intbasmp, shift, Earr, E->c, pts, d);
    nmod_poly_mat_set_from_mat_poly(intbas, intbasmp);
    for (slong k = 0; k < d; k++)
        nmod_mat_clear(Earr + k);
    flint_free(Earr);
    nmod_mat_poly_clear(intbasmp);
}


void nmod_poly_mat_pmintbasis(nmod_poly_mat_t intbas,
                              slong * shift,
                              const nmod_poly_mat_t E,
                              const ulong * pts,
                              slong d)
{
    if (d <= PMINTBASIS_THRES)
    {
        nmod_poly_mat_mintbasis(intbas, shift, E, pts, d);
        return;
    }

    const slong d1 = (d + 1) / 2;
    const slong d2 = d - d1;

    nmod_poly_mat_t E1;
    nmod_poly_mat_init(E1, E->r, E->c, E->modulus);
    nmod_poly_mat_set_trunc(E1, E, d1);

    nmod_poly_mat_t P1;
    nmod_poly_mat_init(P1, E->r, E->r, E->modulus);
    nmod_poly_mat_pmintbasis(P1, shift, E1, pts, d1);
    nmod_poly_mat_clear(E1);

    /* R_i = P1(pts[d1+i]) * E_{d1+i}, i = 0..d2-1 -- unlike pmbasis's
       residual (a middle product), this is d2 independent evaluate-then-
       multiply operations; no polynomial-arithmetic trick applies here
       (see this file's header comment / nmod_poly_mat_interpolant.h). */
    nmod_poly_mat_t R;
    nmod_poly_mat_init(R, E->r, E->c, E->modulus);

    nmod_mat_t evalP1, Ei, Ri;
    nmod_mat_init(evalP1, E->r, E->r, E->modulus);
    nmod_mat_init(Ei, E->r, E->c, E->modulus);
    nmod_mat_init(Ri, E->r, E->c, E->modulus);
    for (slong idx = 0; idx < d2; idx++)
    {
        nmod_poly_mat_evaluate_nmod(evalP1, P1, pts[d1 + idx]);
        nmod_poly_mat_get_coeff_mat(Ei, E, d1 + idx);
        nmod_mat_mul(Ri, evalP1, Ei);
        nmod_poly_mat_set_coeff_mat(R, Ri, idx);
    }
    nmod_mat_clear(evalP1);
    nmod_mat_clear(Ei);
    nmod_mat_clear(Ri);

    nmod_poly_mat_t P2;
    nmod_poly_mat_init(P2, E->r, E->r, E->modulus);
    nmod_poly_mat_pmintbasis(P2, shift, R, pts + d1, d2);
    nmod_poly_mat_clear(R);

    nmod_poly_mat_multiply(intbas, P2, P1);

    nmod_poly_mat_clear(P1);
    nmod_poly_mat_clear(P2);
}

/* -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

