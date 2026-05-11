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


void mat_to_xy(nmod_mpoly_t P, nmod_mpoly_ctx_t ctx, const nmod_poly_mat_t PT);


#endif // GFUN_H
