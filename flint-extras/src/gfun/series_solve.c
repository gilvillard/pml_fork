/*
    Copyright (C) 2026 Gilles Villard

    This file is part of PML.

    PML is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License version 2.0 (GPL-2.0-or-later)
    as published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version. See
    <https://www.gnu.org/licenses/>.
*/

#include <flint/nmod_vec.h>
#include <flint/nmod_mat.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_mat.h>

#include "nmod_poly_mat_extra.h"

#include "gfun.h"

/*
    Linear systems over power series (2026-10-15), for the naive
    pseudo-Krylov route (claude-pseudoKrylov: the kernel of the naive K by
    lifting instead of a minimal kernel basis). Generic polynomial-matrix
    code, kept in gfun for now; the names follow nmod_poly_mat_extra so that
    it can move there unchanged.

      nmod_poly_mat_inv_trunc_newton   A^{-1} mod x^order, Newton iteration
      nmod_poly_mat_solve_series_dc    A^{-1} B mod x^prec, divide and conquer
      nmod_poly_mat_kernel_vector_lift generator of the right kernel of an
                                       m x (m+1) matrix of rank m, by lifting
                                       and Pade approximation (2026-10-15)

    The divide and conquer is the algorithm of Bostan, Chyzak, Ollivier,
    Salvy, Schost, Sedoglavic (SODA 2007, Sec. 3, Fig. 3, Prop. 1) without
    the derivative: to solve A z = c mod x^m, solve for the lower half z0
    mod x^h, h = floor(m/2), then for the upper half z1 with the right-hand
    side R = coefficients h..m-1 of c - A z0; z = z0 + x^h z1. Below the
    base size s0, z = S c mod x^m with S = A^{-1} mod x^s0 computed once.
*/

/*------------------------------------------------------------*/
/* truncated inverse by Newton iteration                      */
/*------------------------------------------------------------*/

int
nmod_poly_mat_inv_trunc_newton(nmod_poly_mat_t S, const nmod_poly_mat_t A, slong order)
{
    const slong n = A->r;
    const ulong prime = A->modulus;
    nmod_t mod;

    if (A->c != n)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_inv_trunc_newton: A must be square\n");
    if (order < 1)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_inv_trunc_newton: order must be >= 1\n");

    nmod_init(&mod, prime);

    nmod_mat_t A0, A0inv;
    nmod_mat_init(A0, n, n, prime);
    nmod_mat_init(A0inv, n, n, prime);
    nmod_poly_mat_evaluate_nmod(A0, A, 0);
    int ok = nmod_mat_inv(A0inv, A0);
    nmod_mat_clear(A0);
    if (!ok)
    {
        nmod_mat_clear(A0inv);
        return 0;
    }

    /* X: the inverse being built (S may alias A, so S is set at the end) */
    nmod_poly_mat_t X;
    nmod_poly_mat_init(X, n, n, prime);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++)
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(X, i, j), 0, nmod_mat_entry(A0inv, i, j));
    nmod_mat_clear(A0inv);

    /* with X correct mod x^p and q = min(2p, order):
       E = ((I - (A mod x^q) X) mod x^q) div x^p, X <- X + x^p (X E mod x^(q-p)) */
    nmod_poly_mat_t At, E, F;
    nmod_poly_mat_init(At, n, n, prime);
    nmod_poly_mat_init(E, n, n, prime);
    nmod_poly_mat_init(F, n, n, prime);
    for (slong p = 1; p < order; )
    {
        slong q = FLINT_MIN(2 * p, order);

        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < n; j++)
                nmod_poly_set_trunc(nmod_poly_mat_entry(At, i, j), nmod_poly_mat_entry(A, i, j), q);
        nmod_poly_mat_multiply(E, At, X);
        nmod_poly_mat_truncate(E, q);
        nmod_poly_mat_neg(E, E);
        for (slong i = 0; i < n; i++)
            nmod_poly_set_coeff_ui(nmod_poly_mat_entry(E, i, i), 0,
                nmod_add(nmod_poly_get_coeff_ui(nmod_poly_mat_entry(E, i, i), 0), 1, mod));
        nmod_poly_mat_shift_right(E, E, p);

        nmod_poly_mat_multiply(F, X, E);
        nmod_poly_mat_truncate(F, q - p);
        nmod_poly_mat_shift_left(F, F, p);
        nmod_poly_mat_add(X, X, F);

        p = q;
    }
    nmod_poly_mat_clear(At);
    nmod_poly_mat_clear(E);
    nmod_poly_mat_clear(F);

    nmod_poly_mat_swap(S, X);
    nmod_poly_mat_clear(X);
    return 1;
}

/*------------------------------------------------------------*/
/* divide and conquer solve                                   */
/*------------------------------------------------------------*/

/* acc[0 .. nhi-nlo) += coefficients nlo .. nhi-1 of a*b; tmp has length >= nhi-nlo.
   Covers the cases _nmod_poly_mulmid does not (zero lengths, nhi beyond the
   product, nlo past it). */
static void
_mulmid_add(nn_ptr acc, nn_srcptr a, slong alen, nn_srcptr b, slong blen,
            slong nlo, slong nhi, nn_ptr tmp, nmod_t mod)
{
    while (alen > 0 && a[alen - 1] == 0)
        alen--;
    while (blen > 0 && b[blen - 1] == 0)
        blen--;
    if (alen == 0 || blen == 0)
        return;

    slong top = FLINT_MIN(nhi, alen + blen - 1);
    if (nlo >= top)
        return;

    _nmod_poly_mulmid(tmp, a, alen, b, blen, nlo, top, mod);
    _nmod_vec_add(acc, acc, tmp, top - nlo, mod);
}

typedef struct
{
    const nmod_poly_mat_struct * A;   /* n x n */
    const nmod_poly_mat_struct * S;   /* A^{-1} mod x^s0 */
    slong n, k, s0;
    nn_ptr acc, tmp;                  /* scratch, length >= prec */
    nmod_t mod;
}
_solve_dc_ctx_struct;

/* Z, C: n*k coefficient arrays, entry (i,j) at index i*k+j, of length >= m.
   On output Z[.][0..m) solves A Z = C mod x^m. C[.][h..m) is used as
   workspace for the right-hand side of the upper half. */
static void
_solve_dc_rec(nn_ptr * Z, nn_ptr * C, slong m, const _solve_dc_ctx_struct * ctx)
{
    const slong n = ctx->n, k = ctx->k;

    if (m <= ctx->s0)
    {
        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < k; j++)
            {
                nn_ptr z = Z[i * k + j];
                _nmod_vec_zero(z, m);
                for (slong l = 0; l < n; l++)
                {
                    const nmod_poly_struct * s = nmod_poly_mat_entry(ctx->S, i, l);
                    _mulmid_add(z, s->coeffs, FLINT_MIN(s->length, m), C[l * k + j], m,
                                0, m, ctx->tmp, ctx->mod);
                }
            }
        return;
    }

    const slong h = m / 2;

    /* lower half: Z mod x^h */
    _solve_dc_rec(Z, C, h, ctx);

    /* right-hand side of the upper half: C[h..m) -= coefficients h..m-1 of A Z[0..h) */
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < k; j++)
        {
            _nmod_vec_zero(ctx->acc, m - h);
            for (slong l = 0; l < n; l++)
            {
                const nmod_poly_struct * a = nmod_poly_mat_entry(ctx->A, i, l);
                _mulmid_add(ctx->acc, a->coeffs, FLINT_MIN(a->length, m), Z[l * k + j], h,
                            h, m, ctx->tmp, ctx->mod);
            }
            _nmod_vec_sub(C[i * k + j] + h, C[i * k + j] + h, ctx->acc, m - h, ctx->mod);
        }

    /* upper half: Z[h..m) */
    nn_ptr * Z1 = flint_malloc(2 * n * k * sizeof(nn_ptr));
    nn_ptr * C1 = Z1 + n * k;
    for (slong t = 0; t < n * k; t++)
    {
        Z1[t] = Z[t] + h;
        C1[t] = C[t] + h;
    }
    _solve_dc_rec(Z1, C1, m - h, ctx);
    flint_free(Z1);
}

int
nmod_poly_mat_solve_series_dc(nmod_poly_mat_t X, const nmod_poly_mat_t A,
                              const nmod_poly_mat_t B, slong prec, slong s0)
{
    const slong n = A->r;
    const slong k = B->c;
    const ulong prime = A->modulus;

    if (A->c != n || B->r != n)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_solve_series_dc: A must be n x n and B n x k\n");
    if (X->r != n || X->c != k)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_solve_series_dc: X must be n x k\n");
    if (prec < 0)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_solve_series_dc: prec must be >= 0\n");

    if (s0 < 1)
        s0 = 1;
    if (prec > 0 && s0 > prec)
        s0 = prec;

    nmod_poly_mat_t S;
    nmod_poly_mat_init(S, n, n, prime);
    if (!nmod_poly_mat_inv_trunc_newton(S, A, s0))
    {
        nmod_poly_mat_clear(S);
        return 0;
    }

    if (prec == 0)
    {
        nmod_poly_mat_zero(X);
        nmod_poly_mat_clear(S);
        return 1;
    }

    /* T: the solution, built in place (X may alias A or B);
       C: copy of B mod x^prec, zero padded, overwritten during the recursion */
    nmod_poly_mat_t T;
    nmod_poly_mat_init(T, n, k, prime);
    nn_ptr * Z = flint_malloc(2 * n * k * sizeof(nn_ptr));
    nn_ptr * C = Z + n * k;
    nn_ptr Cdata = _nmod_vec_init(n * k * prec);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < k; j++)
        {
            nmod_poly_struct * t = nmod_poly_mat_entry(T, i, j);
            nmod_poly_fit_length(t, prec);
            Z[i * k + j] = t->coeffs;

            const nmod_poly_struct * b = nmod_poly_mat_entry(B, i, j);
            nn_ptr c = Cdata + (i * k + j) * prec;
            slong len = FLINT_MIN(b->length, prec);
            _nmod_vec_set(c, b->coeffs, len);
            _nmod_vec_zero(c + len, prec - len);
            C[i * k + j] = c;
        }

    _solve_dc_ctx_struct ctx;
    ctx.A = A;
    ctx.S = S;
    ctx.n = n;
    ctx.k = k;
    ctx.s0 = s0;
    ctx.acc = _nmod_vec_init(prec);
    ctx.tmp = _nmod_vec_init(prec);
    nmod_init(&ctx.mod, prime);

    _solve_dc_rec(Z, C, prec, &ctx);

    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < k; j++)
        {
            nmod_poly_struct * t = nmod_poly_mat_entry(T, i, j);
            _nmod_poly_set_length(t, prec);
            _nmod_poly_normalise(t);
        }

    nmod_poly_mat_swap(X, T);

    _nmod_vec_clear(ctx.acc);
    _nmod_vec_clear(ctx.tmp);
    _nmod_vec_clear(Cdata);
    flint_free(Z);
    nmod_poly_mat_clear(T);
    nmod_poly_mat_clear(S);
    return 1;
}

/*------------------------------------------------------------*/
/* generator of a one-dimensional right kernel, by lifting    */
/*------------------------------------------------------------*/

/* [q, p] of minimal degree with q w = p mod x^N (2 x 1 approximant basis of
   [w ; -1], zero shift) and q(0) != 0; returns 0 if no basis row has q(0) != 0 */
static int
_pade_pmbasis(nmod_poly_t q, nmod_poly_t p, const nmod_poly_t w, slong N)
{
    const ulong prime = w->mod.n;
    nmod_poly_mat_t F, P;
    nmod_poly_mat_init(F, 2, 1, prime);
    nmod_poly_mat_init(P, 2, 2, prime);
    nmod_poly_set(nmod_poly_mat_entry(F, 0, 0), w);
    nmod_poly_set_coeff_ui(nmod_poly_mat_entry(F, 1, 0), 0, prime - 1);
    slong shift[2] = {0, 0};
    nmod_poly_mat_pmbasis(P, shift, F, N);

    slong best = -1, bdeg = WORD_MAX;
    for (slong i = 0; i < 2; i++)
    {
        const nmod_poly_struct * qi = nmod_poly_mat_entry(P, i, 0);
        slong dg = FLINT_MAX(nmod_poly_degree(qi), nmod_poly_degree(nmod_poly_mat_entry(P, i, 1)));
        if (nmod_poly_get_coeff_ui(qi, 0) != 0 && dg < bdeg)
        {
            best = i;
            bdeg = dg;
        }
    }
    if (best >= 0)
    {
        nmod_poly_set(q, nmod_poly_mat_entry(P, best, 0));
        nmod_poly_set(p, nmod_poly_mat_entry(P, best, 1));
    }
    nmod_poly_mat_clear(F);
    nmod_poly_mat_clear(P);
    return best >= 0;
}

int
nmod_poly_mat_kernel_vector_lift(nmod_poly_mat_t eta, const nmod_poly_mat_t K,
                                 slong deg_bound, slong s0, flint_rand_t state)
{
    const slong m = K->r, n = K->c;
    const ulong prime = K->modulus;
    nmod_t mod;

    if (m < 1 || n != m + 1)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_kernel_vector_lift: K must be m x (m+1), m >= 1\n");
    if (eta->r != n || eta->c != 1)
        flint_throw(FLINT_ERROR, "nmod_poly_mat_kernel_vector_lift: eta must be (m+1) x 1\n");

    nmod_init(&mod, prime);
    const slong D = nmod_poly_mat_degree(K);
    if (D < 0)
        return 0;

    /* 1. a point a (0 first, then random) and a column c with
          (K without column c)(a) invertible; none: K has rank < m (or bad luck) */
    slong c = -1;
    ulong a = 0;
    {
        nmod_mat_t Ka, Am;
        nmod_mat_init(Ka, m, n, prime);
        nmod_mat_init(Am, m, m, prime);
        for (slong tries = 0; tries < 8 && c < 0; tries++)
        {
            a = (tries == 0) ? 0 : n_randint(state, prime);
            nmod_poly_mat_evaluate_nmod(Ka, K, a);
            for (slong cc = n - 1; cc >= 0 && c < 0; cc--)
            {
                for (slong i = 0; i < m; i++)
                    for (slong j = 0, jj = 0; j < n; j++)
                        if (j != cc)
                            nmod_mat_entry(Am, i, jj++) = nmod_mat_entry(Ka, i, j);
                if (nmod_mat_rank(Am) == m)
                    c = cc;
            }
        }
        nmod_mat_clear(Ka);
        nmod_mat_clear(Am);
    }
    if (c < 0)
        return 0;

    /* Kx = K(x + a): a copy only when a != 0 */
    nmod_poly_mat_t Ks;
    const nmod_poly_mat_struct * Kx = K;
    if (a != 0)
    {
        nmod_poly_mat_init(Ks, m, n, prime);
        for (slong i = 0; i < m; i++)
            for (slong j = 0; j < n; j++)
                nmod_poly_taylor_shift(nmod_poly_mat_entry(Ks, i, j), nmod_poly_mat_entry(K, i, j), a);
        Kx = Ks;
    }

    /* 2. A = Kx without column c, as a read-only view on Kx's entries (no copy;
          never cleared as a matrix), and b = -Kx[:, c] */
    nmod_poly_struct * Aent = flint_malloc(m * m * sizeof(nmod_poly_struct));
    for (slong i = 0; i < m; i++)
        for (slong jj = 0; jj < m; jj++)
            Aent[i * m + jj] = *nmod_poly_mat_entry(Kx, i, jj < c ? jj : jj + 1);
    nmod_poly_mat_struct Aview;
    Aview.entries = Aent;
    Aview.r = m;
    Aview.c = m;
    Aview.stride = m;
    Aview.modulus = prime;

    nmod_poly_mat_t Bm, Z, E;
    nmod_poly_mat_init(Bm, m, 1, prime);
    nmod_poly_mat_init(Z, m, 1, prime);
    nmod_poly_mat_init(E, n, 1, prime);
    for (slong i = 0; i < m; i++)
        nmod_poly_neg(nmod_poly_mat_entry(Bm, i, 0), nmod_poly_mat_entry(Kx, i, c));

    /* 3. lifting to precision 2B+1, Pade, check; B doubled up to the Cramer
          bound m*D, then a few fresh random combinations */
    slong B = (deg_bound > 0) ? deg_bound : FLINT_MAX(D, 1);
    const slong cap = FLINT_MAX(B, m * D);
    nmod_poly_t w, q, p, t, acc;
    nmod_poly_init(w, prime);
    nmod_poly_init(q, prime);
    nmod_poly_init(p, prime);
    nmod_poly_init(t, prime);
    nmod_poly_init(acc, prime);
    int ok = 0;
    for (slong retries = 0; retries <= 3; )
    {
        const slong N = 2 * B + 1;
        nmod_poly_mat_solve_series_dc(Z, &Aview, Bm, N, s0);

        nmod_poly_zero(w);
        for (slong i = 0; i < m; i++)
        {
            nmod_poly_scalar_mul_nmod(t, nmod_poly_mat_entry(Z, i, 0), n_randint(state, prime));
            nmod_poly_add(w, w, t);
        }
        ok = _pade_pmbasis(q, p, w, N) && nmod_poly_degree(q) <= B && nmod_poly_degree(p) <= B;

        for (slong j = 0; j < n && ok; j++)
        {
            nmod_poly_struct * e = nmod_poly_mat_entry(E, j, 0);
            if (j == c)
                nmod_poly_set(e, q);
            else
            {
                nmod_poly_mullow(e, q, nmod_poly_mat_entry(Z, j < c ? j : j - 1, 0), N);
                if (nmod_poly_degree(e) > B)
                    ok = 0;
            }
        }
        for (slong i = 0; i < m && ok; i++)   /* Kx E = 0 */
        {
            nmod_poly_zero(acc);
            for (slong j = 0; j < n; j++)
            {
                nmod_poly_mul(t, nmod_poly_mat_entry(Kx, i, j), nmod_poly_mat_entry(E, j, 0));
                nmod_poly_add(acc, acc, t);
            }
            ok = nmod_poly_is_zero(acc);
        }

        if (ok)
            break;
        if (B < cap)
            B = FLINT_MIN(2 * B, cap);
        else
            retries++;
    }

    if (ok)
    {
        /* 4. back to x, primitive part, monic on the entry of highest degree
              (lowest index on a tie) */
        if (a != 0)
            for (slong j = 0; j < n; j++)
                nmod_poly_taylor_shift(nmod_poly_mat_entry(E, j, 0), nmod_poly_mat_entry(E, j, 0),
                                       nmod_neg(a, mod));
        nmod_poly_zero(t);
        for (slong j = 0; j < n; j++)
            nmod_poly_gcd(t, t, nmod_poly_mat_entry(E, j, 0));
        if (nmod_poly_degree(t) > 0)
            for (slong j = 0; j < n; j++)
                nmod_poly_div(nmod_poly_mat_entry(E, j, 0), nmod_poly_mat_entry(E, j, 0), t);
        slong best = 0;
        for (slong j = 1; j < n; j++)
            if (nmod_poly_degree(nmod_poly_mat_entry(E, j, 0)) > nmod_poly_degree(nmod_poly_mat_entry(E, best, 0)))
                best = j;
        const nmod_poly_struct * eb = nmod_poly_mat_entry(E, best, 0);
        ulong inv = n_invmod(eb->coeffs[eb->length - 1], prime);
        for (slong j = 0; j < n; j++)
            nmod_poly_scalar_mul_nmod(nmod_poly_mat_entry(E, j, 0), nmod_poly_mat_entry(E, j, 0), inv);
        nmod_poly_mat_swap(eta, E);
    }

    nmod_poly_clear(w);
    nmod_poly_clear(q);
    nmod_poly_clear(p);
    nmod_poly_clear(t);
    nmod_poly_clear(acc);
    nmod_poly_mat_clear(Bm);
    nmod_poly_mat_clear(Z);
    nmod_poly_mat_clear(E);
    flint_free(Aent);
    if (a != 0)
        nmod_poly_mat_clear(Ks);
    return ok;
}
