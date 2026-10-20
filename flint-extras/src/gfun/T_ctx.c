/*
    Generic routines on a T context (nmod_T_ctx_struct, gfun.h, 2026-10-20):
    written once, for any problem providing apply(R, A, D, data) = scale * T(A).
    Ported so far: phi1 (from nmod_phi1, algeqtodiffeq.c), phi1/phi2 (from nmod_phi_T,
    same file), the rank-one pair, the recurrence step and the width-1 description (from find_uv,
    nmod_width1_recurrence_step, nmod_width1_description, algeqtodiffeq_width1.c), the naive pseudo-Krylov
    matrix (from nmod_pseudo_Krylov_naive, algeqtodiffeq_naive.c) and the left
    description of T (from nmod_algeqtodiffeq_T_left_description,
    algeqtodiffeq_recursive.c, with its row sort); the old functions are now
    wrappers building an AlgeqtoDiffeq context. Also the core of Algorithm 4
    (_nmod_T_ctx_width1_core, from _nmod_algeq_to_diffeq_width1, algeqtodiffeq_width1.c).
*/

#include <flint/nmod_poly.h>
#include "nmod_poly_mat_extra.h"
#include "gfun.h"

void nmod_T_ctx_phi1(nmod_poly_t phi1, const nmod_T_ctx_struct * T, flint_rand_t state)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;

    nmod_poly_mat_t randz, Bz;
    nmod_poly_mat_init(randz, r, 1, prime);
    nmod_poly_mat_init(Bz, r, 1, prime);

    /* random constant projection z, drawn exactly as nmod_phi1 always did */
    for (slong i = 0; i < r; i++)
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randz, i, 0), 0,
                                n_randbits(state, FLINT_BITS - 2));

    /* Bz = scale * T(z), fraction free; for AlgeqtoDiffeq (deg_bound = (2r-1)d) the
       bound is nmod_phi1's former (2r-1)d - 1 + margin */
    T->apply(Bz, randz, T->deg_bound - 1 + NMOD_GFUN_NONPROPER_MARGIN, T->data);

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_gcd_hgcd(g, nmod_poly_mat_entry(Bz, 0, 0), T->scale);
    for (slong i = 1; i < r; i++)
        nmod_poly_gcd_hgcd(g, g, nmod_poly_mat_entry(Bz, i, 0));

    nmod_poly_div(phi1, T->scale, g);

    nmod_poly_clear(g);
    nmod_poly_mat_clear(randz);
    nmod_poly_mat_clear(Bz);
}

void nmod_T_ctx_phi_T(nmod_poly_t phi1, nmod_poly_t phi2, const nmod_T_ctx_struct * T,
                      flint_rand_t state)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;
    /* bound: nmod_phi_T's former nmod_gfun_delta_T_degree_bound(r, d) + margin for
       AlgeqtoDiffeq (deg_bound = (2r-1)d) */
    slong D = T->deg_bound - 1 + NMOD_GFUN_NONPROPER_MARGIN;

    nmod_poly_mat_t randT1, randT2;
    nmod_poly_mat_init(randT1, r, 1, prime);
    nmod_poly_mat_init(randT2, r, 1, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT1, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randT2, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
    }

    nmod_poly_mat_t colT1, colT2;
    nmod_poly_mat_init(colT1, r, 1, prime);
    nmod_poly_mat_init(colT2, r, 1, prime);
    T->apply(colT1, randT1, D, T->data);
    T->apply(colT2, randT2, D, T->data);

    nmod_poly_t g;
    nmod_poly_init(g, prime);
    nmod_poly_gcd_hgcd(g, nmod_poly_mat_entry(colT1, 0, 0), T->scale);
    for (slong i = 1; i < r; i++)
        nmod_poly_gcd_hgcd(g, g, nmod_poly_mat_entry(colT1, i, 0));
    nmod_poly_div(phi1, T->scale, g);

    /* phi2: a random 2x2 minor of scale * T, via two random row combinations
       (randU) applied to the same two random columns above */
    nmod_poly_mat_t randU;
    nmod_poly_mat_init(randU, 2, r, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 0, i), 0, n_randtest(state) % prime);
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(randU, 1, i), 0, n_randtest(state) % prime);
    }

    nmod_poly_mat_t P1, P2;
    nmod_poly_mat_init(P1, 2, 1, prime);
    nmod_poly_mat_init(P2, 2, 1, prime);
    nmod_poly_mat_mul(P1, randU, colT1);
    nmod_poly_mat_mul(P2, randU, colT2);

    nmod_poly_t tp1, tp2;
    nmod_poly_init(tp1, prime);
    nmod_poly_init(tp2, prime);
    nmod_poly_mul(tp1, nmod_poly_mat_entry(P1, 0, 0), nmod_poly_mat_entry(P2, 1, 0));
    nmod_poly_mul(tp2, nmod_poly_mat_entry(P1, 1, 0), nmod_poly_mat_entry(P2, 0, 0));
    nmod_poly_sub(tp1, tp1, tp2);

    nmod_poly_div(tp1, tp1, T->scale);
    nmod_poly_gcd_hgcd(g, tp1, T->scale);
    nmod_poly_div(phi2, T->scale, g);

    nmod_poly_clear(g);
    nmod_poly_clear(tp1);
    nmod_poly_clear(tp2);
    nmod_poly_mat_clear(randT1);
    nmod_poly_mat_clear(randT2);
    nmod_poly_mat_clear(colT1);
    nmod_poly_mat_clear(colT2);
    nmod_poly_mat_clear(randU);
    nmod_poly_mat_clear(P1);
    nmod_poly_mat_clear(P2);
}

void nmod_T_ctx_find_uv(nmod_poly_mat_t U, nmod_poly_mat_t V, const nmod_poly_t phi1,
                        const nmod_T_ctx_struct * T, flint_rand_t state)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;
    /* bound for scale * T(e_j): find_uv's former nmod_gfun_delta_T_degree_bound(r, d) + margin
       for AlgeqtoDiffeq (scale = phi1, deg phi1 <= deg_bound = (2r-1)d) */
    slong D = FLINT_MAX(T->deg_bound, nmod_poly_degree(T->scale)) - 1 + NMOD_GFUN_NONPROPER_MARGIN;

    /* random constant projections z (column) and w (row), drawn as find_uv always did */
    nmod_poly_mat_t z, w;
    nmod_poly_mat_init(z, r, 1, prime);
    nmod_poly_mat_init(w, 1, r, prime);
    for (slong i = 0; i < r; i++)
    {
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(z, i, 0), 0, n_randbits(state, FLINT_BITS - 2));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(w, 0, i), 0, n_randbits(state, FLINT_BITS - 2));
    }

    /* phi1 T, column by column: (scale T(e_j)) / (scale / phi1) */
    nmod_poly_mat_t Tmat, Yk, col;
    nmod_poly_t s, rem;
    nmod_poly_mat_init(Tmat, r, r, prime);
    nmod_poly_mat_init(Yk, r, 1, prime);
    nmod_poly_mat_init(col, r, 1, prime);
    nmod_poly_init(s, prime);
    nmod_poly_init(rem, prime);
    nmod_poly_div(s, T->scale, phi1);
    int s_is_one = nmod_poly_is_one(s);
    for (slong j = 0; j < r; j++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_zero(nmod_poly_mat_entry(Yk, i, 0));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Yk, j, 0), 0, 1);

        T->apply(col, Yk, D, T->data);

        for (slong i = 0; i < r; i++)
        {
            if (s_is_one)
                nmod_poly_set(nmod_poly_mat_entry(Tmat, i, j), nmod_poly_mat_entry(col, i, 0));
            else
            {
                nmod_poly_divrem(nmod_poly_mat_entry(Tmat, i, j), rem, nmod_poly_mat_entry(col, i, 0), s);
                if (!nmod_poly_is_zero(rem))
                    flint_throw(FLINT_ERROR, "nmod_T_ctx_find_uv: phi1 is not a denominator "
                                "of T (division by scale/phi1 not exact)\n");
            }
        }
    }

    /* U = T z, V = w T (Lemma 3.3's u, v^T) */
    nmod_poly_mat_multiply(U, Tmat, z);
    nmod_poly_mat_multiply(V, w, Tmat);

    /* e = w T z = w U; V := V e^{-1} mod phi1, so that U V = T mod phi1 */
    nmod_poly_mat_t e;
    nmod_poly_mat_init(e, 1, 1, prime);
    nmod_poly_mat_multiply(e, w, U);

    nmod_poly_t einv;
    nmod_poly_init(einv, prime);
    nmod_poly_set(einv, nmod_poly_mat_entry(e, 0, 0));
    nmod_poly_invmod(einv, einv, phi1);

    for (slong i = 0; i < r; i++)
    {
        nmod_poly_mul(nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(V, 0, i), einv);
        nmod_poly_rem(nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(V, 0, i), phi1);
    }

    nmod_poly_mat_clear(z);
    nmod_poly_mat_clear(w);
    nmod_poly_mat_clear(Tmat);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(e);
    nmod_poly_clear(einv);
    nmod_poly_clear(s);
    nmod_poly_clear(rem);
}

void _nmod_T_ctx_width1_recurrence_step(nmod_poly_mat_t N_next, nmod_poly_t beta_next,
                                        const nmod_poly_mat_t N_prev, const nmod_poly_mat_t U,
                                        const nmod_poly_mat_t V, const nmod_poly_t phi1,
                                        const nmod_T_ctx_struct * T, slong D)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;

    nmod_poly_t tpol, s, rem;
    nmod_poly_init(tpol, prime);
    nmod_poly_init(s, prime);
    nmod_poly_init(rem, prime);

    nmod_poly_mul(beta_next, nmod_poly_mat_entry(V, 0, 0), nmod_poly_mat_entry(N_prev, 0, 0));
    for (slong i = 1; i < r; i++)
    {
        nmod_poly_mul(tpol, nmod_poly_mat_entry(V, 0, i), nmod_poly_mat_entry(N_prev, i, 0));
        nmod_poly_add(beta_next, beta_next, tpol);
    }
    nmod_poly_rem(beta_next, beta_next, phi1);
    nmod_poly_neg(beta_next, beta_next);

    /* temp = phi1 T(N_prev) = (scale T(N_prev)) / (scale / phi1) */
    nmod_poly_mat_t temp;
    nmod_poly_mat_init(temp, r, 1, prime);
    T->apply(temp, N_prev, D, T->data);
    nmod_poly_div(s, T->scale, phi1);
    if (!nmod_poly_is_one(s))
        for (slong i = 0; i < r; i++)
        {
            nmod_poly_divrem(nmod_poly_mat_entry(temp, i, 0), rem, nmod_poly_mat_entry(temp, i, 0), s);
            if (!nmod_poly_is_zero(rem))
                flint_throw(FLINT_ERROR, "_nmod_T_ctx_width1_recurrence_step: phi1 is not a "
                            "denominator of T (division by scale/phi1 not exact)\n");
        }

    for (slong i = 0; i < r; i++)
    {
        nmod_poly_derivative(tpol, nmod_poly_mat_entry(N_prev, i, 0));
        nmod_poly_mul(tpol, tpol, phi1);
        nmod_poly_add(tpol, tpol, nmod_poly_mat_entry(temp, i, 0));

        nmod_poly_mul(nmod_poly_mat_entry(N_next, i, 0), beta_next, nmod_poly_mat_entry(U, i, 0));
        nmod_poly_add(nmod_poly_mat_entry(N_next, i, 0), nmod_poly_mat_entry(N_next, i, 0), tpol);
        nmod_poly_div(nmod_poly_mat_entry(N_next, i, 0), nmod_poly_mat_entry(N_next, i, 0), phi1);
    }

    nmod_poly_mat_clear(temp);
    nmod_poly_clear(tpol);
    nmod_poly_clear(s);
    nmod_poly_clear(rem);
}

void nmod_T_ctx_width1_description(nmod_poly_mat_t NN, nmod_poly_mat_t DD, const ulong n,
                                   const nmod_poly_mat_t U, const nmod_poly_mat_t V,
                                   const nmod_poly_t phi1, const nmod_T_ctx_struct * T,
                                   const nmod_poly_t alpha, const nmod_poly_mat_t N_ini)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;
    slong deg_phi1 = nmod_poly_degree(phi1);

    /* bound for phi1 T(N_{j-1}): nmod_width1_description's former
       nmod_gfun_delta_T_degree_bound(r, d) + deg(phi1) + margin for AlgeqtoDiffeq
       (scale = phi1, deg phi1 <= deg_bound = (2r-1)d); see its doc for the margin */
    slong D = FLINT_MAX(T->deg_bound, nmod_poly_degree(T->scale)) - 1 + deg_phi1
              + NMOD_GFUN_NONPROPER_MARGIN;

    /* column 0: the base case, given directly */
    nmod_poly_set(nmod_poly_mat_entry(DD, 0, 0), alpha);
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(NN, i, 0), nmod_poly_mat_entry(N_ini, i, 0));

    nmod_poly_mat_t N_prev, N_next;
    nmod_poly_mat_init(N_prev, r, 1, prime);
    nmod_poly_mat_init(N_next, r, 1, prime);

    nmod_poly_t b, tpol;
    nmod_poly_init(b, prime);
    nmod_poly_init(tpol, prime);

    for (slong j = 1; j < (slong) n; j++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(N_prev, i, 0), nmod_poly_mat_entry(NN, i, j - 1));

        _nmod_T_ctx_width1_recurrence_step(N_next, b, N_prev, U, V, phi1, T, D);

        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(NN, i, j), nmod_poly_mat_entry(N_next, i, 0));

        /* column j of DD: beta_{1,j+1} = b_{j+1} + d/dx(beta_{1,j});
           beta_{i+1,j+1} = beta_{i,j} + d/dx(beta_{i+1,j}) for 1 <= i <= j */
        nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, 0, j - 1));
        nmod_poly_add(nmod_poly_mat_entry(DD, 0, j), b, tpol);

        for (slong i = 1; i <= j; i++)
        {
            nmod_poly_derivative(tpol, nmod_poly_mat_entry(DD, i, j - 1));
            nmod_poly_add(nmod_poly_mat_entry(DD, i, j), nmod_poly_mat_entry(DD, i - 1, j - 1), tpol);
        }
    }

    nmod_poly_mat_clear(N_prev);
    nmod_poly_mat_clear(N_next);
    nmod_poly_clear(b);
    nmod_poly_clear(tpol);
}

void nmod_T_ctx_pseudo_Krylov_naive(nmod_poly_mat_t K, ulong n, const nmod_T_ctx_struct * T,
                                    const nmod_poly_mat_t a, const nmod_poly_t phi1)
{
    if (n == 0)
        flint_throw(FLINT_DOMERR, "nmod_T_ctx_pseudo_Krylov_naive: n must be >= 1\n");

    ulong prime = T->scale->mod.n;
    slong r = T->r;

    nmod_poly_mat_t tempN, temp;
    nmod_poly_mat_init(tempN, r, 1, prime);
    nmod_poly_mat_init(temp, r, 1, prime);

    nmod_poly_t p1, p2, dphi1, s;
    nmod_poly_init(p1, prime);
    nmod_poly_init(p2, prime);
    nmod_poly_init(dphi1, prime);
    nmod_poly_init(s, prime);
    nmod_poly_derivative(dphi1, phi1);

    slong deg_phi1 = nmod_poly_degree(phi1);
    slong deg_a = FLINT_MAX(nmod_poly_mat_degree(a), 0);
    slong deg_scale = nmod_poly_degree(T->scale);

    /* s = scale / phi1: phi1 T(.) = (scale T(.)) / s; s = 1 when scale = phi1 */
    nmod_poly_div(s, T->scale, phi1);
    int s_is_one = nmod_poly_is_one(s);
    int s_is_const = (nmod_poly_degree(s) == 0);
    ulong s_inv = s_is_const ? n_invmod(nmod_poly_get_coeff_ui(s, 0), prime) : 0;

    /* K_0 = a */
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(K, i, 0), nmod_poly_mat_entry(a, i, 0));

    for (slong k = 0; k < (slong) n - 1; k++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_set(nmod_poly_mat_entry(tempN, i, 0), nmod_poly_mat_entry(K, i, k));

        /* bound on deg(scale T(K_k)), deg K_k <= k deg phi1 + deg a; for AlgeqtoDiffeq
           (scale = phi1, a = y) this is nmod_pseudo_Krylov_naive's former
           k deg phi1 + deg phi1 + margin, see its doc for the margin */
        slong D = k * deg_phi1 + deg_a + deg_scale + NMOD_GFUN_NONPROPER_MARGIN;

        T->apply(temp, tempN, D, T->data);

        for (slong i = 0; i < r; i++)
        {
            nmod_poly_struct * Ki1 = nmod_poly_mat_entry(K, i, k + 1);
            if (s_is_one)
                nmod_poly_set(Ki1, nmod_poly_mat_entry(temp, i, 0));
            else if (s_is_const)
                nmod_poly_scalar_mul_nmod(Ki1, nmod_poly_mat_entry(temp, i, 0), s_inv);
            else
            {
                nmod_poly_divrem(Ki1, p1, nmod_poly_mat_entry(temp, i, 0), s);
                if (!nmod_poly_is_zero(p1))
                    flint_throw(FLINT_ERROR, "nmod_T_ctx_pseudo_Krylov_naive: phi1 is not "
                                "a denominator of T (division by scale/phi1 not exact)\n");
            }

            nmod_poly_derivative(p1, nmod_poly_mat_entry(K, i, k));
            nmod_poly_mul(p1, p1, phi1);

            nmod_poly_mul(p2, dphi1, nmod_poly_mat_entry(K, i, k));
            nmod_poly_scalar_mul_nmod(p2, p2, k);

            nmod_poly_add(Ki1, Ki1, p1);
            nmod_poly_sub(Ki1, Ki1, p2);
        }
    }

    nmod_poly_mat_clear(temp);
    nmod_poly_mat_clear(tempN);
    nmod_poly_clear(dphi1);
    nmod_poly_clear(p1);
    nmod_poly_clear(p2);
    nmod_poly_clear(s);
}

/* Rows of a left description (Q, P) by increasing row degree of Q (stable
 * insertion sort; a row permutation, so still an irreducible reduced
 * description). Used by nmod_T_ctx_left_description (hence
 * nmod_algeqtodiffeq_T_left_description) and nmod_algeqtodiffeq_Tb_left_description. pmbasis returns the rows of degree
 * ceil(delta/r) first; with that order the first shifted kernel of
 * nmod_pseudo_Krylov_recursive (shift t = rdeg Q) returns a Q_2 that is
 * t-reduced but not row reduced when r does not divide delta = deg det Q:
 * sum rdeg Q_2 - delta = min(delta mod r, (r - 2 delta mod r) mod r),
 * deg Q_2 = ceil(delta/r)+1. The kernel puts its larger shifted degrees on
 * the last rows; with Q's high rows last as well, the low-shift columns lie
 * left of the high pivots, where the weak Popov bound is strict, and Q_2 is
 * row reduced (observed, r = 2..12, d = 1..12; claude-pseudoKrylov
 * ongoing-M2Pro.md, 2026-10-16). */
void _nmod_poly_mat_description_sort_rows(nmod_poly_mat_t Q, nmod_poly_mat_t P)
{
    slong n = Q->r;
    slong * rd = flint_malloc(n * sizeof(slong));
    nmod_poly_mat_row_degree(rd, Q, NULL);
    for (slong i = 1; i < n; i++)
        for (slong k = i; k > 0 && rd[k - 1] > rd[k]; k--)
        {
            FLINT_SWAP(slong, rd[k - 1], rd[k]);
            for (slong j = 0; j < n; j++)
            {
                nmod_poly_swap(nmod_poly_mat_entry(Q, k - 1, j), nmod_poly_mat_entry(Q, k, j));
                nmod_poly_swap(nmod_poly_mat_entry(P, k - 1, j), nmod_poly_mat_entry(P, k, j));
            }
        }
    flint_free(rd);
}

void nmod_T_ctx_left_description(nmod_poly_mat_t Q, nmod_poly_mat_t P,
                                 const nmod_T_ctx_struct * T, const nmod_poly_t phi1)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;

    /* (a) Tmat = phi1 T, column j = (scale T(e_j)) / (scale / phi1); bound for apply
       deg_bound - 1 + deg scale (AlgeqtoDiffeq: (2r-1)d - 1 + deg phi1, as before) */
    slong Dbound = T->deg_bound - 1 + nmod_poly_degree(T->scale);

    nmod_poly_mat_t Tmat, Yk, col;
    nmod_poly_t s, rem;
    nmod_poly_mat_init(Tmat, r, r, prime);
    nmod_poly_mat_init(Yk, r, 1, prime);
    nmod_poly_mat_init(col, r, 1, prime);
    nmod_poly_init(s, prime);
    nmod_poly_init(rem, prime);
    nmod_poly_div(s, T->scale, phi1);
    int s_is_one = nmod_poly_is_one(s);
    slong live = 0;   /* nonzero columns (AlgeqtoDiffeq: r - 1, T(1) = 0) */
    for (slong j = 0; j < r; j++)
    {
        for (slong i = 0; i < r; i++)
            nmod_poly_zero(nmod_poly_mat_entry(Yk, i, 0));
        nmod_poly_set_coeff_ui(nmod_poly_mat_entry(Yk, j, 0), 0, 1);

        T->apply(col, Yk, Dbound, T->data);

        int nonzero = 0;
        for (slong i = 0; i < r; i++)
        {
            if (s_is_one)
                nmod_poly_set(nmod_poly_mat_entry(Tmat, i, j), nmod_poly_mat_entry(col, i, 0));
            else
            {
                nmod_poly_divrem(nmod_poly_mat_entry(Tmat, i, j), rem, nmod_poly_mat_entry(col, i, 0), s);
                if (!nmod_poly_is_zero(rem))
                    flint_throw(FLINT_ERROR, "nmod_T_ctx_left_description: phi1 is not a "
                                "denominator of T (division by scale/phi1 not exact)\n");
            }
            if (!nmod_poly_is_zero(nmod_poly_mat_entry(Tmat, i, j)))
                nonzero = 1;
        }
        live += nonzero;
    }
    if (live == 0)
        live = 1;

    /* (b) Irreducible left description via a truncated approximant basis.
     *
     * 2026-10-19: target from max(deg phi1, (2r-1)d) instead of deg phi1 alone,
     * (2r-1)d being the bound degM(T) <= (2r-1)d (G2026 Lem. 4, pk_main.tex
     * lem:real_T_algeqtodiffe). deg phi1 = degM(T) only when T is proper of
     * width 1 (the generic case, where deg phi1 = (2r-1)d and nothing changes);
     * when width(T) > 1 (e.g. trinomials p_r y^r + p_1 y + p_0: deg phi1 =
     * (r+1)d, degM(T) = (2r-1)d) the old target was too small and selected rows
     * that were only approximants, not relations -- LeftDesc then returned a
     * WRONG operator, silently. The selected rows are now checked (Q (phi1 T) v
     * = phi1 P v for a random constant vector v: a wrong (Q, P) passes with
     * probability <= 1/p), and on failure the target and sigma are doubled, up
     * to 4 times the bound; this also covers non-proper T (e.g. P of total
     * degree d), where degM(T) has a part at infinity that deg phi1 misses. */
    slong deg_phi1 = nmod_poly_degree(phi1);
    slong deg_bound = FLINT_MAX(deg_phi1, T->deg_bound);
    // Counting on the nonzero columns only (AlgeqtoDiffeq: one zero column in T, r - 1 live)
    slong target_degree = (deg_bound + NMOD_GFUN_NONPROPER_MARGIN + live - 1) / live; /* ceil */
    const slong target_cap = 4 * (deg_bound + NMOD_GFUN_NONPROPER_MARGIN);

    nmod_poly_mat_t B, ker, v, Tv, QTv, Pv;
    nmod_poly_mat_init(B, 2 * r, r, prime);
    nmod_poly_mat_init(ker, 2 * r, 2 * r, prime);
    nmod_poly_mat_init(v, r, 1, prime);
    nmod_poly_mat_init(Tv, r, 1, prime);
    nmod_poly_mat_init(QTv, r, 1, prime);
    nmod_poly_mat_init(Pv, r, 1, prime);
    slong * shift = flint_malloc(2 * r * sizeof(slong));
    flint_rand_t vstate;
    flint_rand_init(vstate);
    flint_rand_set_seed(vstate, (ulong) deg_phi1 + 1, (ulong) prime);

    int exact = 0;
    while (1)
    {
        slong sigma = (2 * r * target_degree + r - 1) / r + 1; /* ceil(2r*target_degree/r) + 1 */
        for (slong i = 0; i < r; i++)
        {
            for (slong j = 0; j < r; j++)
            {
                nmod_poly_neg(nmod_poly_mat_entry(B, i, j), nmod_poly_mat_entry(Tmat, i, j));
                nmod_poly_truncate(nmod_poly_mat_entry(B, i, j), sigma);
            }
            nmod_poly_set_trunc(nmod_poly_mat_entry(B, i + r, i), phi1, sigma);
        }
        for (slong i = 0; i < 2 * r; i++)
            shift[i] = 0;
        nmod_poly_mat_pmbasis(ker, shift, B, sigma);

        slong nbrows = 0;
        for (slong i = 0; i < 2 * r && nbrows < r; i++)
        {
            if (shift[i] <= target_degree)
            {
                for (slong j = 0; j < r; j++)
                {
                    nmod_poly_set(nmod_poly_mat_entry(Q, nbrows, j), nmod_poly_mat_entry(ker, i, j));
                    nmod_poly_set(nmod_poly_mat_entry(P, nbrows, j), nmod_poly_mat_entry(ker, i, j + r));
                }
                nbrows++;
            }
        }

        if (nbrows == r)
        {
            /* exactness check: Q (Tmat v) == phi1 (P v), v random constant */
            for (slong i = 0; i < r; i++)
                nmod_poly_set_coeff_ui(nmod_poly_mat_entry(v, i, 0), 0, n_randint(vstate, prime));
            nmod_poly_mat_mul(Tv, Tmat, v);
            nmod_poly_mat_mul(QTv, Q, Tv);
            nmod_poly_mat_mul(Pv, P, v);
            nmod_poly_mat_scalar_mul_nmod_poly(Pv, Pv, phi1);
            exact = nmod_poly_mat_equal(QTv, Pv);
        }
        if (exact || target_degree >= target_cap)
            break;
        target_degree = FLINT_MIN(2 * target_degree, target_cap);
    }
    if (!exact)
        flint_throw(FLINT_ERROR, "nmod_T_ctx_left_description: no exact "
                    "description of degree at most %wd found\n", target_degree);

    /* (c) Rows by increasing degree, see _nmod_poly_mat_description_sort_rows. */
    _nmod_poly_mat_description_sort_rows(Q, P);

    flint_rand_clear(vstate);
    nmod_poly_mat_clear(v);
    nmod_poly_mat_clear(Tv);
    nmod_poly_mat_clear(QTv);
    nmod_poly_mat_clear(Pv);
    flint_free(shift);
    nmod_poly_clear(s);
    nmod_poly_clear(rem);
    nmod_poly_mat_clear(Tmat);
    nmod_poly_mat_clear(Yk);
    nmod_poly_mat_clear(col);
    nmod_poly_mat_clear(B);
    nmod_poly_mat_clear(ker);
}

slong _nmod_T_ctx_width1_core(nmod_poly_mat_t Y, const nmod_poly_mat_t a, const ulong m,
                              const nmod_T_ctx_struct * T, const nmod_poly_t phi1,
                              flint_rand_t state)
{
    ulong prime = T->scale->mod.n;
    slong r = T->r;

    nmod_poly_mat_t U, V;
    nmod_poly_mat_init(U, r, 1, prime);
    nmod_poly_mat_init(V, 1, r, prime);
    nmod_T_ctx_find_uv(U, V, phi1, T, state);

    /* bound for phi1 T(a) in Lemma 3.7's step: _nmod_algeq_to_diffeq_width1's former
       nmod_gfun_delta_T_degree_bound(r, d) + deg(phi1) for AlgeqtoDiffeq (scale = phi1,
       a = y of degree 0) */
    slong D = FLINT_MAX(T->deg_bound, nmod_poly_degree(T->scale)) - 1 + nmod_poly_degree(phi1)
              + FLINT_MAX(nmod_poly_mat_degree(a), 0);

    /* Phase 2: realisations (N,D) of Q (Proposition 3.2) and (F,A) of a
     * (Proposition 3.3, via Lemma 3.7's beta/f). */
    nmod_poly_mat_t N, D_mat;
    nmod_poly_mat_init(N, r, m, prime);
    nmod_poly_mat_init(D_mat, m, m, prime);
    nmod_T_ctx_width1_description(N, D_mat, m, U, V, phi1, T, phi1, U);

    nmod_poly_t beta;
    nmod_poly_init(beta, prime);
    nmod_poly_mat_t f;
    nmod_poly_mat_init(f, r, 1, prime);
    _nmod_T_ctx_width1_recurrence_step(f, beta, a, U, V, phi1, T, D);

    nmod_poly_mat_t F, A;
    nmod_poly_mat_init(F, r, m, prime);
    nmod_poly_mat_init(A, m, m, prime);
    nmod_T_ctx_width1_description(F, A, m, U, V, phi1, T, beta, f);

    /* Phase 3: augmented block matrix P (eq. 19):
     *   Dbar=Diag(1,D), Nbar=[0 N], Abar=Diag(1,A), Fbar=[a F]
     *   P = [[Dbar, Abar], [Nbar, Fbar]], size (r + n1) x 2*n1, n1 = m+1.
     * P is freshly initialized (zero), so only the nonzero entries below
     * need to be set explicitly. */
    slong n1 = m + 1;
    nmod_poly_mat_t P;
    nmod_poly_mat_init(P, r + n1, 2 * n1, prime);

    nmod_poly_one(nmod_poly_mat_entry(P, 0, 0));   /* Dbar[0][0] = 1 */
    nmod_poly_one(nmod_poly_mat_entry(P, 0, n1));  /* Abar[0][0] = 1 */
    for (slong i = 0; i < r; i++)
        nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, n1), nmod_poly_mat_entry(a, i, 0)); /* Fbar[:,0] = a */
    /* Nbar[:,0] stays zero (P's own fresh zero-init). */

    for (slong i = 0; i < (slong) m; i++)
        for (slong j = 0; j < (slong) m; j++)
        {
            nmod_poly_set(nmod_poly_mat_entry(P, 1 + i, 1 + j), nmod_poly_mat_entry(D_mat, i, j));
            nmod_poly_set(nmod_poly_mat_entry(P, 1 + i, n1 + 1 + j), nmod_poly_mat_entry(A, i, j));
        }
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < (slong) m; j++)
        {
            nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, 1 + j), nmod_poly_mat_entry(N, i, j));
            nmod_poly_set(nmod_poly_mat_entry(P, n1 + i, n1 + 1 + j), nmod_poly_mat_entry(F, i, j));
        }

    /* Nullspace of P: W = [Z^T Y^T]^T (Lemma 3.8), Y = W's last n1 rows.
     *
     * `shift` is an explicit ZEROED array, not NULL (changed 2026-09-18, per
     * the user, measured by claude-pseudoKrylov/experiments/
     * bench-width1-kernel-shift.c). NULL does not mean "zero shift": per
     * nmod_poly_mat_kernel.h it means weak Popov with respect to pmat's own
     * COLUMN degrees here (COL_UPPER = right kernel), and P's column degrees
     * run from 0 (its first column, essentially the lone 1 entry) up to
     * deg(phi1) -- a strongly non-uniform shift, the kind that inflates this
     * computation. Uniform zero measured 5-8% faster on the identical P at
     * r,d = 8,5 / 12,8 / 16,10, stable across trials.
     *
     * Same answer either way, and not merely "a different valid basis":
     * generically nz = 1, so the kernel is one-dimensional and its basis
     * vector is unique up to a scalar whatever the shift -- verified equal
     * (after normalization) on every trial.
     *
     * Length is 2*n1 = ncols(P), the COL_UPPER convention (the doc's
     * "number of rows" wording describes the left-kernel orientation).
     * shift is in/out -- the call overwrites it with the output shifted
     * degrees -- so it is freshly zeroed here and not reused as an input. */
    slong *pivind = flint_malloc(2 * n1 * sizeof(slong));
    slong *shift = flint_malloc(2 * n1 * sizeof(slong));
    for (slong i = 0; i < 2 * n1; i++)
        shift[i] = 0;
    nmod_poly_mat_t W;
    nmod_poly_mat_init(W, 2 * n1, 2 * n1, prime);
    slong nz = nmod_poly_mat_kernel(W, pivind, shift, P, ORD_WEAK_POPOV, COL_UPPER);
    flint_free(pivind);
    flint_free(shift);

    for (slong i = 0; i < n1; i++)
        for (slong j = 0; j < n1; j++)
            nmod_poly_set(nmod_poly_mat_entry(Y, i, j), nmod_poly_mat_entry(W, n1 + i, j));

    nmod_poly_mat_clear(U);
    nmod_poly_mat_clear(V);
    nmod_poly_mat_clear(N);
    nmod_poly_mat_clear(D_mat);
    nmod_poly_mat_clear(f);
    nmod_poly_mat_clear(F);
    nmod_poly_mat_clear(A);
    nmod_poly_mat_clear(P);
    nmod_poly_mat_clear(W);
    nmod_poly_clear(beta);

    return nz;
}
