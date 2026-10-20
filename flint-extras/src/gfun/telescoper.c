/*
    Creative telescoping for bivariate rational functions (CTBivRatFun, telescopers),
    step 1 (2026-10-19): the map T of Hermite reduction, fraction free, and its T context.
    (Named ctbivratfun.c until 2026-10-20; the K + lifting driver is in telescoper_naive.c.)

    Setting (pk_main.tex Sec. 3.5.1, "CTBivRatFun"; G2026 Sec. 9.1): f = p/q,
    q square-free in y, r = deg_y q, d = deg_x q, A = k(x)[y]_{<r} with the
    monomial basis. For a in A, Hermite reduction of q_x a / q^2: solve the
    extended Sylvester system (lem:real_T_hermite)

        U q + V q_y = q_x a,   deg_y U, deg_y V < r,

    then q_x a / q^2 = d_y(-V/q) + (U + d_y V)/q, so
    herm(q_x a/q^2) = U + d_y V and T(a) = -herm(q_x a/q^2) = -(U + d_y V).

    Common denominator. The extended Sylvester map M*: (U,V) -> U q + V q_y
    (2r x 2r) has, in its top row (coefficient of y^(2r-1)), the single
    nonzero entry q_r (from y^(r-1) q), hence det M* = +-q_r Res_y(q, q_y),
    of degree <= d + (2r-1)d = 2rd (the bound degM(T) <= 2rd of
    lem:real_T_hermite). With Delta' = q_r Res_y(q, q_y), Cramer gives
    U' = Delta' U, V' = Delta' V polynomial, and Delta' T(a) = -(U' + d_y V').

    Conventions as in algeqtodiffeq.c: q is QT, an (r+1) x 1 nmod_poly_mat of
    its y-coefficients (QT[i] = coefficient of y^i); an element of A is an
    r x 1 nmod_poly_mat.
*/

#include <flint/nmod_poly.h>
#include "nmod_poly_mat_extra.h"
#include "gfun.h"

/* Delta = q_r Res_y(q, q_y) (= +-det M*), CT = Delta (q_x / q_y mod q) (r x 1).
   Delta and CT are initialized here (as nmod_algeqtodiffeq_setup does). */
void nmod_telescoper_setup(nmod_poly_t Delta, nmod_poly_mat_t CT, const nmod_poly_mat_t QT)
{
    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;
    slong d = nmod_poly_mat_degree(QT);

    /* Res = Res_y(q, q_y), iqy = Res q_y^{-1} mod q */
    nmod_poly_t Res;
    nmod_poly_mat_t iqyT, QxT;
    nmod_poly_init(Res, prime);
    nmod_poly_mat_init(iqyT, r, 1, prime);
    nmod_biv_resultant_geometric(Res, iqyT, QT);

    nmod_poly_init(Delta, prime);
    nmod_poly_mul(Delta, Res, nmod_poly_mat_entry(QT, r, 0));

    /* q_r iqy = Delta q_y^{-1} mod q */
    for (slong i = 0; i < r; i++)
        nmod_poly_mul(nmod_poly_mat_entry(iqyT, i, 0), nmod_poly_mat_entry(iqyT, i, 0),
                      nmod_poly_mat_entry(QT, r, 0));

    nmod_poly_mat_init(QxT, r + 1, 1, prime);
    for (slong i = 0; i <= r; i++)
        nmod_poly_derivative(nmod_poly_mat_entry(QxT, i, 0), nmod_poly_mat_entry(QT, i, 0));

    /* CT = q_x (Delta q_y^{-1}) mod q = Delta (q_x/q_y mod q): its entries are, by
       Cramer on M*, minors of size 2r-1 (degree <= (2r-1)d) times q_x (degree <= d-1) */
    nmod_poly_mat_init(CT, r, 1, prime);
    nmod_biv_mulmod_geometric(CT, QxT, iqyT, QT, 2 * r * d + NMOD_GFUN_NONPROPER_MARGIN);

    nmod_poly_clear(Res);
    nmod_poly_mat_clear(iqyT);
    nmod_poly_mat_clear(QxT);
}

/* V' = Delta V = a C mod q, U' = Delta U = (Delta q_x a - V' q_y) / q (exact division in
   k[x][y]), for a in A (AT r x 1), with (Delta, CT) from nmod_telescoper_setup. UT, VT
   r x 1, initialized by the caller. Throws if the division by q is not exact. */
void nmod_telescoper_hermite_UV(nmod_poly_mat_t UT, nmod_poly_mat_t VT, const nmod_poly_mat_t AT,
                                 const nmod_poly_mat_t CT, const nmod_poly_mat_t QT,
                                 const nmod_poly_t Delta)
{
    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;
    slong d = nmod_poly_mat_degree(QT);
    slong da = FLINT_MAX(nmod_poly_mat_degree(AT), 0);

    /* V' = Delta V: by Cramer, minors of M* (degree <= (2r-1)d) times q_x a (degree <= d-1+da) */
    nmod_biv_mulmod_geometric(VT, AT, CT, QT, 2 * r * d + da + NMOD_GFUN_NONPROPER_MARGIN);

    /* numerator Nm = Delta q_x a - V' q_y, as y-coefficients 0..2r-1 */
    slong len = 2 * r;
    nmod_poly_struct * Nm = flint_malloc(len * sizeof(nmod_poly_struct));
    nmod_poly_t t, qx, qy, c;
    nmod_poly_init(t, prime);
    nmod_poly_init(qx, prime);
    nmod_poly_init(qy, prime);
    nmod_poly_init(c, prime);
    for (slong k = 0; k < len; k++)
        nmod_poly_init(Nm + k, prime);
    for (slong j = 0; j <= r; j++)          /* q_x = sum_j (q_j)' y^j */
    {
        nmod_poly_derivative(qx, nmod_poly_mat_entry(QT, j, 0));
        if (nmod_poly_is_zero(qx)) continue;
        nmod_poly_mul(qx, qx, Delta);
        for (slong i = 0; i < r; i++)
        {
            nmod_poly_mul(t, qx, nmod_poly_mat_entry(AT, i, 0));
            nmod_poly_add(Nm + i + j, Nm + i + j, t);
        }
    }
    for (slong j = 1; j <= r; j++)          /* q_y = sum_j j q_j y^(j-1) */
    {
        nmod_poly_scalar_mul_nmod(qy, nmod_poly_mat_entry(QT, j, 0), (ulong) j % prime);
        for (slong i = 0; i < r; i++)
        {
            nmod_poly_mul(t, qy, nmod_poly_mat_entry(VT, i, 0));
            nmod_poly_sub(Nm + i + j - 1, Nm + i + j - 1, t);
        }
    }

    /* exact division by q in y: U'_k = Nm_{k+r} / q_r, then Nm -= U'_k y^k q, k = r-1..0 */
    const nmod_poly_struct * qr = nmod_poly_mat_entry(QT, r, 0);
    for (slong k = r - 1; k >= 0; k--)
    {
        nmod_poly_divrem(nmod_poly_mat_entry(UT, k, 0), c, Nm + k + r, qr);
        if (!nmod_poly_is_zero(c))
            flint_throw(FLINT_ERROR, "nmod_telescoper_hermite_UV: division by q not exact\n");
        for (slong j = 0; j <= r; j++)
        {
            nmod_poly_mul(t, nmod_poly_mat_entry(UT, k, 0), nmod_poly_mat_entry(QT, j, 0));
            nmod_poly_sub(Nm + k + j, Nm + k + j, t);
        }
    }
    for (slong k = 0; k < len; k++)
    {
        if (!nmod_poly_is_zero(Nm + k))
            flint_throw(FLINT_ERROR, "nmod_telescoper_hermite_UV: division by q not exact\n");
        nmod_poly_clear(Nm + k);
    }
    flint_free(Nm);
    nmod_poly_clear(t);
    nmod_poly_clear(qx);
    nmod_poly_clear(qy);
    nmod_poly_clear(c);
}

/* T context (2026-10-20, gfun.h nmod_T_ctx_struct): apply = nmod_telescoper_apply_T */
static void _nmod_telescoper_T_apply(nmod_poly_mat_t R, const nmod_poly_mat_t A, slong D,
                                      const void * data)
{
    (void) D;   /* not needed: nmod_telescoper_hermite_UV sizes its own evaluation */
    const nmod_telescoper_T_data_struct * dat = data;
    nmod_telescoper_apply_T(R, A, dat->CT, dat->QT, dat->Delta);
}

void nmod_telescoper_T_ctx_init(nmod_T_ctx_struct * T, nmod_telescoper_T_data_struct * data,
                                 const nmod_poly_mat_t QT, const nmod_poly_mat_t CT,
                                 const nmod_poly_t Delta)
{
    slong r = QT->r - 1;
    data->QT = QT;
    data->CT = CT;
    data->Delta = Delta;
    T->r = r;
    T->apply = _nmod_telescoper_T_apply;
    T->data = data;
    T->scale = Delta;
    T->deg_bound = 2 * r * nmod_poly_mat_degree(QT);
}

/* RT = Delta T(a) = -(U' + d_y V') for a in A (AT r x 1); RT r x 1, initialized by the caller. */
void nmod_telescoper_apply_T(nmod_poly_mat_t RT, const nmod_poly_mat_t AT, const nmod_poly_mat_t CT,
                              const nmod_poly_mat_t QT, const nmod_poly_t Delta)
{
    ulong prime = nmod_poly_mat_modulus(QT);
    slong r = QT->r - 1;
    nmod_poly_mat_t UT, VT;
    nmod_poly_mat_init(UT, r, 1, prime);
    nmod_poly_mat_init(VT, r, 1, prime);
    nmod_telescoper_hermite_UV(UT, VT, AT, CT, QT, Delta);
    nmod_poly_t t;
    nmod_poly_init(t, prime);
    for (slong i = 0; i < r; i++)
    {
        /* coefficient of y^i in d_y V' is (i+1) V'_{i+1} */
        if (i + 1 < r)
            nmod_poly_scalar_mul_nmod(t, nmod_poly_mat_entry(VT, i + 1, 0), (ulong) (i + 1) % prime);
        else
            nmod_poly_zero(t);
        nmod_poly_add(t, t, nmod_poly_mat_entry(UT, i, 0));
        nmod_poly_neg(nmod_poly_mat_entry(RT, i, 0), t);
    }
    nmod_poly_clear(t);
    nmod_poly_mat_clear(UT);
    nmod_poly_mat_clear(VT);
}
