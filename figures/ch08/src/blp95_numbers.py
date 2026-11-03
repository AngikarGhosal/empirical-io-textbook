"""Derived quantities used in the BLP (1995) case study (Section 8.12).

All inputs come from blp95_data.py (transcribed from the paper).  Every printed number that the
text uses is computed here.  Run:  python blp95_numbers.py
"""
import numpy as np
from blp95_data import (CARS, PARENT, PRICE, OWN_ELAS, SEMI, OUT_LOGIT, OUT_BLP, MARKUP,
                        VARPROF, T9, ALPHA_OLS, SE_OLS, ALPHA_IV, SE_IV, ALPHA_CD, GAMMA_Q,
                        IVLOGIT_MARKUP_RANGE)

p = np.array(PRICE)            # $1000
eps = np.array(OWN_ELAS)
S = np.array(SEMI)
mk = np.array(MARKUP) / 1000.0  # $1000
vp = np.array(VARPROF)          # $1000

print("1. Own elasticities implied by the Table VI diagonal: -semi * p / 100")
for c, pj, sj, e in zip(CARS, p, np.diag(S), eps):
    print(f"   {c:18s} from VI {-sj * pj / 100:6.3f}   Table V {e:6.3f}")

print("\n2. Lerner index, inverse elasticity, gap (multi-product term), implied sales")
for c, f, pj, m, e, v in zip(CARS, PARENT, p, mk, eps, vp):
    L = m / pj
    print(f"   {c:18s} {f:7s} L={L:.3f} 1/eps={1/e:.3f} gap={L-1/e:+.3f} "
          f"q={v/m:8.0f} cars")

print("\n3. Logit inelasticity cutoffs p < 1/alpha (thousand 1983 $)")
for a, se, lab in [(ALPHA_OLS, SE_OLS, "OLS"), (ALPHA_IV, SE_IV, "IV")]:
    print(f"   {lab}: alpha={a} cutoff={1/a:.3f}; alpha-2se={a-2*se:.3f} alpha+2se={a+2*se:.3f}")
    if a - 2 * se > 0:
        print(f"       cutoffs at +/-2se: {1/(a+2*se):.3f} {1/(a-2*se):.3f}")

print("\n4. IV-logit markups: single-product 1/alpha, GM share from 1/(alpha(1-S_f))")
print(f"   1/alpha_IV = {1/ALPHA_IV*1000:.0f} dollars")
lo, hi = IVLOGIT_MARKUP_RANGE
print(f"   BMW: 1 - 1/(alpha*{lo/1000}) = {1-1/(ALPHA_IV*lo/1000):.4f}")
print(f"   GM : 1 - 1/(alpha*{hi/1000}) = {1-1/(ALPHA_IV*hi/1000):.4f}")

print("\n5. Implied outside share and market size, 1990")
sales1990 = 131 * 66.377  # thousand cars (Table I)
print(f"   1990 sales = {sales1990:.0f} thousand")
for s0 in [min(OUT_LOGIT) / 100, max(OUT_LOGIT) / 100]:
    print(f"   s0={s0:.4f}: M = {sales1990/(1-s0)/1000:.1f} million")

print("\n6. ln q specification: AVC = mc/(1+gamma_q)")
print(f"   1/(1+gamma_q) = {1/(1+GAMMA_Q):.3f}")
for c, pj, row in zip(CARS, p, T9):
    m2 = row[1] / 1000
    mc = pj - m2
    avc = mc / (1 + GAMMA_Q)
    print(f"   {c:18s} p={pj:6.3f} mc={mc:6.3f} AVC={avc:6.3f} p-AVC={pj-avc:+.3f}")

print("\n7. Upper bound on the harmonic-mean residual income of buyers: alpha*100/|semi_jj| (thousand $)")
for c, sj in zip(CARS, np.diag(S)):
    hm = ALPHA_CD * 100 / abs(sj)
    print(f"   {c:18s} y-p <= {hm:7.1f}")

print("\n8. Logit benchmark semi-elasticities (IV logit): own -100*alpha(1-s) ~", -100 * ALPHA_IV)

print("\n9. Table VI: largest cross entry in each column (closest substitute by column)")
for j, c in enumerate(CARS):
    col = S[:, j].copy()
    col[j] = -np.inf
    i = int(np.argmax(col))
    print(f"   price of {c:18s} -> largest gain {CARS[i]} ({col[i]:.3f})")

print("\n10. Table VI: column max/min ratio of cross entries (where min > 0)")
for j, c in enumerate(CARS):
    col = np.delete(S[:, j], j)
    pos = col[col > 0]
    print(f"   {c:18s} max={col.max():.3f} min={col.min():.3f} "
          f"ratio={'inf' if pos.size < col.size else col.max()/col.min():}")

print("\n11. Cross elasticities from semi-elasticities: semi_ij * p_j / 100 (examples)")
for (i, j) in [(1, 0), (0, 3), (10, 9), (12, 11), (4, 5), (5, 4)]:
    print(f"   share {CARS[i]} wrt price {CARS[j]}: {S[i, j] * p[j] / 100:.4f}")

print("\n12. Variable profit ranking")
for k in np.argsort(-vp):
    print(f"   {CARS[k]:18s} {vp[k]:8.0f}")

print("\n13. Markup range across Table IX columns, ratio max/min per car")
T = np.array(T9)
for c, row in zip(CARS, T):
    print(f"   {c:18s} min={row.min():6d} max={row.max():6d} ratio={row.max()/row.min():.2f}")
print("   correlation of each column with base:",
      [f"{np.corrcoef(T[:,0], T[:,k])[0,1]:.3f}" for k in range(T.shape[1])])
print("\n14. Markup/price ratios: min, max", (mk / p).min(), (mk / p).max())
