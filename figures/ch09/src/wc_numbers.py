"""Arithmetic behind the Whinston and Collins (1992) magnitudes in Section sec:09-wc.

All inputs are printed in the paper (Tables 1-2, Sections 3-4, footnotes 26-30).
Output: figures/ch09/src/wc_numbers.txt
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = []


def p(s):
    out.append(s)
    print(s)


# Table 1 (one-, two-, three-day windows): SM coefficient and s.e., dollars per annualized seat-mile.
t1 = {"one": (-0.0109, 0.0035), "two": (-0.0147, 0.0049), "three": (-0.0191, 0.0061)}
for w, (b, se) in t1.items():
    p(f"Table 1 {w}-day: t = {b / se:.2f}")

# Implied annual seat-miles of the average incumbent from the paper's $3.7m and $6.5m.
sm_bar_lo = 3.7e6 / 0.0109
sm_bar_hi = 6.5e6 / 0.0191
p(f"implied mean incumbent annual seat-miles: {sm_bar_lo / 1e6:.1f}m (one-day), {sm_bar_hi / 1e6:.1f}m (three-day)")
prod_means = 7100 * 892 * 52
p(f"product of means 7,100 x 892 x 52 = {prod_means / 1e6:.1f}m seat-miles; "
  f"x .0109 = ${0.0109 * prod_means / 1e6:.2f}m; x .0191 = ${0.0191 * prod_means / 1e6:.2f}m")

# Per event, all incumbents: 1.68 incumbents per event with incumbents.
for loss in (3.7, 6.5):
    p(f"1.68 x {loss} = {1.68 * loss:.2f}m per event")
p(f"1.68 x 19/24 = {1.68 * 19 / 24:.3f} (compare 1.33 incumbents per event over all 24 events)")
p(f"1.33 x 7,100 x 892 = {1.33 * 7100 * 892 / 1e6:.2f}m weekly seat-miles (paper: 8.718m)")

# Pretax at a 46% tax rate.
for loss in (3.7, 6.5):
    p(f"pretax {loss}/(1-.46) = {loss / 0.54:.2f}m")

# Relative to annual revenue per seat-mile of .081.
for b in (0.0109, 0.0191):
    frac = b / 0.54 / 0.081
    p(f"pretax loss per seat-mile {b}/.54 = {b / 0.54:.4f}; / .081 = {frac:.3f} years = {12 * frac:.1f} months")

# Table 2, column 5.
b5, se5 = -0.0126, 0.0058
p(f"Table 2 col 5 SM t = {b5 / se5:.2f}")
fe_sum2, fe_sum5 = -30841, -36238          # $1,000s
p(f"general effect per firm: col 2 {fe_sum2 / 19:.0f}k, col 5 {fe_sum5 / 19:.0f}k")
p(f"t of FE sums: col 2 {fe_sum2 / 17621:.2f}, col 5 {fe_sum5 / 19136:.2f}")
inc5 = 0.0126 * sm_bar_lo / 1e6
p(f"col 5 incumbency effect, mean incumbent: .0126 x {sm_bar_lo / 1e6:.1f}m = ${inc5:.2f}m")
p(f"incumbent total = {inc5:.2f} + {-fe_sum5 / 19 / 1e3:.2f} = {inc5 - fe_sum5 / 19 / 1e3:.2f}m; "
  f"ratio = {(inc5 - fe_sum5 / 19 / 1e3) / (-fe_sum5 / 19 / 1e3):.2f}; "
  f"general share of incumbent loss = {(-fe_sum5 / 19 / 1e3) / (inc5 - fe_sum5 / 19 / 1e3):.2f}")
# Industry total per event.
sm_event = 8.718e6 * 52
inc_event = 0.0126 * sm_event / 1e6
p(f"incumbency loss per event (all incumbents) = .0126 x 52 x 8.718m = ${inc_event:.2f}m; "
  f"general total = ${-fe_sum5 / 1e3:.2f}m; general share = {(-fe_sum5 / 1e3) / (-fe_sum5 / 1e3 + inc_event):.2f}")
# Weighted alternative (fn. 29): general loss 3.75m for the average incumbent.
p(f"fn 29: ({inc5:.2f} + 3.75)/3.75 = {(inc5 + 3.75) / 3.75:.2f}")

# EPORT: -33.8 dollars per annual departure.
p(f"EPORT: 33.8 x 3,444 = ${33.8 * 3444 / 1e6:.3f}m; 33.8 x 29,616 = ${33.8 * 29616 / 1e6:.3f}m")

# SM vs OSM, column 5, ignoring the covariance of the two estimates.
d = -0.0126 - 0.0026
se_d = (0.0058 ** 2 + 0.0029 ** 2) ** 0.5
p(f"SM - OSM = {d:.4f}, se (zero covariance) = {se_d:.4f}, z = {d / se_d:.2f}")

# Averages across windows (fn. 28).
p(f"col 5 window average = {(-0.0126 - 0.0154 - 0.0142) / 3:.4f}; Table 1 average = {(-0.0109 - 0.0147 - 0.0191) / 3:.4f}")

# Table 2 t statistics.
t2 = {"(2)": (-0.0127, 0.0040), "(3)": (-0.0158, 0.0053), "(4)": (-0.0144, 0.0054)}
for c, (b, se) in t2.items():
    p(f"Table 2 col {c} SM t = {b / se:.2f}")

(HERE / "wc_numbers.txt").write_text("\n".join(out) + "\n")
