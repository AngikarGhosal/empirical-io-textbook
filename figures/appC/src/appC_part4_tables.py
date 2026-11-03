"""Print the LaTeX table rows used in Appendix C, Part 4 (C.7 and C.8).

Reads the JSON outputs of appC_rc.py, appC_rc_merger.py and
autos_markups.py (run those first) and prints booktabs rows, so that no
number in the tables is typed by hand. Writes appC_part4_tables.txt.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIRM = {0: "A", 1: "B", 2: "C", 4: "D"}


def num(x, nd=3):
    """Signed number in math mode with nd decimals."""
    s = "%.*f" % (nd, x)
    if s.startswith("-"):
        return "$-%s$" % s[1:]
    return s


def se(x, nd=3):
    return "(%.*f)" % (nd, x)


def main():
    rc = json.loads((HERE / "appC_rc_results.json").read_text())
    mg = json.loads((HERE / "appC_rc_merger_results.json").read_text())
    au = json.loads((HERE / "autos_results.json").read_text())
    lines = []
    out = lines.append

    # ---------------- tab:C-delta0 ----------------
    out("% tab:C-delta0")
    m0 = rc["m0"]
    for k, pid in enumerate(m0["product_id"]):
        out(" & ".join([
            str(pid), FIRM[m0["owner"][k]], str(m0["buyers"][k]),
            num(m0["logit_start"][k]), num(m0["delta"][k]), num(m0["delta_100k"][k]),
            num(m0["delta_true"][k]), num(m0["simulation_part"][k]),
            num(m0["sampling_part"][k]), "%.3f" % m0["approx_sd_log_share"][k],
        ]) + r" \\")

    # ---------------- tab:C-rc-demand ----------------
    out("% tab:C-rc-demand (rows: const, sugar, organic, alpha)")
    lg, r = rc["logit"], rc["rc"]
    names = ["Constant", "Sugar content", "Organic", r"Price coefficient $\alpha$"]
    truth = [-1.0, 0.2, 0.15, 3.0]
    for i, nm in enumerate(names):
        sgn = -1.0 if i == 3 else 1.0
        cells = [nm]
        for b, s in ((lg["iv"]["b"], lg["iv"]["se"]), (r["ols"]["b"], r["ols"]["se"]),
                     (r["iv"]["b"], r["iv"]["se"]), (r["z_only"], r.get("z_only_se"))):
            v = num(sgn * b[i])
            if s is not None:
                v += " " + se(s[i])
            cells.append(v)
        cells.append(num(truth[i], 2))
        out(" & ".join(cells) + r" \\")

    # ---------------- tab:C-rcelast ----------------
    out("% tab:C-rcelast panel A (rc_est) and panel B (true)")
    el = rc["elasticities_m0"]
    pid = m0["product_id"]
    for key in ("rc_est", "true"):
        out("% " + key)
        E = np.array(el[key])
        for k in range(E.shape[0]):
            lab = "Product %d" % pid[k] if k < len(pid) else "Outside good"
            out(lab + " & " + " & ".join(num(v, 2) for v in E[k]) + r" \\")

    # ---------------- diversion and own elasticities ----------------
    out("% own elasticities and diversion to outside good (percent)")
    for key in ("vertical", "logit", "rc_est", "true"):
        own = np.array(el["own"][key])
        div = 100 * np.array(el["diversion_outside"][key])
        out("%s own %s ; div0 %s" % (key, np.round(own, 2), np.round(div, 1)))

    # ---------------- tab:C-merger-m0 ----------------
    out("% tab:C-merger-m0")
    for k, p in enumerate(mg["product_id"]):
        cells = [str(p), FIRM[mg["owner"][k]], "%.4f" % mg["price_pre"][k]]
        for key in ("logit", "rc", "true"):
            cells.append("%.4f" % mg[key]["dp"][k])
        for key in ("logit", "rc", "true"):
            cells.append("%.4f" % mg[key]["upp"][k] if mg[key]["upp"][k] > 0 else "")
        out(" & ".join(cells) + r" \\")
    for key in ("logit", "rc", "true"):
        o = mg[key]
        out("%% %s: mean dp merging %.4f rivals %.4f; outside %.4f -> %.4f; dCS total %.2f; "
            "dprofit merged %.2f; rivals %.2f; dW %.2f; diversion to partner %s" % (
                key, o["mean_dp_merging"], o["mean_dp_rivals"], o["outside_pre"],
                o["outside_post"], o["dcs_total"], o["dprofit_merged_total"],
                o["rival_profit_change_total"], o["dwelfare_total"],
                np.round(o["diversion_to_partner"], 3)))

    # ---------------- tab:C-autos ----------------
    out("% tab:C-autos")
    mk = au["markups"]
    rows = [
        ("Plain logit", "logit", "%s %s" % (num(au["logit"]["price_coef"]),
                                         se(au["logit"]["se_make_cluster"])), "0"),
        ("Instrumented nested logit", "nested_2sls",
         "%s %s" % (num(au["nested_2sls"]["make_cluster"]["price"]["coef"]),
                    se(au["nested_2sls"]["make_cluster"]["price"]["se"])),
         "%s %s" % (num(au["nested_2sls"]["make_cluster"]["ln_within_nest_share"]["coef"]),
                    se(au["nested_2sls"]["make_cluster"]["ln_within_nest_share"]["se"]))),
        ("Survey nested logit", "nested_survey",
         "%s %s" % (num(au["nested_survey"]["price_coef"]), se(au["nested_survey"]["se_two_step"])),
         "%s %s" % (num(au["second_choice"]["sigma_hat"]), se(au["second_choice"]["sigma_boot_se"]))),
        ("Log-price logit", "logprice",
         "%s %s" % (num(au["logprice_logit"]["log_price_coef"]),
                    se(au["logprice_logit"]["se_make_cluster"])), "0"),
        ("Period log-price logit", "logprice_period",
         "%s to %s" % (num(-au["logprice_period"]["alpha_2sls"][0]),
                       num(-au["logprice_period"]["alpha_2sls"][-1])), "0"),
        ("Period log-price logit, LIML", "logprice_period_liml",
         "%s to %s" % (num(-au["logprice_period"]["alpha_liml"][0]),
                       num(-au["logprice_period"]["alpha_liml"][-1])), "0"),
    ]
    for lab, key, coef, sig in rows:
        t = mk[key]["trend"]
        out(" & ".join([lab, coef, sig, "%.3f" % t["lerner_first"], "%.3f" % t["lerner_last"],
                        "%.3f" % t["lerner_ratio_last_first"],
                        "{:,.0f}".format(t["markup_usd_first"]),
                        "{:,.0f}".format(t["markup_usd_last"]),
                        num(t["markup_usd_pct_change"], 1)]) + r" \\")

    # ---------------- tab:C-autos-decomp ----------------
    out("% tab:C-autos-decomp: dlogL = dlog m - dlog p ; dlog m = dlog(alpha m) - dlog alpha")
    for lab, key, _, _ in rows:
        t = mk[key]["trend"]
        dlogm = t["dlog_alpha_markup"] - t["dlog_alpha"]
        out(" & ".join([lab, num(t["dlog_lerner"]), num(dlogm), num(t["dlog_alpha_markup"]),
                        num(-t["dlog_alpha"]), num(-t["dlog_price"]),
                        "%.0f" % (100 * t["dlog_price"] / -t["dlog_lerner"])]) + r" \\")

    # ---------------- tab:C-autos-nl ----------------
    out("% tab:C-autos-nl")
    lab_i = {"count": "Products in type-year", "rival_count": "Rival products in type-year",
             "rival_chars": "Rival characteristics", "differentiation": "Differentiation"}
    lab_t = {"none": "none", "linear": "linear", "quadratic": "quadratic"}
    for g in au["nested_2sls_sensitivity"]:
        out(" & ".join([lab_i[g["instruments"]], lab_t[g["nest_trends"]],
                        "%s %s" % (num(g["price_coef"]), se(g["price_se"])),
                        "%s %s" % (num(g["sigma"]), se(g["sigma_se"])),
                        "%.1f" % g["sw_F"][0], "%.1f" % g["sw_F"][1]]) + r" \\")

    # ---------------- tab:C-autos-periods ----------------
    out("% tab:C-autos-periods")
    lp = au["logprice_period"]
    for k, per in enumerate(lp["periods"]):
        out(" & ".join([per.replace("-", "--"),
                        "%.3f %s" % (lp["alpha_2sls"][k], se(lp["se_2sls_make_cluster"][k])),
                        "%.3f" % lp["alpha_liml"][k], "%.3f" % lp["alpha_ols"][k],
                        "%.3f %s" % (lp["alpha_2sls_rxr_only"][k], se(lp["se_2sls_rxr_only"][k])),
                        "%.1f" % lp["sw_F_rxr_only"][k]]) + r" \\")
    # mean log price by period (sales weighted, from by_year rows)
    by = mk["logprice_period"]["by_year"]
    out("%% sales-weighted mean log price by year (1980, 2018): %.3f %.3f" % (
        by[0]["sw_mean_log_price"], by[-1]["sw_mean_log_price"]))

    text = "\n".join(lines) + "\n"
    (HERE / "appC_part4_tables.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
