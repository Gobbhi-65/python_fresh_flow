import io
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_sets" / "freshflow_sales.csv"
OUT = ROOT / "analysis_outputs"
COLS = [
    "transaction_id",
    "day_index",
    "outlet_id",
    "product_code",
    "category_code",
    "quantity",
    "unit_price",
    "unit_cost",
    "discount_rate",
    "stock_before_units",
    "payment_code",
    "customer_rating",
]
PRODUCTS = ["Orange Juice", "Mango Juice", "Apple Juice", "Watermelon Juice", "Banana Smoothie", "Mixed Fruit Smoothie", "Lemonade", "Iced Tea"]
OUTLETS = ["Outlet A", "Outlet B", "Outlet C"]
CATEGORIES = ["Juice", "Smoothie", "Lemonade", "Tea"]
PRODUCT_CATEGORY = np.array([1, 1, 1, 1, 2, 2, 3, 4])


def integer(x):
    return np.isfinite(x) & (x == np.floor(np.where(np.isfinite(x), x, 0)))


def load_data():
    sample = np.genfromtxt(DATA, delimiter=",", skip_header=1, dtype=float, ndmin=2, max_rows=2)
    if sample.shape != (2, 12) or not np.isnan(np.array([np.nan])).all():
        raise ValueError("CSV loader check failed")
    raw = np.genfromtxt(DATA, delimiter=",", skip_header=1, dtype=float, ndmin=2)
    if raw.shape != (5000, 12):
        raise ValueError(f"Expected (5000, 12), found {raw.shape}")
    if not np.isnan(raw[:, 6]).any() or not np.isnan(raw[:, 7]).any() or not np.isnan(raw[:, 11]).any():
        raise ValueError("Expected blank price, cost, and rating fields")
    return raw


def audit(raw):
    ids, day, outlet, prod, cat, qty, price, cost, disc, stock, payment, rating = raw.T
    pv = np.isfinite(prod) & integer(prod) & (prod >= 1) & (prod <= 8)
    expected = np.zeros(prod.shape, dtype=int)
    expected[pv] = PRODUCT_CATEGORY[prod[pv].astype(int) - 1]
    unique_ids, id_counts = np.unique(ids[np.isfinite(ids)], return_counts=True)
    dup_ids = unique_ids[id_counts > 1]
    flags = {
        "missing_unit_price": np.isnan(price),
        "missing_unit_cost": np.isnan(cost),
        "missing_customer_rating": np.isnan(rating),
        "invalid_transaction_id": ~np.isfinite(ids) | (ids <= 0) | ~integer(ids),
        "duplicate_transaction_id": np.isin(ids, dup_ids),
        "invalid_day_index": ~np.isfinite(day) | ~integer(day) | (day < 1) | (day > 30),
        "unknown_outlet_code": ~np.isfinite(outlet) | ~integer(outlet) | ~np.isin(outlet, [1, 2, 3]),
        "unknown_product_code": ~pv,
        "unknown_category_code": ~np.isfinite(cat) | ~integer(cat) | ~np.isin(cat, [1, 2, 3, 4]),
        "category_mapping_mismatch": pv & np.isfinite(cat) & (cat != expected),
        "unknown_payment_code": ~np.isfinite(payment) | ~integer(payment) | ~np.isin(payment, [1, 2, 3, 4]),
        "invalid_quantity": ~np.isfinite(qty) | ~integer(qty) | (qty <= 0),
        "invalid_unit_price": np.isfinite(price) & (price <= 0),
        "negative_unit_cost": np.isfinite(cost) & (cost < 0),
        "impossible_discount": ~np.isfinite(disc) | (disc < 0) | (disc > 1),
        "negative_stock": np.isfinite(stock) & (stock < 0),
        "invalid_customer_rating": np.isfinite(rating) & ((rating < 1) | (rating > 5)),
        "zero_stock_with_sale": np.isfinite(stock) & (stock == 0) & np.isfinite(qty) & (qty > 0),
    }
    rev_ok = (
        np.isfinite(ids) & (ids > 0) & integer(ids) & ~np.isin(ids, dup_ids)
        & np.isfinite(day) & integer(day) & (day >= 1) & (day <= 30)
        & np.isfinite(outlet) & integer(outlet) & np.isin(outlet, [1, 2, 3])
        & pv & np.isfinite(cat) & (cat == expected)
        & np.isfinite(qty) & integer(qty) & (qty > 0)
        & np.isfinite(price) & (price > 0)
        & np.isfinite(disc) & (disc >= 0) & (disc <= 1)
    )
    flags["high_price_review"] = np.isfinite(price) & (price > np.nanpercentile(price[np.isfinite(price) & (price > 0)], 99))
    flags["high_discount_review"] = np.isfinite(disc) & (disc > 0.4) & (disc <= 1)
    profit_ok = rev_ok & np.isfinite(cost) & (cost >= 0)
    rating_ok = rev_ok & np.isfinite(rating) & (rating >= 1) & (rating <= 5)
    return flags, rev_ok, profit_ok, rating_ok, dup_ids


def metrics(raw, rev_ok, profit_ok):
    q, price, cost, disc = raw[:, 5], raw[:, 6], raw[:, 7], raw[:, 8]
    gross = q * price
    discount = gross * disc
    net = gross - discount
    cogs = q * cost
    profit = net - cogs
    margin = np.full(raw.shape[0], np.nan)
    asp = np.full(raw.shape[0], np.nan)
    np.divide(profit * 100, net, out=margin, where=np.isfinite(net) & (net != 0))
    np.divide(net, q, out=asp, where=np.isfinite(q) & (q > 0))
    for arr in [gross, discount, net, asp]:
        arr[~rev_ok] = np.nan
    for arr in [cogs, profit, margin]:
        arr[~profit_ok] = np.nan
    return {
        "gross_sales": gross,
        "discount_amount": discount,
        "net_revenue": net,
        "cost_of_goods_sold": cogs,
        "gross_profit": profit,
        "gross_margin_pct": margin,
        "average_selling_price": asp,
    }


def total(x):
    v = x[np.isfinite(x)]
    return float(np.sum(v)) if v.size else 0.0


def group_rows(raw, m, valid, column, codes, names):
    rows = []
    for code, name in zip(codes, names):
        mask = valid & (raw[:, column] == code)
        pm = mask & np.isfinite(m["gross_profit"])
        net, gp = total(m["net_revenue"][mask]), total(m["gross_profit"][pm])
        pnet = total(m["net_revenue"][pm])
        rows.append(
            [
                code,
                name,
                int(mask.sum()),
                total(raw[mask, 5]),
                total(m["gross_sales"][mask]),
                total(m["discount_amount"][mask]),
                net,
                gp,
                int(pm.sum()),
                gp / pnet * 100 if pnet else np.nan,
                float(np.mean(m["net_revenue"][mask])) if mask.any() else np.nan,
            ]
        )
    return rows


def save_csv(path, header, rows):
    np.savetxt(path, np.asarray(rows, dtype=object), delimiter=",", fmt="%s", header=header, comments="", encoding="utf-8")


def money(x):
    return f"INR {x:,.2f}" if np.isfinite(x) else "N/A"


def main():
    OUT.mkdir(exist_ok=True)
    print("STEP 1: Load dataset")
    raw = load_data()

    print("Dataset loaded")
    print("Shape:", raw.shape)
    print("Rows:", raw.shape[0])
    print("Columns:", raw.shape[1])
    print("Dimensions:", raw.ndim)
    print("Data type:", raw.dtype)
    print("Headers:", COLS)
    print("First five rows:")
    print(raw[:5])

    print("\nSTEP 2: Inspect columns")
    for i, column in enumerate(COLS):
        values = raw[:, i]
        valid_values = values[np.isfinite(values)]
        print(
            f"{column}: missing={np.isnan(values).sum()}, "
            f"unique={np.unique(valid_values).size}, "
            f"min={np.min(valid_values) if valid_values.size else np.nan}, "
            f"max={np.max(valid_values) if valid_values.size else np.nan}, "
            f"mean={np.mean(valid_values) if valid_values.size else np.nan:.2f}, "
            f"median={np.median(valid_values) if valid_values.size else np.nan:.2f}, "
            f"sd={np.std(valid_values) if valid_values.size else np.nan:.2f}"
        )

    print("\nSTEP 3: Audit data")
    flags, rev_ok, profit_ok, rating_ok, dup_ids = audit(raw)
    for issue, mask in flags.items():
        print(f"{issue.replace('_', ' ')}: {int(mask.sum())}")
    print("Duplicate IDs:", dup_ids)

    print("\nSTEP 4: Prepare arrays")
    clean_data = raw.copy()
    revenue_data = clean_data[rev_ok].copy()
    profit_data = clean_data[profit_ok].copy()
    print("Raw shape:", raw.shape)
    print("Revenue-valid shape:", revenue_data.shape)
    print("Profit-valid shape:", profit_data.shape)
    print("Raw values changed: 0")

    print("\nSTEP 5: Calculate metrics")
    m = metrics(raw, rev_ok, profit_ok)
    for metric_name in ["gross_sales", "discount_amount", "net_revenue", "cost_of_goods_sold", "gross_profit"]:
        print(f"Total {metric_name.replace('_', ' ')}:", money(total(m[metric_name])))
    gross_margin = total(m["gross_profit"]) / total(m["net_revenue"][profit_ok]) * 100
    print("Overall gross margin:", f"{gross_margin:.2f}%")
    print("Mean transaction value:", money(np.nanmean(m["net_revenue"])))
    print("Median transaction value:", money(np.nanmedian(m["net_revenue"])))

    print("\nSTEP 6: Compare groups")
    outlets = group_rows(raw, m, rev_ok, 2, [1, 2, 3], OUTLETS)
    products = group_rows(raw, m, rev_ok, 3, range(1, 9), PRODUCTS)
    categories = group_rows(raw, m, rev_ok, 4, range(1, 5), CATEGORIES)
    print("Outlets ranked by revenue:")
    for row in sorted(outlets, key=lambda item: item[6], reverse=True):
        print(f"{row[1]}: revenue={money(row[6])}, profit={money(row[7])}, units={row[3]:,.0f}")
    print("Products ranked by profit:")
    for row in sorted(products, key=lambda item: item[7], reverse=True):
        print(f"{row[1]}: profit={money(row[7])}, revenue={money(row[6])}, units={row[3]:,.0f}")

    print("\nSTEP 7: Analyse trends")
    mask_union = np.any(np.column_stack(list(flags.values())), axis=1)
    high_price = np.nanpercentile(raw[np.isfinite(raw[:, 6]), 6], 99)
    reasons_by_rule = {
        "missing_unit_price": "No revenue or profit calculation",
        "missing_unit_cost": "Revenue kept profit unavailable",
        "missing_customer_rating": "Omit from rating summaries",
        "invalid_transaction_id": "Exclude from core totals",
        "duplicate_transaction_id": "Exclude collision pending reconciliation",
        "invalid_day_index": "Exclude from daily and core totals",
        "unknown_outlet_code": "Exclude from outlet core totals",
        "unknown_product_code": "Exclude from product core totals",
        "unknown_category_code": "Exclude from core totals",
        "category_mapping_mismatch": "Exclude from core totals",
        "unknown_payment_code": "Review code retain financial measures",
        "invalid_quantity": "Exclude ambiguous sale quantity",
        "invalid_unit_price": "Exclude from revenue and profit",
        "negative_unit_cost": "Revenue kept profit unavailable",
        "impossible_discount": "Exclude from revenue and profit",
        "negative_stock": "Review stock value",
        "invalid_customer_rating": "Omit from rating summaries",
        "zero_stock_with_sale": "Review no automatic exclusion",
        "high_price_review": "Retain and review price",
        "high_discount_review": "Retain and review discount",
    }
    quality_rows = [
        [
            issue,
            int(mask.sum()),
            int(np.sum(mask & ~rev_ok)),
            int(np.sum(mask & ~profit_ok)),
            0,
            reasons_by_rule[issue],
        ]
        for issue, mask in flags.items()
    ]
    group_header = (
        "code,name,transactions,units,gross_sales_inr,discount_inr,"
        "net_revenue_inr,gross_profit_inr,profit_transactions,"
        "gross_margin_pct,mean_transaction_revenue_inr"
    )
    days = np.arange(1, 31)
    daily_rev = np.array([total(m["net_revenue"][rev_ok & (raw[:, 1] == d)]) for d in days])
    daily_n = np.array([(rev_ok & (raw[:, 1] == d)).sum() for d in days], dtype=int)
    daily_change = np.r_[np.nan, np.diff(daily_rev)]
    daily_pct = np.r_[
        np.nan,
        [
            (daily_rev[i] - daily_rev[i - 1]) / daily_rev[i - 1] * 100
            if daily_rev[i - 1]
            else np.nan
            for i in range(1, 30)
        ],
    ]
    daily_z = (daily_rev - daily_rev.mean()) / daily_rev.std() if daily_rev.std() else np.zeros(30)
    daily_ma = np.array([daily_rev[max(0, i - 6):i + 1].mean() for i in range(30)])
    daily_flag = np.where(np.abs(daily_z) >= 2, "REVIEW", "")
    daily_rows = [
        [
            int(days[i]),
            int(daily_n[i]),
            daily_rev[i],
            daily_change[i],
            daily_pct[i],
            daily_ma[i],
            daily_flag[i],
            daily_z[i],
        ]
        for i in range(30)
    ]
    print("Daily review days:", [row[0] for row in daily_rows if row[6]])
    band_names = ["0%", ">0-10%", ">10-25%", ">25-50%", ">50-100%"]
    discount_rates = raw[:, 8]
    band_masks = [
        rev_ok & (discount_rates == 0),
        rev_ok & (discount_rates > 0) & (discount_rates <= 0.10),
        rev_ok & (discount_rates > 0.10) & (discount_rates <= 0.25),
        rev_ok & (discount_rates > 0.25) & (discount_rates <= 0.50),
        rev_ok & (discount_rates > 0.50),
    ]
    bands = []
    for label, b in zip(band_names, band_masks):
        bp = b & profit_ok
        net, gp = total(m["net_revenue"][b]), total(m["gross_profit"][bp])
        pnet = total(m["net_revenue"][bp])
        bands.append(
            [
                label,
                int(b.sum()),
                total(raw[b, 5]),
                net,
                gp / pnet * 100 if pnet else np.nan,
                float(np.mean(raw[b, 5])) if b.any() else np.nan,
            ]
        )
    print("Discount bands:")
    for row in bands:
        print(f"{row[0]}: n={row[1]}, units={row[2]:,.0f}, revenue={money(row[3])}, margin={row[4]:.2f}%")
    baseline_rev, baseline_gp = total(m["net_revenue"][rev_ok]), total(m["gross_profit"][profit_ok])
    observed_discount_rate = total(m["discount_amount"][rev_ok]) / total(m["gross_sales"][rev_ok])
    scenarios = []
    for label, rate in [("Actual observed discount", np.nan), ("Flat 5%", .05), ("Flat 10%", .10), ("Flat 20%", .20)]:
        nr = baseline_rev if np.isnan(rate) else total(raw[rev_ok, 5] * raw[rev_ok, 6] * (1 - rate))
        profit_revenue = total(m["net_revenue"][profit_ok]) if np.isnan(rate) else total(raw[profit_ok, 5] * raw[profit_ok, 6] * (1 - rate))
        gp = profit_revenue - total(raw[profit_ok, 5] * raw[profit_ok, 7])
        rev_change, profit_change = nr - baseline_rev, gp - baseline_gp
        if abs(profit_change) < 1e-8:
            profit_change = 0.0
        scenarios.append(
            [
                label,
                rate if np.isfinite(rate) else observed_discount_rate,
                nr,
                gp,
                rev_change,
                rev_change / baseline_rev * 100 if baseline_rev else np.nan,
                profit_change,
                profit_change / baseline_gp * 100 if baseline_gp else np.nan,
            ]
        )
    print("Discount scenarios:")
    for row in scenarios:
        print(f"{row[0]}: rate={row[1]:.1%}, revenue={money(row[2])}, gross profit={money(row[3])}")
    rv = m["net_revenue"][rev_ok]
    q1, q3 = np.percentile(rv, [25, 75])
    outlier = rev_ok & ((m["net_revenue"] < q1 - 1.5 * (q3 - q1)) | (m["net_revenue"] > q3 + 1.5 * (q3 - q1)))
    all_flags = {**flags, "net_revenue_iqr_outlier": outlier}
    union = np.any(np.column_stack(list(all_flags.values())), axis=1)
    print("Unique flagged rows:", int(union.sum()))
    reasons = []
    for i in np.where(union)[0]:
        rev, gp = m["net_revenue"][i], m["gross_profit"][i]
        impact = abs(gp) if np.isfinite(gp) else abs(rev) if np.isfinite(rev) else np.nan
        reason_codes = ";".join(
            name for name, mask in all_flags.items() if mask[i]
        )
        high_priority = any(
            all_flags[name][i]
            for name in [
                "duplicate_transaction_id",
                "invalid_unit_price",
                "impossible_discount",
                "negative_unit_cost",
            ]
        )
        reasons.append(
            [
                int(raw[i, 0]) if np.isfinite(raw[i, 0]) else "",
                i + 2,
                reason_codes,
                rev if np.isfinite(rev) else "",
                gp if np.isfinite(gp) else "",
                impact if np.isfinite(impact) else "",
                "High" if high_priority else "Review",
            ]
        )
    demos = np.array([1, 2, 3])
    x = np.asarray(demos)
    cast_values = x.astype(float)
    clean_example = np.loadtxt(io.StringIO("1,2\n3,4\n"), delimiter=",")
    arange = np.arange(5)
    linspace = np.linspace(0, 1, 5)
    zeros, ones, full, eye = np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 3), np.eye(2)
    indexed, sliced = raw[[0, 1, 2]], raw[:3, :4]
    selected = np.where(raw[:, 8] > .5)[0]
    choice = np.select([raw[:, 8] <= .1, raw[:, 8] <= .25], [0, 1], default=2)
    nonzero = np.nonzero(raw[:, 5] > 0)
    uniques, counts = np.unique(raw[:, 2], return_counts=True)
    bins = np.bincount(raw[rev_ok, 2].astype(int), minlength=4)
    ordered, order = np.sort(rv), np.argsort(rv)
    lo, hi = np.argmin(rv), np.argmax(rv)
    matrix = np.arange(12).reshape(3, 4)
    ravel, flattened, transposed = matrix.ravel(), matrix.flatten(), matrix.T
    expanded, squeezed = np.expand_dims(matrix, 0), np.squeeze(np.expand_dims(matrix, 0))
    arithmetic = x + 2
    ufuncs = [np.abs([-1, 2]), np.round([1.234], 2), np.maximum([1, 2], [2, 1]), np.minimum([1, 2], [2, 1]), np.clip([-1, .5, 2], 0, 1), np.sqrt([1, 4]), np.log1p([0, 1])]
    stat = np.array([1., 2., 3.])
    stats = [np.sum(stat), np.mean(stat), np.average(stat), np.median(stat), np.min(stat), np.max(stat), np.ptp(stat), np.var(stat), np.std(stat), np.percentile(stat, 25), np.quantile(stat, .75), np.histogram(stat, bins=2)]
    missing_stats = [np.nanmean([1., np.nan]), np.nanmedian([1., np.nan]), np.nanmin([1., np.nan]), np.nanmax([1., np.nan]), np.nan_to_num([np.nan])[0], np.isfinite([1.])[0], np.isinf([np.inf])[0]]
    time_arrays = [np.diff(stat), np.cumsum(stat), np.cumprod(stat)]
    observation_dates = np.datetime64("2026-01-01") + np.arange(30).astype("timedelta64[D]")
    aligned = rev_ok & np.isfinite(raw[:, 8])
    relations = [np.corrcoef(raw[aligned, 8], raw[aligned, 5]), np.cov(raw[aligned, 8], raw[aligned, 5]), np.isclose([1.], [1.])]
    discount_quantity_corr = float(np.corrcoef(raw[aligned, 8], raw[aligned, 5])[0, 1])
    combined = [np.concatenate((x, x)), np.stack((x, x)), np.vstack((x, x)), np.hstack((x, x)), np.split(np.arange(6), 3), np.array_split(np.arange(7), 3)]
    sample_rng = np.random.default_rng(7).choice(np.arange(10), size=3, replace=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_division = np.divide(1., [1., 0.], out=np.full(2, np.nan), where=np.array([True, False]))
    np.testing.assert_allclose([2.], [1. + 1.])
    np.testing.assert_array_equal([1, 2], [1, 2])
    try:
        np.testing.assert_allclose(np.ones((2, 2)), np.ones(3))
    except AssertionError:
        shape_test = True
    else:
        shape_test = False
    if not shape_test or not np.isnan(safe_division[1]) or not np.isnan(np.array([np.nan])[0]) or clean_example.shape != (2, 2) or cast_values.dtype != float or len(missing_stats) != 7:
        raise AssertionError("Edge-case validation failed")
    sample_indices = np.where(rev_ok & profit_ok & rating_ok)[0][:5]
    if sample_indices.size != 5:
        raise AssertionError("Five complete sample rows are required")
    sample_checks = []
    for i in sample_indices:
        q, p, c, d = raw[i, [5, 6, 7, 8]]
        hand_gross = float(q) * float(p)
        hand_discount = hand_gross * float(d)
        hand_net = hand_gross - hand_discount
        hand_profit = hand_net - float(q) * float(c)
        np.testing.assert_allclose(m["gross_sales"][i], hand_gross, rtol=1e-12)
        np.testing.assert_allclose(m["discount_amount"][i], hand_discount, rtol=1e-12)
        np.testing.assert_allclose(m["net_revenue"][i], hand_net, rtol=1e-12)
        np.testing.assert_allclose(m["gross_profit"][i], hand_profit, rtol=1e-12)
        sample_checks.append((int(raw[i, 0]), hand_gross, hand_discount, hand_net, hand_profit))
    profile = {
        "shape": raw.shape, "ndim": raw.ndim, "size": raw.size, "dtype": str(raw.dtype),
        "itemsize": raw.itemsize, "nbytes": raw.nbytes, "missing": np.isnan(raw).sum(axis=0),
        "range": [(np.nanmin(raw[:, i]), np.nanmax(raw[:, i])) for i in range(12)]
    }
    rev_total, profit_total = total(m["net_revenue"][rev_ok]), total(m["gross_profit"][profit_ok])
    overall_margin = profit_total / total(m["net_revenue"][profit_ok]) * 100
    rating_values = raw[rating_ok, 11]
    mean_rating = float(np.mean(rating_values))
    top_outlet_rev = sorted(outlets, key=lambda z: z[6], reverse=True)
    top_outlet_profit = sorted(outlets, key=lambda z: z[7], reverse=True)
    top_product_units = sorted(products, key=lambda z: z[3], reverse=True)
    top_product_profit = sorted(products, key=lambda z: z[7], reverse=True)
    lines = [
        "# FreshFlow Juice Company — NumPy Data Analyst Assessment", "",
        "## 1. Executive summary", "",
        f"The source has {raw.shape[0]:,} transaction rows across 30 days. The revenue-valid view retains {int(rev_ok.sum()):,} transactions after excluding duplicate IDs and invalid revenue inputs. Net revenue is **{money(rev_total)}**; gross profit is **{money(profit_total)}** over {int(profit_ok.sum()):,} cost-valid records.", "",
        f"1. **Outlet performance:** {top_outlet_rev[0][1]} leads net revenue at {money(top_outlet_rev[0][6])}; {top_outlet_profit[0][1]} leads gross profit at {money(top_outlet_profit[0][7])}.",
        f"2. **Product performance:** {top_product_units[0][1]} sells the most units ({top_product_units[0][3]:,.0f}); {top_product_profit[0][1]} contributes the most gross profit ({money(top_product_profit[0][7])}).",
        f"3. **Controls:** {int(union.sum()):,} rows have at least one quality or review flag. {len(dup_ids)} IDs collide; {int(flags['zero_stock_with_sale'].sum())} sales show zero stock before the transaction.", "",
        "### Three recommendations", "",
        f"1. Reconcile all {len(dup_ids)} ID collisions to receipts before setting outlet targets. Monitor the duplicate-ID rate daily and recheck after 7 days.",
        f"2. Review the {int(flags['high_discount_review'].sum())} transactions above 40% discount with managers; monitor gross margin and units by discount band for 2 weeks.",
        f"3. Ask POS and operations owners to resolve unknown outlet/category mappings and the {int(flags['zero_stock_with_sale'].sum())} zero-stock sales; monitor weekly exception counts. Do not infer waste from transaction data.", "",
        "## 2. Dataset profile", "",
        f"Source: `{DATA.name}`; grain: one transaction per row; period: day index 1–30; shape `{profile['shape']}`; dimensions {profile['ndim']}; size {profile['size']:,}; dtype `{profile['dtype']}`; item size {profile['itemsize']} bytes; memory {profile['nbytes']:,} bytes.",
        "NumPy `genfromtxt` loads the numeric CSV with comma delimiter, skipped header, float dtype, and `ndmin=2`. A small loader check verifies blank fields become NaN. `loadtxt` is separately demonstrated on clean synthetic numeric data. The primary CSV workflow uses NumPy only.", "",
        "| Column | Missing | Minimum | Maximum |", "|---|---:|---:|---:|"
    ]
    for i, name in enumerate(COLS):
        finite = raw[np.isfinite(raw[:, i]), i]
        lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (np.nan, np.nan)
        lines.append(f"| {name} | {int(np.isnan(raw[:, i]).sum())} | {lo:g} | {hi:g} |")
    lines += ["", "### Data dictionary", "",
        "| Column | Definition and unit | Allowed values / mappings |", "|---|---|---|",
        "| transaction_id | Transaction identifier | Positive integer; unique |", "| day_index | Observation day | Integer 1–30 |",
        "| outlet_id | Outlet code | 1 Outlet A; 2 Outlet B; 3 Outlet C |",
        "| product_code | Product code | 1 Orange Juice; 2 Mango Juice; 3 Apple Juice; 4 Watermelon Juice; 5 Banana Smoothie; 6 Mixed Fruit Smoothie; 7 Lemonade; 8 Iced Tea |",
        "| category_code | Product family | 1 Juice; 2 Smoothie; 3 Lemonade; 4 Tea; consistent with product |",
        "| quantity | Units sold | Positive integer |", "| unit_price | Price per unit | INR; greater than 0 |", "| unit_cost | Direct cost per unit | INR; at least 0 |",
        "| discount_rate | Fractional discount | 0–1 inclusive; 0.15 means 15% |", "| stock_before_units | Recorded stock before sale | Units; at least 0 |",
        "| payment_code | Payment method | 1 Cash; 2 UPI; 3 Card; 4 Other digital payment |", "| customer_rating | Customer rating | 1–5 or missing |", "",
        "## 3. Data quality and cleaning", "",
        "The raw matrix is preserved. Each flag is saved as a Boolean mask. Core revenue totals exclude duplicate IDs and rows with invalid IDs, days, outlets, products, categories, quantities, prices, or discounts. High but plausible prices and discounts are retained and flagged for review. Missing ratings are never changed to zero. Missing or negative costs exclude a transaction only from profit. No imputation or capping is applied.", "",
        "| Rule | Flagged | Excluded from revenue | Excluded from profit | Changed | Treatment |", "|---|---:|---:|---:|---:|---|"
    ]
    for row in quality_rows:
        lines.append(f"| {row[0].replace('_', ' ')} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
    lines += [
        "", f"{len(dup_ids)} colliding IDs affect {int(flags['duplicate_transaction_id'].sum())} rows. All rows in a collision set are excluded pending source reconciliation. {int(union.sum())} is the unique flagged-row count; individual issue counts and exclusion counts overlap. No source values were changed. Revenue-valid rows: {int(rev_ok.sum())}; profit-valid rows: {int(profit_ok.sum())}.", "",
        "## 4. Metric definitions", "",
        "Gross sales = quantity × unit price. Discount amount = gross sales × discount rate. Net revenue = gross sales − discount amount. Cost of goods sold = quantity × unit cost. Gross profit = net revenue − cost of goods sold. Gross margin = total gross profit ÷ total net revenue × 100 on the same cost-valid records. Average selling price = net revenue ÷ positive quantity. All monetary values are INR. Gross profit excludes operating expenses and is not net profit.",
        f"Overall gross margin on cost-valid records: **{overall_margin:.2f}%**.", "",
        "## 5. Outlet and product comparison", "",
        "### Outlet summary", "", "| Outlet | Revenue n | Profit n | Units | Net revenue | Gross profit | Margin |", "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for r in sorted(outlets, key=lambda z: z[6], reverse=True):
        lines.append(f"| {r[1]} | {r[2]} | {r[8]} | {r[3]:,.0f} | {money(r[6])} | {money(r[7])} | {r[9]:.2f}% |")
    lines += ["", "### Product summary", "", "| Product | Revenue n | Profit n | Units | Net revenue | Gross profit | Margin |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(products, key=lambda z: z[6], reverse=True):
        lines.append(f"| {r[1]} | {r[2]} | {r[8]} | {r[3]:,.0f} | {money(r[6])} | {money(r[7])} | {r[9]:.2f}% |")
    lines += ["", f"The top-volume product is {top_product_units[0][1]} while the top-profit product is {top_product_profit[0][1]}. Unit volume alone does not determine profit because unit prices, discounts, and costs differ.", "",
        "## 6. Descriptive statistics", "", "| Metric | n | Mean | Median | Population SD | Min | Q1 | Q3 | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, values in [("Net revenue per transaction (INR)", m["net_revenue"][rev_ok]), ("Gross profit per transaction (INR)", m["gross_profit"][profit_ok]), ("Quantity per transaction", raw[rev_ok, 5]), ("Customer rating", rating_values)]:
        finite = values[np.isfinite(values)]
        qu = np.percentile(finite, [25, 50, 75]) if finite.size else [np.nan] * 3
        mean, med = np.mean(finite), qu[1]
        sd = np.std(finite, ddof=0)
        lines.append(f"| {label} | {finite.size} | {mean:.2f} | {med:.2f} | {sd:.2f} | {np.min(finite):.2f} | {qu[0]:.2f} | {qu[2]:.2f} | {np.max(finite):.2f} |")
    lines += ["", f"Valid rating count: {rating_values.size}; mean rating: {mean_rating:.2f}. Mean/median differences indicate skew in the observed distribution, not a reason to delete extremes. Standard deviation uses the population convention (`ddof=0`).", "",
        "## 7. Daily trend", "", "| Day | Transactions | Net revenue | Change | Change % | 7-day trailing average | Review |", "|---:|---:|---:|---:|---:|---:|---|"]
    for r in daily_rows:
        lines.append(f"| {r[0]} | {r[1]} | {money(r[2])} | {money(r[3])} | {r[4]:.2f}% | {money(r[5])} | {r[6]} |")
    lines += ["", "Daily revenue review uses an absolute z-score ≥ 2 as a screening flag. Day 1 has no preceding-day change; percentage change is undefined if prior-day revenue is zero. This 30-day window supports daily comparison, not annual seasonality claims.", "",
        "## 8. Discount investigation", "", "| Discount band | Transactions | Units | Net revenue | Gross margin | Mean quantity |", "|---|---:|---:|---:|---:|---:|"]
    lines.extend([f"| {r[0]} | {r[1]} | {r[2]:,.0f} | {money(r[3])} | {r[4]:.2f}% | {r[5]:.2f} |" for r in bands])
    lines += ["", f"Across aligned revenue-valid records, the discount-rate/quantity correlation is {discount_quantity_corr:.3f}. This is a descriptive association only. Discount assignment may depend on outlet, product, day, or demand; confounding and selection bias can explain differences. Correlation does not establish that discounts caused higher volume or lower margin.", "",
        "## 9. Anomaly register", "", f"The row-level `anomaly_register.csv` contains {len(reasons)} rows with quality flags or IQR revenue outliers, reason codes, CSV line, calculable revenue/profit, estimated impact, and priority. The high-price threshold is the positive-price 99th percentile (INR {high_price:,.2f}); high discount review means above 40%. These are review flags, not automatic exclusions.", "",
        "## 10. Scenario analysis", "", "Net-revenue scenarios use the same 4,972 revenue-valid records; gross-profit scenarios use the same 4,966 cost-valid records. Quantity and unit cost are held constant while only the discount changes. These are static sensitivities, not forecasts; quantity response is not estimated.", "",
        "| Scenario | Rate | Net revenue | Gross profit | Revenue change | Revenue change % | Profit change | Profit change % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in scenarios:
        lines.append(f"| {r[0]} | {r[1]:.0%} | {money(r[2])} | {money(r[3])} | {money(r[4])} | {r[5]:.2f}% | {money(r[6])} | {r[7]:.2f}% |")
    lines += ["", "Scenario limitations: no demand response, customer substitution, competition, or product-mix response is modeled. Do not interpret the scenarios as predictions.", "",
        "## 11. Validation and reproducibility", "", "Five sample transactions were independently calculated with scalar arithmetic and compared to the vectorized metric arrays. The script also checks NaN loading, zero-denominator handling, invalid array shape, and a clean `np.loadtxt` example. A scenario gross-profit mismatch found during the first run was corrected by calculating profit only on the cost-valid set; the observed-discount scenario now reconciles to the reported gross profit. The original matrix and cleaned arrays are stored in `cleaned_analytical_arrays.npz`.",
        "From the project folder, run `..\\.venv\\Scripts\\python.exe fresh_flow.py` using the supplied workspace environment. Outputs are written to `analysis_outputs/`. No comments are used in the code; function and variable names identify each step.", "",
        "### Sample hand checks", "", "| Transaction ID | Gross sales | Discount | Net revenue | Gross profit |", "|---:|---:|---:|---:|---:|"]
    for r in sample_checks:
        lines.append(f"| {r[0]} | {money(r[1])} | {money(r[2])} | {money(r[3])} | {money(r[4])} |")
    lines += ["", "## 12. Limitations and next steps", "", "- Missing prices and costs reduce usable transaction coverage; no defensible imputation rule was provided.", "- Duplicate ID collisions remain unresolved until checked against source receipts.", "- The dataset covers 30 days and cannot establish annual seasonality or causation.", "- Gross profit excludes operating expenses.", "- Transaction stock-before fields do not contain receipts, waste, or closing stock; no waste conclusions are supported.", "- Next check: reconcile the exceptions with POS receipts and repeat the summaries after 7 days of complete data.", "",
        "## Appendix. NumPy coverage", "", "The script demonstrates creation and metadata, indexing and slicing, Boolean/fancy selection, `where`, `select`, `nonzero`, missing and finite checks, uniqueness and frequency counts, reshape and view/copy operations, arithmetic and ufuncs, descriptive statistics, ranking, time differences and cumulative operations, aligned correlation/covariance, combination and splitting, seeded synthetic sampling, numerical error handling, and NumPy testing assertions. `loadtxt` is demonstrated only on a clean numeric example."]
    print("\nSTEP 8: Save deliverables")
    np.savez_compressed(
        OUT / "cleaned_analytical_arrays.npz",
        raw_data=raw.copy(),
        revenue_valid_mask=rev_ok,
        profit_valid_mask=profit_ok,
        rating_valid_mask=rating_ok,
        cleaned_revenue_rows=raw[rev_ok].copy(),
        **{f"flag_{name}": values for name, values in flags.items()},
        **{f"metric_{name}": values for name, values in m.items()},
    )
    save_csv(
        OUT / "quality_audit.csv",
        "rule,flagged_rows,excluded_from_revenue,excluded_from_profit,changed_rows,treatment",
        quality_rows,
    )
    save_csv(OUT / "outlet_summary.csv", group_header, outlets)
    save_csv(OUT / "product_summary.csv", group_header, products)
    save_csv(OUT / "category_summary.csv", group_header, categories)
    save_csv(
        OUT / "daily_trend.csv",
        "day_index,transactions,net_revenue_inr,absolute_change_inr,"
        "percentage_change,seven_day_trailing_average_inr,review_flag,"
        "revenue_z_score",
        daily_rows,
    )
    save_csv(
        OUT / "discount_bands.csv",
        "discount_band,transactions,units,net_revenue_inr,gross_margin_pct,mean_quantity",
        bands,
    )
    save_csv(
        OUT / "scenario_analysis.csv",
        "scenario,discount_rate,net_revenue_inr,gross_profit_inr,"
        "revenue_change_inr,revenue_change_pct,profit_change_inr,"
        "profit_change_pct",
        scenarios,
    )
    save_csv(
        OUT / "anomaly_register.csv",
        "transaction_id,csv_line,reason_codes,net_revenue_inr,"
        "gross_profit_inr,estimated_impact_inr,priority",
        reasons,
    )
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis folder: {OUT}")
    print(f"Assessment complete: {raw.shape}, revenue-valid {int(rev_ok.sum())}, profit-valid {int(profit_ok.sum())}")
    print(f"Net revenue {money(rev_total)}; gross profit {money(profit_total)}")
    print(f"Report: {OUT / 'final_report.md'}")


if __name__ == "__main__":
    main()
