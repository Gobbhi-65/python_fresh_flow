# FreshFlow Juice Company — NumPy Data Analyst Assessment

## 1. Executive summary

The source has 5,000 transaction rows across 30 days. The revenue-valid view retains 4,972 transactions after excluding duplicate IDs and invalid revenue inputs. Net revenue is **INR 2,156,048.65**; gross profit is **INR 1,094,496.23** over 4,966 cost-valid records.

1. **Outlet performance:** Outlet A leads net revenue at INR 792,483.89; Outlet A leads gross profit at INR 403,456.01.
2. **Product performance:** Orange Juice sells the most units (3,816); Mango Juice contributes the most gross profit (INR 192,740.29).
3. **Controls:** 209 rows have at least one quality or review flag. 5 IDs collide; 5 sales show zero stock before the transaction.

### Three recommendations

1. Reconcile all 5 ID collisions to receipts before setting outlet targets. Monitor the duplicate-ID rate daily and recheck after 7 days.
2. Review the 5 transactions above 40% discount with managers; monitor gross margin and units by discount band for 2 weeks.
3. Ask POS and operations owners to resolve unknown outlet/category mappings and the 5 zero-stock sales; monitor weekly exception counts. Do not infer waste from transaction data.

## 2. Dataset profile

Source: `freshflow_sales.csv`; grain: one transaction per row; period: day index 1–30; shape `(5000, 12)`; dimensions 2; size 60,000; dtype `float64`; item size 8 bytes; memory 480,000 bytes.
NumPy `genfromtxt` loads the numeric CSV with comma delimiter, skipped header, float dtype, and `ndmin=2`. A small loader check verifies blank fields become NaN. `loadtxt` is separately demonstrated on clean synthetic numeric data. The primary CSV workflow uses NumPy only.

| Column | Missing | Minimum | Maximum |
|---|---:|---:|---:|
| transaction_id | 0 | 100001 | 105000 |
| day_index | 0 | 1 | 30 |
| outlet_id | 0 | 0 | 9 |
| product_code | 0 | 1 | 8 |
| category_code | 0 | 1 | 4 |
| quantity | 0 | -3 | 8 |
| unit_price | 4 | -79.48 | 224.77 |
| unit_cost | 4 | -24.32 | 112.74 |
| discount_rate | 0 | 0 | 1.19 |
| stock_before_units | 0 | 0 | 120 |
| payment_code | 0 | 1 | 4 |
| customer_rating | 20 | 0 | 7 |

### Data dictionary

| Column | Definition and unit | Allowed values / mappings |
|---|---|---|
| transaction_id | Transaction identifier | Positive integer; unique |
| day_index | Observation day | Integer 1–30 |
| outlet_id | Outlet code | 1 Outlet A; 2 Outlet B; 3 Outlet C |
| product_code | Product code | 1 Orange Juice; 2 Mango Juice; 3 Apple Juice; 4 Watermelon Juice; 5 Banana Smoothie; 6 Mixed Fruit Smoothie; 7 Lemonade; 8 Iced Tea |
| category_code | Product family | 1 Juice; 2 Smoothie; 3 Lemonade; 4 Tea; consistent with product |
| quantity | Units sold | Positive integer |
| unit_price | Price per unit | INR; greater than 0 |
| unit_cost | Direct cost per unit | INR; at least 0 |
| discount_rate | Fractional discount | 0–1 inclusive; 0.15 means 15% |
| stock_before_units | Recorded stock before sale | Units; at least 0 |
| payment_code | Payment method | 1 Cash; 2 UPI; 3 Card; 4 Other digital payment |
| customer_rating | Customer rating | 1–5 or missing |

## 3. Data quality and cleaning

The raw matrix is preserved. Each flag is saved as a Boolean mask. Core revenue totals exclude duplicate IDs and rows with invalid IDs, days, outlets, products, categories, quantities, prices, or discounts. High but plausible prices and discounts are retained and flagged for review. Missing ratings are never changed to zero. Missing or negative costs exclude a transaction only from profit. No imputation or capping is applied.

| Rule | Flagged | Excluded from revenue | Excluded from profit | Changed | Treatment |
|---|---:|---:|---:|---:|---|
| missing unit price | 4 | 4 | 4 | 0 | No revenue or profit calculation |
| missing unit cost | 4 | 0 | 4 | 0 | Revenue kept profit unavailable |
| missing customer rating | 20 | 0 | 0 | 0 | Omit from rating summaries |
| invalid transaction id | 0 | 0 | 0 | 0 | Exclude from core totals |
| duplicate transaction id | 10 | 10 | 10 | 0 | Exclude collision pending reconciliation |
| invalid day index | 0 | 0 | 0 | 0 | Exclude from daily and core totals |
| unknown outlet code | 2 | 2 | 2 | 0 | Exclude from outlet core totals |
| unknown product code | 0 | 0 | 0 | 0 | Exclude from product core totals |
| unknown category code | 0 | 0 | 0 | 0 | Exclude from core totals |
| category mapping mismatch | 4 | 4 | 4 | 0 | Exclude from core totals |
| unknown payment code | 0 | 0 | 0 | 0 | Review code retain financial measures |
| invalid quantity | 4 | 4 | 4 | 0 | Exclude ambiguous sale quantity |
| invalid unit price | 2 | 2 | 2 | 0 | Exclude from revenue and profit |
| negative unit cost | 2 | 0 | 2 | 0 | Revenue kept profit unavailable |
| impossible discount | 2 | 2 | 2 | 0 | Exclude from revenue and profit |
| negative stock | 0 | 0 | 0 | 0 | Review stock value |
| invalid customer rating | 3 | 0 | 0 | 0 | Omit from rating summaries |
| zero stock with sale | 5 | 0 | 0 | 0 | Review no automatic exclusion |
| high price review | 50 | 0 | 0 | 0 | Retain and review price |
| high discount review | 5 | 0 | 0 | 0 | Retain and review discount |

5 colliding IDs affect 10 rows. All rows in a collision set are excluded pending source reconciliation. 209 is the unique flagged-row count; individual issue counts and exclusion counts overlap. No source values were changed. Revenue-valid rows: 4972; profit-valid rows: 4966.

## 4. Metric definitions

Gross sales = quantity × unit price. Discount amount = gross sales × discount rate. Net revenue = gross sales − discount amount. Cost of goods sold = quantity × unit cost. Gross profit = net revenue − cost of goods sold. Gross margin = total gross profit ÷ total net revenue × 100 on the same cost-valid records. Average selling price = net revenue ÷ positive quantity. All monetary values are INR. Gross profit excludes operating expenses and is not net profit.
Overall gross margin on cost-valid records: **50.81%**.

## 5. Outlet and product comparison

### Outlet summary

| Outlet | Revenue n | Profit n | Units | Net revenue | Gross profit | Margin |
|---|---:|---:|---:|---:|---:|---:|
| Outlet A | 1861 | 1860 | 8,285 | INR 792,483.89 | INR 403,456.01 | 50.94% |
| Outlet B | 1706 | 1704 | 7,726 | INR 764,868.82 | INR 387,868.60 | 50.77% |
| Outlet C | 1405 | 1402 | 6,404 | INR 598,695.94 | INR 303,171.62 | 50.70% |

### Product summary

| Product | Revenue n | Profit n | Units | Net revenue | Gross profit | Margin |
|---|---:|---:|---:|---:|---:|---:|
| Mixed Fruit Smoothie | 514 | 513 | 2,356 | INR 379,847.93 | INR 165,941.15 | 43.70% |
| Mango Juice | 769 | 766 | 3,505 | INR 379,468.27 | INR 192,740.29 | 50.98% |
| Banana Smoothie | 590 | 590 | 2,628 | INR 357,032.19 | INR 166,852.69 | 46.73% |
| Orange Juice | 848 | 848 | 3,816 | INR 306,061.62 | INR 161,439.87 | 52.75% |
| Apple Juice | 584 | 583 | 2,705 | INR 268,188.16 | INR 139,426.86 | 52.06% |
| Watermelon Juice | 623 | 623 | 2,726 | INR 208,181.30 | INR 115,640.57 | 55.55% |
| Lemonade | 610 | 609 | 2,750 | INR 136,767.91 | INR 83,250.24 | 60.97% |
| Iced Tea | 434 | 434 | 1,929 | INR 120,501.27 | INR 69,204.56 | 57.43% |

The top-volume product is Orange Juice while the top-profit product is Mango Juice. Unit volume alone does not determine profit because unit prices, discounts, and costs differ.

## 6. Descriptive statistics

| Metric | n | Mean | Median | Population SD | Min | Q1 | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Net revenue per transaction (INR) | 4972 | 433.64 | 382.05 | 285.36 | 35.91 | 209.88 | 596.90 | 1599.19 |
| Gross profit per transaction (INR) | 4966 | 220.40 | 202.35 | 137.57 | -21.83 | 110.23 | 304.71 | 889.70 |
| Quantity per transaction | 4972 | 4.51 | 5.00 | 2.30 | 1.00 | 2.00 | 7.00 | 8.00 |
| Customer rating | 4949 | 3.84 | 4.00 | 1.06 | 1.00 | 3.00 | 5.00 | 5.00 |

Valid rating count: 4949; mean rating: 3.84. Mean/median differences indicate skew in the observed distribution, not a reason to delete extremes. Standard deviation uses the population convention (`ddof=0`).

## 7. Daily trend

| Day | Transactions | Net revenue | Change | Change % | 7-day trailing average | Review |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 181 | INR 83,667.60 | N/A | nan% | INR 83,667.60 |  |
| 2 | 148 | INR 64,264.78 | INR -19,402.82 | -23.19% | INR 73,966.19 |  |
| 3 | 136 | INR 55,241.78 | INR -9,023.00 | -14.04% | INR 67,724.72 | REVIEW |
| 4 | 170 | INR 85,842.05 | INR 30,600.26 | 55.39% | INR 72,254.05 |  |
| 5 | 166 | INR 71,792.80 | INR -14,049.24 | -16.37% | INR 72,161.80 |  |
| 6 | 189 | INR 84,952.83 | INR 13,160.03 | 18.33% | INR 74,293.64 |  |
| 7 | 177 | INR 82,589.38 | INR -2,363.46 | -2.78% | INR 75,478.75 |  |
| 8 | 148 | INR 67,075.07 | INR -15,514.31 | -18.78% | INR 73,108.38 |  |
| 9 | 175 | INR 77,057.84 | INR 9,982.77 | 14.88% | INR 74,935.97 |  |
| 10 | 160 | INR 70,358.24 | INR -6,699.60 | -8.69% | INR 77,095.46 |  |
| 11 | 175 | INR 81,229.69 | INR 10,871.44 | 15.45% | INR 76,436.55 |  |
| 12 | 152 | INR 63,516.71 | INR -17,712.97 | -21.81% | INR 75,254.25 |  |
| 13 | 170 | INR 74,578.86 | INR 11,062.14 | 17.42% | INR 73,772.26 |  |
| 14 | 167 | INR 69,837.43 | INR -4,741.42 | -6.36% | INR 71,950.55 |  |
| 15 | 166 | INR 71,543.89 | INR 1,706.46 | 2.44% | INR 72,588.95 |  |
| 16 | 163 | INR 74,918.04 | INR 3,374.15 | 4.72% | INR 72,283.27 |  |
| 17 | 177 | INR 80,978.55 | INR 6,060.51 | 8.09% | INR 73,800.45 |  |
| 18 | 180 | INR 75,328.59 | INR -5,649.96 | -6.98% | INR 72,957.44 |  |
| 19 | 143 | INR 61,069.80 | INR -14,258.79 | -18.93% | INR 72,607.88 |  |
| 20 | 170 | INR 72,083.65 | INR 11,013.85 | 18.03% | INR 72,251.42 |  |
| 21 | 168 | INR 74,326.11 | INR 2,242.46 | 3.11% | INR 72,892.66 |  |
| 22 | 173 | INR 69,810.92 | INR -4,515.19 | -6.07% | INR 72,645.10 |  |
| 23 | 163 | INR 62,585.84 | INR -7,225.09 | -10.35% | INR 70,883.35 |  |
| 24 | 164 | INR 70,206.63 | INR 7,620.80 | 12.18% | INR 69,344.51 |  |
| 25 | 171 | INR 73,356.76 | INR 3,150.12 | 4.49% | INR 69,062.82 |  |
| 26 | 161 | INR 66,163.49 | INR -7,193.27 | -9.81% | INR 69,790.49 |  |
| 27 | 159 | INR 61,292.09 | INR -4,871.40 | -7.36% | INR 68,248.84 |  |
| 28 | 150 | INR 62,949.00 | INR 1,656.90 | 2.70% | INR 66,623.53 |  |
| 29 | 165 | INR 69,064.39 | INR 6,115.39 | 9.71% | INR 66,516.88 |  |
| 30 | 185 | INR 78,365.83 | INR 9,301.44 | 13.47% | INR 68,771.17 |  |

Daily revenue review uses an absolute z-score ≥ 2 as a screening flag. Day 1 has no preceding-day change; percentage change is undefined if prior-day revenue is zero. This 30-day window supports daily comparison, not annual seasonality claims.

## 8. Discount investigation

| Discount band | Transactions | Units | Net revenue | Gross margin | Mean quantity |
|---|---:|---:|---:|---:|---:|
| 0% | 859 | 3,832 | INR 404,719.15 | 56.24% | 4.46 |
| >0-10% | 2171 | 9,858 | INR 986,287.69 | 52.13% | 4.54 |
| >10-25% | 1937 | 8,697 | INR 763,897.13 | 46.30% | 4.49 |
| >25-50% | 1 | 5 | INR 212.90 | 36.47% | 5.00 |
| >50-100% | 4 | 23 | INR 931.78 | 1.93% | 5.75 |

Across aligned revenue-valid records, the discount-rate/quantity correlation is 0.001. This is a descriptive association only. Discount assignment may depend on outlet, product, day, or demand; confounding and selection bias can explain differences. Correlation does not establish that discounts caused higher volume or lower margin.

## 9. Anomaly register

The row-level `anomaly_register.csv` contains 209 rows with quality flags or IQR revenue outliers, reason codes, CSV line, calculable revenue/profit, estimated impact, and priority. The high-price threshold is the positive-price 99th percentile (INR 193.67); high discount review means above 40%. These are review flags, not automatic exclusions.

## 10. Scenario analysis

Net-revenue scenarios use the same 4,972 revenue-valid records; gross-profit scenarios use the same 4,966 cost-valid records. Quantity and unit cost are held constant while only the discount changes. These are static sensitivities, not forecasts; quantity response is not estimated.

| Scenario | Rate | Net revenue | Gross profit | Revenue change | Revenue change % | Profit change | Profit change % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actual observed discount | 11% | INR 2,156,048.65 | INR 1,094,496.23 | INR 0.00 | 0.00% | INR 0.00 | 0.00% |
| Flat 5% | 5% | INR 2,289,003.74 | INR 1,227,285.35 | INR 132,955.09 | 6.17% | INR 132,789.12 | 12.13% |
| Flat 10% | 10% | INR 2,168,529.86 | INR 1,106,930.49 | INR 12,481.21 | 0.58% | INR 12,434.26 | 1.14% |
| Flat 20% | 20% | INR 1,927,582.10 | INR 866,220.77 | INR -228,466.55 | -10.60% | INR -228,275.46 | -20.86% |

Scenario limitations: no demand response, customer substitution, competition, or product-mix response is modeled. Do not interpret the scenarios as predictions.

## 11. Validation and reproducibility

Five sample transactions were independently calculated with scalar arithmetic and compared to the vectorized metric arrays. The script also checks NaN loading, zero-denominator handling, invalid array shape, and a clean `np.loadtxt` example. A scenario gross-profit mismatch found during the first run was corrected by calculating profit only on the cost-valid set; the observed-discount scenario now reconciles to the reported gross profit. The original matrix and cleaned arrays are stored in `cleaned_analytical_arrays.npz`.
From the project folder, run `..\.venv\Scripts\python.exe fresh_flow.py` using the supplied workspace environment. Outputs are written to `analysis_outputs/`. No comments are used in the code; function and variable names identify each step.

### Sample hand checks

| Transaction ID | Gross sales | Discount | Net revenue | Gross profit |
|---:|---:|---:|---:|---:|
| 100001 | INR 444.85 | INR 44.49 | INR 400.37 | INR 216.72 |
| 100002 | INR 527.52 | INR 0.00 | INR 527.52 | INR 313.67 |
| 100003 | INR 147.40 | INR 14.74 | INR 132.66 | INR 62.24 |
| 100004 | INR 478.31 | INR 47.83 | INR 430.48 | INR 250.44 |
| 100005 | INR 527.55 | INR 52.75 | INR 474.79 | INR 211.78 |

## 12. Limitations and next steps

- Missing prices and costs reduce usable transaction coverage; no defensible imputation rule was provided.
- Duplicate ID collisions remain unresolved until checked against source receipts.
- The dataset covers 30 days and cannot establish annual seasonality or causation.
- Gross profit excludes operating expenses.
- Transaction stock-before fields do not contain receipts, waste, or closing stock; no waste conclusions are supported.
- Next check: reconcile the exceptions with POS receipts and repeat the summaries after 7 days of complete data.

## Appendix. NumPy coverage

The script demonstrates creation and metadata, indexing and slicing, Boolean/fancy selection, `where`, `select`, `nonzero`, missing and finite checks, uniqueness and frequency counts, reshape and view/copy operations, arithmetic and ufuncs, descriptive statistics, ranking, time differences and cumulative operations, aligned correlation/covariance, combination and splitting, seeded synthetic sampling, numerical error handling, and NumPy testing assertions. `loadtxt` is demonstrated only on a clean numeric example.
