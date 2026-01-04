import pandas as pd
import numpy as np
from scipy import stats
import os


def generate_latex_stats_table(file_path):
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    col_left = "frcoe.accuracy"
    col_right = "qfrcore.accuracy"
    name_col = "name"

    # Filter and Copy
    all_data = df.dropna(subset=[col_left, col_right]).copy()

    # Transform to [0, 100]
    all_data[col_left] = all_data[col_left] * 100
    all_data[col_right] = all_data[col_right] * 100

    # --- 1. EXTRACT BASELINE ---
    baseline_row = all_data[
        all_data[name_col]
        .astype(str)
        .str.contains("RandomBaseline", case=False, na=False)
    ]

    baseline_frcoe = 0
    baseline_qfrcore = 0
    has_baseline = False

    if not baseline_row.empty:
        baseline_frcoe = baseline_row.iloc[0][col_left]
        baseline_qfrcore = baseline_row.iloc[0][col_right]
        has_baseline = True
        # Remove baseline from statistics
        valid_data = all_data[all_data.index != baseline_row.index[0]].copy()
    else:
        valid_data = all_data.copy()

    # --- 2. COMPUTE STATISTICS ---
    n_left = 4938
    n_right = 4633
    alpha = 0.001

    # Counters
    count_sig_frcoe_better = 0
    count_sig_qfrcore_better = 0
    count_ns = 0

    count_green = 0  # > 65% on both
    count_red = 0  # < Baseline on either
    count_blue = 0  # Intermediate

    for index, row in valid_data.iterrows():
        p1_pct = row[col_left]
        p2_pct = row[col_right]

        # Z-Test
        p1_prop = p1_pct / 100.0
        p2_prop = p2_pct / 100.0
        x1 = p1_prop * n_left
        x2 = p2_prop * n_right
        p_pool = (x1 + x2) / (n_left + n_right)

        if p_pool <= 0 or p_pool >= 1:
            z = 0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * ((1 / n_left) + (1 / n_right)))
            z = (p1_prop - p2_prop) / se if se != 0 else 0

        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        is_significant = p_val < alpha

        # Significance Direction Counts
        if not is_significant:
            count_ns += 1
        elif z > 0:
            count_sig_frcoe_better += 1
        else:
            count_sig_qfrcore_better += 1

        # Classification Counts
        is_below_baseline = False
        if has_baseline:
            if p1_pct < baseline_frcoe or p2_pct < baseline_qfrcore:
                is_below_baseline = True

        if p1_pct > 65.0 and p2_pct > 65.0:
            count_green += 1
        elif is_below_baseline:
            count_red += 1
        else:
            count_blue += 1

    total_models = len(valid_data)

    # --- 3. GENERATE LATEX TABLE ---
    latex_table = rf"""\begin{{table}}[ht]
    \centering
    \begin{{tabular}}{{lr}}
        \toprule
        & \textbf{{Count}} \\
        \midrule
        Significantly better on \textbf{{FrCoE}} & {count_sig_frcoe_better} \small{{({count_sig_frcoe_better / total_models * 100:.1f}\%)}} \\
        Significantly better on \textbf{{QFrCoRE}} & {count_sig_qfrcore_better} \small{{({count_sig_qfrcore_better / total_models * 100:.1f}\%)}} \\
         No Significant Difference & {count_ns} \small{{({count_ns / total_models * 100:.1f}\%)}} \\
        \bottomrule
    \end{{tabular}}\
    \caption{{Summary of model performance and statistical comparison (Z-test, $\alpha={alpha}$). Comparisons are between FrCoE ($N={n_left}$) and QFrCoRE ($N={n_right}$).}}
    \label{{tab:z_test_stats}}
\end{{table}}
"""

    print("-" * 40)
    print(latex_table)
    print("-" * 40)

    with open(os.path.join("figures_tables", "z_test_statistics.tex"), "w") as f:
        f.write(latex_table)
    print("Saved table to 'z_test_statistics.tex'")


if __name__ == "__main__":
    INPUT_FILE = os.path.join("results", "filtered_accuracies.csv")
    generate_latex_stats_table(INPUT_FILE)
