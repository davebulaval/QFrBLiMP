import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = os.path.join("results", "comparaison_intersection_clean.csv")
OUTPUT_DIR = "figures_tables"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "z_score_scatter_plot_minimal_pairs.png")

# Noms EXACTS des colonnes
COL_Y_FR = "test_minimal_pair_analysis_fr"  # Axe Y
COL_X_MULTI = "test_minimal_pair_analysis_multiblimp"  # Axe X

# Paramètres
N_FR = 1000
N_MULTIBLIMP = 1000
HIGH_PERF_THRESHOLD = 80.0
ALPHA = 0.001


def extract_accuracy(val):
    """
    Extrait la valeur 'accuracy' d'une chaîne format dictionnaire ou d'un nombre.
    """
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val = val.strip()
            # On utilise ast pour convertir la chaîne "{'acc': 10}" en dict
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                # On cherche les clés probables
                for key in ["accuracy", "acc", "score", "f1"]:
                    if key in parsed:
                        return float(parsed[key])
            return np.nan
        return np.nan
    except:
        return np.nan


def compute_z_test_and_plot(file_path):
    print(f"Chargement des données depuis {file_path}...")

    if not os.path.exists(file_path):
        print(f"Erreur : Fichier introuvable.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lecture CSV : {e}")
        return

    # Vérification colonnes
    if COL_Y_FR not in df.columns or COL_X_MULTI not in df.columns:
        print(f"Erreur : Colonnes manquantes.")
        return

    # 1. Extraction des valeurs numériques
    print("Extraction des scores...")
    df[COL_Y_FR] = df[COL_Y_FR].apply(extract_accuracy)
    df[COL_X_MULTI] = df[COL_X_MULTI].apply(extract_accuracy)

    # 2. Nettoyage (suppression des lignes vides)
    df = df.dropna(subset=[COL_Y_FR, COL_X_MULTI]).copy()

    # 3. Mise à l'échelle 0-100
    if df[COL_Y_FR].max() <= 1.0:
        df[COL_Y_FR] = df[COL_Y_FR] * 100
    if df[COL_X_MULTI].max() <= 1.0:
        df[COL_X_MULTI] = df[COL_X_MULTI] * 100

    # ---------------------------------------------------------
    # GESTION DE LA BASELINE (RANDOM)
    # ---------------------------------------------------------
    # On cherche une ligne dont le nom contient "Random" (ex: RandomBaseline)
    baseline_mask = (
        df["name"].astype(str).str.contains("Aléatoire", case=False, na=False)
    )
    baseline_rows = df[baseline_mask]

    if not baseline_rows.empty:
        # On prend le premier modèle trouvé comme référence
        ref_row = baseline_rows.iloc[0]
        baseline_fr = ref_row[COL_Y_FR]
        baseline_multi = ref_row[COL_X_MULTI]
        baseline_name = ref_row["name"]

        print(f"-> Baseline détectée : '{baseline_name}'")
        print(f"   Score FR: {baseline_fr:.2f}% | Score Multi: {baseline_multi:.2f}%")

        # On retire ce modèle du DataFrame pour qu'il ne soit pas tracé comme un point
        df = df[~baseline_mask].copy()
    else:
        print(
            "-> Avertissement : Aucune 'RandomBaseline' trouvée. Utilisation de 50% par défaut."
        )
        baseline_fr = 50.0
        baseline_multi = 50.0
        baseline_name = "Random Baseline"

    # ---------------------------------------------------------
    # CALCULS STATISTIQUES
    # ---------------------------------------------------------
    critical_z = stats.norm.ppf(1 - (ALPHA / 2))
    results = []

    for index, row in df.iterrows():
        p_fr = row[COL_Y_FR]
        p_multi = row[COL_X_MULTI]
        model_name = row.get("name", f"Model_{index}")

        # Props
        p1 = p_fr / 100.0
        p2 = p_multi / 100.0

        # Z-score poolé
        p_pool = (p1 * N_FR + p2 * N_MULTIBLIMP) / (N_FR + N_MULTIBLIMP)
        if p_pool <= 0 or p_pool >= 1:
            z = 0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * ((1 / N_FR) + (1 / N_MULTIBLIMP)))
            z = (p1 - p2) / se if se != 0 else 0

        # Classification basée sur la Baseline dynamique
        if p_fr < baseline_fr or p_multi < baseline_multi:
            classification = "Below Random Baseline"
        elif p_fr > HIGH_PERF_THRESHOLD and p_multi > HIGH_PERF_THRESHOLD:
            classification = f"High Performance (>{int(HIGH_PERF_THRESHOLD)}%)"
        else:
            classification = "Intermediate"

        results.append(
            {
                "name": model_name,
                "acc_fr": p_fr,
                "acc_multi": p_multi,
                "z_score": z,
                "Classification": classification,
            }
        )

    results_df = pd.DataFrame(results)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if results_df.empty:
        print("Aucun résultat à tracer.")
        return

    # ---------------------------------------------------------
    # GRAPHIQUE
    # ---------------------------------------------------------
    print("Génération du graphique...")
    sns.set_style("white")
    plt.figure(figsize=(10, 10))

    custom_palette = {
        f"High Performance (>{int(HIGH_PERF_THRESHOLD)}%)": "green",
        "Below Random Baseline": "red",
        "Intermediate": "blue",
    }

    hue_order = [
        f"High Performance (>{int(HIGH_PERF_THRESHOLD)}%)",
        "Intermediate",
        "Below Random Baseline",
    ]
    present_categories = [
        cat for cat in hue_order if cat in results_df["Classification"].unique()
    ]

    # Scatter plot
    sns.scatterplot(
        data=results_df,
        x="acc_multi",
        y="acc_fr",
        hue="Classification",
        palette=custom_palette,
        hue_order=present_categories,
        alpha=0.6,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
    )

    # Limites
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Ligne Diagonale
    plt.plot(
        [0, 100],
        [0, 100],
        color="black",
        linestyle="-.",
        alpha=0.4,
        linewidth=1,
        label="Equal Performance",
    )

    # Lignes de Baseline (Dynamiques)
    # Verticale (MultiBlimp)
    plt.axvline(
        x=baseline_multi, color="#444444", linestyle="--", linewidth=1, alpha=0.8
    )
    # Horizontale (FR)
    plt.axhline(
        y=baseline_fr,
        color="#444444",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label=baseline_name,
    )

    # Bandes de Significativité
    x_range = np.linspace(0, 100, 500)
    x_prop = x_range / 100.0
    se_curve = np.sqrt(x_prop * (1 - x_prop) * ((1 / N_FR) + (1 / N_MULTIBLIMP)))
    margin_curve = critical_z * se_curve * 100

    plt.plot(
        x_range,
        x_range + margin_curve,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
        label=f"Sig. Interval (α={ALPHA})",
    )
    plt.plot(
        x_range,
        x_range - margin_curve,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
    )

    plt.title("Comparaison : Minimal Pair FR vs MultiBlimp")
    plt.xlabel(f"MultiBlimp Accuracy (%)")
    plt.ylabel(f"Minimal Pair FR Accuracy (%)")

    plt.grid(False)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Graphique sauvegardé : {OUTPUT_PLOT}")
    plt.close()


if __name__ == "__main__":
    compute_z_test_and_plot(INPUT_FILE)
