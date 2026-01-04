import pandas as pd
import numpy as np
from scipy import stats
import os
import ast

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = os.path.join("results", "comparaison_intersection_clean.csv")
OUTPUT_DIR = "figures_tables"
DATA_SUBDIR = os.path.join(OUTPUT_DIR, "data")
OUTPUT_TEX = os.path.join(OUTPUT_DIR, "z_score_minimal_pairs.tex")

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
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                for key in ["accuracy", "acc", "score", "f1"]:
                    if key in parsed:
                        return float(parsed[key])
            return np.nan
        return np.nan
    except:
        return np.nan


def compute_and_generate_pgf(file_path):
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
    # GESTION DE LA BASELINE (Aléatoire)
    # ---------------------------------------------------------
    baseline_mask = (
        df["name"].astype(str).str.contains("Aléatoire|Random", case=False, na=False)
    )
    baseline_rows = df[baseline_mask]

    has_baseline = False
    baseline_fr = 50.0
    baseline_multi = 50.0

    if not baseline_rows.empty:
        ref_row = baseline_rows.iloc[0]
        baseline_fr = ref_row[COL_Y_FR]
        baseline_multi = ref_row[COL_X_MULTI]
        has_baseline = True

        print(
            f"-> Baseline détectée (pour tracé des lignes) : FR={baseline_fr:.2f}, Multi={baseline_multi:.2f}"
        )

        # On retire la baseline du dataset de plot (pour ne pas avoir un point sur la croix)
        df = df[~baseline_mask].copy()
    else:
        print("-> Avertissement : Aucune baseline trouvée.")

    # ---------------------------------------------------------
    # CLASSIFICATION & PRÉPARATION PGF
    # ---------------------------------------------------------
    results = []

    for index, row in df.iterrows():
        p_fr = row[COL_Y_FR]
        p_multi = row[COL_X_MULTI]

        # Classification pour couleur
        if p_fr < baseline_fr or p_multi < baseline_multi:
            tex_class = "red_dot"  # Below Random Baseline
        elif p_fr > HIGH_PERF_THRESHOLD and p_multi > HIGH_PERF_THRESHOLD:
            tex_class = "green_dot"  # High Performance
        else:
            tex_class = "blue_dot"  # Intermediate

        results.append({"acc_fr": p_fr, "acc_multi": p_multi, "tex_class": tex_class})

    results_df = pd.DataFrame(results)

    # Création des dossiers de sortie
    if not os.path.exists(DATA_SUBDIR):
        os.makedirs(DATA_SUBDIR)

    # 1. Sauvegarde des données Scatter
    scatter_csv_name = "pgf_scatter_minimal_pairs.csv"
    scatter_csv_path = os.path.join(DATA_SUBDIR, scatter_csv_name)
    results_df.to_csv(scatter_csv_path, index=False)
    print(f"Données scatter sauvegardées : {scatter_csv_path}")

    # 2. Calcul des Courbes de Significativité
    x_range = np.linspace(0, 100, 500)
    x_prop = x_range / 100.0

    critical_z = stats.norm.ppf(1 - (ALPHA / 2))
    se_curve = np.sqrt(x_prop * (1 - x_prop) * ((1 / N_FR) + (1 / N_MULTIBLIMP)))
    margin_curve = critical_z * se_curve * 100

    y_upper = x_range + margin_curve
    y_lower = x_range - margin_curve

    sig_df = pd.DataFrame({"x": x_range, "y_upper": y_upper, "y_lower": y_lower})
    sig_csv_name = "pgf_sig_curves_minimal_pairs.csv"
    sig_csv_path = os.path.join(DATA_SUBDIR, sig_csv_name)
    sig_df.to_csv(sig_csv_path, index=False)
    print(f"Courbes significativité sauvegardées : {sig_csv_path}")

    # ---------------------------------------------------------
    # GÉNÉRATION LATEX
    # ---------------------------------------------------------
    print("Génération du code LaTeX...")

    rel_scatter = os.path.join("data", scatter_csv_name).replace("\\", "/")
    rel_sig = os.path.join("data", sig_csv_name).replace("\\", "/")

    tex_content = rf"""\documentclass[border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=12cm, height=12cm,
    % Titres et Labels
    xlabel={{MultiBlimp Accuracy (\%)}},
    ylabel={{Minimal Pair FR Accuracy (\%)}},
    % Limites
    xmin=0, xmax=100,
    ymin=0, ymax=100,
    % Styles
    grid=none,
    scatter/classes={{
        green_dot={{mark=*, draw=green!60!black, fill=green!60!black, opacity=0.6, mark size=3pt}},
        red_dot={{mark=*, draw=red!60!black, fill=red!60!black, opacity=0.6, mark size=3pt}},
        blue_dot={{mark=*, draw=blue!60!black, fill=blue!60!black, opacity=0.6, mark size=3pt}}
    }}
]

% 1. Ligne de performance égale (Diagonale)
\addplot [black, dashdotted, domain=0:100] {{x}};

% 2. Lignes de Baseline (SANS TEXTE)
"""
    if has_baseline:
        tex_content += rf"""
\draw [black!70, dashed, line width=1pt] (axis cs:{baseline_multi:.2f}, 0) -- (axis cs:{baseline_multi:.2f}, 100);
\draw [black!70, dashed, line width=1pt] (axis cs:0, {baseline_fr:.2f}) -- (axis cs:100, {baseline_fr:.2f});
% Texte supprimé ici
"""

    tex_content += rf"""
% 3. Bandes de Significativité (Alpha={ALPHA})
\addplot [gray, dotted, line width=1.5pt] table [x=x, y=y_upper, col sep=comma] {{{rel_sig}}};
\addplot [gray, dotted, line width=1.5pt] table [x=x, y=y_lower, col sep=comma] {{{rel_sig}}};

% 4. Points Scatter
\addplot [scatter, only marks, scatter src=explicit symbolic] 
    table [x=acc_multi, y=acc_fr, meta=tex_class, col sep=comma] {{{rel_scatter}}};

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"Succès ! Fichier généré : {OUTPUT_TEX}")
    print("Compilez le avec : pdflatex z_score_minimal_pairs.tex")


if __name__ == "__main__":
    compute_and_generate_pgf(INPUT_FILE)
