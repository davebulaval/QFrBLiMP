import os
import wandb
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WANDB_ENTITY = "davebulaval"
PROJECT_1 = "minimal_pair_analysis_fr"
PROJECT_2 = "minimal_pair_analysis_multiblimp"

# La clé commune (ex: 'name') nécessaire dans les deux projets
JOIN_KEY = "name"


def get_project_data(entity, project):
    """
    Récupère les runs, filtre OLMo et retire les doublons.
    """
    print(f"Récupération des runs depuis {entity}/{project}...")

    api = wandb.Api()

    try:
        # Récupère les runs (souvent du plus récent au plus ancien)
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Erreur d'accès au projet {project}: {e}")
        return pd.DataFrame()

    summary_list = []

    for run in runs:
        run_data = {
            JOIN_KEY: getattr(run, JOIN_KEY, "N/A"),
            "run_id": run.id,
            "state": run.state,
        }
        run_data.update(run.config)
        run_data.update(run.summary._json_dict)
        summary_list.append(run_data)

    df = pd.DataFrame(summary_list)

    if df.empty:
        return df

    # 1. FILTRAGE : RETIRER LES MODÈLES OLMO
    if JOIN_KEY in df.columns:
        df = df[~df[JOIN_KEY].astype(str).str.contains("olmo", case=False, na=False)]

    # 2. DÉDUPLICATION : RETIRER LES DOUBLONS
    # On garde la première occurrence (le run le plus récent)
    if JOIN_KEY in df.columns:
        df = df.drop_duplicates(subset=[JOIN_KEY], keep="first")

    print(f" -> {len(df)} modèles uniques (sans OLMo) retenus pour {project}.")
    return df


def main():
    # 1. Récupérer les données propres des deux projets
    df_p1 = get_project_data(WANDB_ENTITY, PROJECT_1)
    df_p2 = get_project_data(WANDB_ENTITY, PROJECT_2)

    # Si l'un des deux est vide, l'intersection sera impossible
    if df_p1.empty or df_p2.empty:
        print(
            "L'un des projets est vide ou inaccessible. Impossible de faire l'intersection."
        )
        return

    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Fusionner les DataFrames : INTERSECTION SEULEMENT
    print(
        f"Fusion des données (Intersection des modèles présents dans les DEUX projets)..."
    )

    # how='inner' ne garde que les clés présentes dans les deux DataFrames
    merged_df = pd.merge(
        df_p1,
        df_p2,
        on=JOIN_KEY,
        how="inner",
        suffixes=(f"_{PROJECT_1}", f"_{PROJECT_2}"),
    )

    if merged_df.empty:
        print("Aucun modèle commun trouvé entre les deux projets.")
        return

    # 3. Nettoyage colonnes
    if JOIN_KEY in merged_df.columns:
        cols = [JOIN_KEY] + [c for c in merged_df.columns if c != JOIN_KEY]
        merged_df = merged_df[cols]

    # 4. Export
    output_filename = os.path.join(output_dir, "comparaison_intersection_clean.csv")
    merged_df.to_csv(output_filename, index=False)

    print(f"Succès ! Données exportées vers {output_filename}")
    print(f"Total de modèles communs : {len(merged_df)}")
    print(merged_df.head())


if __name__ == "__main__":
    main()
