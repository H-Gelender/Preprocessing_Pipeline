import pandas as pd
from col_to_remove import remove_hest_columns, remove_stimage_columns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_datasets")

def merge_data(path_stimage: str, path_hest: str, path_sortie: str = "merged_dataset.csv"):
    """
    Nettoie, harmonise, fusionne les datasets STimage et HEST, et enregistre le résultat.

    Cette fonction lit les deux fichiers CSV, filtre les données, supprime les colonnes
    inutiles, standardise les colonnes communes, fusionne les deux datasets en un seul,
    et l'enregistre dans un nouveau fichier CSV.

    Args:
        path_stimage (str): Le chemin vers le fichier CSV de STimage.
        path_hest (str): Le chemin vers le fichier CSV de HEST.
        path_sortie (str): Le chemin où enregistrer le fichier CSV final fusionné.

    Returns:
        pd.DataFrame: Le DataFrame final, nettoyé et fusionné.
    """
    # Listes des colonnes à supprimer

    # --- Traitement du fichier STimage ---
    df_sti = pd.read_csv(path_stimage)
    df_sti = df_sti[df_sti["tech"] == "Visium"]
    df_sti_clean = df_sti.drop(columns=remove_stimage_columns)
    
    # --- Traitement du fichier HEST ---
    df_hest = pd.read_csv(path_hest)
    df_hest = df_hest[df_hest["st_technology"] == "Visium"]
    df_hest_clean = df_hest.drop(columns=remove_hest_columns)

    df_hest_clean['organ'] = df_hest_clean['organ'].fillna('Muscle')
    df_hest_clean = df_hest_clean.drop(columns=['tissue'])
    df_hest_clean = df_hest_clean.rename(columns={'organ': 'tissue'})
    
    df_hest_clean['tissue'] = df_hest_clean['tissue'].str.lower()
    df_hest_clean['species'] = df_hest_clean['species'].replace('Homo sapiens', 'human')
    
    df_hest_clean = df_hest_clean.rename(columns={
        'id': 'slide',
    })

    # --- NOUVELLE ÉTAPE : Fusion des deux DataFrames ---
    df_final = pd.concat([df_sti_clean, df_hest_clean], ignore_index=True)
    
    # --- NOUVELLE ÉTAPE : Enregistrement du fichier CSV final ---
    try:
        df_final.to_csv(path_sortie, index=False)
        print(f"Traitement terminé. Fichier fusionné enregistré sous : {path_sortie}")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'enregistrement du fichier : {e}")

    return df_final
# --- Exemple d'utilisation de la fonction ---
def main():
    metadata_path = Path(__file__).parent.parent / "metadata"

    path1 = metadata_path / "STImage_no_overlapp.csv"
    path2 = metadata_path / "HEST_v1_1_0.csv"
    output_path = metadata_path / "dataset_harmonise.csv"

    df_fusionne = merge_data(path1, path2, output_path)
    logger.info("\nAperçu du DataFrame final fusionné :")
    logger.info(df_fusionne.head())
    logger.info(df_fusionne.tail())

if __name__ == "__main__":
    main()