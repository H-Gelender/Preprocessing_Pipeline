# tissue_masker.py

OPENSLIDE_PATH = r"C:\Users\henry\Desktop\Asclepios\code\bin"

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import numpy as np
import cv2
from PIL import Image

def generate_tissue_mask(wsi_path: str, level: int = -1, threshold_method: str = 'otsu', min_saturation: int = 30, kernel_size: int = 7) -> np.ndarray:
    """
    Génère un masque binaire pour les régions tissulaires d'une image WSI.

    Args:
        wsi_path (str): Chemin vers le fichier WSI.
        level (int): Le niveau de résolution de la pyramide WSI à utiliser pour la
                     génération du masque. Un niveau plus élevé (nombre plus grand)
                     correspond à une résolution plus basse. Si -1, choisit un niveau
                     avec une dimension maximale d'environ 2048 pixels.
        threshold_method (str): Méthode de seuillage ('otsu' ou 'saturation').
                                'otsu' utilise le seuillage d'Otsu sur l'image en niveaux de gris.
                                'saturation' utilise un seuil sur le canal Saturation de l'image HSV.
        min_saturation (int): Seuil de saturation minimum (utilisé si threshold_method='saturation').
        kernel_size (int): Taille du noyau pour les opérations morphologiques (doit être impair).

    Returns:
        np.ndarray: Un tableau NumPy 2D représentant le masque binaire, où les pixels
                    du tissu sont généralement > 0 (par exemple, 255) et le fond est 0.
                    Les dimensions correspondent à celles du niveau choisi.

    Raises:
        FileNotFoundError: Si le fichier WSI n'est pas trouvé.
        openslide.OpenSlideError: Si le fichier ne peut pas être ouvert par OpenSlide.
        ValueError: Si le niveau spécifié est invalide ou si la méthode de seuillage est inconnue.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
    except FileNotFoundError:
        print(f"Erreur: Fichier non trouvé - {wsi_path}")
        raise
    except openslide.OpenSlideError as e:
        print(f"Erreur: Impossible d'ouvrir le fichier WSI {wsi_path} - {e}")
        raise

    if level == -1:
        # Choisir automatiquement un niveau approprié (basse résolution)
        target_max_dim = 2048
        best_level = slide.level_count - 1
        for i in range(slide.level_count):
            dims = slide.level_dimensions[i]
            if max(dims) <= target_max_dim:
                best_level = i
                break
        level = best_level
        print(f"Niveau de traitement choisi automatiquement : {level} (Dimensions: {slide.level_dimensions[level]})")
    elif level < 0 or level >= slide.level_count:
        slide.close()
        raise ValueError(f"Niveau invalide spécifié : {level}. Le WSI a {slide.level_count} niveaux (0 à {slide.level_count - 1}).")

    try:
        # Lire l'image au niveau choisi
        level_dims = slide.level_dimensions[level]
        # read_region prend (location_x, location_y, level, size_x, size_y)
        # Location (0,0) est le coin supérieur gauche au niveau 0.
        # Size doit être les dimensions du niveau choisi.
        img_pil = slide.read_region((0, 0), level, level_dims)

        # Convertir en NumPy array et s'assurer qu'elle est en RGB (ignorer l'alpha si présent)
        img_rgb = np.array(img_pil.convert('RGB'))

        # --- Méthode 1: Seuillage d'Otsu sur l'image en niveaux de gris ---
        if threshold_method.lower() == 'otsu':
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            # Appliquer le seuillage d'Otsu.
            # Le tissu est généralement plus sombre que le fond (blanc/gris clair),
            # donc on inverse le seuil (THRESH_BINARY_INV) pour que le tissu soit blanc (255).
            threshold_value, mask_binary = cv2.threshold(
                img_gray,
                0, # Ignoré quand Otsu est utilisé
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            print(f"Seuil d'Otsu calculé : {threshold_value}")

        # --- Méthode 2: Seuillage sur la saturation ---
        elif threshold_method.lower() == 'saturation':
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            saturation_channel = img_hsv[:, :, 1] # Canal S
            # Appliquer un seuil simple sur la saturation
            _, mask_binary = cv2.threshold(
                saturation_channel,
                min_saturation,
                255,
                cv2.THRESH_BINARY
            )
            print(f"Seuillage appliqué sur la saturation avec seuil = {min_saturation}")

        else:
            raise ValueError(f"Méthode de seuillage inconnue : {threshold_method}. Choisir 'otsu' ou 'saturation'.")

        # --- Opérations Morphologiques pour nettoyer le masque ---
        if kernel_size > 0:
            # Noyau pour les opérations morphologiques (elliptique est souvent bon pour les formes biologiques)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # 1. Fermeture : Dilatation suivie d'Érosion
            #    Utile pour remplir les petits trous dans les objets (tissu).
            mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

            # 2. Ouverture : Érosion suivie de Dilatation
            #    Utile pour supprimer les petits objets/bruits de fond.
            mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
            final_mask = mask_opened
            print(f"Opérations morphologiques (Fermeture + Ouverture) appliquées avec kernel_size={kernel_size}")
        else:
            final_mask = mask_binary # Pas d'opérations morphologiques
            print("Aucune opération morphologique appliquée.")

        return final_mask

    finally:
        # Toujours fermer le slide après utilisation
        slide.close()

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    # Créez un dossier 'output' s'il n'existe pas
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ** REMPLACEZ CECI par le chemin vers votre fichier WSI **
    # exemple_wsi = "chemin/vers/votre/image.svs"
    # Pour tester sans WSI réel, vous pouvez utiliser un placeholder,
    # mais la fonction lèvera une FileNotFoundError.
    exemple_wsi = "../images/image_test.svs" # Mettez un vrai chemin ici !

    try:
        print(f"Traitement du fichier WSI : {exemple_wsi}")

        # Générer le masque en utilisant la méthode d'Otsu (par défaut)
        tissue_mask_otsu = generate_tissue_mask(exemple_wsi, level=-1, threshold_method='otsu', kernel_size=7)

        # Sauvegarder le masque Otsu en image
        mask_otsu_path = os.path.join(output_dir, "tissue_mask_otsu.png")
        cv2.imwrite(mask_otsu_path, tissue_mask_otsu)
        print(f"Masque (Otsu) sauvegardé dans : {mask_otsu_path}")

        # Générer le masque en utilisant la méthode de saturation
        tissue_mask_sat = generate_tissue_mask(exemple_wsi, level=-1, threshold_method='saturation', min_saturation=40, kernel_size=7)

        # Sauvegarder le masque Saturation en image
        mask_sat_path = os.path.join(output_dir, "tissue_mask_saturation.png")
        cv2.imwrite(mask_sat_path, tissue_mask_sat)
        print(f"Masque (Saturation) sauvegardé dans : {mask_sat_path}")

        # Afficher les masques (optionnel, nécessite matplotlib)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(tissue_mask_otsu, cmap='gray')
        plt.title("Masque Tissulaire (Otsu)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(tissue_mask_sat, cmap='gray')
        plt.title("Masque Tissulaire (Saturation > 40)")
        plt.axis('off')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "tissue_masks_comparison.png")
        plt.savefig(plot_path)
        print(f"Comparaison des masques sauvegardée dans : {plot_path}")
        plt.show() # Décommentez pour afficher interactivement

    except FileNotFoundError:
        print(f"\nErreur: Le fichier WSI d'exemple '{exemple_wsi}' n'a pas été trouvé.")
        print("Veuillez remplacer 'placeholder_image.svs' par le chemin réel de votre fichier WSI dans le script.")
    except openslide.OpenSlideError as e:
         print(f"\nErreur OpenSlide : {e}")
         print("Assurez-vous que le fichier est un WSI valide et que la bibliothèque C OpenSlide est correctement installée.")
    except ImportError:
        print("\nErreur: Matplotlib n'est pas installé. L'affichage des images est désactivé.")
        print("Installez-le avec : pip install matplotlib")
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue : {e}")