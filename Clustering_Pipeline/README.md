# WSI Feature Clustering Pipeline

> An end-to-end pipeline to cluster histopathological regions in whole-slide images using patch features extracted from a foundation model.

This project provides an automated pipeline for the analysis of whole-slide images (WSI). It identifies tissue regions, extracts patches, generates feature embeddings using a pre-trained foundation model, and groups these patches into clusters of morphologically similar regions.

## 🚀 Features

  * **Automated End-to-End Workflow**: From the raw WSI to a final cluster visualization map.
  * **Foundation Model Integration**: Uses powerful, pre-trained models (e.g., from `Hugging Face Hub`) for high-quality feature extraction without needing to train a model from scratch.
  * **Smart Patch Extraction**: A two-step process first identifies tissue regions on a low-resolution image to then intelligently extract high-resolution patches, saving time and computational resources.
  * [cite\_start]**Intelligent Clustering**: Implements K-Medoids clustering, which is robust to outliers and provides actual patches as cluster exemplars (medoids)[cite: 1]. It also automatically finds the optimal number of clusters (`k`).
  * **Highly Configurable**: All pipeline parameters (patch size, model name, thresholds, etc.) are managed in a single `config.json` file for easy experimentation.
  * [cite\_start]**Organized Outputs**: Automatically organizes the resulting patch images into folders by cluster, ready for review or for use in downstream machine learning tasks[cite: 1].
  * [cite\_start]**Modular Architecture**: The code is split into logical modules (`masking`, `patching`, `features`, `clustering`), making it easy to understand, maintain, and extend[cite: 1].

## ⚙️ Pipeline Workflow

The pipeline executes the following steps in sequence:

1.  **Tissue Mask Generation**: A low-resolution tissue mask is created from the WSI to isolate tissue from the background, using methods like Otsu's thresholding.
2.  **Patch Extraction**: High-resolution patches are extracted only from the areas identified as tissue by the mask. The coordinates of every patch are systematically saved.
3.  **Feature Extraction**: Each patch is processed by a foundation model (`timm` compatible) to generate a high-dimensional feature vector, or embedding. The resulting features and their corresponding coordinates are saved to `.npy` files.
4.  [cite\_start]**Optimal `k` Search**: The pipeline evaluates a range of cluster numbers (`k`) and determines the optimal value by analyzing the inertia of the clusters[cite: 1].
5.  [cite\_start]**Clustering**: The feature vectors are grouped using the K-Medoids algorithm with the optimal `k`[cite: 1]. A JSON file mapping each cluster to its patch coordinates is generated.
6.  **Output Organization & Visualization**:
      * [cite\_start]The patch image files are copied into sub-folders named after the cluster they belong to[cite: 1].
      * A final scatter plot visualization is created, showing the spatial location of each cluster on the original slide.

## 🔧 Installation

### Prerequisites

1.  **Python 3.8+**
2.  **OpenSlide C Library**: This project uses the `openslide-python` library, which requires the OpenSlide C library to be installed on your system. Please follow the instructions on the [OpenSlide website](https://openslide.org/download/). On Windows, you may need to add the `bin` folder to your system's PATH.
3.  **(Optional) NVIDIA GPU**: A CUDA-compatible GPU is highly recommended for accelerating the feature extraction step. Ensure you have the appropriate NVIDIA drivers and CUDA Toolkit installed.

### Dependencies

Clone the repository and install the required Python packages using the `requirements.txt` file:

```bash
git clone https://github.com/votre-nom/votre-repo.git
cd votre-repo
pip install -r requirements.txt
```

## ▶️ How to Use

1.  **Configure the Pipeline**
    Modify the `config.json` file to set your desired parameters. This is where you define the model to use, the patch size, the range of `k` to test, and other important settings.

    ```json
    {
        "output_base_dir": "pipeline_output",
        "patch_extraction_level": 0,
        "patch_size": 224,
        "feature_model_name": "hf-hub:bioptimus/H-optimus-0",
        "clustering_method": "BanditPam",
        "clustering_k_range": [2, 11]
      }
    ```

2.  **Run the Pipeline**
    Execute the main script from your terminal, providing the path to your WSI file.

    ```bash
    python main.py /path/to/your/slide.svs -c config.json
    ```

    If you omit the `-c` flag, the script will look for `config.json` in the current directory.

## 📂 Output Structure

The pipeline will generate a structured output folder for each processed slide, as defined by `output_base_dir` in your configuration.

```
pipeline_output/
└── <slide_name>/
    ├── tissue_mask.png
    ├── patches/
    │   ├── patch_L0_x..._y..._s224.png
    │   └── ...
    ├── features/
    │   ├── extracted_features.npy
    │   └── extracted_positions.npy
    └── clustering/
        ├── clustered_coordinates.json
        ├── cluster_visualization.png
        └── clustered_patches/
            ├── cluster_0/
            ├── cluster_1/
            └── ...
```

## 🛠️ Key Technologies

  * **PyTorch**: For deep learning model inference.
  * **timm**: For easy access to a wide range of pre-trained computer vision models.
  * **OpenSlide**: For reading and processing whole-slide images.
  * **OpenCV & scikit-image**: For image processing and mask generation.
  * **scikit-learn & sklearn-extra**: For clustering algorithms (K-Medoids).
  * **NumPy**: For efficient numerical operations.
  * **Hugging Face Hub**: For downloading pre-trained foundation models.
