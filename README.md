# Image to Spatial Knowledge Graph

This project provides a complete Python pipeline to analyze a static image, perform object detection, estimate depth, and calculate spatial relationships between all detected objects. The final output is a formal Knowledge Graph (KG) saved in the **Turtle (TTL)** format, ready for use in RDF databases or semantic applications.

This script uses:
* **YOLOv8** for state-of-the-art object detection.
* **Depth Anything v2** for high-quality monocular depth estimation.
* **Shapely** for robust 2D geometric relationship calculations.
* **RDFLib** for generating the final Knowledge Graph.

## 🚀 Features

* **Object Detection:** Identifies multiple objects in an image (e.g., `person`, `dog`, `bicycle`).
* **Depth Estimation:** Calculates the relative depth for each detected object.
* **Spatial Relationship Logic:** Determines how objects are positioned relative to each other (e.g., `is_left_of`, `is_above`, `contains`, `is_overlapping`).
* **Knowledge Graph Export:** Saves the entire scene, including all objects, their properties (bbox, depth, confidence), and their inter-relationships, as a `.ttl` file.
* **Visualization:** Displays the source image with bounding boxes and a colorized depth map for visual verification.



## 🛠️ Installation & Setup (with Miniconda)

This guide assumes you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and are in the project's root directory.

1.  **Create a Conda Environment:**
    We will create a new environment named `scene-kg` with Python 3.10.

    ```bash
    conda create -n scene-kg python=3.10
    conda activate scene-kg
    ```

2.  **Install PyTorch (Essential):**
    For CUDA (NVIDIA GPU) support, which is **highly recommended** for performance, install PyTorch using Conda's specific channel:

    ```bash
    # For NVIDIA GPU (Recommended)
    conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -c nvidia
    ```
    
    If you only have a CPU, use this command instead:
    ```bash
    # For CPU-Only
    conda install pytorch torchvision -c pytorch
    ```

3.  **Install Remaining Dependencies:**
    Install the rest of the required Python packages using `pip`:

    ```bash
    pip install ultralytics shapely opencv-python transformers rdflib
    ```

## 🏃 Usage

Run `main.py` from your terminal, passing the path to your image as an argument.

```bash
python main.py "path/to/your/image.jpg"