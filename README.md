# Multimodal Object Locator System (Final-Year Project)

This repository contains the full-stack application and system evaluation framework for a **Multimodal Object Locator System** developed as a final-year project. The system combines computer vision models, text/image embeddings, and semantic retrieval to help users locate objects through multimodal queries.
## Repository Structure

* **`FYP/`**: Contains the main full-stack application (FastAPI backend and frontend interface).
* **`FYP-analysis/`**: Contains Jupyter Notebooks and evaluation scripts used for model training, testing, image/text embedding experiments, and system-level evaluations.

## Core Features
* **Object Detection & Cropping:** Automatically detects target objects in images and extracts bounding box crops.
* **Scene Classification:** Classifies indoor environments using Places365 architecture.
* **Semantic & Image Embedding:** Performs vector-based similarity matching for textual queries and visual content.
* **Spatial & Color Analysis:** Calculates spatial positioning and color properties of targeted items.
