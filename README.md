# Ship Mast Defect Detection using Siamese Network (ResNet50)

## Overview

This project focuses on detecting defects in ship masts using a combination of synthetic data generation and deep learning.

The idea is simple:

* Generate controlled examples of **good and damaged ship masts**
* Train a **Siamese Network (ResNet50 backbone)** to learn similarity between them

Instead of a standard classifier, the model learns to compare two images and decide whether they belong to the same condition or not.

---

## What the project includes

* Synthetic dataset generation using **Diffusion Models + ControlNet**
* Image preprocessing (cropping mast regions)
* Manual dataset cleaning
* Pair creation for Siamese training
* Siamese Network with **ResNet50**

---

## Dataset Preparation (What was actually done)

The dataset wasn’t just generated and used directly — some practical steps were involved:

* Ship images were first collected, and the **mast region was manually cropped** so the model focuses only on the relevant part.
* Synthetic images were generated using prompts (good condition and different types of damage).
* Not all generated images were usable. Some had:

  * unrealistic shapes
  * broken or distorted structures
  * poor alignment

These were **manually removed**.

So there is a **human-in-the-loop cleaning step** to keep the dataset reasonably clean.

---

## Pair Creation (Siamese Learning)

The model is trained on image pairs:

* **Label 0 (Similar)**

  * good ↔ good

* **Label 1 (Dissimilar)**

  * good ↔ damaged

This helps the network learn a feature space where similar conditions are closer.

---

## Model Details

* Backbone: **ResNet50 (pretrained)**
* Siamese architecture (shared weights)
* Outputs feature embeddings for comparison

The goal is not classification directly, but learning **distance between images**.

---

## Tech Stack

* PyTorch
* Diffusers (Stable Diffusion)
* ControlNet
* OpenCV
* NumPy / Pandas

---

## Files

* `ship_mast_dataset_colab.ipynb` → Dataset generation
* `ShipMast_Siamese_ResNet50.ipynb` → Model training

---

## How to Run

### 1. Dataset Generation

* Open the dataset notebook in Google Colab
* Set paths for input/output
* Run all cells

### 2. Model Training

* Load the generated dataset
* Run the Siamese network notebook

---

## Note!

* This dataset is **synthetically generated and manually curated**
* Model performance depends heavily on:

  * prompt quality
  * dataset diversity
  * manual filtering
* GPU is recommended (T4 / A100 or similar)

---

## Possible Improvements

* Add real-world ship mast data for validation
* Try contrastive / triplet loss tuning
* Improve prompt diversity for better generalization
* Build a small inference demo (web app)

---

## 🙌 Final Note

This is a practical attempt to combine:

* generative models
* computer vision
* similarity learning

Not a perfect pipeline, but a working approach that can be improved further.
