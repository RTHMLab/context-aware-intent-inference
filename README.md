# Context-Aware Intent Inference

This repository contains the code pipeline for context-aware human intent inference using wearable motion sensing and first-person RGB-D visual context.

This codebase supports the experiments reported in:

**Bayesian Belief Fusion for Intent Inference via Egocentric Vision and Gesture Recognition**

## Overview

The pipeline contains four main components:

1. **Gesture Recognition (GR)**  
   Predicts intent from Xsens joint-angle motion windows.

2. **2D Mapping / Proximity Extraction**  
   Uses synchronized RGB-D frames, object detections, depth values, and Xsens segment pose data to estimate human-object proximity over time.

3. **Bayesian Fusion (BF)**  
   Combines GR confidence scores with object-proximity evidence using manually defined Bayesian likelihoods.

4. **End-to-End Transformer (E2E)**  
   Uses RGB-D frames and Xsens motion data directly in a multimodal deep learning model.

## Data Access

The raw RGB-D and motion-capture data are not included in this repository due to file size and participant privacy.

To request access to the data, please contact:

    ptimilsina@famu.fsu.edu

After receiving the data, place it under the local `data/` directory as described below.

The `data/` folder is expected to exist locally, but its contents are ignored by Git and are not pushed to GitHub.

## Expected Local Data Layout

Expected local structure:

    data/
    ├── Synced-color-depthPNG/
    │   ├── sub2b/
    │   ├── sub3/
    │   ├── sub4-001/
    │   └── ...
    ├── extracted_JointAngles/
    │   ├── sub2b_ja.csv
    │   ├── sub3_ja.csv
    │   └── ...
    ├── xsens_segments/
    │   ├── sub3_Segment Position.csv
    │   ├── sub3_Segment Orientation - Euler.csv
    │   └── ...
    ├── trials/
    │   ├── sub3_trials.csv
    │   └── ...
    └── models/
        └── yolov8lFT.pt

## Install

Create and activate a Python environment, then install dependencies:

    pip install -r requirements.txt

## Run Pipeline

Run all commands from the repository root.

Run the components in the following order.

### 1. Gesture Recognition

    scripts/run_gr_full_pipeline_all_subjects.sh

This runs the GR leave-one-subject-out training, fine-tuning, and inference pipeline.

### 2. 2D Mapping / Proximity Extraction

    scripts/run_2d_mapping_full_pipeline_all_sessions.sh

This runs object-depth extraction and proximity mapping for all sessions.

### 3. Bayesian Fusion

    scripts/run_final_yolov8lFT_bf_pipeline.sh

This combines GR outputs with proximity evidence to generate Bayesian Fusion predictions.

### 4. End-to-End Transformer

    scripts/run_e2e_transformer_full_pipeline_all_subjects.sh

This runs the E2E leave-one-subject-out training, fine-tuning, and inference pipeline.

### 5. Final Evaluation

    scripts/run_final_evaluation.sh

This generates the final method-comparison outputs under:

    docs/experiment_reports/final_method_comparison/

### 6. Final Figures

    scripts/run_final_plots.sh

### 7. Final Statistical Tests

    scripts/run_final_stats.sh

## Outputs

Generated prediction outputs are written locally under `results/` and are ignored by Git.

Selected compact report files and final figures are tracked under:

    docs/experiment_reports/
