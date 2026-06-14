# Context-Aware Intent Inference

This repository contains the cleaned code pipeline for context-aware human intent inference using wearable motion sensing and first-person RGB-D visual context.

The project supports four main components:

1. **Gesture Recognition (GR)**  
   Predicts intent from Xsens joint-angle motion windows.

2. **2D Mapping / Proximity Extraction**  
   Uses synchronized RGB-D frames, YOLO object detections, depth values, and Xsens segment pose data to estimate human-object proximity over time.

3. **Bayesian Fusion (BF)**  
   Combines GR confidence scores with object proximity using manually defined Bayesian conditional probability tables.

4. **End-to-End Transformer (E2E)**  
   Uses RGB-D frames and Xsens motion data directly in a multimodal Transformer-based model.

## Repository Status

This repository has been reconstructed around the Ro-Man accepted context-aware intent inference pipeline.

Current cleaned components include:

- GR LOSO training/inference pipeline
- YOLO-depth proximity extraction pipeline
- Final Bayesian Fusion pipeline
- E2E Transformer LOSO pipeline
- General dynamic evaluation script
- Final method-comparison report scripts

Large raw data, generated model outputs, and generated prediction outputs are kept locally and are not pushed to GitHub.

## Local Data Layout

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

The repository assumes RGB-D PNGs are already synchronized. Raw RealSense `.bin` files, camera alignment code, and raw camera timestamp correction are outside the scope of this repository.

## Trial Window Convention

Each annotated action trial is represented as:

    2.5 seconds before gesture onset
    1.0 second after gesture onset

Dynamic labels are assigned as:

    before onset threshold: Nothing
    after onset threshold: action label

The current cleaned split-sit label space contains 9 classes:

    Clean-table
    Nothing
    Pick-up-backpack
    Push-chair
    Sit-on-chair
    Sit-on-couch
    Sit-on-table
    Stand-on-couch
    Wear-backpack

## Main Pipeline Scripts

### Gesture Recognition

Create subject splits:

    python scripts/create_gr_subject_splits.py

Run the full GR pipeline:

    scripts/run_gr_full_pipeline_all_subjects.sh

Main GR source files:

    src/gr/data_loader.py
    src/gr/preprocessing.py
    src/gr/model.py
    src/gr/train.py
    src/gr/fine_tune.py
    src/gr/inference.py

### 2D Mapping / Proximity Extraction

Run YOLO-depth extraction:

    scripts/run_yolo_depth_all_sessions.sh

Run proximity mapping after YOLO detections:

    scripts/run_proximity_after_yolo_all_sessions.sh

Run the final YOLOv8lFT + Bayesian Fusion pipeline:

    scripts/run_final_yolov8lFT_bf_pipeline.sh

Main proximity source files:

    src/proximity_mapping/processRawData.py
    src/proximity_mapping/extractObjectPositions.py
    src/proximity_mapping/mapper.py
    src/proximity_mapping/normalizeTime.py

### Bayesian Fusion

Main BF source file:

    src/bayesian_fusion/run_bayesian_fusion.py

Bayesian Fusion combines GR confidence scores with object-proximity evidence using interpretable expert-defined likelihoods.

### End-to-End Transformer

Run the E2E Transformer LOSO pipeline:

    scripts/run_e2e_transformer_full_pipeline_all_subjects.sh

Match E2E outputs to the GR/BF test-trial set:

    scripts/evaluate_e2e_matched_to_gr.sh

Main E2E source files:

    src/e2e_transformer/dataset_module.py
    src/e2e_transformer/model.py
    src/e2e_transformer/train.py
    src/e2e_transformer/fineTune.py
    src/e2e_transformer/predict_and_compare.py

## Evaluation

General dynamic evaluation script:

    python -m src.evaluation.evaluate_predictions \
      --input_glob "results/<method>/inference_*/<prediction_file>.csv" \
      --method_name <method_name> \
      --output_dir results/evaluation \
      --label_mode dynamic \
      --onset_time_ms 2000 \
      --max_time_ms 2500

Final comparison report:

    scripts/create_final_method_comparison_report.py

This generates:

    docs/experiment_reports/final_method_comparison/

The report includes:

    final_overall_metrics.csv
    final_subject_metrics.csv
    final_per_class_f1.csv
    final_early_prediction_lead_time.csv
    final_early_prediction_summary.csv
    final_ambiguity_metrics.csv
    final_conditional_bf_threshold_sweep.csv

## Experiment Reports

Tracked report folders include:

    docs/experiment_reports/final_yolov8lFT_conf06_pipeline/
    docs/experiment_reports/e2e_transformer_loso_pipeline/
    docs/experiment_reports/final_method_comparison/
    docs/experiment_reports/sub3_original_scope_audit/

The `sub3_original_scope_audit` report isolates the current cleaned outputs to subject sub3 to compare with the narrower original accepted-paper evaluation scope.

## Generated Outputs

Generated outputs are local and ignored by Git, including:

    results/gr/
    results/proximity/
    results/bayesian_fusion/
    results/e2e_transformer/
    results/e2e_transformer_matched_gr/
    results/evaluation/

Only selected compact report CSVs and READMEs under `docs/experiment_reports/` are tracked.

## Notes

This repository is intended to preserve the accepted Ro-Man pipeline while also supporting cleaner internal analysis and reproducible follow-up experiments.

The full cleaned 5-subject LOSO evaluation and the narrower sub3-style audit should be treated as separate analysis scopes.
