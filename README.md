# Context-Aware Intent Inference

This repository contains the code pipeline for context-aware human intent inference using wearable motion sensing and first-person RGB-D visual context.

The project is organized around four main modules:

1. **Gesture Recognition (GR)**  
   Uses processed Xsens joint-angle data and ANVIL gesture annotations to predict early gesture intent from motion windows.

2. **2D Mapping / Proximity Extraction**  
   Uses already-synchronized RGB-D PNG frames, YOLO-based object detection, depth values, and Xsens segment position/orientation data to estimate human-object distances over time.

3. **Bayesian Fusion (BF)**  
   Combines gesture-recognition confidence scores with object proximity information using manually defined Bayesian conditional probability tables.

4. **End-to-End Transformer (E2E)**  
   Uses synchronized RGB-D frames and Xsens motion data to directly predict intent with a multimodal Transformer-based model.
