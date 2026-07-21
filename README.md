Crop Disease Detection

A CNN-based image classifier for detecting plant diseases across 15 classes (Pepper, Potato, Tomato — healthy and diseased variants), trained on the PlantVillage dataset (54,000+ labeled leaf images).

Model Weights

Pretrained weights are hosted on Hugging Face (excluded from this repo due to size):

 https://huggingface.co/niranthanand/plant_disease/tree/main

Download best_model.pth and place it at saved_models/best_model.pth before running evaluation or inference.

Results
Best model checkpoint: Epoch 28/30 — Train Acc 94.49%, Val Acc 93.56%
Final epoch (30/30): Train Acc 94.83%, Val Acc 93.20%
Full metrics: see saved_models/classification_report.txt
Confusion matrix: see saved_models/confusion_matrix.png
Confusion matrix highlights

Most classes classify near-perfectly (e.g. Tomato_YellowLeaf_Curl_Virus: 326/331 correct). Two notable confusion points:

Potato_healthy: 15/18 correct, with 3 misclassified as Pepper_bell_healthy — likely due to a small test sample size and visual similarity between healthy leaves across species.
Tomato_Spider_mites: 128/173 correct, with 35 misclassified as Tomato_healthy — early-stage spider mite damage causes subtle stippling that can resemble healthy leaf texture.