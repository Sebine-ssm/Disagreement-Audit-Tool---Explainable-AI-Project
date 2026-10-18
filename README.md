
# When Stable Explanations Disagree: A Stability–Localization Disagreement Audit for Chest X-ray Classification

## Project Overview

Explainable Artificial Intelligence (XAI) methods are increasingly used to interpret predictions made by deep learning models, particularly in high-stakes domains such as medical imaging. However, an explanation that remains visually or quantitatively stable under image perturbations may not necessarily remain aligned with clinically relevant regions.

This project investigates a specific question:

**Can an explanation remain stable under image corruption while losing localization alignment with radiologist-annotated regions?**

Using a ResNet-50 chest X-ray classifier initialized with pretrained weights and three XAI methods, this study evaluates explanation stability and localization jointly under controlled image corruptions.

Rather than treating explanation stability as sufficient evidence of reliability, the project introduces a **stability–localization disagreement audit** to identify cases in which an explanation remains quantitatively stable but its alignment with annotated lung-opacity regions deteriorates.

## Research Objectives

1. Generate explanations using three different XAI methods for chest X-ray classification.
2. Evaluate how explanation maps change under Gaussian blur and Gaussian noise.
3. Quantify explanation stability using Spearman rank correlation.
4. Measure localization alignment using radiologist-provided bounding boxes.
5. Identify and analyze cases where explanation stability remains high despite a meaningful reduction in localization alignment.
6. Examine whether the classifier's predicted label changes in these disagreement cases.
7. Compare explanation stability across XAI methods using paired bootstrap confidence intervals.

## Dataset

**Source:** RSNA Pneumonia Detection Challenge (2018)

The dataset contains chest X-ray images in DICOM format, along with annotations identifying lung-opacity regions.

For this project, images were used to develop a binary classification task distinguishing cases with and without annotated lung opacity.

### Dataset Preparation

- Approximately 29,684 images were included in the modeling dataset.
- Images were divided into training, validation, and test sets.
- Stratified image-level splitting was used.
- The final explainability evaluation used **300 held-out images: 150 ground-truth positive cases and 150 ground-truth negative cases**.

The evaluation groups were defined using the **dataset annotations rather than the classifier's predictions**.

Consequently, the positive group may contain both true positives (correctly classified positive images) and false negatives (positive images incorrectly classified as negative). Similarly, the negative group may contain true negatives and false positives.

The primary stability–localization disagreement audit focuses on the **150 annotated positive images** because their radiologist-provided bounding boxes enable localization alignment measurement.

The full DICOM dataset is not included in this repository because of its size and dataset distribution requirements.

Dataset access: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

## Classification Model

The study uses a ResNet-50 convolutional neural network adapted for binary chest X-ray classification.

The classifier was trained and evaluated before conducting the explainability experiments.

| Evaluation metric | Result |
|---|---:|
| Validation AUROC | Approximately 0.8864 |
| Test AUROC | Approximately 0.8793 |
| Test accuracy | Approximately 77.77% |

The objective of this project is not to introduce a new classification architecture, but to investigate the behavior of explanation methods applied to an existing classifier.

## Explainability Methods

Three XAI methods were evaluated.

### 1. Score-CAM

Score-CAM generates class activation maps by weighting feature maps according to their contribution to the model's target-class score.

It provides spatial explanations derived from intermediate convolutional representations.

### 2. Integrated Gradients

Integrated Gradients attributes a model prediction to input features by accumulating gradients along a path between a reference baseline and the input.

It provides input-level feature attributions based on the model's gradients.

### 3. Occlusion Sensitivity

Occlusion Sensitivity measures changes in the model's output when portions of an image are systematically obscured.

It identifies image regions that influence the prediction through perturbation-based evaluation.

These methods were selected to compare different explanation mechanisms rather than relying on a single attribution technique.

### Explanation Target

For each image, the classifier's **original predicted class** was selected as the explanation target. This target was held fixed when generating explanations for the original image and its corrupted versions.

Holding the target fixed allows attribution maps to be compared for the same class, even when image corruption changes the classifier's predicted label.

However, the explanation target is not necessarily the ground-truth class.

For example, if a ground-truth positive image is incorrectly classified as negative, the XAI methods explain the model's negative-class prediction.

This distinction affects the interpretation of localization alignment. In false-negative cases, low overlap with positive-opacity bounding boxes does not necessarily indicate that an explanation is unfaithful to the model's negative-class prediction.

## Experimental Design

The final evaluation used:

- **300 held-out chest X-rays:** 150 ground-truth positive and 150 ground-truth negative images.
- **3 XAI methods:** Score-CAM, Integrated Gradients, and Occlusion Sensitivity.
- **7 image conditions:** The original image and six corrupted versions.

### Image Corruptions

| Corruption | Severity | Parameter |
|---|---|---|
| Gaussian blur | Mild | σ = 0.5 |
| Gaussian blur | Moderate | σ = 1.5 |
| Gaussian blur | Severe | σ = 3.0 |
| Gaussian noise | Mild | σ = 0.02 |
| Gaussian noise | Moderate | σ = 0.05 |
| Gaussian noise | Severe | σ = 0.10 |

The expanded evaluation produced **6,300 method–image–condition records**.

The primary disagreement audit focused on the 150 annotated positive images across three explanation methods and six corrupted conditions, resulting in **2,700 evaluations**.

These are repeated evaluations of images under different methods and conditions, not 2,700 independent patients.

## Evaluation Metrics

### Explanation Stability

Explanation stability was measured using Spearman rank correlation between the original and corrupted attribution magnitude maps.

A higher correlation indicates greater preservation of the relative spatial ranking of attribution values.

### Localization Alignment

Localization alignment was measured as the proportion of total attribution magnitude falling inside the union of radiologist-annotated lung-opacity bounding boxes.

**Localization alignment = Attribution magnitude inside annotated regions / Total attribution magnitude**

Higher values indicate greater concentration of attribution magnitude within annotated regions.

This measure evaluates agreement with bounding-box regions, not clinical correctness or causal faithfulness.

### Stability–Localization Disagreement

The primary audit identified cases satisfying both conditions:

- Explanation stability ≥ 0.80.
- Localization alignment decreases by at least 0.05 compared with the original image.

These thresholds are exploratory operational definitions rather than clinically validated cutoffs.

### Paired Bootstrap Comparison

A supplementary analysis compared explanation stability between the three XAI methods.

For each corruption type and severity, stability measurements were paired by image identifier so that methods were compared using the same chest X-rays.

For each pair of methods, the analysis calculated image-level differences in explanation stability and performed **5,000 bootstrap resamples with replacement**.

The resulting bootstrap distributions were used to estimate **95% percentile confidence intervals** for the mean stability differences.

Confidence intervals excluding zero provide evidence of a difference in mean explanation stability for the corresponding method pair and corruption condition, subject to the assumptions and limitations of the sampling design.

This analysis evaluates uncertainty in **between-method stability comparisons**. It is separate from the primary stability–localization disagreement audit and does not directly establish the statistical significance of the disagreement rate.

## Key Findings

### 1. Stable Explanations Can Lose Localization Alignment

Of the 2,700 positive-case method–image–corruption evaluations:

- **171 evaluations (6.33%)** met the disagreement criterion.
- Among the 1,076 high-stability evaluations, **171 (15.9%)** also exhibited the specified localization decrease.

These findings show that high explanation stability does not necessarily imply preserved localization alignment.

### 2. Most Flagged Cases Retained the Same Predicted Class

Of the 171 flagged evaluations:

- **169 (98.83%)** retained the original predicted class after corruption.

Thus, stability–localization disagreement frequently occurred without a change in the classifier's predicted label.

**Prediction consistency is not equivalent to classification correctness.**

An unchanged predicted class may be correct or incorrect relative to the ground-truth label.

Therefore, the 98.83% result represents prediction consistency among flagged evaluations, not classification accuracy.

### 3. Disagreement Varies Across Methods and Corruption Conditions

The frequency of flagged cases differed across the three XAI methods and corruption settings.

One notable condition was Occlusion Sensitivity under mild Gaussian noise, with **43 of 150 positive images (28.67%)** meeting the disagreement criterion.

More severe corruption does not necessarily produce more flagged disagreements, because the audit specifically requires explanations to remain highly stable.

### 4. Qualitative Case Study

A selected Score-CAM case under severe Gaussian blur illustrated substantial localization loss despite high explanation stability and an unchanged predicted class.

For this example:

- Explanation stability: approximately 0.810.
- Original localization alignment: approximately 0.488.
- Corrupted localization alignment: approximately 0.161.
- Alignment change: approximately −0.327.

The case study provides an illustrative example of why stability and localization should be evaluated together.

It should not be interpreted as evidence that Score-CAM is consistently the least reliable method.

### 5. Threshold Sensitivity

The disagreement audit was also evaluated using alternative stability and localization-loss thresholds.

This analysis demonstrated that the percentage of flagged evaluations depends on the chosen operational thresholds.

The threshold-sensitivity results help characterize the robustness of the audit's findings but do not establish clinically validated cutoff values.

## Main Contribution

The project's primary contribution is an **empirical stability–localization disagreement audit** that jointly evaluates explanation stability and bounding-box localization alignment under controlled image corruptions.

Instead of asking only whether an explanation changes after image corruption, the audit asks whether an explanation can remain stable while losing alignment with annotated regions.

The study combines quantitative attribution comparisons, localization measurements, corruption experiments, and disagreement analysis to characterize a specific limitation of evaluating explanations through stability alone.

The work is an exploratory empirical audit, not a new attribution algorithm or a clinically validated explanation-reliability metric.

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_ScoreCAM.ipynb
│   ├── 04_Integrated_Gradients.ipynb
│   ├── 05_Occlusion_Sensitivity.ipynb
│   └── 06_XAI_Comparison.ipynb
├── src/
│   └── xai_utils.py
├── results/
│   ├── figures/
│   └── summaries/
├── poster/
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above is the intended repository layout. Filenames and paths should be updated to match the files actually uploaded.*

## Reproducibility and Setup

The experiments were developed in **Google Colab** using Python.

### 1. Obtain the Dataset

Download the RSNA Pneumonia Detection Challenge dataset and its corresponding annotations from the official competition page.

Follow the dataset provider's access and usage requirements.

### 2. Install Dependencies

The notebooks use packages from the Python scientific computing and deep learning ecosystem, including:

- PyTorch and torchvision
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- pydicom
- Captum

Install the project's dependencies using:

```bash
pip install -r requirements.txt
```

The exact package versions and complete dependency list should be documented in `requirements.txt`.

### 3. Configure File Paths

The original experiments were run in Google Colab with files stored in Google Drive.

Users must update the notebook path configuration to point to their own dataset, annotations, model checkpoint, and results directories.

The repository does not automatically reproduce the original Google Drive directory structure.

### 4. Execute the Notebooks

The intended workflow is:

1. `01_Data_Preprocessing.ipynb` — Dataset preparation and preprocessing.
2. `02_Model_Training.ipynb` — ResNet-50 training and evaluation.
3. `03_ScoreCAM.ipynb` — Score-CAM pilot analysis.
4. `04_Integrated_Gradients.ipynb` — Integrated Gradients pilot analysis.
5. `05_Occlusion_Sensitivity.ipynb` — Occlusion Sensitivity pilot analysis.
6. `06_XAI_Comparison.ipynb` — Expanded evaluation, statistical comparisons, disagreement audit, sensitivity analysis, and qualitative case study.

The final comparison notebook requires the trained classifier checkpoint and supporting files generated or used in the earlier stages.

### 5. Inspect Saved Results

The repository's `results/` directory is intended to contain selected summary CSVs and figures so the main findings can be examined without recomputing every attribution map.

The paired bootstrap comparison results are saved as `xai_paired_bootstrap_results.csv`, with a corresponding visualization saved as `xai_paired_bootstrap_comparison.png` in the configured expanded-results directory.

Full end-to-end reproduction may require substantial computation and access to the original data and trained model checkpoint.

## Limitations

This study has several important limitations:

- **Single model:** The experiments focus on one ResNet-50 classifier.
- **Single dataset:** The results may not generalize to other imaging datasets or clinical settings.
- **Limited evaluation sample:** The expanded explainability evaluation includes 300 held-out images.
- **Bounding-box annotations:** Radiologist boxes approximate annotated lung-opacity regions and do not establish complete ground-truth explanations.
- **Classification correctness and explanation targets:** The primary audit includes ground-truth positive images regardless of whether the classifier predicts them correctly. For false-negative cases, the explanation targets the predicted negative class, making localization alignment with positive-opacity annotations harder to interpret. The current analysis does not separately report disagreement rates for true positives and false negatives.
- **Artificial corruptions:** Gaussian blur and noise may not fully represent real-world acquisition artifacts or distribution shifts.
- **Exploratory thresholds:** The disagreement thresholds were selected for analysis and are not clinically validated.
- **Repeated measurements:** Method–image–corruption evaluations are not independent patient-level observations.
- **Bootstrap uncertainty:** The paired bootstrap estimates uncertainty in mean stability differences for matched images. Its confidence intervals do not account for every potential source of uncertainty, such as model training variability or dataset shifts.
- **Explanation faithfulness:** Stability and localization alignment do not, by themselves, establish whether an explanation faithfully represents the model's decision process.

The findings should therefore be interpreted as evidence of a measurable disagreement between explanation-evaluation criteria, not as clinical validation of the model or explanation methods.

## Future Work

This project will be extended in **Project Expo 2**, with a focus on strengthening the interpretation and evaluation of stability–localization disagreement.

Planned extensions include:

- **Classification-stratified analysis:** Compare disagreement patterns between correctly classified positive images (true positives) and incorrectly classified positive images (false negatives).
- **Explanation target interpretation:** Investigate how the relationship between predicted classes and ground-truth labels affects the interpretation of localization alignment.
- **Alternative evaluation metrics:** Explore additional localization and explanation-faithfulness measures to complement bounding-box attribution overlap.
- **Corruption robustness:** Investigate more clinically realistic image corruptions and distribution shifts.
- **Threshold sensitivity:** Further examine how disagreement rates change under alternative stability and localization-loss thresholds.

These extensions aim to better characterize when stability–localization disagreement occurs and how classification correctness influences its interpretation.

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. International Conference on Machine Learning. https://proceedings.mlr.press/v70/sundararajan17a.html

2. Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks*. European Conference on Computer Vision. https://doi.org/10.1007/978-3-319-10590-1_53

3. Wang, H., et al. (2020). *Score-CAM: Score-weighted visual explanations for convolutional neural networks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

4. RSNA. (2018). *RSNA Pneumonia Detection Challenge*. https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge

---

**Project type:** Explainable AI / Responsible AI research project

**Institution:** Duke University

**Status:** Course research project — October 2026
