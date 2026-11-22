# When Stable Explanations Disagree: A Stability–Localization Disagreement Audit for Chest X-ray Classification

## Project Overview

Explainable Artificial Intelligence (XAI) methods are increasingly used to interpret predictions made by deep learning models, particularly in high-stakes domains such as medical imaging. However, an explanation that remains visually or quantitatively stable under image perturbations may not necessarily remain aligned with clinically relevant regions. Across 2,700 positive-case evaluations, 171 (6.33%) met the stability–localization disagreement criterion. Explanation similarity achieved an overall screening AUROC of 0.7911 for substantial localization loss, while the 20 most affected images accounted for 50.88% of disagreement events.

This project investigates a specific question:

**Can an explanation remain stable under image corruption while losing localization alignment with radiologist-annotated regions?**

Using a ResNet-50 chest X-ray classifier initialized with pretrained weights and three XAI methods, this study evaluates explanation stability and localization jointly under controlled image corruptions.

Rather than treating explanation stability as sufficient evidence of reliability, the project introduces a **stability–localization disagreement audit** to identify cases in which an explanation remains quantitatively stable but its alignment with annotated lung-opacity regions deteriorates.

This research extends the initial stability–localization disagreement audit through additional analyses of classification correctness, bounding-box-area-adjusted localization, continuous stability–localization relationships, retrospective screening performance, and image-level disagreement concentration. Together, these analyses investigate not only whether disagreement occurs, but also how it varies across evaluation settings and what limitations arise when explanation stability is used as an indicator of localization preservation.

### Explore the Project

**Interactive Dashboard:** [Stability–Localization Disagreement Audit](https://disagreement-audit-tool---explainable-ai-project-eqrkskucbczxc.streamlit.app/)

The dashboard provides an interactive interface for exploring explanation stability, localization alignment, and supporting audit results.

**Research Notebooks:** See the `Python_Notebooks/` directory for the experimental pipeline, statistical analyses, and supporting visualizations.

## Research Objectives

1. Generate explanations using three XAI methods for chest X-ray classification.
2. Evaluate how explanation maps change under Gaussian blur and Gaussian noise.
3. Quantify explanation stability using Spearman rank correlation.
4. Measure localization alignment using radiologist-provided bounding boxes.
5. Identify cases where explanation stability remains high despite a meaningful reduction in localization alignment.
6. Examine prediction consistency and classification correctness in disagreement cases.
7. Investigate the relationship between continuous explanation stability and localization change, including method-specific and condition-specific patterns.
8. Evaluate whether explanation similarity can reliably screen for substantial localization loss.
9. Examine whether disagreement events are concentrated among particular images beyond what would be expected under a condition-stratified permutation benchmark.
10. Develop an interactive dashboard to communicate and explore the audit's findings.

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

### Similarity-Based Screening

A supplementary analysis evaluated whether explanation similarity could serve as a screening signal for substantial localization loss.

Substantial localization loss was defined as an alignment decrease of at least 0.05. Screening performance was assessed using receiver operating characteristic (ROC) analysis and area under the ROC curve (AUROC), treating lower explanation similarity as indicating greater risk of localization loss.

This analysis evaluates the discriminatory ability of explanation similarity for the specified outcome. It does not establish a clinically validated screening threshold.

### Image-Level Disagreement Concentration

The analysis also examined whether disagreement events were distributed evenly across the 150 annotated positive images or concentrated among a smaller subset.

For each image, disagreement events were counted across the three XAI methods and six corruption conditions.

A condition-stratified permutation analysis independently reassigned disagreement flags across images within each method–corruption–severity condition while preserving the number of flagged evaluations in each condition.

Using 2,000 permutations, the observed concentration of disagreement events among the 20 most affected images was compared with the resulting permutation distribution.

This benchmark evaluates concentration relative to the specified reassignment procedure. It does not establish the causes of repeated disagreement or preserve all forms of within-image dependence.

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

### 6. Continuous Stability–Localization Relationships

Across the 2,700 positive-case evaluations, explanation similarity and localization-alignment change had a pooled Spearman correlation of approximately **0.508**, with an image-cluster bootstrap 95% confidence interval of **[0.453, 0.559]**.

The relationship differed across explanation methods:

- Integrated Gradients: ρ ≈ 0.504
- Occlusion Sensitivity: ρ ≈ 0.336
- Score-CAM: ρ ≈ 0.621

Importantly, the pooled positive association did not consistently describe every corruption condition. For Occlusion Sensitivity, all six condition-specific correlations were negative despite the positive pooled correlation, illustrating the importance of stratified interpretation.

### 7. Explanation Similarity Is an Imperfect Screening Signal

The analysis evaluated whether explanation similarity could identify substantial localization loss, defined as an alignment decrease of at least 0.05.

The overall screening AUROC was approximately **0.7911**, with method-specific performance varying across Integrated Gradients, Occlusion Sensitivity, and Score-CAM.

Of 1,145 substantial localization-loss evaluations, 171 retained explanation similarity of at least 0.80 and therefore would not be flagged by a screening rule that treats lower similarity as the warning signal.

Even among 536 evaluations with similarity of at least 0.90, 39 (7.28%) exhibited substantial localization loss.

These results suggest that explanation similarity contains useful information but should not be treated as sufficient evidence of preserved localization alignment.

### 8. Disagreement Events Are Concentrated Among Particular Images

Of the 150 annotated positive images, **70 experienced at least one disagreement event** across the evaluated methods and corruption conditions.

The 20 images with the most disagreement events accounted for **50.88%** of all 171 events.

A condition-stratified permutation analysis, using 2,000 permutations, produced a mean top-20 concentration of approximately **32.84%**, with a 95% permutation interval of **[29.24%, 36.84%]**.

The observed concentration exceeded this benchmark, suggesting that disagreement events were more concentrated among particular images than expected under the specified permutation procedure.

This exploratory result does not establish why particular images experienced more disagreements, and the permutation benchmark does not preserve every form of within-image dependence.

## Main Contribution

This project's primary contribution is an empirical stability–localization disagreement audit for chest X-ray classification. The audit jointly evaluates explanation stability and localization alignment under controlled image corruptions, identifying cases where explanations remain highly similar despite losing alignment with radiologist-annotated lung-opacity regions.

The research extends this audit through additional evaluations of classification correctness, bounding-box-area-adjusted localization, continuous stability–localization relationships, similarity-based screening performance, and image-level disagreement concentration.

Together, these analyses provide a reproducible framework for investigating the limitations of explanation stability as a standalone evaluation criterion.

The work is intended for XAI researchers, medical-imaging AI developers, and practitioners evaluating model explanations. It is an exploratory research audit, not a new attribution algorithm, a clinically validated reliability measure, or a diagnostic system.

## Responsible AI Implications

Explanation methods are often used to make complex machine learning predictions more understandable, particularly in high-stakes applications such as medical imaging. However, a stable explanation does not necessarily indicate that the explanation remains concentrated on annotated abnormalities.

This project demonstrates why explanation stability and localization alignment should be evaluated as complementary properties rather than interchangeable measures of explanation reliability.

The intended audience includes XAI researchers, model evaluators, and medical-imaging AI developers who want to investigate explanation behavior under controlled perturbations.

The audit may help identify cases requiring closer investigation, but its results should not be interpreted as evidence of clinical correctness, explanation faithfulness, or suitability for deployment. Additional validation, including evaluation across datasets, models, and clinically relevant conditions, would be necessary before considering operational use.


## Repository Structure
```
├── Python_Notebooks/
│   ├── 01_RSNA_Data_Preparation.ipynb
│   ├── 02_ResNet50_Training.ipynb
│   ├── 03_Score-CAM.ipynb
│   ├── 04_Integrated_Gradients.ipynb
│   ├── 05_Occlusion_Sensitivity.ipynb
│   ├── 06_XAI_Comparison.ipynb
│   ├── 07_Disagreement_Validation_Notebook.ipynb
│   ├── 08_Localization_Analysis_and_Attribution_Visualization.ipynb
│   ├── 09_Continuous_Stability_Localization_Analysis.ipynb
│   ├── 10_Stability_Screening_and_Localization_Sensitivity.ipynb
│   └── 11_Image_Level_Failure_Concentration_and_Reaesrch_Synthesis.ipynb
│
├── Results/
│   └── ... Research outputs and saved analysis results
│
├── chest_xray_xai_streamlit/
│   ├── data/
│   ├── images/
│   ├── app.py
│   ├── continuous_analysis.py
│   └── requirements.txt
│
├── poster/
│   └── XAI_Expo1_Poster.pdf
│
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducibility and Setup

| Required Resource | Purpose |
|---|---|
| RSNA DICOM images and annotations | Data preparation and localization evaluation |
| Trained ResNet-50 checkpoint | Generate model predictions and attribution maps |
| Saved XAI evaluation results | Inputs for subsequent statistical analyses |
| Python dependencies | Execute research notebooks |
| Streamlit data and dependencies | Run the interactive dashboard |

The experiments were developed in **Google Colab** using Python.

### 1. Obtain the Dataset

Download the RSNA Pneumonia Detection Challenge dataset and its corresponding annotations from the official competition page.

Follow the dataset provider's access and usage requirements.

### Install Dependencies

The repository maintains separate dependency files for the research notebooks and the interactive Streamlit dashboard.

**Research environment**

From the repository root:

```bash
python -m pip install -r requirements.txt
```

**Streamlit dashboard environment**

From the repository root:

```bash
python -m pip install -r chest_xray_xai_streamlit/requirements.txt
```

The research notebooks were primarily developed in Google Colab and may require additional environment-specific configuration, dataset access, and model checkpoints.

The Streamlit application uses its own dependency file to support dashboard deployment independently of the full research environment.

### 3. Configure File Paths

The original experiments were run in Google Colab with files stored in Google Drive.

Users must update the notebook path configuration to point to their own dataset, annotations, model checkpoint, and results directories.

The repository does not automatically reproduce the original Google Drive directory structure.

### 4. Execute the Notebooks

The research workflow consists of 11 notebooks organized into data preparation, model training, explanation generation, statistical analysis, and research synthesis.

1. `01_RSNA_Data_Preparation.ipynb` — Dataset preparation, preprocessing, and train/validation/test splitting.
2. `02_ResNet50_Training.ipynb` — ResNet-50 training and classification evaluation.
3. `03_Score-CAM.ipynb` — Score-CAM pilot analysis.
4. `04_Integrated_Gradients.ipynb` — Integrated Gradients pilot analysis.
5. `05_Occlusion_Sensitivity.ipynb` — Occlusion Sensitivity pilot analysis.
6. `06_XAI_Comparison.ipynb` — Expanded XAI evaluation, paired comparisons, and primary disagreement audit.
7. `07_Disagreement_Validation_Notebook.ipynb` — Validates disagreement calculations, compares true-positive and false-negative cases, examines prediction changes under corruption, and evaluates uncertainty using image-level bootstrap analysis.
8. `08_Localization_Analysis_and_Attribution_Visualization.ipynb` — Investigates localization behavior using bounding-box-area-adjusted attribution enrichment, compares explanation methods, and examines representative attribution visualizations.
9. `09_Continuous_Stability_Localization_Analysis.ipynb` — Continuous stability–localization relationships and stratified analysis.
10. `10_Stability_Screening_and_Localization_Sensitivity.ipynb` — Screening reliability and localization-loss sensitivity.
11. `11_Image_Level_Failure_Concentration_and_Reaesrch_Synthesis.ipynb` — Image-level disagreement concentration, permutation analysis, and research synthesis.

The notebooks are organized in the order of the research workflow. However, complete execution requires the appropriate input files, saved intermediate artifacts, model checkpoint, and configured paths. Some later notebooks reuse outputs generated during earlier experiments.

Full end-to-end reproduction may require substantial computational resources, particularly for generating attribution maps.

### 5. Inspect Saved Results

The repository's Results/ directory contains saved research outputs from the experiments. Selected summary files and visualizations can be inspected without recomputing every attribution map, where those artifacts are available.

The paired bootstrap comparison results are saved as `xai_paired_bootstrap_results.csv`, with a corresponding visualization saved as `xai_paired_bootstrap_comparison.png` in the configured expanded-results directory.

Full end-to-end reproduction may require substantial computation and access to the original data and trained model checkpoint.

## Interactive Streamlit Dashboard

**Link**: https://disagreement-audit-tool---explainable-ai-project-eqrkskucbczxc.streamlit.app/

In addition to the research notebooks, this repository includes a Streamlit application for exploring the stability–localization disagreement audit.

The application is implemented in `chest_xray_xai_streamlit/app.py`, with additional analysis functionality in `continuous_analysis.py`.

The dashboard provides an interactive interface for examining the project's explainability results and supporting analyses.

### Running the Dashboard Locally

From the repository root, install the dashboard dependencies:

```bash
python -m pip install -r chest_xray_xai_streamlit/requirements.txt
```

Launch the application:

```bash
streamlit run chest_xray_xai_streamlit/app.py
```

The application relies on supporting files within the `chest_xray_xai_streamlit/` directory. These files should retain their expected locations when running the dashboard.

The dashboard is a research visualization tool, not a clinically validated diagnostic system.

## Limitations

This study has several important limitations:

- **Single model:** The experiments focus on one ResNet-50 classifier.
- **Single dataset:** The results may not generalize to other imaging datasets or clinical settings.
- **Limited evaluation sample:** The expanded explainability evaluation includes 300 held-out images.
- **Bounding-box annotations:** Radiologist boxes approximate annotated lung-opacity regions and do not establish complete ground-truth explanations.
- **Classification correctness and explanation targets:** The primary audit includes ground-truth positive images regardless of whether the classifier predicts them correctly. The 150-image sample contains 128 true positives and 22 false negatives. Classification-stratified analyses provide additional context, but the small false-negative subgroup limits the strength of comparisons. In false-negative cases, explanations target the predicted negative class, so alignment with positive-opacity annotations requires particular caution.
- **Artificial corruptions:** Gaussian blur and noise may not fully represent real-world acquisition artifacts or distribution shifts.
- **Exploratory thresholds:** The disagreement thresholds were selected for analysis and are not clinically validated.
- **Repeated measurements:** Method–image–corruption evaluations are not independent patient-level observations.
- **Bootstrap uncertainty:** The paired bootstrap estimates uncertainty in mean stability differences for matched images. Its confidence intervals do not account for every potential source of uncertainty, such as model training variability or dataset shifts.
- **Explanation faithfulness:** Stability and localization alignment do not, by themselves, establish whether an explanation faithfully represents the model's decision process.
- **Screening reliability:** The screening AUROC evaluates how well explanation similarity discriminates substantial localization loss under the chosen definition. It does not establish a calibrated or clinically validated reliability measure.
- **Permutation benchmark:** The image-level concentration analysis preserves disagreement totals within method–corruption–severity conditions but does not preserve all within-image dependence. Its results should be interpreted relative to the specified exploratory null model.

The findings should therefore be interpreted as evidence of a measurable disagreement between explanation-evaluation criteria, not as clinical validation of the model or explanation methods.

## Future Work

Several extensions could strengthen the generalizability and interpretation of this exploratory audit:

- **External validation:** Evaluate stability–localization disagreement on additional chest X-ray datasets and model architectures.
- **Clinical interpretation:** Investigate whether images with repeated disagreement events share identifiable imaging characteristics or annotation-related challenges.
- **Alternative explanation evaluation:** Incorporate complementary localization, perturbation-faithfulness, and explanation-reliability measures.
- **More realistic distribution shifts:** Examine acquisition-related artifacts and other clinically relevant perturbations beyond synthetic Gaussian blur and noise.
- **Larger subgroup evaluations:** Expand the false-negative sample to better characterize differences associated with classification correctness.
- **Threshold validation:** Investigate how operational thresholds might be selected and validated for specific evaluation objectives.

Future work should assess whether these patterns generalize beyond the present dataset, model, and experimental conditions. The current audit is not intended for clinical decision-making.

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. International Conference on Machine Learning. https://proceedings.mlr.press/v70/sundararajan17a.html

2. Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks*. European Conference on Computer Vision. https://doi.org/10.1007/978-3-319-10590-1_53

3. Wang, H., et al. (2020). *Score-CAM: Score-weighted visual explanations for convolutional neural networks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

4. RSNA. (2018). *RSNA Pneumonia Detection Challenge*. https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge

---

**Project type:** Explainable AI / Responsible AI research project

**Institution:** Duke University

**Status:** Final research project — November 2026
