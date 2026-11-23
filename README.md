# EnsembleLab: An Interactive Ensemble Learning Dashboard for Sri Lankan Tourism Review Analytics

## 1. Project Overview

**EnsembleLab** is an interactive machine learning dashboard developed to explore and compare ensemble learning techniques for sentiment classification of Sri Lankan tourism reviews.

The project investigates how different ensemble learning approaches can classify tourism reviews into Positive, Negative, and Neutral sentiment categories. It also explores a hybrid architecture that combines textual and metadata-based machine learning models.

The project provides an interactive Streamlit dashboard for exploring model performance, comparing experimental results, and predicting sentiment from new tourism reviews.

## 2. Project Objectives

- Implement and evaluate different ensemble learning techniques.
- Compare Bagging, Boosting, and Stacking approaches.
- Investigate Random Forest and XGBoost performance.
- Develop a hybrid sentiment classification approach combining text and metadata.
- Evaluate models using accuracy, macro precision, macro recall, and macro F1-score.
- Develop an interactive dashboard for presenting experimental findings and predictions.

## 3. Dataset

The project uses the **Sri Lankan Tourism Review Sentiment-Rating Incongruence Dataset**, published by `UOM-CSE-E23` on Hugging Face.

Dataset characteristics:

- 16,155 review records after exact duplicate-row removal.
- Covers the period 2010–2023.
- Includes 11 tourism attraction categories.
- Contains review text, titles, location information, user country, and other metadata.
- Sentiment classes: Positive, Negative, and Neutral.
- Dataset license: CC BY 4.0.

### Sentiment Distribution

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 13,040 | 80.72% |
| Negative | 1,602 | 9.92% |
| Neutral | 1,513 | 9.37% |
| **Total** | **16,155** | **100%** |

The dataset has a substantial class imbalance, with Positive reviews representing the majority class.

### Data Preprocessing

The preprocessing workflow includes:

- Dataset inspection and cleaning.
- Exact duplicate-row removal.
- Feature selection.
- Text vectorization using TF-IDF.
- Categorical feature encoding using One-Hot Encoding.
- Numerical feature scaling.
- Grouped stratified train-test splitting to prevent identical review texts from appearing in both training and testing sets.

The final split contains 12,924 training records and 3,231 testing records.

The selected features include review text, title, location type, province, location name, user country, review length, and title length.

Rating-related fields and other potentially leakage-prone fields were excluded from model training.

## 4. Machine Learning Models

The project implements and evaluates the following approaches:

### Decision Tree
A baseline tree-based classification model.

### Bagging
A Bootstrap Aggregating approach using multiple decision trees to improve prediction stability.

### Random Forest
An ensemble of decision trees trained using randomized subsets of data and features.

### AdaBoost
A boosting technique that combines weak learners by assigning greater attention to previously misclassified observations.

### Gradient Boosting
A sequential boosting approach that builds models to reduce errors made by previous learners.

### XGBoost
An optimized gradient boosting implementation used for multiclass sentiment classification.

### Stacking
A stacked ensemble that combines predictions from multiple base learners using a meta-classifier.

### Hybrid Ensemble
A hybrid architecture combining two separate prediction branches:

- **Text branch:** TF-IDF feature extraction with Logistic Regression.
- **Metadata branch:** One-Hot Encoding and numerical scaling with a Random Forest classifier.
- **Fusion:** Equal-weight soft voting using the predicted class probabilities from both branches.

This architecture explores the combination of textual information and structured review metadata.

## 5. Experimental Results

The models were evaluated using a held-out test dataset containing 3,231 records.

### Model Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 0.5587 | 0.4521 | 0.5277 | 0.4285 |
| Bagging | 0.7249 | 0.5547 | 0.5987 | 0.5532 |
| Random Forest | 0.8205 | 0.6192 | 0.6426 | **0.6299** |
| AdaBoost | 0.8124 | 0.5706 | 0.3520 | 0.3340 |
| Gradient Boosting | **0.8298** | 0.5871 | 0.4222 | 0.4385 |
| XGBoost | 0.7753 | 0.5839 | **0.6698** | 0.6101 |
| Stacking | 0.7406 | 0.5941 | 0.6660 | 0.6008 |
| Hybrid Ensemble | 0.8066 | 0.5841 | 0.5984 | 0.5909 |

### Key Findings

- Random Forest achieved the highest Macro F1-score of **0.6299**, providing the strongest overall balance across sentiment classes among the evaluated models.
- Gradient Boosting achieved the highest accuracy of **0.8298**.
- XGBoost achieved the highest macro recall of **0.6698**.
- The Hybrid Ensemble achieved an accuracy of **0.8066** and a Macro F1-score of **0.5909**.
- Accuracy alone does not fully represent performance on this imbalanced dataset. Macro-averaged metrics provide additional insight into performance across minority sentiment classes.

These results demonstrate the importance of evaluating ensemble approaches using multiple performance metrics rather than relying exclusively on accuracy.

## 6. Interactive Dashboard

The dashboard is developed using Streamlit and provides the following sections:

### Overview
- Summary of the implemented machine learning models.
- Dataset and experiment information.
- Key model evaluation metrics.

### Model Comparison
- Comparison of model accuracy.
- Macro F1-score comparison.
- Training time comparison.
- Detailed evaluation metrics.

### Random Forest Prediction
- Interactive sentiment prediction using the trained Random Forest model.
- Review input and predicted sentiment output.

### Hybrid Prediction
- Interactive sentiment prediction using the hybrid text and metadata architecture.
- Combines textual and structured metadata information.

## 7. Technologies Used

- **Programming Language:** Python
- **Frontend / Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Text Processing:** TF-IDF
- **Data Visualization:** Plotly
- **Model Serialization:** Joblib
- **Development Environment:** Jupyter Notebook and Visual Studio Code
- **Version Control:** Git and GitHub

## 8. Project Structure

```text
EnsembleLab/
│
├── app/
│   └── dashboard.py
│
├── datasets/
│   └── Processed_Reviews_with_Sentiment.csv
│
├── models/
│   ├── ensemble_models.pkl              # Download from GitHub Release
│   ├── hybrid_metadata_model.pkl        # Download from GitHub Release
│   ├── hybrid_text_model.pkl
│   ├── label_encoder.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   └── 02_Hybrid_Ensemble_Experiment.ipynb
│
├── results/
│   ├── hybrid_evaluation.pkl
│   ├── hybrid_model_comparison.csv
│   ├── model_accuracy.png
│   ├── model_comparison.csv
│   ├── model_macro_f1.png
│   ├── model_training_time.png
│   └── test_predictions.pkl
│
├── .gitignore
└── README.md
```

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- Git.
- The project repository.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dhanuja416/EnsembleLab.git
cd EnsembleLab
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib
```

Use compatible library versions when loading the serialized models.

### Step 4: Download the Large Model Files

The two large model files are distributed separately through the GitHub Release to keep the repository lightweight.

Visit:

[EnsembleLab v1.0.0 Release](https://github.com/Dhanuja416/EnsembleLab/releases/tag/v1.0.0)

Download:

- `ensemble_models.pkl`
- `hybrid_metadata_model.pkl`

Place both files inside the project's `models/` directory.

Ensure the final directory contains all five required model files.

### Step 5: Run the Dashboard

```bash
python -m streamlit run app/dashboard.py
```

The dashboard will open in your default web browser.

## 10. Limitations and Future Improvements

Potential future improvements include:

- Exploring systematic hyperparameter optimization.
- Investigating additional class imbalance handling techniques.
- Evaluating additional ensemble algorithms.
- Improving minority-class sentiment classification.
- Exploring more advanced transformer-based text representations.
- Deploying the dashboard to a public hosting platform.
- Investigating additional explainability and feature-importance techniques.

The experimental results should be interpreted in the context of the dataset's class imbalance and the dataset's sentiment annotation methodology.

## 11. Generative AI Usage Acknowledgement

Generative AI tools, including ChatGPT, were used during the development of this project to assist with Python code development, debugging, explanation of machine learning concepts, Streamlit dashboard implementation, and interpretation of experimental results.

The tools were used as development and learning assistants. The project code and suggestions were reviewed, adapted, and tested by the project author. Model training, predictions, evaluation metrics, visualizations, and experimental findings were generated through the implemented Python experiments.

The author takes responsibility for the final implementation, verification of results, and submitted work.

## 12. Repository

**GitHub:** [https://github.com/Dhanuja416/EnsembleLab](https://github.com/Dhanuja416/EnsembleLab)

**Release:** [EnsembleLab v1.0.0](https://github.com/Dhanuja416/EnsembleLab/releases/tag/v1.0.0)