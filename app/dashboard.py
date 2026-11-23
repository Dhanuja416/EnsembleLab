
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import urllib.request
import shutil

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EnsembleLab | Tourism Analytics",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 0px;
        color: #2563EB;
    }

    .sub-header {
        font-size: 15px;
        color: #8B949E;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, #172554, #1E3A8A);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        min-height: 120px;
    }

    .metric-label {
        color: #CBD5E1;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 27px;
        font-weight: 750;
    }

    .metric-description {
        color: #CBD5E1;
        font-size: 12px;
        margin-top: 5px;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "results")


# ==================================================
# LOAD SAVED FILES
# ==================================================

@st.cache_resource
def download_model_file(filename):
    """
    Return a local model path. If a large model is not present
    in the repository checkout, download it from the GitHub Release.
    """

    os.makedirs(MODEL_PATH, exist_ok=True)
    file_path = os.path.join(MODEL_PATH, filename)

    # Keep an existing non-empty local model file.
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        return file_path

    release_url = (
        "https://github.com/Dhanuja416/EnsembleLab/"
        "releases/download/v1.0.0/"
        + filename
    )

    temporary_path = file_path + ".part"

    try:
        request = urllib.request.Request(
            release_url,
            headers={"User-Agent": "EnsembleLab-Streamlit-App"}
        )

        with urllib.request.urlopen(request, timeout=600) as response:
            with open(temporary_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)

        # Do not leave a partial model file if the download is interrupted.
        if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
            raise RuntimeError("The downloaded file is empty.")

        os.replace(temporary_path, file_path)

    except Exception as e:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

        raise RuntimeError(
            f"Unable to download {filename} from the GitHub Release. "
            f"Check that release tag v1.0.0 contains an asset with this "
            f"exact filename. Details: {e}"
        ) from e

    return file_path


@st.cache_resource
def load_models():

    # These two large files are stored as GitHub Release assets.
    ensemble_model_path = download_model_file("ensemble_models.pkl")
    hybrid_metadata_path = download_model_file("hybrid_metadata_model.pkl")

    models = joblib.load(ensemble_model_path)

    preprocessor = joblib.load(
        os.path.join(MODEL_PATH, "preprocessor.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_PATH, "label_encoder.pkl")
    )

    hybrid_text_model = joblib.load(
        os.path.join(MODEL_PATH, "hybrid_text_model.pkl")
    )

    hybrid_metadata_model = joblib.load(hybrid_metadata_path)

    return (
        models,
        preprocessor,
        label_encoder,
        hybrid_text_model,
        hybrid_metadata_model
    )


@st.cache_data
def load_results():

    return pd.read_csv(
        os.path.join(RESULTS_PATH, "hybrid_model_comparison.csv")
    )


# Initialize every variable before loading. If loading fails, stop
# execution here so the rest of the dashboard cannot use undefined names.
models = None
preprocessor = None
label_encoder = None
hybrid_text_model = None
hybrid_metadata_model = None
results_df = None

try:
    (
        models,
        preprocessor,
        label_encoder,
        hybrid_text_model,
        hybrid_metadata_model
    ) = load_models()

    results_df = load_results()

    if results_df is None or results_df.empty:
        raise ValueError("The model comparison CSV is empty or could not be loaded.")

except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# ==================================================
# VALIDATE MODEL RESULTS
# ==================================================

required_columns = [
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Training Time (s)",
    "Prediction Time (s)"
]

missing_columns = [
    column for column in required_columns
    if column not in results_df.columns
]

if missing_columns:

    st.error(
        f"The model comparison file is missing columns: {missing_columns}"
    )
    st.stop()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📊 EnsembleLab")

st.sidebar.caption(
    "Sri Lankan Tourism Review Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Model Comparison",
        "Sentiment Prediction",
        "About the Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.caption("Dataset: Sri Lankan Tourism Reviews")
st.sidebar.caption("Models: 8")
st.sidebar.caption("Task: 3-Class Sentiment Classification")

st.sidebar.markdown("### Developed By")

st.sidebar.markdown(
    """
    **Dhanuja S.A.**

    Index No: 224038C

    Faculty of Information Technology
    """
)

st.sidebar.caption("EnsembleLab | 2026")


# ==================================================
# DASHBOARD OVERVIEW
# ==================================================

if page == "Dashboard Overview":

    st.markdown(
        '<div class="main-header">EnsembleLab</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-header">'
        'An Interactive Ensemble Learning Dashboard for '
        'Sri Lankan Tourism Review Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # CALCULATE BEST MODELS

    best_f1_row = results_df.loc[
        results_df["Macro F1"].idxmax()
    ]

    best_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    best_recall_row = results_df.loc[
        results_df["Macro Recall"].idxmax()
    ]

    # KPI CARDS

    st.markdown(
        '<div class="section-heading">Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">DATASET RECORDS</div>
            <div class="metric-value">16,155</div>
            <div class="metric-description">Tourism reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TRAINED MODELS</div>
            <div class="metric-value">8</div>
            <div class="metric-description">Seven standard models + hybrid</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BEST MACRO F1</div>
            <div class="metric-value">{best_f1_row["Macro F1"]:.4f}</div>
            <div class="metric-description">{best_f1_row["Model"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HIGHEST ACCURACY</div>
            <div class="metric-value">{best_accuracy_row["Accuracy"]:.2%}</div>
            <div class="metric-description">{best_accuracy_row["Model"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # PERFORMANCE HIGHLIGHTS

    st.markdown(
        '<div class="section-heading">Performance Highlights</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.info(
            f"**Best balanced performance:** "
            f"{best_f1_row['Model']} achieved a Macro F1 score "
            f"of {best_f1_row['Macro F1']:.4f}."
        )

    with right:

        st.success(
            f"**Highest accuracy:** {best_accuracy_row['Model']} "
            f"achieved {best_accuracy_row['Accuracy']:.2%} accuracy."
        )

    st.info(
        f"**Highest macro recall:** {best_recall_row['Model']} "
        f"achieved {best_recall_row['Macro Recall']:.4f}."
    )

    st.divider()

    # MODEL PERFORMANCE CHARTS

    st.markdown(
        '<div class="section-heading">Model Performance Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Macro F1 Comparison")

        f1_data = results_df.sort_values(
            "Macro F1",
            ascending=True
        )

        fig_f1 = px.bar(
            f1_data,
            x="Macro F1",
            y="Model",
            orientation="h",
            text_auto=".3f",
            color="Macro F1",
            color_continuous_scale="Blues",
            title="Balanced Classification Performance"
        )

        fig_f1.update_layout(
            height=450,
            xaxis_range=[0, 1],
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=55, b=10)
        )

        st.plotly_chart(
            fig_f1,
            use_container_width=True
        )

    with chart_col2:

        st.markdown("#### Accuracy Comparison")

        accuracy_data = results_df.sort_values(
            "Accuracy",
            ascending=True
        )

        fig_accuracy = px.bar(
            accuracy_data,
            x="Accuracy",
            y="Model",
            orientation="h",
            text_auto=".3f",
            color="Accuracy",
            color_continuous_scale="Teal",
            title="Overall Classification Accuracy"
        )

        fig_accuracy.update_layout(
            height=450,
            xaxis_range=[0, 1],
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=55, b=10)
        )

        st.plotly_chart(
            fig_accuracy,
            use_container_width=True
        )

    st.divider()

    # MODEL RESULTS TABLE

    st.markdown(
        '<div class="section-heading">Complete Model Performance</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        results_df.sort_values(
            "Macro F1",
            ascending=False
        ).style.format({
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "Training Time (s)": "{:.2f}",
            "Prediction Time (s)": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Macro F1 assigns equal importance to all sentiment classes. "
        "It is particularly useful for evaluating this imbalanced dataset."
    )


# ==================================================
# MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    st.markdown(
        "Explore and compare seven standard machine learning "
        "models and one hybrid ensemble approach."
    )

    st.divider()

    metric = st.selectbox(
        "Select Performance Metric",
        [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
            "Training Time (s)",
            "Prediction Time (s)"
        ]
    )

    comparison_data = results_df.sort_values(
        metric,
        ascending=True
    )

    fig = px.bar(
        comparison_data,
        x=metric,
        y="Model",
        orientation="h",
        color=metric,
        text_auto=".3f",
        title=f"{metric} Comparison",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=550,
        showlegend=False,
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detailed Results")

    st.dataframe(
        results_df.sort_values(
            metric,
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The hybrid prediction time includes both model branches "
        "and probability fusion. Timing comparisons should be "
        "interpreted with the respective prediction pipelines in mind."
    )


# ==================================================
# SENTIMENT PREDICTION
# ==================================================

elif page == "Sentiment Prediction":

    st.title("Tourism Review Sentiment Prediction")

    st.markdown(
        "Enter a tourism review and its details to predict "
        "the review's sentiment using one of the trained models."
    )

    st.divider()

    # MODEL SELECTION

    selected_model_name = st.selectbox(
        "Select Prediction Model",
        [
            "Random Forest",
            "Hybrid (Text + Metadata)"
        ]
    )

    if selected_model_name == "Random Forest":

        st.caption(
            "Uses the trained Random Forest classifier with "
            "TF-IDF text features and review metadata."
        )

    else:

        st.caption(
            "Combines a text-based Logistic Regression model and "
            "a metadata-based Random Forest model using equal-weight "
            "soft voting."
        )

    st.divider()

    # REVIEW INPUTS

    review_text = st.text_area(
        "Enter a Tourism Review",
        placeholder="Write your tourism experience here...",
        height=150
    )

    review_title = st.text_input(
        "Review Title",
        placeholder="Enter a title for your review"
    )

    col1, col2 = st.columns(2)

    with col1:

        location_type = st.selectbox(
            "Location Type",
            [
                "Beach",
                "Historical Site",
                "Nature",
                "Religious Site",
                "Waterfall",
                "Wildlife",
                "Other"
            ]
        )

        province = st.selectbox(
            "Province",
            [
                "Western",
                "Central",
                "Southern",
                "Northern",
                "Eastern",
                "North Western",
                "North Central",
                "Uva",
                "Sabaragamuwa"
            ]
        )

        location_name = st.text_input(
            "Location Name",
            placeholder="e.g. Mirissa Beach"
        )

    with col2:

        user_country = st.text_input(
            "User Country",
            value="Sri Lanka"
        )

    st.write("")

    # PREDICTION BUTTON

    if st.button(
        "Predict Sentiment",
        type="primary",
        use_container_width=True
    ):

        if not review_text.strip():

            st.warning("Please enter a tourism review first.")

        else:

            input_data = pd.DataFrame([{
                "Text": review_text,
                "Title": review_title,
                "Location_Type": location_type,
                "Province": province,
                "Location_Name": location_name,
                "User_Country": user_country,
                "Review_Length": len(review_text),
                "Title_Length": len(review_title)
            }])

            try:

                if selected_model_name == "Random Forest":

                    processed_input = preprocessor.transform(
                        input_data
                    )

                    selected_model = models["Random Forest"]

                    probabilities = selected_model.predict_proba(
                        processed_input
                    )[0]

                    predicted_class = selected_model.classes_[
                        np.argmax(probabilities)
                    ]

                else:

                    text_probabilities = (
                        hybrid_text_model.predict_proba(input_data)
                    )[0]

                    metadata_probabilities = (
                        hybrid_metadata_model.predict_proba(input_data)
                    )[0]

                    # Equal-weight soft voting

                    hybrid_probabilities = (
                        text_probabilities + metadata_probabilities
                    ) / 2

                    predicted_class = np.argmax(
                        hybrid_probabilities
                    )

                    probabilities = hybrid_probabilities

                predicted_sentiment = label_encoder.inverse_transform(
                    [int(predicted_class)]
                )[0]

                confidence = float(np.max(probabilities))

                st.divider()

                st.subheader("Prediction Result")

                if predicted_sentiment == "POSITIVE":

                    st.success(
                        f"Predicted Sentiment: {predicted_sentiment}"
                    )

                elif predicted_sentiment == "NEGATIVE":

                    st.error(
                        f"Predicted Sentiment: {predicted_sentiment}"
                    )

                else:

                    st.info(
                        f"Predicted Sentiment: {predicted_sentiment}"
                    )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2%}"
                )

                st.caption(
                    "Confidence represents the highest predicted "
                    "class probability. It is not a guarantee of correctness."
                )

                # CLASS PROBABILITY BREAKDOWN

                probability_labels = label_encoder.inverse_transform(
                    np.arange(len(probabilities))
                )

                probability_df = pd.DataFrame({
                    "Sentiment": probability_labels,
                    "Probability": probabilities
                })

                st.subheader("Class Probability Breakdown")

                probability_chart = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Probability",
                    color="Sentiment",
                    text_auto=".2%",
                    title="Predicted Probability by Sentiment"
                )

                probability_chart.update_layout(
                    yaxis_range=[0, 1],
                    showlegend=False
                )

                st.plotly_chart(
                    probability_chart,
                    use_container_width=True
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# ==================================================
# ABOUT THE PROJECT
# ==================================================

elif page == "About the Project":

    st.title("About EnsembleLab")

    st.markdown("""
    **EnsembleLab** is an interactive machine learning dashboard
    developed to investigate ensemble learning techniques using a
    Sri Lankan tourism review dataset.

    ---

    ### Dataset Information

    - 16,155 review records after exact duplicate removal
    - 11 tourism attraction categories
    - Data covering 2010–2023
    - Three sentiment classes: Positive, Negative and Neutral

    ### Machine Learning Models

    1. Decision Tree
    2. Bagging
    3. Random Forest
    4. AdaBoost
    5. Gradient Boosting
    6. XGBoost
    7. Stacking
    8. Hybrid (Text + Metadata)

    ### Hybrid Ensemble Architecture

    The hybrid approach combines two specialized classification
    branches:

    - **Text branch:** TF-IDF feature extraction followed by
      Logistic Regression.
    - **Metadata branch:** One-Hot Encoding and numerical
      preprocessing followed by Random Forest.

    The probability outputs of both branches are combined using
    equal-weight soft voting. The final sentiment is selected
    from the class with the highest combined probability.

    ### Research Focus

    The dashboard compares model performance using:

    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-score
    - Weighted F1-score
    - Training time
    - Prediction time

    ### Important Methodological Considerations

    The dataset has an imbalanced sentiment distribution.
    Therefore, accuracy alone is not sufficient to evaluate
    model performance. Macro F1 is used as an important
    comparison metric because it gives equal importance to
    each sentiment class.

    The hybrid model is an experimental approach and does not
    necessarily outperform every individual model.

    The sentiment labels are largely model-generated rather
    than manually verified for every review. This is an
    important limitation when interpreting the results.

    Model comparisons should also consider differences in
    class-imbalance handling and prediction-time measurement.
    """)

    st.divider()

    st.caption(
        "EnsembleLab | Machine Learning Assignment | "
        "Sri Lankan Tourism Analytics | 2026"
    )
