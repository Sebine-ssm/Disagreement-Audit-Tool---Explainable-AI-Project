import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from continuous_analysis import render_continuous_analysis

st.set_page_config(
    page_title="Chest X-ray XAI Disagreement Audit",
    page_icon="🩻",
    layout="wide",
)

# Presentation-only styling. All analyses below remain unchanged.
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] .block-container {padding-top: 1.4rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px;
    padding: .85rem 1rem;
}
div[data-testid="stTabs"] button[role="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🩻 XAI Disagreement Audit")
    st.caption("ResNet-50 · RSNA chest X-rays")
    st.divider()
    st.markdown("**Explore the research**")
    st.caption(
        "Use the four tabs in the main panel to explore disagreement, "
        "continuous relationships, statistical evidence, and the case study."
    )
    st.markdown("**How to read the results**")
    st.caption(
        "The study metrics at the top are fixed full-sample results. "
        "Filters below them update the exploratory analyses, not the "
        "original permutation test or published summary findings."
    )
    st.divider()
    st.caption("Academic research demonstration · Not for clinical use")

st.title("When Stable Explanations Disagree")
st.markdown("#### A Stability–Localization Disagreement Audit for Chest X-ray Classification")
st.markdown(
    "Can an explanation remain similar after an image corruption while becoming "
    "**less aligned with the radiologist-annotated opacity region**? "
    "This dashboard investigates that question using a ResNet-50 classifier "
    "and three explanation methods: **Integrated Gradients**, "
    "**Occlusion Sensitivity**, and **Score-CAM**."
)

DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "stability_localization_public.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Verify the original, fixed disagreement definition.
baseline_disagreement = (
    (df["explanation_similarity"] >= 0.80)
    & (df["alignment_change"] <= -0.05)
)
assert (
    baseline_disagreement == df["disagreement"].astype(bool)
).all(), "Baseline disagreement labels do not match!"

st.markdown("### Study at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evaluations", f"{len(df):,}")
m2.metric("Unique chest X-rays", f"{df['sop_uid'].nunique():,}")
m3.metric("Disagreement events", f"{int(baseline_disagreement.sum()):,}")
m4.metric("Disagreement rate", f"{baseline_disagreement.mean():.2%}")
st.caption(
    "Fixed full-sample summary · Explanation similarity ≥ 0.80 and "
    "localization alignment decrease ≥ 0.05. Multiple evaluations come "
    "from each image; these are not independent patient observations."
)

with st.expander("Study design and metric definitions", expanded=False):
    st.markdown(
        "**Study design:** Opacity-positive chest X-rays from the RSNA "
        "Pneumonia Detection Challenge were evaluated using three XAI methods "
        "across six image-corruption conditions. Each record represents an "
        "image–method–corruption evaluation.\n\n"
        "**Explanation similarity:** Spearman correlation between the original "
        "and corrupted explanation maps.\n\n"
        "**Localization alignment:** Spatial alignment of an explanation with "
        "the annotated opacity region. **Alignment change** is corrupted "
        "alignment minus original alignment; negative values indicate loss.\n\n"
        "**Disagreement event:** Similarity at or above the selected threshold "
        "despite alignment loss at or above the selected loss threshold. "
        "These metrics describe different properties and do not by themselves "
        "establish explanation faithfulness."
    )

st.divider()
st.markdown("### Explore the results")
st.caption(
    "Select a subset of the evaluations below. The same method, corruption, "
    "and severity selections apply to the interactive charts across tabs."
)

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_methods = st.multiselect(
            "XAI method",
            options=sorted(df["method"].dropna().unique()),
            default=sorted(df["method"].dropna().unique()),
        )
    with col2:
        selected_corruptions = st.multiselect(
            "Corruption type",
            options=sorted(df["corruption"].dropna().unique()),
            default=sorted(df["corruption"].dropna().unique()),
        )
    with col3:
        selected_severities = st.multiselect(
            "Severity",
            options=sorted(df["severity"].dropna().unique()),
            default=sorted(df["severity"].dropna().unique()),
        )

filtered_df = df[
    df["method"].isin(selected_methods)
    & df["corruption"].isin(selected_corruptions)
    & df["severity"].isin(selected_severities)
].copy()

st.markdown("**Exploratory disagreement thresholds**")
left, right = st.columns(2)
with left:
    similarity_threshold = st.slider(
        "Minimum explanation similarity",
        min_value=0.50, max_value=1.00, value=0.80, step=0.05,
    )
with right:
    loss_threshold = st.slider(
        "Minimum localization loss",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
    )

st.caption(
    "Thresholds change the exploratory disagreement counts and screening "
    "results. Continuous correlations do not use these binary thresholds. "
    "The fixed original permutation results and findings do not update."
)

filtered_df["dynamic_disagreement"] = (
    (filtered_df["explanation_similarity"] >= similarity_threshold)
    & (filtered_df["alignment_change"] <= -loss_threshold)
)
n_disagreements = int(filtered_df["dynamic_disagreement"].sum())
n_evaluations = len(filtered_df)
disagreement_rate = 100 * n_disagreements / n_evaluations if n_evaluations else 0

with st.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected evaluations", f"{n_evaluations:,}")
    c2.metric("Selected disagreements", f"{n_disagreements:,}")
    c3.metric("Selected disagreement rate", f"{disagreement_rate:.2f}%")

if filtered_df.empty:
    st.warning("No records match the current filters. Select at least one option in each filter.")

# Shared filters and thresholds are intentionally outside the tabs.
# Streamlit retains their values as visitors move between analyses.
st.divider()
tab_audit, tab_continuous, tab_statistics, tab_findings = st.tabs([
    "1 · Disagreement Audit",
    "2 · Continuous Analysis",
    "3 · Statistical Evidence",
    "4 · Case Study & Findings",
])

with tab_audit:
    st.header("Stability–Localization Explorer")
    highlight_disagreements = st.checkbox("Highlight disagreement events", value=False)

    if filtered_df.empty:
        st.warning("No records match your selected filters.")
    else:
        fig = px.scatter(
            filtered_df,
            x="explanation_similarity",
            y="alignment_change",
            color="method",
            hover_data=[
                "corruption",
                "severity",
                "original_alignment",
                "localization_alignment",
            ],
            labels={
                "explanation_similarity": "Explanation Similarity (Spearman)",
                "alignment_change": "Change in Localization Alignment",
                "method": "XAI Method",
            },
            title="Explanation Stability vs. Localization Change",
            opacity=0.65,
        )

        # Original disagreement thresholds
        fig.add_vline(x=similarity_threshold, line_dash="dash", line_color="gray")

        fig.add_hline(y=-loss_threshold, line_dash="dash", line_color="red")

        fig.update_layout(height=600)

        # Highlight disagreement events
        if highlight_disagreements:

            disagreement_df = filtered_df[filtered_df["dynamic_disagreement"]]

            if not disagreement_df.empty:

                # Fade the background observations
                for trace in fig.data:
                    trace.update(marker=dict(opacity=0.15, size=5))

                highlight_fig = px.scatter(
                    disagreement_df,
                    x="explanation_similarity",
                    y="alignment_change",
                    hover_data=[
                        "method",
                        "corruption",
                        "severity",
                    ],
                )

                for trace in highlight_fig.data:
                    trace.update(
                        name="Disagreement events",
                        marker=dict(
                            color="black",
                            size=9,
                            symbol="circle-open",
                            line=dict(width=2),
                        ),
                        showlegend=True,
                    )

                    fig.add_trace(trace)

            else:
                st.info(
                    "No disagreement events meet the selected thresholds " "and filters."
                )

        st.plotly_chart(fig, use_container_width=True, key="stability_localization_scatter")

    # -------------------------------------------
    # XAI Method Comparison
    # -------------------------------------------

    st.divider()
    st.header("XAI Method Comparison")

    st.markdown(
        "Compare the frequency of stability–localization disagreement "
        "across the selected XAI methods and corruption conditions."
    )

    if filtered_df.empty:
        st.warning("Select at least one method and corruption condition.")

    else:
        method_summary = (
            filtered_df.groupby("method")
            .agg(
                evaluations=("dynamic_disagreement", "size"),
                disagreements=("dynamic_disagreement", "sum"),
            )
            .reset_index()
        )

        method_summary["disagreement_rate"] = (
            100 * method_summary["disagreements"] / method_summary["evaluations"]
        )

        fig_methods = px.bar(
            method_summary,
            x="method",
            y="disagreement_rate",
            color="method",
            text="disagreement_rate",
            custom_data=["evaluations", "disagreements"],
            labels={
                "method": "XAI Method",
                "disagreement_rate": "Disagreement Rate (%)",
            },
            title="Disagreement Rate by XAI Method",
        )

        fig_methods.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            hovertemplate=(
                "Method: %{x}<br>"
                "Disagreement rate: %{y:.2f}%<br>"
                "Disagreements: %{customdata[1]}<br>"
                "Evaluations: %{customdata[0]}"
                "<extra></extra>"
            ),
        )

        fig_methods.update_layout(
            height=450,
            showlegend=False,
            yaxis_range=[
                0,
                max(1, method_summary["disagreement_rate"].max() * 1.2),
            ],
        )

        st.plotly_chart(
            fig_methods, use_container_width=True, key="method_comparison_chart"
        )

        st.subheader("Method-Level Summary")

        st.dataframe(
            method_summary.rename(
                columns={
                    "method": "XAI Method",
                    "evaluations": "Evaluations",
                    "disagreements": "Disagreements",
                    "disagreement_rate": "Disagreement Rate (%)",
                }
            ).round({"Disagreement Rate (%)": 2}),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Rates are calculated per method–image–corruption evaluation, "
            "not per unique chest X-ray. Comparisons are descriptive."
        )


    # -------------------------------------------
    # Screening Performance Explorer
    # -------------------------------------------

    st.divider()
    st.header("Screening Performance Explorer")

    st.markdown(
        "Investigate whether explanation similarity can identify "
        "substantial localization deterioration."
    )

    if filtered_df.empty:
        st.warning("Select at least one method and corruption condition.")

    else:
        # A substantial loss is defined by the selected loss threshold
        actual_loss = filtered_df["alignment_change"] <= -loss_threshold

        # A stability-based screening rule flags low-similarity explanations
        screening_flag = filtered_df["explanation_similarity"] < similarity_threshold

        true_positives = int((screening_flag & actual_loss).sum())
        false_negatives = int((~screening_flag & actual_loss).sum())
        false_positives = int((screening_flag & ~actual_loss).sum())

        total_losses = int(actual_loss.sum())

        sensitivity = true_positives / total_losses if total_losses > 0 else 0

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Substantial Losses", f"{total_losses:,}")
        col2.metric("Detected Losses", f"{true_positives:,}")
        col3.metric("Missed Losses", f"{false_negatives:,}")
        col4.metric("Sensitivity", f"{sensitivity:.1%}")

        st.caption(
            "A loss is considered substantial when alignment decreases "
            "by at least the selected loss threshold. The screening rule "
            "flags explanations with similarity below the selected "
            "similarity threshold."
        )

        st.write(f"Screening precision: {precision:.1%}")

        # -------------------------------------------
        # Screening Outcomes Visualization
        # -------------------------------------------

        st.subheader("Detected vs. Missed Localization Losses")

        screening_summary = pd.DataFrame(
            {
                "Outcome": ["Detected Losses", "Missed Losses"],
                "Count": [true_positives, false_negatives],
            }
        )

        fig_screening = px.bar(
            screening_summary,
            x="Outcome",
            y="Count",
            color="Outcome",
            text="Count",
            title="Screening Outcomes for Substantial Localization Losses",
            color_discrete_map={
                "Detected Losses": "#269477",
                "Missed Losses": "#DC7951",
            },
        )

        fig_screening.update_traces(
            textposition="outside",
            hovertemplate="%{x}: %{y:,} evaluations<extra></extra>",
        )

        fig_screening.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Number of Evaluations",
        )

        st.plotly_chart(
            fig_screening,
            use_container_width=True,
            key="screening_outcomes_chart",
        )

        missed_loss_rate = 100 * false_negatives / total_losses if total_losses > 0 else 0

        st.metric("Percentage of Substantial Losses Missed", f"{missed_loss_rate:.2f}%")

        st.caption(
            "The screening rule flags evaluations with explanation "
            "similarity below the selected threshold. Missed losses "
            "are evaluations with substantial localization deterioration "
            "that remain above the similarity threshold. "
            "Results are descriptive and do not establish clinical reliability."
        )



with tab_continuous:
    render_continuous_analysis(df, filtered_df)

with tab_statistics:
    # -------------------------------------------
    # Image-Level Disagreement Analysis
    # -------------------------------------------

    st.divider()
    st.header("Image-Level Disagreement Analysis")

    st.markdown(
        "Explore whether stability–localization disagreement events "
        "are distributed evenly across chest X-ray images or "
        "concentrated among particular images."
    )

    if filtered_df.empty:
        st.warning("No evaluations match the selected filters.")

    else:
        # Count disagreement events for each unique image
        image_summary = (
            filtered_df.groupby("sop_uid")
            .agg(
                evaluations=("dynamic_disagreement", "size"),
                disagreements=("dynamic_disagreement", "sum"),
            )
            .reset_index()
        )

        # Count how many images have each disagreement count
        recurrence_summary = (
            image_summary["disagreements"]
            .value_counts()
            .sort_index()
            .rename_axis("Disagreements per Image")
            .reset_index(name="Number of Images")
        )

        fig_recurrence = px.bar(
            recurrence_summary,
            x="Disagreements per Image",
            y="Number of Images",
            text="Number of Images",
            title="Distribution of Disagreement Events Across Images",
        )

        fig_recurrence.update_traces(textposition="outside")

        fig_recurrence.update_layout(
            height=450,
            xaxis=dict(dtick=1),
            yaxis_title="Number of Chest X-ray Images",
        )

        st.plotly_chart(
            fig_recurrence,
            use_container_width=True,
            key="image_recurrence_chart",
        )

        images_with_disagreement = int((image_summary["disagreements"] > 0).sum())

        total_images = len(image_summary)

        col1, col2 = st.columns(2)

        col1.metric("Images Evaluated", f"{total_images:,}")
        col2.metric(
            "Images with ≥1 Disagreement",
            f"{images_with_disagreement:,}",
        )

        st.caption(
            "Each image can contribute multiple evaluations across "
            "XAI methods and corruption conditions. Counts reflect "
            "the currently selected filters and disagreement thresholds."
        )


    # -------------------------------------------
    # Disagreement Concentration Explorer
    # -------------------------------------------

    st.divider()
    st.header("Disagreement Concentration Explorer")

    st.markdown(
        "Investigate how much of the total disagreement burden "
        "is concentrated among the images with the most events."
    )

    if filtered_df.empty:
        st.warning("No evaluations match the selected filters.")

    else:
        image_counts = (
            filtered_df.groupby("sop_uid")["dynamic_disagreement"]
            .sum()
            .sort_values(ascending=False)
        )

        total_events = int(image_counts.sum())

        max_top_n = len(image_counts)

        top_n = st.slider(
            "Number of highest-disagreement images",
            min_value=1,
            max_value=max_top_n,
            value=min(20, max_top_n),
            step=1,
        )

        top_events = int(image_counts.head(top_n).sum())

        concentration = 100 * top_events / total_events if total_events > 0 else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Disagreement Events", f"{total_events:,}")
        col2.metric("Events in Top Images", f"{top_events:,}")
        col3.metric("Share of Events", f"{concentration:.2f}%")

        concentration_df = pd.DataFrame(
            {
                "Group": [
                    f"Top {top_n} Images",
                    "Remaining Images",
                ],
                "Events": [
                    top_events,
                    total_events - top_events,
                ],
            }
        )

        fig_concentration = px.bar(
            concentration_df,
            x="Group",
            y="Events",
            color="Group",
            text="Events",
            title="Concentration of Disagreement Events",
        )

        fig_concentration.update_traces(textposition="outside")

        fig_concentration.update_layout(
            height=420,
            showlegend=False,
            yaxis_title="Number of Disagreement Events",
        )

        st.plotly_chart(
            fig_concentration,
            use_container_width=True,
            key="disagreement_concentration_chart",
        )

        st.caption(
            "Images are ranked by their number of disagreement events "
            "under the currently selected filters and thresholds. "
            "This is an exploratory descriptive comparison."
        )


    # -------------------------------------------
    # Original Research: Permutation Test
    # -------------------------------------------

    st.divider()
    st.header("Original Research: Concentration Test")

    st.markdown(
        "The original analysis tested whether disagreement events "
        "were more concentrated among images than expected under "
        "a condition-stratified permutation null model."
    )

    research_results = pd.DataFrame(
        {
            "Measure": [
                "Observed Top-20 Concentration",
                "Permutation Null Mean",
            ],
            "Concentration (%)": [
                50.88,
                32.84,
            ],
        }
    )

    fig_permutation = px.bar(
        research_results,
        x="Measure",
        y="Concentration (%)",
        color="Measure",
        text="Concentration (%)",
        title="Observed Concentration vs. Permutation Null Mean",
    )

    fig_permutation.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_permutation.update_layout(
        height=420,
        showlegend=False,
        yaxis_range=[0, 60],
    )

    st.plotly_chart(
        fig_permutation,
        use_container_width=True,
        key="original_permutation_comparison",
    )

    st.markdown(
        "The observed top-20 concentration (50.88%) exceeded "
        "the permutation null 95% interval (29.24%–36.84%). "
        "This provides evidence that disagreement events were "
        "more concentrated among images than expected under "
        "the specified condition-stratified null model."
    )

    st.caption(
        "These values come from the original research analysis "
        "using explanation similarity ≥ 0.80 and localization "
        "loss ≥ 0.05, with all methods and corruption conditions. "
        "They do not update when dashboard filters change."
    )



with tab_findings:
    # -------------------------------------------
    # Research Findings and Limitations
    # -------------------------------------------

    st.divider()
    st.header("Research Findings and Limitations")

    st.caption(
        "The findings below summarize the original full-sample "
        "analysis at explanation similarity ≥ 0.80 and "
        "localization loss ≥ 0.05. They do not change "
        "when dashboard filters are adjusted."
    )

    st.subheader("Key Research Findings")

    st.markdown("""
    **1. Stable explanations can lose spatial alignment.**

    At the original thresholds, 171 of 2,700 evaluations
    (6.33%) showed high explanation similarity despite
    a substantial decrease in localization alignment.

    **2. Explanation similarity is an incomplete screening signal.**

    A screening rule based on explanation similarity below 0.80
    detected 974 of 1,145 substantial localization losses,
    but missed 171 (14.93%).

    **3. Disagreement varies across XAI methods.**

    At the original thresholds, disagreement rates were:
    - Integrated Gradients: 1.33%
    - Occlusion Sensitivity: 11.33%
    - Score-CAM: 6.33%

    These differences are descriptive and should not be interpreted
    as a definitive ranking of explanation quality.

    **4. Disagreement events are concentrated among images.**

    The 20 images with the most disagreement events accounted
    for 50.88% of all 171 events, exceeding the permutation
    null mean of 32.84%.
    """)

    st.subheader("Important Limitations")

    st.markdown("""
    - **Localization is not faithfulness:** Spatial alignment with
      radiologist-annotated opacity regions does not establish
      that an explanation faithfully represents model behavior.

    - **Repeated evaluations:** The 2,700 evaluation records come
      from 150 positive chest X-rays evaluated across multiple
      methods and corruption conditions. They are not independent
      patient-level observations.

    - **Fixed prediction targets:** Explanations were generated
      for the original predicted class, held fixed under corruption.
      This matters when interpreting subgroup differences.

    - **Dataset and model scope:** Findings are based on a single
      ResNet-50 classifier and the selected RSNA lung-opacity
      evaluation sample. Generalization to other models, datasets,
      and clinical settings remains untested.

    - **Threshold dependence:** Disagreement rates change when
      similarity or localization-loss thresholds are adjusted.

    - **No clinical validation:** This dashboard is a research
      demonstration, not a diagnostic or clinical decision-support
      tool.
    """)

    # -------------------------------------------
    # Illustrative Case Study
    # -------------------------------------------

    st.divider()
    st.header("Illustrative Case Study")

    st.markdown(
        "This example illustrates how an explanation can retain "
        "relatively high similarity under image corruption while "
        "losing spatial alignment with radiologist-annotated "
        "opacity regions."
    )

    CASE_IMAGE_PATH = (
        Path(__file__).resolve().parent / "images" / "occlusion_case_study.png"
    )

    if CASE_IMAGE_PATH.exists():
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.image(
                str(CASE_IMAGE_PATH),
                caption=(
                    "Occlusion Sensitivity under severe Gaussian blur. "
                    "Original and corrupted images with their explanations."
                ),
                width=660,
            )
    else:
        st.warning("Case-study figure not found.")

    st.subheader("What Does This Example Show?")

    col1, col2, col3 = st.columns(3)

    col1.metric("Explanation Similarity", "0.843")
    col2.metric("Original Alignment", "0.121")
    col3.metric("Corrupted Alignment", "0.038")

    st.markdown("""
        Under severe Gaussian blur, the Occlusion Sensitivity
        explanation retained a Spearman similarity of **0.843**
        relative to the original explanation.

        However, localization alignment decreased from
        approximately **0.121 to 0.038**, a loss of about
        **0.083 (8.3 percentage points)**.

        This example therefore meets the original disagreement
        criteria:

        - Explanation similarity ≥ 0.80
        - Localization alignment decrease ≥ 0.05

        **Interpretation:** A relatively stable attribution pattern
        does not necessarily preserve its spatial overlap with
        radiologist-annotated opacity regions.
        """)

    st.caption(
        "This is one illustrative evaluation, not evidence that "
        "all images or explanation methods behave similarly. "
        "Localization alignment does not establish explanation "
        "faithfulness or clinical validity."
    )

    st.info(
        "The central finding is that explanation stability and "
        "spatial localization alignment measure different properties. "
        "High similarity alone should not be treated as sufficient "
        "evidence of explanation reliability."
    )

    st.divider()

    st.subheader("Dataset Attribution")

    st.markdown("""
    This research uses chest X-ray images and opacity annotations
    from the **RSNA Pneumonia Detection Challenge**, based on the
    NIH Clinical Center ChestX-ray dataset.

    - **RSNA Pneumonia Detection Challenge:** [RSNA AI Challenges](https://www.rsna.org/artificial-intelligence/ai-image-challenge)
    - **Dataset source:** NIH Clinical Center ChestX-ray dataset
    - **Usage:** Academic research and educational demonstration.

    This dashboard is intended for research and educational purposes only.
    It is not a medical diagnostic or clinical decision-support tool.
    """)
