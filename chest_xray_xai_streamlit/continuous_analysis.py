"""Interactive continuous stability–localization analysis for the XAI dashboard.

Place beside app.py and call render_continuous_analysis(df, filtered_df)
from app.py after its existing filter controls have been defined.
"""
import pandas as pd
import plotly.express as px
import streamlit as st


def _spearman(frame):
    if len(frame) < 3 or frame['explanation_similarity'].nunique() < 2 or frame['alignment_change'].nunique() < 2:
        return float('nan')
    return frame['explanation_similarity'].corr(frame['alignment_change'], method='spearman')


def render_continuous_analysis(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.divider()
    st.header('Continuous Stability–Localization Analysis')
    st.markdown(
        'Explore the association between explanation similarity and changes in '
        'localization alignment without first converting either measure into a binary label. '
        'These analyses are descriptive; repeated evaluations of the same image are not independent.'
    )
    st.caption('Method, corruption, and severity selections are inherited from the Stability–Localization Explorer above.')
    if filtered_df.empty:
        st.warning('Select at least one method, corruption type, and severity above.')
        return

    subset = filtered_df.copy()
    rho = _spearman(subset)
    high_cutoff = st.slider('High-similarity cutoff (continuous analysis)', 0.50, 1.00, 0.90, 0.01, key='continuous_similarity_cutoff')
    loss_cutoff = st.slider('Substantial localization loss cutoff (continuous analysis)', 0.01, 0.20, 0.05, 0.01, key='continuous_loss_cutoff')
    high = subset['explanation_similarity'] >= high_cutoff
    exceptions = high & (subset['alignment_change'] <= -loss_cutoff)
    a, b, c, e = st.columns(4)
    a.metric('Selected evaluations', f'{len(subset):,}')
    b.metric('Spearman ρ', f'{rho:.3f}' if pd.notna(rho) else 'N/A')
    c.metric('High-similarity evaluations', f'{int(high.sum()):,}')
    e.metric('High-similarity losses', f'{int(exceptions.sum()):,} ({exceptions.sum()/high.sum():.1%})' if high.sum() else '0 (N/A)')

    st.subheader('Explore individual evaluations')
    subset['Result'] = 'Other evaluation'
    subset.loc[exceptions, 'Result'] = 'High similarity + substantial loss'
    fig = px.scatter(
        subset, x='explanation_similarity', y='alignment_change',
        color='Result', symbol='method',
        hover_data=['sop_uid', 'method', 'corruption', 'severity', 'original_alignment', 'localization_alignment'],
        color_discrete_map={'Other evaluation': '#5289BA', 'High similarity + substantial loss': '#D18E35'},
        labels={'explanation_similarity': 'Explanation similarity (Spearman)', 'alignment_change': 'Change in localization alignment'},
        opacity=0.7,
    )
    fig.add_vline(x=high_cutoff, line_dash='dash', line_color='gray')
    fig.add_hline(y=-loss_cutoff, line_dash='dash', line_color='gray')
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True, key='continuous_scatter')
    st.caption('The bottom-right region identifies observations with high explanation similarity and substantial localization deterioration.')

    st.subheader('Correlation by XAI method')
    method_rows = []
    for method, group in subset.groupby('method'):
        method_rows.append({'Method': method, 'Spearman ρ': _spearman(group), 'Evaluations': len(group), 'Unique images': group['sop_uid'].nunique()})
    methods = pd.DataFrame(method_rows)
    if methods['Spearman ρ'].notna().any():
        fig_method = px.bar(methods.dropna(subset=['Spearman ρ']), x='Method', y='Spearman ρ', color='Method',
                            hover_data=['Evaluations', 'Unique images'], text='Spearman ρ')
        fig_method.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_method.update_layout(showlegend=False, height=390)
        st.plotly_chart(fig_method, use_container_width=True, key='continuous_method_correlation')
    st.dataframe(methods.round(3), hide_index=True, use_container_width=True)

    st.subheader('Correlation by corruption and severity')
    group_rows = []
    for (method, corruption, severity), group in subset.groupby(['method', 'corruption', 'severity']):
        group_rows.append({'Method': method, 'Condition': f'{corruption} · {severity}', 'Spearman ρ': _spearman(group), 'Evaluations': len(group)})
    strata = pd.DataFrame(group_rows)
    if strata['Spearman ρ'].notna().any():
        pivot = strata.pivot(index='Method', columns='Condition', values='Spearman ρ')
        fig_heat = px.imshow(pivot, color_continuous_scale='RdBu', color_continuous_midpoint=0,
                             aspect='auto', labels={'color': 'Spearman ρ', 'x': 'Corruption condition', 'y': 'XAI method'},
                             text_auto='.2f')
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True, key='continuous_condition_heatmap')
    else:
        st.info('Not enough variation in the selected subsets to calculate correlations.')
    with st.expander('View condition-level correlation table'):
        st.dataframe(strata.round(3), hide_index=True, use_container_width=True)

    st.subheader('High-similarity localization losses by method')
    exception_rows = []
    for method, group in subset.groupby('method'):
        qualifying = group['explanation_similarity'] >= high_cutoff
        losses = qualifying & (group['alignment_change'] <= -loss_cutoff)
        denominator = int(qualifying.sum())
        exception_rows.append({'Method': method, 'High-similarity evaluations': denominator,
                               'Substantial losses': int(losses.sum()),
                               'Loss rate (%)': 100 * losses.sum() / denominator if denominator else float('nan')})
    exception_table = pd.DataFrame(exception_rows)
    valid = exception_table.dropna(subset=['Loss rate (%)'])
    if not valid.empty:
        fig_exc = px.bar(valid, x='Method', y='Loss rate (%)', color='Method',
                         hover_data=['High-similarity evaluations', 'Substantial losses'], text='Loss rate (%)')
        fig_exc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_exc.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_exc, use_container_width=True, key='continuous_exception_rates')
    st.dataframe(exception_table.round(2), hide_index=True, use_container_width=True)
    st.caption('Loss rates use only high-similarity evaluations as the denominator; they are not the overall disagreement rates shown elsewhere in the app.')

    st.info('Statistical interpretation: correlations in this explorer are recalculated for the current filters, but no confidence interval or p-value is claimed for those filtered subsets. Image-cluster bootstrap inference from the original notebook must be presented separately as a fixed full-sample result.')
    with st.expander('Download currently filtered continuous-analysis data'):
        st.download_button('Download CSV', data=subset.drop(columns=['Result']).to_csv(index=False),
                           file_name='continuous_stability_localization_filtered.csv', mime='text/csv',
                           key='continuous_download')
