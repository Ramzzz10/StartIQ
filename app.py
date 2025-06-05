import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from io import BytesIO
import joblib
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.model_utils import initialize_models, load_data, get_startup_by_id, get_user_metrics_by_startup_id
from utils.visualization import (
    create_radar_chart, create_success_histogram, create_trend_analysis,
    create_pmf_distribution, create_comparison_chart, create_investment_dashboard
)

# Set page configuration
st.set_page_config(
    page_title="StartIQ - Платформа анализа стартапов",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4257B2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3C9D9B;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.05);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4257B2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .success-tag {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .fail-tag {
        background-color: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .unclear-tag {
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4257B2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'startups_df' not in st.session_state:
    st.session_state.startups_df = None
if 'user_metrics_df' not in st.session_state:
    st.session_state.user_metrics_df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'selected_startups' not in st.session_state:
    st.session_state.selected_startups = []
if 'comparison_startups' not in st.session_state:
    st.session_state.comparison_startups = []

# Function to load data and models
@st.cache_resource
def load_models_and_data():
    with st.spinner('Loading data and initializing models...'):
        # Set paths relative to current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize models and load data
        models, startups_df, user_metrics_df = initialize_models(data_dir, models_dir)
        
        # Apply scoring model to startups
        startups_df = models['scoring_model'].score_startups(startups_df)
        
        # Apply ML predictor to startups
        startups_df = models['ml_predictor'].predict_batch(startups_df)
        
        # Apply PMF analyzer to startups
        startups_df = models['pmf_analyzer'].analyze_startups(startups_df, user_metrics_df)
        
        return models, startups_df, user_metrics_df

# Function to convert dataframe to CSV download link
def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to generate PDF report
def generate_report(startup_id, startups_df, user_metrics_df, models):
    # This is a placeholder for PDF generation
    # In a real implementation, you would use a library like ReportLab or WeasyPrint
    
    # For now, we'll just return a success message
    return "Report generated successfully!"

# Load data and models
models, startups_df, user_metrics_df = load_models_and_data()

# Store data and models in session state
st.session_state.startups_df = startups_df
st.session_state.user_metrics_df = user_metrics_df
st.session_state.models = models

# Sidebar
st.sidebar.markdown("<h1 style='text-align: center;'>🚀 StartIQ</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>Платформа анализа стартапов</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Навигация",
    ["📌 Обзор стартапов", "📈 Прогноз успеха", "💼 PMF анализ", "🧭 Карта стартапов", "📊 Скоринг и рекомендации"]
)

# Filters
st.sidebar.markdown("## Фильтры")

# Industry filter
industries = ["Все"] + sorted(startups_df["industry"].unique().tolist())
selected_industry = st.sidebar.selectbox("Отрасль", industries)

# Country filter
countries = ["Все"] + sorted(startups_df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Страна", countries)

# Year range filter
min_year = int(startups_df["year_founded"].min())
max_year = int(startups_df["year_founded"].max())
year_range = st.sidebar.slider("Год основания", min_year, max_year, (min_year, max_year))

# Success filter
success_options = ["Все", "Успех", "Неудача", "Неопределенно"]
selected_success = st.sidebar.selectbox("Статус успеха", success_options)

# Apply filters
filtered_df = startups_df.copy()

if selected_industry != "Все":
    filtered_df = filtered_df[filtered_df["industry"] == selected_industry]
    
if selected_country != "Все":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]
    
filtered_df = filtered_df[
    (filtered_df["year_founded"] >= year_range[0]) & 
    (filtered_df["year_founded"] <= year_range[1])
]

# Маппинг переведенных статусов успеха на оригинальные значения в данных
success_mapping = {
    "Все": "All",
    "Успех": "Success",
    "Неудача": "Fail",
    "Неопределенно": "Unclear"
}

if selected_success != "Все":
    original_success_value = success_mapping[selected_success]
    filtered_df = filtered_df[filtered_df["success"] == original_success_value]

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### О проекте")
st.sidebar.info(
    "StartIQ - это интеллектуальная платформа для анализа и оценки стартапов. "
    "Она использует продвинутую аналитику и машинное обучение, чтобы помочь инвесторам и "
    "предпринимателям принимать решения на основе данных."
)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 StartIQ")

# Main content
if page == "📌 Обзор стартапов":
    st.markdown("<h1 class='main-header'>Обзор стартапов</h1>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(filtered_df)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Стартапов</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        success_rate = filtered_df[filtered_df['success'] != 'Unclear']['success'].value_counts(normalize=True).get('Success', 0) * 100
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{success_rate:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Уровень успеха</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        avg_investment = filtered_df['total_investment'].mean()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${avg_investment/1000000:.1f}M</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Средние инвестиции</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        avg_score = filtered_df['overall_score'].mean()
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_score:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Средний скоринг</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Success by Industry
    st.markdown("<h2 class='sub-header'>Успех по отраслям</h2>", unsafe_allow_html=True)
    fig1 = create_success_histogram(filtered_df, group_by='industry', title="Успех по отраслям")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Success by Country
    st.markdown("<h2 class='sub-header'>Успех по странам</h2>", unsafe_allow_html=True)
    fig2 = create_success_histogram(filtered_df, group_by='country', title="Успех по странам")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Trend Analysis
    st.markdown("<h2 class='sub-header'>Анализ трендов</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = create_trend_analysis(filtered_df, metric='success_rate', title="Тренд уровня успеха по годам")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = create_trend_analysis(filtered_df, metric='avg_investment', title="Тренд средних инвестиций по годам")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Investment Dashboard
    st.markdown("<h2 class='sub-header'>Дашборд инвестиций</h2>", unsafe_allow_html=True)
    fig5 = create_investment_dashboard(filtered_df, title="Дашборд инвестиций")
    st.plotly_chart(fig5, use_container_width=True)
    
    # Startup Table
    st.markdown("<h2 class='sub-header'>База данных стартапов</h2>", unsafe_allow_html=True)
    
    # Select columns to display
    display_cols = ['id', 'name', 'industry', 'country', 'year_founded', 'product_stage', 
                    'total_investment', 'revenue', 'overall_score', 'success']
    
    # Display table
    st.dataframe(filtered_df[display_cols], use_container_width=True)
    
    # Download link
    st.markdown(get_csv_download_link(filtered_df, "startups_data.csv", "Скачать отфильтрованные данные"), unsafe_allow_html=True)

elif page == "📈 Прогноз успеха":
    st.markdown("<h1 class='main-header'>Прогноз успеха</h1>", unsafe_allow_html=True)
    
    # Model information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>ML Success Predictor</h3>", unsafe_allow_html=True)
    st.markdown(
        "This model uses machine learning to predict the probability of startup success "
        "based on various features. It has been trained on historical startup data and "
        "can provide insights into the likelihood of success for new ventures."
    )
    
    # Model metrics
    if models['ml_predictor'].metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{models['ml_predictor'].metrics['accuracy']:.2f}")
            st.metric("F1 Score", f"{models['ml_predictor'].metrics['f1_score']:.2f}")
        
        with col2:
            st.markdown("### Confusion Matrix")
            fig = models['ml_predictor'].plot_confusion_matrix()
            st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction for existing startup
    st.markdown("<h2 class='sub-header'>Predict Success for Existing Startup</h2>", unsafe_allow_html=True)
    
    # Startup selector
    startup_options = filtered_df[['id', 'name']].copy()
    startup_options['display'] = startup_options['name'] + " (ID: " + startup_options['id'].astype(str) + ")"
    selected_startup = st.selectbox("Select a startup", startup_options['display'].tolist())
    
    if selected_startup:
        # Extract ID from selection
        selected_id = int(selected_startup.split("(ID: ")[1].split(")")[0])
        
        # Get startup data
        startup_data = get_startup_by_id(selected_id, filtered_df)
        
        # Make prediction
        prediction = models['ml_predictor'].predict(startup_data)
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>{startup_data['name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Industry:** {startup_data['industry']}")
            st.markdown(f"**Country:** {startup_data['country']}")
            st.markdown(f"**Founded:** {startup_data['year_founded']}")
            st.markdown(f"**Product Stage:** {startup_data['product_stage']}")
            st.markdown(f"**Investment:** ${startup_data['total_investment']:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            
            # Success probability
            success_prob = prediction['probability'].get('Success', 0) * 100
            fail_prob = prediction['probability'].get('Fail', 0) * 100
            
            # Display gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = success_prob,
                title = {'text': "Success Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(66, 87, 178, 0.8)"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(220, 53, 69, 0.3)"},
                        {'range': [30, 70], 'color': "rgba(255, 193, 7, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(40, 167, 69, 0.3)"}
                    ]
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction label
            if success_prob > 70:
                st.markdown("<div style='text-align: center;'><span class='success-tag'>High Chance of Success</span></div>", unsafe_allow_html=True)
            elif success_prob > 40:
                st.markdown("<div style='text-align: center;'><span class='unclear-tag'>Moderate Chance of Success</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center;'><span class='fail-tag'>Low Chance of Success</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
    
    if models['ml_predictor'].feature_importances is not None:
        # Get feature names
        categorical_features = ['country', 'industry', 'product_stage']
        numerical_features = [col for col in filtered_df.columns if col not in categorical_features + ['id', 'name', 'success']]
        feature_names = numerical_features + categorical_features
        
        # Plot feature importance
        fig = models['ml_predictor'].plot_feature_importance(feature_names=feature_names)
        st.pyplot(fig)
        
        st.markdown(
            "Feature importance shows which factors have the most influence on the prediction model. "
            "Higher values indicate that the feature has a stronger effect on the model's predictions."
        )
    else:
        st.info("Feature importance not available for the current model.")

elif page == "💼 PMF анализ":
    st.markdown("<h1 class='main-header'>Анализ Product-Market Fit</h1>", unsafe_allow_html=True)
    
    # PMF overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Product-Market Fit (PMF)</h3>", unsafe_allow_html=True)
    st.markdown(
        "Product-Market Fit is the degree to which a product satisfies strong market demand. "
        "This analysis uses retention, NPS, user reviews, and growth metrics to determine "
        "if a startup has achieved PMF and provides recommendations for improvement."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # PMF Distribution
    st.markdown("<h2 class='sub-header'>PMF Distribution</h2>", unsafe_allow_html=True)
    fig = create_pmf_distribution(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # PMF Analysis for specific startup
    st.markdown("<h2 class='sub-header'>PMF Analysis for Startup</h2>", unsafe_allow_html=True)
    
    # Startup selector
    startup_options = filtered_df[['id', 'name']].copy()
    startup_options['display'] = startup_options['name'] + " (ID: " + startup_options['id'].astype(str) + ")"
    selected_startup = st.selectbox("Select a startup", startup_options['display'].tolist(), key="pmf_startup_selector")
    
    if selected_startup:
        # Extract ID from selection
        selected_id = int(selected_startup.split("(ID: ")[1].split(")")[0])
        
        # Get startup data
        startup_data = get_startup_by_id(selected_id, filtered_df)
        
        # Get user metrics if available
        try:
            user_metrics = get_user_metrics_by_startup_id(selected_id, user_metrics_df)
        except ValueError:
            user_metrics = None
        
        # Calculate PMF score
        pmf_result = models['pmf_analyzer'].calculate_pmf_score(startup_data, user_metrics)
        
        # Display PMF analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>{startup_data['name']}</h3>", unsafe_allow_html=True)
            
            # PMF score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pmf_result['pmf_score'],
                title = {'text': "PMF Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(66, 87, 178, 0.8)"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(220, 53, 69, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(255, 193, 7, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(40, 167, 69, 0.3)"}
                    ]
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PMF category
            if pmf_result['pmf_category'] == "High PMF":
                st.markdown("<div style='text-align: center;'><span class='success-tag'>High PMF</span></div>", unsafe_allow_html=True)
            elif pmf_result['pmf_category'] == "Medium PMF":
                st.markdown("<div style='text-align: center;'><span class='unclear-tag'>Medium PMF</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center;'><span class='fail-tag'>Low PMF</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
            
            # Display metrics
            if user_metrics is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Retention D1", f"{user_metrics['retention_d1']:.1f}%")
                    st.metric("Retention D7", f"{user_metrics['retention_d7']:.1f}%")
                    st.metric("Retention D30", f"{user_metrics['retention_d30']:.1f}%")
                
                with col2:
                    st.metric("NPS", f"{user_metrics['nps']:.1f}")
                    st.metric("User Growth", f"{user_metrics['user_growth_rate']:.1f}%")
                    st.metric("Viral Coefficient", f"{user_metrics['viral_coefficient']:.2f}")
            else:
                st.metric("Retention", f"{startup_data['retention']:.1f}%")
                st.metric("NPS", f"{startup_data['nps']:.1f}")
                st.metric("User Growth", f"{startup_data['user_growth_rate']:.1f}%")
                st.metric("User Reviews", f"{startup_data['user_reviews']:.1f}/100")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
        
        if 'pmf_recommendations' in startup_data and len(startup_data['pmf_recommendations']) > 0:
            for i, recommendation in enumerate(startup_data['pmf_recommendations']):
                st.markdown(f"**{i+1}.** {recommendation}")
        else:
            st.info("No specific recommendations available for this startup.")

elif page == "🧭 Карта стартапов":
    st.markdown("<h1 class='main-header'>Карта ландшафта стартапов</h1>", unsafe_allow_html=True)
    
    # Map overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Startup Landscape Map</h3>", unsafe_allow_html=True)
    st.markdown(
        "This map visualizes startups based on their innovation score and risk score. "
        "It helps identify different types of startups: Disruptors (high innovation, low risk), "
        "Moonshots (high innovation, high risk), Conservatives (low innovation, low risk), "
        "and Gamblers (low innovation, high risk)."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Industry filter for map
    map_industry = st.selectbox("Filter by Industry", industries)
    
    # Create landscape map
    industry_filter = None if map_industry == "All" else map_industry
    fig = models['landscape_map'].create_landscape_plot(filtered_df, industry_filter=industry_filter)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant statistics
    st.markdown("<h2 class='sub-header'>Quadrant Statistics</h2>", unsafe_allow_html=True)
    
    # Get quadrant statistics
    quadrant_stats = models['landscape_map'].get_quadrant_statistics(filtered_df)
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Disruptors</h3>", unsafe_allow_html=True)
        st.markdown("High innovation, low risk")
        
        stats = quadrant_stats.get('Disruptors', {})
        if stats:
            st.metric("Count", stats['count'])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            st.metric("Avg Investment", f"${stats['avg_investment']/1000000:.2f}M")
            
            st.markdown("**Top Industries:**")
            for industry, count in stats['top_industries'].items():
                st.markdown(f"- {industry}: {count}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Conservatives</h3>", unsafe_allow_html=True)
        st.markdown("Low innovation, low risk")
        
        stats = quadrant_stats.get('Conservatives', {})
        if stats:
            st.metric("Count", stats['count'])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            st.metric("Avg Investment", f"${stats['avg_investment']/1000000:.2f}M")
            
            st.markdown("**Top Industries:**")
            for industry, count in stats['top_industries'].items():
                st.markdown(f"- {industry}: {count}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Moonshots</h3>", unsafe_allow_html=True)
        st.markdown("High innovation, high risk")
        
        stats = quadrant_stats.get('Moonshots', {})
        if stats:
            st.metric("Count", stats['count'])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            st.metric("Avg Investment", f"${stats['avg_investment']/1000000:.2f}M")
            
            st.markdown("**Top Industries:**")
            for industry, count in stats['top_industries'].items():
                st.markdown(f"- {industry}: {count}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Gamblers</h3>", unsafe_allow_html=True)
        st.markdown("Low innovation, high risk")
        
        stats = quadrant_stats.get('Gamblers', {})
        if stats:
            st.metric("Count", stats['count'])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            st.metric("Avg Investment", f"${stats['avg_investment']/1000000:.2f}M")
            
            st.markdown("**Top Industries:**")
            for industry, count in stats['top_industries'].items():
                st.markdown(f"- {industry}: {count}")
        
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Скоринг и рекомендации":
    st.markdown("<h1 class='main-header'>Скоринг стартапов и рекомендации</h1>", unsafe_allow_html=True)
    
    # Scoring overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Startup Scoring Model</h3>", unsafe_allow_html=True)
    st.markdown(
        "This model evaluates startups based on four key categories: Team, Product, Market, and Finance. "
        "Each category has a set of weighted metrics that contribute to the overall score. "
        "The model also provides a risk category (High Risk, Medium, Strong Bet) based on the overall score."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Startup selector
    startup_options = filtered_df[['id', 'name']].copy()
    startup_options['display'] = startup_options['name'] + " (ID: " + startup_options['id'].astype(str) + ")"
    selected_startup = st.selectbox("Select a startup", startup_options['display'].tolist(), key="scoring_startup_selector")
    
    if selected_startup:
        # Extract ID from selection
        selected_id = int(selected_startup.split("(ID: ")[1].split(")")[0])
        
        # Get startup data
        startup_data = get_startup_by_id(selected_id, filtered_df)
        
        # Display startup information
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>{startup_data['name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Industry:** {startup_data['industry']}")
            st.markdown(f"**Country:** {startup_data['country']}")
            st.markdown(f"**Founded:** {startup_data['year_founded']}")
            st.markdown(f"**Product Stage:** {startup_data['product_stage']}")
            st.markdown(f"**Investment:** ${startup_data['total_investment']:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add to comparison
            if st.button("Add to Comparison"):
                if selected_id not in st.session_state.comparison_startups:
                    st.session_state.comparison_startups.append(selected_id)
                    st.success(f"Added {startup_data['name']} to comparison")
                else:
                    st.warning(f"{startup_data['name']} is already in comparison")
        
        with col2:
            # Overall score gauge
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Overall Score</h3>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = startup_data['overall_score'],
                title = {'text': "Overall Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(66, 87, 178, 0.8)"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(220, 53, 69, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(255, 193, 7, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(40, 167, 69, 0.3)"}
                    ]
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk category
            if startup_data['risk_category'] == "Strong Bet":
                st.markdown("<div style='text-align: center;'><span class='success-tag'>Strong Bet</span></div>", unsafe_allow_html=True)
            elif startup_data['risk_category'] == "Medium":
                st.markdown("<div style='text-align: center;'><span class='unclear-tag'>Medium Risk</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center;'><span class='fail-tag'>High Risk</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Category scores
        st.markdown("<h3>Category Scores</h3>", unsafe_allow_html=True)
        
        # Create radar chart
        category_scores = {
            'Team': startup_data['team_score'],
            'Product': startup_data['product_score'],
            'Market': startup_data['market_score'],
            'Finance': startup_data['finance_score']
        }
        
        fig = create_radar_chart(category_scores, title=f"{startup_data['name']} - Category Scores")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores
        st.markdown("<h3>Detailed Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Team</h4>", unsafe_allow_html=True)
            st.metric("Score", f"{startup_data['team_score']:.1f}")
            st.metric("Founders", startup_data['founder_count'])
            st.metric("Experience", f"{startup_data['founder_experience']} years")
            st.metric("Previous Startups", "Yes" if startup_data['previous_startups'] > 0 else "No")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Product</h4>", unsafe_allow_html=True)
            st.metric("Score", f"{startup_data['product_score']:.1f}")
            st.metric("Stage", startup_data['product_stage'])
            st.metric("Uniqueness", f"{startup_data['product_uniqueness']}/10")
            st.metric("Innovation", f"{startup_data['innovation_score']}/10")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Market</h4>", unsafe_allow_html=True)
            st.metric("Score", f"{startup_data['market_score']:.1f}")
            st.metric("Market Size", f"${startup_data['market_size']/1000000:.1f}M")
            st.metric("Growth Rate", f"{startup_data['market_growth_rate']:.1f}%")
            st.metric("Competitors", startup_data['competitors_count'])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Finance</h4>", unsafe_allow_html=True)
            st.metric("Score", f"{startup_data['finance_score']:.1f}")
            st.metric("Revenue", f"${startup_data['revenue']/1000000:.2f}M")
            st.metric("Burn Rate", f"${startup_data['burn_rate']/1000000:.2f}M")
            st.metric("Cash Reserves", f"${startup_data['cash_reserves']/1000000:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate report button
        st.markdown("<h3>Export Report</h3>", unsafe_allow_html=True)
        if st.button("Generate Investment Report"):
            with st.spinner("Generating report..."):
                # Simulate report generation
                time.sleep(2)
                st.success("Report generated successfully!")
                
                # Create download link
                st.markdown(f"<a href='#' download='startup_report_{selected_id}.pdf'>Download Report</a>", unsafe_allow_html=True)
    
    # Startup comparison
    st.markdown("<h2 class='sub-header'>Startup Comparison</h2>", unsafe_allow_html=True)
    
    if len(st.session_state.comparison_startups) > 0:
        # Get startup data
        comparison_data = []
        for startup_id in st.session_state.comparison_startups:
            try:
                startup = get_startup_by_id(startup_id, startups_df)
                comparison_data.append(startup)
            except ValueError:
                pass
        
        if len(comparison_data) > 0:
            # Create comparison dataframe
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(comparison_df[['name', 'industry', 'overall_score', 'team_score', 'product_score', 'market_score', 'finance_score']], use_container_width=True)
            
            # Create comparison chart
            fig = create_comparison_chart(comparison_df, comparison_df['id'].tolist())
            st.plotly_chart(fig, use_container_width=True)
            
            # Clear comparison button
            if st.button("Clear Comparison"):
                st.session_state.comparison_startups = []
                st.experimental_rerun()
        else:
            st.info("No startups to compare")
    else:
        st.info("Add startups to comparison by clicking 'Add to Comparison' button when viewing a startup")