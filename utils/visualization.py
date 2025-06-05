import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_radar_chart(category_scores, title="Радар скоринга стартапа"):
    """
    Create a radar chart for startup category scores
    
    Parameters:
    -----------
    category_scores : dict
        Dictionary containing category scores
    title : str
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Перевод категорий
    category_translation = {
        'Team': 'Команда',
        'Product': 'Продукт',
        'Market': 'Рынок',
        'Finance': 'Финансы'
    }
    
    # Создаем новый словарь с переведенными категориями
    translated_categories = {}
    for key, value in category_scores.items():
        translated_key = category_translation.get(key, key)
        translated_categories[translated_key] = value
    
    categories = list(translated_categories.keys())
    values = list(translated_categories.values())
    
    # Add the first value at the end to close the polygon
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Оценка',
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=title,
        showlegend=False
    )
    
    return fig

def create_success_histogram(startups_df, group_by='industry', title=None):
    """
    Create a histogram of startup success by industry or country
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data
    group_by : str
        Column to group by ('industry' or 'country')
    title : str, optional
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Group data
    grouped = pd.crosstab(startups_df[group_by], startups_df['success'])
    
    # Sort by success rate
    if 'Success' in grouped.columns:
        success_rate = grouped['Success'] / grouped.sum(axis=1)
        grouped = grouped.loc[success_rate.sort_values(ascending=False).index]
    
    # Create figure
    fig = go.Figure()
    
    # Перевод категорий успеха
    success_translation = {
        'Success': 'Успех',
        'Fail': 'Неудача',
        'Unclear': 'Неопределенно'
    }
    
    # Add traces for each success category
    for success_category in grouped.columns:
        color = '#28a745' if success_category == 'Success' else '#dc3545' if success_category == 'Fail' else '#6c757d'
        
        fig.add_trace(go.Bar(
            x=grouped.index,
            y=grouped[success_category],
            name=success_translation.get(success_category, success_category),
            marker_color=color
        ))
    
    # Перевод названий осей
    axis_labels = {
        'industry': 'Отрасль',
        'country': 'Страна'
    }
    
    # Update layout
    fig.update_layout(
        title=title or f'Успех стартапов по {axis_labels.get(group_by, group_by)}',
        xaxis_title=axis_labels.get(group_by, group_by.capitalize()),
        yaxis_title='Количество',
        barmode='stack',
        legend_title='Статус успеха'
    )
    
    return fig

def create_trend_analysis(startups_df, metric='success_rate', title=None):
    """
    Create a trend analysis chart by year
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data
    metric : str
        Metric to analyze ('success_rate', 'avg_investment', 'avg_revenue')
    title : str, optional
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Group data by year
    yearly_data = startups_df.groupby('year_founded')
    
    # Calculate metric
    if metric == 'success_rate':
        # Calculate success rate (excluding 'Unclear')
        # Filter out 'Unclear' first
        filtered_df = startups_df[startups_df['success'] != 'Unclear']
        # Then group by year
        success_by_year = filtered_df.groupby('year_founded')['success'].apply(
            lambda x: (x == 'Success').sum() / len(x) * 100 if len(x) > 0 else 0
        )
        y_values = success_by_year
        y_label = 'Уровень успеха (%)'
    elif metric == 'avg_investment':
        y_values = yearly_data['total_investment'].mean()
        y_label = 'Средние инвестиции ($)'
    elif metric == 'avg_revenue':
        y_values = yearly_data['revenue'].mean()
        y_label = 'Средняя выручка ($)'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_values.index,
        y=y_values.values,
        mode='lines+markers',
        name=y_label,
        line=dict(color='rgba(31, 119, 180, 0.8)', width=3),
        marker=dict(size=8)
    ))
    
    # Add trend line
    x_numeric = np.array(range(len(y_values)))
    z = np.polyfit(x_numeric, y_values.values, 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=y_values.index,
        y=p(x_numeric),
        mode='lines',
        name='Тренд',
        line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=title or f'Тренд {y_label} по годам',
        xaxis_title='Год основания',
        yaxis_title=y_label,
        legend_title='Метрика'
    )
    
    return fig

def create_pmf_distribution(startups_df, title="Распределение Product-Market Fit"):
    """
    Create a pie chart of PMF categories
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data with PMF categories
    title : str
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Count PMF categories
    pmf_counts = startups_df['pmf_category'].value_counts()
    
    # Перевод категорий PMF
    pmf_translation = {
        'High PMF': 'Высокий PMF',
        'Medium PMF': 'Средний PMF',
        'Low PMF': 'Низкий PMF'
    }
    
    # Создаем новый индекс с переведенными категориями
    translated_index = [pmf_translation.get(category, category) for category in pmf_counts.index]
    
    # Create color map
    color_map = {
        'High PMF': '#28a745',  # Green
        'Medium PMF': '#ffc107',  # Yellow
        'Low PMF': '#dc3545'  # Red
    }
    
    colors = [color_map.get(category, '#6c757d') for category in pmf_counts.index]
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=translated_index,
        values=pmf_counts.values,
        marker_colors=colors,
        textinfo='percent+label',
        hole=0.4
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        legend_title='Категория PMF'
    )
    
    return fig

def create_comparison_chart(startups_df, startup_ids, metrics=None):
    """
    Create a comparison chart for multiple startups
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data
    startup_ids : list
        List of startup IDs to compare
    metrics : list, optional
        List of metrics to compare
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = [
            'overall_score', 'team_score', 'product_score', 'market_score', 'finance_score',
            'pmf_score', 'retention', 'nps', 'user_growth_rate'
        ]
    
    # Перевод метрик
    metric_translation = {
        'overall_score': 'Общий скоринг',
        'team_score': 'Команда',
        'product_score': 'Продукт',
        'market_score': 'Рынок',
        'finance_score': 'Финансы',
        'pmf_score': 'PMF',
        'retention': 'Удержание',
        'nps': 'NPS',
        'user_growth_rate': 'Рост пользователей'
    }
    
    # Создаем новый список с переведенными метриками
    translated_metrics = [metric_translation.get(metric, metric) for metric in metrics]
    
    # Filter startups
    startups = startups_df[startups_df['id'].isin(startup_ids)]
    
    if len(startups) == 0:
        raise ValueError("Стартапы с указанными ID не найдены")
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each startup
    for _, startup in startups.iterrows():
        values = [startup[metric] if metric in startup else 0 for metric in metrics]
        
        fig.add_trace(go.Bar(
            x=translated_metrics,
            y=values,
            name=startup['name']
        ))
    
    # Update layout
    fig.update_layout(
        title='Сравнение стартапов',
        xaxis_title='Метрика',
        yaxis_title='Значение',
        barmode='group',
        legend_title='Стартап'
    )
    
    return fig

def create_investment_dashboard(startups_df, title="Дашборд инвестиций"):
    """
    Create an investment dashboard with multiple charts
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data
    title : str
        Dashboard title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Инвестиции по отраслям',
            'Выручка по отраслям',
            'Инвестиции vs. Выручка',
            'Инвестиции vs. Уровень успеха'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Investment by Industry
    industry_investment = startups_df.groupby('industry')['total_investment'].mean().sort_values(ascending=False)
    
    fig.add_trace(
        go.Bar(
            x=industry_investment.index,
            y=industry_investment.values,
            marker_color='rgba(31, 119, 180, 0.8)'
        ),
        row=1, col=1
    )
    
    # 2. Revenue by Industry
    industry_revenue = startups_df.groupby('industry')['revenue'].mean().sort_values(ascending=False)
    
    fig.add_trace(
        go.Bar(
            x=industry_revenue.index,
            y=industry_revenue.values,
            marker_color='rgba(44, 160, 44, 0.8)'
        ),
        row=1, col=2
    )
    
    # 3. Investment vs. Revenue
    fig.add_trace(
        go.Scatter(
            x=startups_df['total_investment'],
            y=startups_df['revenue'],
            mode='markers',
            marker=dict(
                size=8,
                color=startups_df['overall_score'] if 'overall_score' in startups_df.columns else 'blue',
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Score')
            ),
            text=startups_df['name'],
            hovertemplate='<b>%{text}</b><br>Investment: $%{x:,.2f}<br>Revenue: $%{y:,.2f}'
        ),
        row=2, col=1
    )
    
    # 4. Investment vs. Success Rate (by industry)
    industry_success = startups_df[startups_df['success'] != 'Unclear'].groupby('industry').apply(
        lambda x: (x['success'] == 'Success').mean() * 100
    ).sort_values(ascending=False)
    
    industry_investment = startups_df.groupby('industry')['total_investment'].mean()
    
    fig.add_trace(
        go.Scatter(
            x=industry_investment.loc[industry_success.index],
            y=industry_success.values,
            mode='markers+text',
            marker=dict(size=12),
            text=industry_success.index,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Avg Investment: $%{x:,.2f}<br>Success Rate: %{y:.1f}%'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text='Отрасль', row=1, col=1)
    fig.update_yaxes(title_text='Средние инвестиции ($)', row=1, col=1)
    
    fig.update_xaxes(title_text='Отрасль', row=1, col=2)
    fig.update_yaxes(title_text='Средняя выручка ($)', row=1, col=2)
    
    fig.update_xaxes(title_text='Инвестиции ($)', row=2, col=1)
    fig.update_yaxes(title_text='Выручка ($)', row=2, col=1)
    
    fig.update_xaxes(title_text='Средние инвестиции ($)', row=2, col=2)
    fig.update_yaxes(title_text='Уровень успеха (%)', row=2, col=2)
    
    return fig
