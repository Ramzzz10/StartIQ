import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

class StartupLandscapeMap:
    """
    Creates a visual map of startups based on innovation and risk scores.
    Allows for filtering by industry and visualizing success status.
    """
    
    def __init__(self):
        """Initialize the landscape map"""
        # Define quadrant boundaries
        self.quadrants = {
            'Disruptors': {'x_min': 5, 'x_max': 10, 'y_min': 0, 'y_max': 5},  # High innovation, low risk
            'Moonshots': {'x_min': 5, 'x_max': 10, 'y_min': 5, 'y_max': 10},  # High innovation, high risk
            'Conservatives': {'x_min': 0, 'x_max': 5, 'y_min': 0, 'y_max': 5},  # Low innovation, low risk
            'Gamblers': {'x_min': 0, 'x_max': 5, 'y_min': 5, 'y_max': 10}  # Low innovation, high risk
        }
    
    def assign_quadrant(self, innovation, risk):
        """
        Assign a startup to a quadrant based on innovation and risk scores
        
        Parameters:
        -----------
        innovation : float
            Innovation score (0-10)
        risk : float
            Risk score (0-10)
            
        Returns:
        --------
        str
            Quadrant name
        """
        for quadrant, bounds in self.quadrants.items():
            if (bounds['x_min'] <= innovation <= bounds['x_max'] and 
                bounds['y_min'] <= risk <= bounds['y_max']):
                return quadrant
        
        return "Unknown"  # Fallback
    
    def prepare_map_data(self, startups_df):
        """
        Prepare data for the landscape map
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added quadrant column
        """
        # Create a copy of the input DataFrame
        result_df = startups_df.copy()
        
        # Ensure innovation and risk scores are present
        if 'innovation_score' not in result_df.columns:
            result_df['innovation_score'] = np.random.randint(0, 11, len(result_df))
        
        if 'risk_score' not in result_df.columns:
            result_df['risk_score'] = np.random.randint(0, 11, len(result_df))
        
        # Assign quadrants
        result_df['quadrant'] = result_df.apply(
            lambda row: self.assign_quadrant(row['innovation_score'], row['risk_score']), 
            axis=1
        )
        
        return result_df
    
    def create_landscape_plot(self, startups_df, industry_filter=None, figsize=(1000, 700)):
        """
        Create a landscape plot of startups
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
        industry_filter : str, optional
            Industry to filter by
        figsize : tuple
            Figure size in pixels
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Prepare data
        plot_df = self.prepare_map_data(startups_df)
        
        # Apply industry filter if provided
        if industry_filter is not None:
            plot_df = plot_df[plot_df['industry'] == industry_filter]
        
        # Create color mapping for success status
        color_map = {
            'Success': '#28a745',  # Green
            'Fail': '#dc3545',     # Red
            'Unclear': '#6c757d'   # Gray
        }
        
        # Create hover text
        plot_df['hover_text'] = plot_df.apply(
            lambda row: f"<b>{row['name']}</b><br>" +
                       f"Industry: {row['industry']}<br>" +
                       f"Founded: {row['year_founded']}<br>" +
                       f"Stage: {row['product_stage']}<br>" +
                       f"Revenue: ${row['revenue']:,.2f}<br>" +
                       f"Quadrant: {row['quadrant']}<br>" +
                       f"Success: {row['success']}",
            axis=1
        )
        
        # Create the scatter plot
        fig = px.scatter(
            plot_df, 
            x='innovation_score', 
            y='risk_score',
            color='success',
            color_discrete_map=color_map,
            hover_name='name',
            hover_data={
                'innovation_score': True,
                'risk_score': True,
                'name': False,  # Hide name as it's in hover_name
                'success': False,  # Hide success as it's in color
                'hover_text': True
            },
            custom_data=['id', 'quadrant'],
            size='total_investment',
            size_max=30,
            opacity=0.8,
            title=f'Startup Landscape Map {f"- {industry_filter}" if industry_filter else ""}',
            width=figsize[0],
            height=figsize[1]
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='%{customdata[1]}<br>%{hovertext}<extra></extra>'
        )
        
        # Add quadrant boundaries
        fig.add_shape(
            type="line", x0=5, y0=0, x1=5, y1=10,
            line=dict(color="gray", width=1, dash="dash")
        )
        fig.add_shape(
            type="line", x0=0, y0=5, x1=10, y1=5,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Add quadrant labels
        fig.add_annotation(x=2.5, y=2.5, text="Conservatives", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=7.5, y=2.5, text="Disruptors", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=2.5, y=7.5, text="Gamblers", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=7.5, y=7.5, text="Moonshots", showarrow=False, font=dict(size=14))
        
        # Update layout
        fig.update_layout(
            xaxis_title="Innovation Score",
            yaxis_title="Risk Score",
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            legend_title="Success Status",
            font=dict(size=12),
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        
        return fig
    
    def get_quadrant_statistics(self, startups_df):
        """
        Calculate statistics for each quadrant
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
            
        Returns:
        --------
        dict
            Dictionary containing statistics for each quadrant
        """
        # Prepare data
        plot_df = self.prepare_map_data(startups_df)
        
        # Calculate statistics for each quadrant
        stats = {}
        for quadrant in self.quadrants:
            quadrant_df = plot_df[plot_df['quadrant'] == quadrant]
            
            if len(quadrant_df) > 0:
                success_rate = (quadrant_df['success'] == 'Success').mean() * 100
                avg_investment = quadrant_df['total_investment'].mean()
                avg_revenue = quadrant_df['revenue'].mean()
                top_industries = quadrant_df['industry'].value_counts().head(3).to_dict()
                
                stats[quadrant] = {
                    'count': len(quadrant_df),
                    'success_rate': success_rate,
                    'avg_investment': avg_investment,
                    'avg_revenue': avg_revenue,
                    'top_industries': top_industries
                }
            else:
                stats[quadrant] = {
                    'count': 0,
                    'success_rate': 0,
                    'avg_investment': 0,
                    'avg_revenue': 0,
                    'top_industries': {}
                }
        
        return stats
    
    def save(self, filepath):
        """Save the model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        return joblib.load(filepath)
