import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from forecasting_models import TimeSeriesForecaster
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Vehicle Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .dataset-info {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚗 Vehicle Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Dataset information
    st.markdown("""
    <div class="dataset-info">
        <h3>📊 Dataset Information</h3>
        <p><strong>Source:</strong> New Motor Vehicle Sales Data (Canada)</p>
        <p><strong>Description:</strong> Monthly sales data for new motor vehicles in Canada</p>
        <p><strong>Scope:</strong> National-level data, unadjusted for seasonality</p>
        <p><strong>Analysis Period:</strong> Last 72 months of available data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Model Configuration")
    
    # Check if data file exists
    default_file = "New motor vehicle sales data.csv"
    if not os.path.exists(default_file):
        st.error(f"❌ Data file '{default_file}' not found. Please ensure the file is in the same directory as the application.")
        st.stop()
    
    # Parameters
    st.sidebar.subheader("🔧 Forecasting Parameters")
    test_period = st.sidebar.slider(
        "Test Period (months)", 
        min_value=12, 
        max_value=48, 
        value=36,
        help="Number of months to use for testing the models"
    )
    
    lookback_window = st.sidebar.slider(
        "LSTM Lookback Window", 
        min_value=6, 
        max_value=24, 
        value=12,
        help="Number of previous time steps to use for LSTM predictions"
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Select Models to Train")
    selected_models = {
        'ARIMA': st.sidebar.checkbox("ARIMA", value=True, help="Auto-Regressive Integrated Moving Average"),
        'Prophet': st.sidebar.checkbox("Prophet", value=True, help="Facebook's Prophet model"),
        'NeuralProphet': st.sidebar.checkbox("NeuralProphet", value=True, help="Neural network-based Prophet"),
        'LSTM': st.sidebar.checkbox("LSTM", value=True, help="Long Short-Term Memory neural network")
    }
    
    # Model descriptions
    with st.sidebar.expander("ℹ️ Model Information"):
        st.markdown("""
        **ARIMA**: Traditional statistical model good for linear trends
        
        **Prophet**: Handles seasonality and holidays well
        
        **NeuralProphet**: Neural network version of Prophet
        
        **LSTM**: Deep learning model for complex patterns
        """)
    
    # Validation
    if not any(selected_models.values()):
        st.sidebar.warning("⚠️ Please select at least one model to train.")
        return
    
    # Run button
    if st.sidebar.button("🚀 Run Forecasting Analysis", type="primary", use_container_width=True):
        run_forecasting(default_file, test_period, lookback_window, selected_models)
    
    # Display sample data preview
    display_data_preview(default_file)

def display_data_preview(file_path):
    """Display a preview of the dataset"""
    try:
        st.subheader("👀 Data Preview")
        
        # Load a small sample of data for preview
        data = pd.read_csv(file_path)
        data['REF_DATE'] = pd.to_datetime(data['REF_DATE'])
        
        # Filter for Canada data
        canada_data = data[data['GEO'] == 'Canada']
        
        # Show basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Date Range", f"{data['REF_DATE'].min().year} - {data['REF_DATE'].max().year}")
        with col3:
            st.metric("Canada Records", f"{len(canada_data):,}")
        with col4:
            unique_geos = data['GEO'].nunique()
            st.metric("Geographic Areas", unique_geos)
        
        # Show sample data
        st.subheader("📋 Sample Data")
        sample_data = data.head(10)[['REF_DATE', 'GEO', 'Vehicle type', 'VALUE']]
        st.dataframe(sample_data, use_container_width=True)
        
        # Show data structure
        with st.expander("🔍 Data Structure Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Unique Values by Column:**")
                for col in ['GEO', 'Sales', 'Seasonal adjustment', 'Vehicle type', 'Origin of manufacture']:
                    if col in data.columns:
                        unique_count = data[col].nunique()
                        st.write(f"- {col}: {unique_count} unique values")
            
            with col2:
                st.write("**Data Types:**")
                st.write(data.dtypes.to_frame('Data Type'))
        
    except Exception as e:
        st.error(f"Error loading data preview: {str(e)}")

def run_forecasting(file_path, test_period, lookback_window, selected_models):
    """Run the forecasting process"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize forecaster
        status_text.text("🔄 Initializing forecaster...")
        progress_bar.progress(10)
        
        forecaster = TimeSeriesForecaster(
            file_path=file_path,
            test_period=test_period,
            lookback_window=lookback_window
        )
        
        # Load data
        status_text.text("📊 Loading and preprocessing data...")
        progress_bar.progress(20)
        
        if not forecaster.load_and_preprocess_data():
            st.error("❌ Failed to load data. Please check your file format.")
            return
        
        # Display data info
        st.subheader("📈 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Data Points", len(forecaster.ts_data))
        with col2:
            st.metric("🎯 Training Points", len(forecaster.train_set))
        with col3:
            st.metric("🧪 Test Points", len(forecaster.test_set))
        with col4:
            train_years = len(forecaster.train_set) / 12
            st.metric("📅 Training Years", f"{train_years:.1f}")
        
        # Show date ranges
        st.info(f"**Training Period:** {forecaster.train_set.index[0].strftime('%Y-%m')} to {forecaster.train_set.index[-1].strftime('%Y-%m')}")
        st.info(f"**Test Period:** {forecaster.test_set.index[0].strftime('%Y-%m')} to {forecaster.test_set.index[-1].strftime('%Y-%m')}")
        
        # Train selected models
        total_models = sum(selected_models.values())
        progress_step = 60 / total_models if total_models > 0 else 0
        current_progress = 20
        
        models_success = {}
        
        if selected_models['ARIMA']:
            status_text.text("🔄 Training ARIMA model...")
            models_success['ARIMA'] = forecaster.train_arima()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['Prophet']:
            status_text.text("🔄 Training Prophet model...")
            models_success['Prophet'] = forecaster.train_prophet()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['NeuralProphet']:
            status_text.text("🔄 Training NeuralProphet model...")
            models_success['NeuralProphet'] = forecaster.train_neural_prophet()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['LSTM']:
            status_text.text("🔄 Training LSTM model...")
            models_success['LSTM'] = forecaster.train_lstm()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        status_text.text("📊 Generating results...")
        progress_bar.progress(90)
        
        # Display results
        display_results(forecaster, models_success)
        
        progress_bar.progress(100)
        status_text.text("✅ Forecasting analysis completed successfully!")
        
        # Add download option for results
        add_download_options(forecaster)
        
    except Exception as e:
        st.error(f"❌ An error occurred during forecasting: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)

def display_results(forecaster, models_success):
    """Display forecasting results"""
    
    # Results table
    st.subheader("📊 Model Performance Comparison")
    
    results_df = forecaster.get_results_dataframe()
    if results_df is not None and len(results_df) > 0:
        # Add ranking
        results_df['MAE_Rank'] = results_df['MAE'].rank()
        results_df['RMSE_Rank'] = results_df['RMSE'].rank()
        results_df['MAPE_Rank'] = results_df['MAPE'].rank()
        results_df['Average_Rank'] = (results_df['MAE_Rank'] + results_df['RMSE_Rank'] + results_df['MAPE_Rank']) / 3
        
        # Sort by average rank
        results_df = results_df.sort_values('Average_Rank')
        
        # Style the dataframe
        styled_df = results_df[['Model', 'MAE', 'RMSE', 'MAPE']].style.format({
            'MAE': '{:.2f}',
            'RMSE': '{:.2f}',
            'MAPE': '{:.2f}%'
        }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model
        best_model = results_df.iloc[0]['Model']
        best_mae = results_df.iloc[0]['MAE']
        st.success(f"🏆 **Best performing model:** {best_model} (MAE: {best_mae:.2f})")
        
        # Performance insights
        with st.expander("📈 Performance Insights"):
            st.write("**Model Rankings (1 = Best):**")
            for idx, row in results_df.iterrows():
                st.write(f"**{int(row['Average_Rank'])}. {row['Model']}** - Average Rank: {row['Average_Rank']:.1f}")
    
    # Interactive plot using Plotly
    st.subheader("📈 Forecast Visualization")
    
    fig = create_interactive_plot(forecaster)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Model status
    st.subheader("🔧 Model Training Status")
    cols = st.columns(len(models_success))
    
    for i, (model, success) in enumerate(models_success.items()):
        with cols[i]:
            if success:
                st.success(f"✅ {model}")
                if model in forecaster.results:
                    mae = forecaster.results[model]['mae']
                    st.metric("MAE", f"{mae:.2f}")
            else:
                st.error(f"❌ {model}")
                st.write("Training failed")

def create_interactive_plot(forecaster):
    """Create interactive Plotly chart"""
    if not forecaster.results:
        return None
    
    fig = go.Figure()
    
    # Actual values
    full_data = pd.concat([forecaster.train_set, forecaster.test_set])
    
    # Training data
    fig.add_trace(go.Scatter(
        x=forecaster.train_set.index,
        y=forecaster.train_set['VALUE'],
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Training</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
    ))
    
    # Test data (actual)
    fig.add_trace(go.Scatter(
        x=forecaster.test_set.index,
        y=forecaster.test_set['VALUE'],
        mode='lines',
        name='Actual (Test)',
        line=dict(color='black', width=3),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
    ))
    
    # Predictions
    colors = ['red', 'green', 'orange', 'purple']
    for i, (model_name, model_results) in enumerate(forecaster.results.items()):
        fig.add_trace(go.Scatter(
            x=forecaster.test_set.index,
            y=model_results['predictions'],
            mode='lines',
            name=f'{model_name} (MAE: {model_results["mae"]:.2f})',
            line=dict(color=colors[i % len(colors)], dash='dash', width=2),
            hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Predicted: %{{y:,.0f}}<br>MAE: {model_results["mae"]:.2f}<extra></extra>'
        ))
    
    # Add vertical line for train/test split
    fig.add_vline(
        x=forecaster.train_set.index[-1],
        line_dash="dot",
        line_color="red",
        annotation_text="Train/Test Split",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Model Forecast Comparison - Vehicle Sales",
        xaxis_title="Date",
        yaxis_title="Sales (Units)",
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def add_download_options(forecaster):
    """Add download options for results"""
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download performance metrics
        results_df = forecaster.get_results_dataframe()
        if results_df is not None:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Performance Metrics (CSV)",
                data=csv,
                file_name="model_performance_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        # Download predictions
        if forecaster.results:
            predictions_df = pd.DataFrame(index=forecaster.test_set.index)
            predictions_df['Actual'] = forecaster.test_set['VALUE']
            
            for model_name, model_results in forecaster.results.items():
                predictions_df[f'{model_name}_Prediction'] = model_results['predictions']
            
            csv = predictions_df.to_csv()
            st.download_button(
                label="🔮 Download Predictions (CSV)",
                data=csv,
                file_name="model_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
