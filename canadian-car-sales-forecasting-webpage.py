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
    page_title="Time Series Forecasting Dashboard",
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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚗 Vehicle Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Model Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your time series data CSV file"
    )
    
    # Parameters
    test_period = st.sidebar.slider("Test Period (months)", 12, 60, 36)
    lookback_window = st.sidebar.slider("LSTM Lookback Window", 6, 24, 12)
    
    # Model selection
    st.sidebar.subheader("Select Models to Train")
    selected_models = {
        'ARIMA': st.sidebar.checkbox("ARIMA", value=True),
        'Prophet': st.sidebar.checkbox("Prophet", value=True),
        'NeuralProphet': st.sidebar.checkbox("NeuralProphet", value=True),
        'LSTM': st.sidebar.checkbox("LSTM", value=True)
    }
    
    # Main content
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.sidebar.button("🚀 Run Forecasting", type="primary"):
            run_forecasting("temp_data.csv", test_period, lookback_window, selected_models)
        
        # Clean up temp file
        if os.path.exists("temp_data.csv"):
            os.remove("temp_data.csv")
    
    else:
        # Default file path (if you have a default dataset)
        default_file = "New motor vehicle sales data.csv"
        if os.path.exists(default_file):
            st.info("📁 Using default dataset. Upload your own file to analyze different data.")
            if st.sidebar.button("🚀 Run Forecasting with Default Data", type="primary"):
                run_forecasting(default_file, test_period, lookback_window, selected_models)
        else:
            st.warning("⚠️ Please upload a CSV file to begin forecasting.")
            st.markdown("""
            ### Expected Data Format:
            Your CSV should contain columns:
            - `REF_DATE`: Date column
            - `VALUE`: Numeric values to forecast
            - `GEO`: Geographic information
            - `Sales`: Sales type
            - `Seasonal adjustment`: Seasonal adjustment info
            - `Vehicle type`: Vehicle type
            - `Origin of manufacture`: Origin information
            """)

def run_forecasting(file_path, test_period, lookback_window, selected_models):
    """Run the forecasting process"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize forecaster
        status_text.text("Initializing forecaster...")
        progress_bar.progress(10)
        
        forecaster = TimeSeriesForecaster(
            file_path=file_path,
            test_period=test_period,
            lookback_window=lookback_window
        )
        
        # Load data
        status_text.text("Loading and preprocessing data...")
        progress_bar.progress(20)
        
        if not forecaster.load_and_preprocess_data():
            st.error("Failed to load data. Please check your file format.")
            return
        
        # Display data info
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Data Points", len(forecaster.ts_data))
        with col2:
            st.metric("Training Points", len(forecaster.train_set))
        with col3:
            st.metric("Test Points", len(forecaster.test_set))
        
        # Train selected models
        progress_step = 60 / sum(selected_models.values())
        current_progress = 20
        
        models_success = {}
        
        if selected_models['ARIMA']:
            status_text.text("Training ARIMA model...")
            models_success['ARIMA'] = forecaster.train_arima()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['Prophet']:
            status_text.text("Training Prophet model...")
            models_success['Prophet'] = forecaster.train_prophet()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['NeuralProphet']:
            status_text.text("Training NeuralProphet model...")
            models_success['NeuralProphet'] = forecaster.train_neural_prophet()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        if selected_models['LSTM']:
            status_text.text("Training LSTM model...")
            models_success['LSTM'] = forecaster.train_lstm()
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        status_text.text("Generating results...")
        progress_bar.progress(90)
        
        # Display results
        display_results(forecaster, models_success)
        
        progress_bar.progress(100)
        status_text.text("✅ Forecasting completed!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

def display_results(forecaster, models_success):
    """Display forecasting results"""
    
    # Results table
    st.subheader("📈 Model Performance Comparison")
    
    results_df = forecaster.get_results_dataframe()
    if results_df is not None:
        # Style the dataframe
        styled_df = results_df.style.format({
            'MAE': '{:.2f}',
            'RMSE': '{:.2f}',
            'MAPE': '{:.2f}%'
        }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model
        best_model_idx = results_df['MAE'].idxmin()
        best_model = results_df.loc[best_model_idx, 'Model']
        st.success(f"🏆 Best performing model: **{best_model}** (lowest MAE)")
    
    # Interactive plot using Plotly
    st.subheader("📊 Forecast Visualization")
    
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
            else:
                st.error(f"❌ {model}")

def create_interactive_plot(forecaster):
    """Create interactive Plotly chart"""
    if not forecaster.results:
        return None
    
    fig = go.Figure()
    
    # Actual values
    full_data = pd.concat([forecaster.train_set, forecaster.test_set])
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data['VALUE'],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=3)
    ))
    
    # Predictions
    colors = ['red', 'blue', 'green', 'orange']
    for i, (model_name, model_results) in enumerate(forecaster.results.items()):
        fig.add_trace(go.Scatter(
            x=forecaster.test_set.index,
            y=model_results['predictions'],
            mode='lines',
            name=f'{model_name} (MAE: {model_results["mae"]:.2f})',
            line=dict(color=colors[i % len(colors)], dash='dash', width=2)
        ))
    
    # Add vertical line for train/test split
    fig.add_vline(
        x=forecaster.train_set.index[-1],
        line_dash="dot",
        line_color="red",
        annotation_text="Train/Test Split"
    )
    
    fig.update_layout(
        title="Model Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified',
        height=600
    )
    
    return fig

if __name__ == "__main__":
    main()
