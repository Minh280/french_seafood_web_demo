# 🇫🇷 French Seafood Export — Hierarchical Forecast Dashboard

A Streamlit web application for forecasting French seafood export volumes using hierarchical time series forecasting with MinTrace reconciliation.

## 📋 Features

- **Hierarchical Forecasting**: Multi-level forecasting (Species → Port → Species/Port combinations)
- **MinTrace Reconciliation**: Ensures forecast coherence across hierarchy levels
- **Interactive Dashboard**: Select species, ports, and forecast horizons dynamically
- **Visual Analytics**: Interactive Plotly charts showing historical data and forecasts
- **Ensemble Models**: Combines Holt-Winters and ARIMA models for robust predictions

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

## 📦 Installation

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/huynvq02/french-seafood-export-web.git
cd french-seafood-export-web
```

Or download and extract the ZIP file, then navigate to the project directory.

### Step 2: Create a Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies

First, update the `requirements.txt` file with all necessary packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter any issues, install packages individually:

```bash
pip install streamlit==1.51.0
pip install pandas
pip install numpy
pip install plotly
pip install joblib
pip install scikit-learn
pip install statsforecast
pip install hierarchicalforecast
```

### Step 4: Verify Installation

Check that Streamlit is installed correctly:

```bash
streamlit --version
```

You should see output like: `Streamlit, version 1.51.0`

## 🚀 Running the Application

### Step 1: Ensure Model Artifacts are Present

Make sure the `artifacts/model/` directory contains the following files:
- `fcst_model.pkl`
- `S_df.pkl`
- `tags.pkl`
- `Y_fitted_ensemble.pkl`
- `Y_full.pkl`

### Step 2: Start the Streamlit Server

Run the following command from the project root directory:

```bash
streamlit run app_demo.py
```

### Step 3: Access the Dashboard

The application will automatically open in your default web browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

## 📖 Using the Dashboard

1. **Select Species (Level 1)**: Choose a FAO species code from the dropdown
2. **Select Port (Level 2)**: Choose an auction house/port from the dropdown
3. **Set Forecast Horizon**: Use the slider to select how many weeks ahead to forecast (4-24 weeks)
4. **Run Forecast**: Click the "🚀 Run Forecast" button to generate predictions
5. **View Results**: Explore the three hierarchical levels:
   - **Level 1**: Total forecast for the selected species
   - **Level 2**: Total forecast for the selected port
   - **Level 3**: Forecast for the specific species/port combination

## 🎨 Chart Legend

- **Gray Line (History)**: Historical actual data
- **Red Line (Forecast)**: Future predictions with markers

## 🔧 Troubleshooting

### Port Already in Use
If port 8501 is already in use, specify a different port:
```bash
streamlit run app_demo.py --server.port 8502
```

### Module Not Found Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Loading Errors
Verify that all `.pkl` files exist in `artifacts/model/` directory and are not corrupted.

### Performance Issues
For faster loading, the model uses JIT compilation warming. First run may take longer.

## 📁 Project Structure

```
.
├── app_demo.py              # Main Streamlit application
├── forecaster.py            # SeafoodForecaster class
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── artifacts/
│   └── model/              # Trained model artifacts
│       ├── fcst_model.pkl
│       ├── S_df.pkl
│       ├── tags.pkl
│       ├── Y_fitted_ensemble.pkl
│       └── Y_full.pkl
└── README.md               # This file
```

## 🔄 Stopping the Application

Press `Ctrl + C` in the terminal where Streamlit is running to stop the server.

## 📝 Notes

- The application caches the forecaster model for better performance
- Forecasts use an ensemble of Holt-Winters (60%) and ARIMA (40%) models
- All negative predictions are clipped to zero
- The dashboard includes debug information in an expandable section

## 🤝 Support

For issues or questions, please open an issue on the GitHub repository.

## 📄 License

This project is part of a thesis on French seafood export forecasting.
