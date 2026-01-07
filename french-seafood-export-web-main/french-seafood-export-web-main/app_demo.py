import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forecaster import SeafoodForecaster

# ===========================================================
# 1. CONFIG
# ===========================================================
st.set_page_config(
    page_title="French Seafood Export Forecast",
    page_icon="🐟",
    layout="wide"
)

# ===========================================================
# 2. HELPER FUNCTIONS
# ===========================================================
@st.cache_resource
def load_forecaster():
    model_dir = "artifacts/model"  
    return SeafoodForecaster(model_dir=model_dir)

def plot_hierarchical_series(df_all, unique_id, title):
    df = df_all[df_all['unique_id'] == unique_id].copy()
    df = df.sort_values('ds')

    # Gộp loại dữ liệu
    df['group'] = df['type'].replace({
        'TRAIN': 'HISTORY',
        'TEST_ACTUAL': 'HISTORY',
        'TEST_PRED': 'FORECAST',
        'FUTURE': 'FORECAST'
    })

    fig = go.Figure()

    # --- HISTORY (gray) ---
    df_hist = df[df['group'] == 'HISTORY']
    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=df_hist['ds'], y=df_hist['y'],
            mode='lines',
            name='History',
            line=dict(color='gray', width=2),
            opacity=0.8
        ))

    # --- FORECAST (red) ---
    df_fcst = df[df['group'] == 'FORECAST']
    if not df_fcst.empty:
        # nối forecast từ điểm cuối history để mượt hơn
        last_hist = df_hist.iloc[[-1]] if not df_hist.empty else None
        if last_hist is not None:
            df_fcst = pd.concat([last_hist, df_fcst])

        fig.add_trace(go.Scatter(
            x=df_fcst['ds'], y=df_fcst['y'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volume (kg)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        height=400
    )
    return fig


# ===========================================================
# 3. INITIALIZE
# ===========================================================
try:
    forecaster = load_forecaster()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===========================================================
# 4. SIDEBAR CONTROLS
# ===========================================================
st.sidebar.header("⚙️ Forecast Controls")

# Lấy danh sách tags từ model
# Lưu ý: Cần đảm bảo tags trong file pkl có key 'FAO_Code' và 'AuctionHouse'
species_list = sorted(forecaster.tags.get("FAO_Code", ["Unknown"]))
port_list = sorted(forecaster.tags.get("AuctionHouse", ["Unknown"]))

selected_species = st.sidebar.selectbox("Select Species (Level 1)", species_list)
selected_port = st.sidebar.selectbox("Select Port (Level 2)", port_list)

horizon = st.sidebar.slider("Future Horizon (Weeks)", 4, 24, 12)

# Bottom-level ID cấu thành từ Species và Port
# Lưu ý: Cần check xem cấu trúc unique_id của bạn có đúng là Species/Port không
bottom_id = f"{selected_species}/{selected_port}"

run_btn = st.sidebar.button("🚀 Run Forecast", type="primary")

# ===========================================================
# 5. MAIN DASHBOARD
# ===========================================================
st.title("🇫🇷 French Seafood Export — Hierarchical Forecast")
st.markdown("""
Dashboard này hiển thị kết quả dự báo phân cấp sử dụng **MinTrace Reconciliation**.
*   **Actual Test (Xanh):** Dữ liệu thực tế gần đây (để kiểm chứng).
*   **Validation (Cam):** Mô hình dự đoán lại quá khứ (để xem độ khớp).
*   **Future (Đỏ):** Dự báo tương lai.
""")

if run_btn:
    with st.spinner(f"Computing forecast for {horizon} weeks ahead..."):

        viz_data = forecaster.get_visualization_data(horizon=horizon)
        
        # 2. Convert sang DataFrame chung để dễ lọc
        df_hist = pd.DataFrame(viz_data['history'])
        df_pred = pd.DataFrame(viz_data['prediction'])
        
        # Gộp lại thành 1 bảng to (Master Table)
        df_all = pd.concat([df_hist, df_pred], ignore_index=True)
        
        # Chuyển đổi cột ds sang datetime nếu chưa phải
        df_all['ds'] = pd.to_datetime(df_all['ds'])

    st.success("Analysis complete!")

    # --- LEVEL 1: SPECIES ---
    st.subheader(f"📊 Level 1: {selected_species}")
    fig1 = plot_hierarchical_series(
        df_all, 
        unique_id=selected_species, 
        title=f"Total Forecast for {selected_species}",
        # lookback_weeks=150
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- LEVEL 2: PORT ---
    st.subheader(f"⚓ Level 2: {selected_port}")
    fig2 = plot_hierarchical_series(
        df_all, 
        unique_id=selected_port, 
        title=f"Total Forecast for Port {selected_port}",
        # lookback_weeks=150
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- LEVEL 3: BOTTOM ---
    st.subheader(f"🐟 Level 3: {bottom_id}")
    fig3 = plot_hierarchical_series(
        df_all, 
        unique_id=bottom_id, 
        title=f"Forecast for {bottom_id}",
        # lookback_weeks=150
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # --- DEBUG INFO (Optional - Xóa khi chạy thật) ---
    with st.expander("Debug Raw Data"):
        st.write("Unique IDs found in data:", df_all['unique_id'].unique())
        st.write("Sample Data:", df_all.head())

else:
    st.info("👈 Please select parameters and click **Run Forecast**.")