# --- Import Libraries ---
import streamlit as st 
import pandas as pd
import numpy as np 
import pickle 
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime, timedelta

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Analisis Penjualan",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Fungsi untuk Load Data Dummy --
@st.cache_data 
def load_data():
    return pd.read_csv("data/data_dummy_retail_store.csv")

df_sales = load_data()
df_sales['Tanggal_Pesanan'] = pd.to_datetime(df_sales['Tanggal_Pesanan'])

# -- Fungsi untuk Load Model --
@st.cache_resource
def load_model():
    with open("models/model_sales.pkl", "rb") as f:
        sales_prediction_model, model_features, base_month_ordinal = pickle.load(f)
    return sales_prediction_model, model_features, base_month_ordinal

sales_prediction_model, model_features, base_month_ordinal = load_model()
# -- Judul dan deskripsi dashboard --
st.title("Dashboard Penjualan Toko Online")
st.markdown("Dashboard interaktif ini berisi gambaran **performa penjualan, tren, distribusi, dan fitur prediksi sederhana**")

st.markdown("---")

# -- Sidebar Halaman--
st.sidebar.header("Pengaturan & Navigasi")

pilihan_halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ("Overview Dashboard", "Prediksi Penjualan")
)
# Filter untuk Halamman Overview Dashboard
if pilihan_halaman == "Overview Dashboard":
    st.sidebar.markdown("### Filter Data Dashboard")

    # Filter tanggal
    min_date = df_sales['Tanggal_Pesanan'].min().date()
    max_date = df_sales['Tanggal_Pesanan'].max().date()

    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date_filter = pd.to_datetime(date_range[0])
        end_date_filter = pd.to_datetime(date_range[1])
        filtered_df = df_sales[(df_sales['Tanggal_Pesanan'] >= start_date_filter) & (df_sales['Tanggal_Pesanan'] <= end_date_filter)]
    else:
        filtered_df = df_sales

# -- Filter wilayah --
    selected_regions = st.sidebar.multiselect(
        "Pilih Wilayah:",
        options=df_sales['Wilayah'].unique().tolist(),
        default=df_sales['Wilayah'].unique().tolist()
    )  
    filtered_df = filtered_df[filtered_df['Wilayah'].isin(selected_regions)]

    # Filter kategori produk
    selected_categories = st.sidebar.multiselect(
        "Pilih Kategori Produk",
        options=df_sales['Kategori'].unique().tolist(),
        default=df_sales['Kategori'].unique().tolist()
    )
    filtered_df = filtered_df[filtered_df['Kategori'].isin(selected_categories)]

else:  # untuk halaman prediksi, pakai df_sales.copy() atau tidak ada filter
    filtered_df = df_sales.copy()
if pilihan_halaman == "Overview Dashboard":
    # Metrics utama
    st.subheader("Ringkasan Performa Penjualan")

    col1, col2, col3, col4 = st.columns(4)

    # agregat metrics
    total_sales = filtered_df['Total_Penjualan'].sum()
    total_orders = filtered_df['OrderID'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    total_products_sold = filtered_df['Jumlah'].sum()

    with col1:
        st.metric(label="Total Penjualan", value=f"Rp {total_sales:,.2f}")
    with col2:
        st.metric(label="Jumlah Pesanan", value=f"Rp {total_orders:,.2f}")
    with col3:
        st.metric(label="Avg. Order Value", value=f"Rp {avg_order_value:,.2f}")
    with col4:
        st.metric(label="Jumlah Produk Terjual", value=f"Rp {total_products_sold:,.2f}")
# Tren Penjualan
    st.subheader("Tren Penjualan Bulanan")
    
    # agregat sales per bulan
    sales_by_month = filtered_df.groupby('Bulan')['Total_Penjualan'].sum().reset_index()
    
    sales_by_month['Bulan'] = pd.to_datetime(sales_by_month['Bulan']).dt.to_period('M')
    sales_by_month = sales_by_month.sort_values('Bulan')
    sales_by_month['Bulan'] = sales_by_month['Bulan'].astype(str) 

    fig_monthly_sales = px.line(
        sales_by_month,
        x='Bulan',
        y='Total_Penjualan',
        title='Total Penjualan per Bulan',
        markers=True,
        hover_name='Bulan',
        height=400
    )

    st.plotly_chart(fig_monthly_sales, use_container_width=True)

# Penjualan & Produk Terlaris
    st.subheader("Top Product & Distribusi Penjualan")

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.write("#### Top 10 Products")

        # agregat
        top_products_sale = filtered_df.groupby(['Produk'])['Total_Penjualan'].sum().nlargest(10).reset_index()

        fig_top_products = px.bar(
            top_products_sale,
            x='Total_Penjualan',
            y='Produk',
            orientation='h',
            title='Top 10 Produk Berdasarkan Total Penjualan',
            color='Total_Penjualan',
            color_continuous_scale=px.colors.sequential.Plasma[::-1], # gradasi warna,
            height=400
        )

        fig_top_products.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_products, use_container_width=True)
with col_vis2:
        st.write("#### Distribusi Penjualan per Kategori")

        # Agregasi total penjualan berdasarkan kategori
        sales_by_category = filtered_df.groupby('Kategori')['Total_Penjualan'].sum().reset_index()

        # Buat pie chart (donut style) berdasarkan proporsi penjualan tiap kategori
        fig_category_pie = px.pie(
            sales_by_category,
            values='Total_Penjualan',    # Nilai yang diplot (besarannya)
            names='Kategori',            # Label di pie chart
            title='Proporsi Penjualan per Kategori',
            hole=0.3,                    # Membuat pie menjadi donut chart (ada lubangnya)
            color_discrete_sequence=px.colors.qualitative.Set2  # Skema warna yang friendly
        )

        # Tampilkan chart di Streamlit
st.plotly_chart(fig_category_pie, use_container_width=True)






#numpang sementara
# --- Import Libraries ---
import streamlit as st 
import pandas as pd
import numpy as np 
import pickle 
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime, timedelta

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Predict Customer Satisfaction",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data 
def load_data():
    return pd.read_csv("data/df_no_outliers.csv")  # Perbaiki backslash jadi slash

df = load_data()
st.write("✅ File data berhasil dimuat")
st.write(df.shape)

# --- Tampilkan Data ---
st.title("Customer Satisfaction Dashboard")

st.markdown("### Contoh Data")
st.dataframe(df.head())

st.markdown("### Distribusi Satisfaction Level")
fig = px.histogram(df, x='Satisfaction_Level', color='Satisfaction_Level')
st.plotly_chart(fig, use_container_width=True)

# --- Load Models ---
@st.cache_resource
def load_all_models():
    models = {}
    with open("model/best_KNN_model.pkl", "rb") as f:
        models["KNN"] = pickle.load(f)
    with open("model/best_Decusion_Tree_model.pkl", "rb") as f:
        models["Decision Tree"] = pickle.load(f)
    with open("model/best_random_forest_model.pkl", "rb") as f:
        models["Random Forest"] = pickle.load(f)
    return models

model_dict = load_all_models()
model_choice = st.sidebar.selectbox("Pilih Model:", list(model_dict.keys()))
selected_model = model_dict[model_choice]
st.write(f"Model yang digunakan: **{model_choice}**")

# --- Form Input User ---
st.markdown("### Input Data untuk Prediksi")

with st.form("form_predict"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 70, 30)
        income = st.number_input("Income", min_value=0, value=50000)
        product_quality = st.selectbox("Product Quality", [1, 2, 3, 4, 5])
        service_quality = st.selectbox("Service Quality", [1, 2, 3, 4, 5])
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        purchase_frequency = st.slider("Purchase Frequency", 0, 100, 10)
        feedback_score = st.selectbox("Feedback Score", ['Low', 'Medium', 'High'])
        loyalty_level = st.selectbox("Loyalty Level", ['Bronze', 'Silver', 'Gold'])

    submitted = st.form_submit_button("Predict Satisfaction Level")

    if submitted:
        # Mapping kategori ke angka sesuai label encoding
        gender_mapping = {"Male": 1, "Female": 0}
        feedback_mapping = {"Low": 0, "Medium": 1, "High": 2}
        loyalty_mapping = {"Bronze": 0, "Silver": 1, "Gold": 2}

        # Buat dataframe input sesuai format model
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender_mapping[gender],
            "Income": income,
            "ProductQuality": product_quality,
            "ServiceQuality": service_quality,
            "PurchaseFrequency": purchase_frequency,
            "FeedbackScore": feedback_mapping[feedback_score],
            "LoyaltyLevel": loyalty_mapping[loyalty_level]
        }])

        # Prediksi
        prediction = selected_model.predict(input_df)[0]
        st.success(f"✅ Prediksi Satisfaction Level: **{prediction}**")
