# Dataco Smart Supply Chain – ML Analysis

An end-to-end analysis and machine-learning project built on the **DataCo Supply
Chain Dataset**, exploring supply chain performance and predicting **Late
Delivery Risk** using a Random Forest classifier. Includes an interactive
Streamlit dashboard.

## 📁 Project Structure

```
Dataco-Smart-Supply-Chain-ML-Analysis/
│
├── app.py                              # Streamlit dashboard (EDA + ML prediction)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation (this file)
├── Dataco-Smart-Supply-Chain.ipynb     # Jupyter notebook with full analysis
└── DataCoSupplyChainDataset.csv        # Raw dataset
```

## 📊 About the Dataset

The DataCo Supply Chain Dataset contains order-level records including
customer, shipping, product, and sales information, along with a
`Late_delivery_risk` flag indicating whether an order was delivered late.

## 🚀 Getting Started

### 1. Clone / download the project
Make sure all files above are in the same folder, including the dataset CSV.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter notebook (optional, for full EDA & model development)
```bash
jupyter notebook Dataco-Smart-Supply-Chain.ipynb
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## 🧭 Dashboard Sections

| Section | Description |
|---|---|
| **Overview** | Dataset summary, key metrics, missing-value report |
| **Exploratory Analysis** | Sales by region/category, late delivery rate by shipping mode, correlation heatmap |
| **Late Delivery Prediction** | Trains a Random Forest model live and shows accuracy, precision, recall, F1, confusion matrix, and feature importance |

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** – data processing
- **Scikit-learn** – machine learning (Random Forest)
- **Matplotlib / Seaborn / Plotly** – visualization
- **Streamlit** – interactive web dashboard

## 📈 Model

The prediction module uses a `RandomForestClassifier` trained on features such
as shipping mode, order region, product category, sales value, and shipping
duration to predict late delivery risk. Model hyperparameters (test size,
number of trees) are adjustable directly from the dashboard sidebar/sliders.

## 📌 Notes

- If your dataset has different column names than expected, `app.py`
  automatically searches for common alternatives (case-insensitive), but you
  may need to update `get_column()` calls for a fully custom dataset.
- Large datasets are cached with `@st.cache_data` for faster reloads.

## 📄 License

This project is for educational/academic purposes.
