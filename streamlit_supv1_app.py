import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EQ Predictive Release Lead Time", layout="wide")

DATA_FILE = Path(__file__).with_name("Expanded_Synthetic_Historical_Batch_Data_Supplier_Mapping_Updated.xlsx")
RANDOM_STATE = 42

FEATURES = [
    "Material", "Supplier Number", "CMO Site", "QC Backlog at Start",
    "Prior 90-Day OOS Count", "QC Sample Count", "Testing Type",
    "Complexity",
]
CAT_FEATURES = ["Material", "Supplier Number", "CMO Site", "Testing Type", "Complexity"]
NUM_FEATURES = ["QC Backlog at Start", "Prior 90-Day OOS Count", "QC Sample Count"]

st.markdown("""
<style>
.block-container {padding-top:1.4rem; max-width:1250px;}
.snap {background:#06466e;border-radius:10px;padding:28px 24px;color:white;min-height:500px;}
.eyebrow {color:#62c8dc;font-weight:800;letter-spacing:2px;font-size:14px;}
.batchid {font-size:38px;font-weight:800;margin:13px 0 4px;}
.sub {font-size:16px;margin-bottom:16px;}.rule {border-top:1px solid #5f88a1;margin:17px 0;}
.kv {display:flex;justify-content:space-between;gap:12px;margin:17px 0;color:#d3e4ef;}
.kv b {color:white;font-size:16px;text-align:right;}.risk {border-radius:6px;padding:18px 12px;margin:26px 0 23px;font-weight:800;text-align:center;color:white;}
.driverhead {color:#06466e;font-weight:800;letter-spacing:2px;font-size:14px;}.driver-title {font-size:18px;font-weight:700;color:#2d2030;margin:8px 0 2px;}
.impact-note {background:#eaf3f7;color:#06466e;font-weight:700;text-align:center;padding:10px;border-radius:4px;margin-top:5px;}
.small-note {color:#667784;font-size:12px;}
</style>
""", unsafe_allow_html=True)


def to_date(series):
    out = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.between(1, 100000) & out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(numeric.loc[mask], unit="D", origin="1899-12-30", errors="coerce")
    return out


@st.cache_data
def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path.name}")

    batches = pd.read_excel(path, sheet_name="Historical Batch Data", header=3, engine="openpyxl")
    oos = pd.read_excel(path, sheet_name="OOS Events", header=0, engine="openpyxl")

    for c in ["Prediction / Start Date", "Planned Release Date", "Actual Release Date"]:
        batches[c] = to_date(batches[c])
    for c in ["OOS Event Date", "Closure Date"]:
        oos[c] = to_date(oos[c])

    batches["Batch ID"] = batches["Batch ID"].astype(str).str.strip()
    batches["Supplier Number"] = batches["Supplier Number"].astype("Int64").astype(str)
    batches["Supplier Name"] = batches["Supplier Name"].astype(str).str.strip()
    batches["Material"] = batches["Material"].astype(str).str.strip()
    batches["CMO Site"] = batches["CMO Site"].astype(str).str.strip()

    oos["Material"] = oos["Material"].astype(str).str.strip()
    oos["CMO Site"] = oos["CMO Site"].astype(str).str.strip()

    # Leakage-safe OOS history: same material/site, D-90 through D-1.
    def prior_90(row):
        start = row["Prediction / Start Date"]
        if pd.isna(start):
            return 0
        mask = (
            (oos["Material"] == row["Material"])
            & (oos["CMO Site"] == row["CMO Site"])
            & (oos["OOS Event Date"] >= start - pd.Timedelta(days=90))
            & (oos["OOS Event Date"] < start)
        )
        return int(mask.sum())

    batches["Prior 90-Day OOS Count"] = batches.apply(prior_90, axis=1)
    batches["Release Lead Time (Days)"] = (batches["Actual Release Date"] - batches["Prediction / Start Date"]).dt.days

    parts = batches["Testing Type / Complexity"].astype(str).str.split("/", n=1, expand=True)
    batches["Testing Type"] = parts[0].str.strip()
    batches["Complexity"] = parts[1].str.strip() if parts.shape[1] > 1 else "Unknown"

    batches = batches.dropna(subset=["Prediction / Start Date", "Actual Release Date", "Release Lead Time (Days)"]).copy()
    return batches.sort_values("Prediction / Start Date").reset_index(drop=True)


@st.cache_resource
def train_model(df):
    cut = int(len(df) * 0.80)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUM_FEATURES),
    ])

    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE)),
    ])
    pipe.fit(train[FEATURES], train["Release Lead Time (Days)"])
    test_pred = pipe.predict(test[FEATURES])
    mae = float(mean_absolute_error(test["Release Lead Time (Days)"], test_pred))
    return pipe, mae


def shap_business_drivers(model, selected_row):
    x = model.named_steps["prep"].transform(selected_row[FEATURES])
    names = model.named_steps["prep"].get_feature_names_out()
    explanation = shap.TreeExplainer(model.named_steps["model"])(x)
    values = explanation.values[0]

    impacts = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, values):
        clean = name.split("__", 1)[-1]
        matched = None
        for feature in FEATURES:
            if clean == feature or clean.startswith(feature + "_"):
                matched = feature
                break
        if matched:
            impacts[matched] += float(value)

    friendly = {
        "QC Backlog at Start": "Site QC backlog",
        "Prior 90-Day OOS Count": "Recent OOS history",
        "Complexity": "Test complexity",
        "Testing Type": "Testing type",
        "Material": "Material / product history",
        "Supplier Number": "Supplier history",
        "CMO Site": "CMO site history",
        "QC Sample Count": "QC sample count",
    }
    return {friendly.get(k, k): v for k, v in impacts.items()}


def risk_details(delta_days):
    if delta_days >= 5:
        return "HIGH RISK", "#dc0073"
    if delta_days >= 2:
        return "MEDIUM RISK", "#f59e0b"
    return "LOW RISK", "#16866b"


def render_view(row, impacts, mae):
    target = pd.to_datetime(row["Planned Release Date"])
    predicted = pd.to_datetime(row["Predicted Release Date"])
    delta = int((predicted - target).days)
    risk_label, risk_color = risk_details(delta)

    c1, c2 = st.columns([0.40, 1.0], gap="medium")
    with c1:
        st.markdown(f"""
        <div class="snap">
          <div class="eyebrow">BATCH SNAPSHOT</div>
          <div class="batchid">{row['Batch ID']}</div>
          <div class="sub">{row['CMO Site']} &nbsp;|&nbsp; {row['Material']}</div>
          <div class="rule"></div>
          <div class="kv"><span>Supplier</span><b>{row['Supplier Number']}<br>{row['Supplier Name']}</b></div>
          <div class="kv"><span>Target release</span><b>{target.strftime('%d %b %Y')}</b></div>
          <div class="kv"><span>Predicted release</span><b>{predicted.strftime('%d %b %Y')}</b></div>
          <div class="risk" style="background:{risk_color}">{delta:+d} DAYS &nbsp;|&nbsp; {risk_label}</div>
          <div class="kv"><span>Predicted lead time</span><b>{row['Predicted Lead Time (Days)']:.1f} days</b></div>
          <div class="kv"><span>Model MAE</span><b>± {mae:.1f} days</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="driverhead">DRIVER VIEW</div><div class="driver-title">Impact on predicted release lead time</div>', unsafe_allow_html=True)
        s = pd.Series(impacts, dtype=float).dropna()
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(6)
        plot_s = s.iloc[::-1]
        max_abs = max(1.0, float(plot_s.abs().max()))
        colors = ["#319474" if v < 0 else "#cb3e55" for v in plot_s.values]
        if len(plot_s):
            strongest = s.abs().idxmax()
            pos = list(plot_s.index).index(strongest)
            colors[pos] = "#e00073" if s[strongest] >= 0 else "#16866b"

        fig, ax = plt.subplots(figsize=(8.7, 4.5), facecolor="white")
        y = np.arange(len(plot_s))
        ax.barh(y, plot_s.values, color=colors, height=0.55)
        ax.axvline(0, color="#9aa5ad", linewidth=1.2)
        ax.set_xlim(-max_abs * 1.65, max_abs * 1.65)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_s.index, fontsize=10, color="#2d2030")
        for i, value in enumerate(plot_s.values):
            offset = max_abs * 0.05
            ax.text(value + (offset if value >= 0 else -offset), i, f"{value:+.1f} days",
                    va="center", ha="left" if value >= 0 else "right", fontsize=9,
                    fontweight="bold", color="#cb3e55" if value >= 0 else "#25876d")
        ax.text(0.02, 1.03, "Earlier release", transform=ax.transAxes, color="#25876d", fontsize=9)
        ax.text(0.98, 1.03, "Later release", transform=ax.transAxes, color="#cb3e55", fontsize=9, ha="right")
        ax.set_xlabel("SHAP impact on predicted lead time (days)", fontsize=9)
        ax.set_ylabel("")
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        fig.subplots_adjust(left=0.31, right=0.95, top=0.87, bottom=0.17)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        plt.close(fig)

        st.markdown(
            f'<div class="impact-note">Net displayed-driver impact: approximately {float(s.sum()):+.1f} days versus the model baseline</div>',
            unsafe_allow_html=True,
        )
        st.caption("Positive SHAP values push the prediction later; negative values pull it earlier. SHAP explains model behavior, not causality.")


try:
    data = load_data(DATA_FILE)
    model, MODEL_MAE_DAYS = train_model(data)
except Exception as exc:
    st.error(f"Unable to initialize the app: {exc}")
    st.stop()

# Generate predictions for all workbook rows from the trained model.
data = data.copy()
data["Predicted Lead Time (Days)"] = model.predict(data[FEATURES])
data["Predicted Release Date"] = data["Prediction / Start Date"] + pd.to_timedelta(
    np.rint(data["Predicted Lead Time (Days)"]).astype(int), unit="D"
)

st.title("EQ Predictive Release Lead Time")
st.caption("Interactive prediction and SHAP explainability using the supplied historical batch workbook")

supplier_lookup = (
    data[["Supplier Number", "Supplier Name"]]
    .drop_duplicates()
    .sort_values(["Supplier Name", "Supplier Number"])
    .reset_index(drop=True)
)
supplier_options = ["All Suppliers"] + [
    f"{r['Supplier Number']} | {r['Supplier Name']}" for _, r in supplier_lookup.iterrows()
]

f1, f2 = st.columns(2)
with f1:
    supplier_choice = st.selectbox("Select Supplier ID | Supplier Name", supplier_options)

if supplier_choice == "All Suppliers":
    filtered = data.copy()
else:
    supplier_id = supplier_choice.split(" | ", 1)[0]
    filtered = data[data["Supplier Number"] == supplier_id].copy()

batch_options = filtered["Batch ID"].drop_duplicates().sort_values().tolist()
with f2:
    batch_choice = st.selectbox("Select Batch ID", batch_options)

selected = filtered[filtered["Batch ID"] == batch_choice].iloc[[0]].copy()
impacts = shap_business_drivers(model, selected)
render_view(selected.iloc[0], impacts, MODEL_MAE_DAYS)

with st.expander("Point-in-time model inputs for this batch"):
    cols = ["Batch ID", "Supplier Number", "Supplier Name", "Material", "CMO Site", "Prediction / Start Date"] + FEATURES[3:]
    st.dataframe(selected[cols], hide_index=True, use_container_width=True)

with st.expander("Prototype notes"):
    st.write("The model uses a chronological 80/20 split. Actual Release Date and current-batch OOS outcomes are not predictors. Prior 90-day OOS history is rebuilt from event dates before the prediction date.")
