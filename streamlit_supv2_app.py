import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EQ Predictive Release Lead Time", layout="wide")

DATA_FILE = Path(__file__).with_name(
    "Expanded_Synthetic_Historical_Batch_Data_Supplier_Mapping_Updated.xlsx"
)
RANDOM_STATE = 42

FEATURES = [
    "Material", "Supplier Number", "CMO Site", "QC Backlog at Start",
    "Recent OOS Count - Prior 90 Days", "QC Sample Count", "Testing Type",
    "Complexity", "Start Month"
]
CAT = ["Material", "Supplier Number", "CMO Site", "Testing Type", "Complexity", "Start Month"]
NUM = ["QC Backlog at Start", "Recent OOS Count - Prior 90 Days", "QC Sample Count"]

st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1450px}.snap{background:#06466e;border-radius:12px;padding:26px 22px;color:white;min-height:470px}
.eyebrow{color:#62c8dc;font-weight:800;letter-spacing:2px;font-size:13px}.batchid{font-size:36px;font-weight:800;margin:12px 0 4px}.sub{font-size:15px}.rule{border-top:1px solid #5f88a1;margin:16px 0}
.kv{display:flex;justify-content:space-between;gap:10px;margin:16px 0;color:#d3e4ef}.kv b{color:white;text-align:right}.risk{border-radius:6px;padding:17px 10px;margin:22px 0;font-weight:800;text-align:center;color:white}
.driverhead{color:#06466e;font-weight:800;letter-spacing:2px;font-size:13px}.impact-note{background:#eaf3f7;color:#06466e;font-weight:700;text-align:center;padding:9px;border-radius:4px}
</style>
""", unsafe_allow_html=True)


def as_date(series):
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.between(1, 100000) & parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            numeric.loc[mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    return parsed


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Place {DATA_FILE.name} in the same folder as this app.")
    df = pd.read_excel(
        DATA_FILE, sheet_name="Historical Batch Data", header=3, engine="openpyxl"
    )
    required = [
        "Batch ID", "Material", "Supplier Number", "Supplier Name", "CMO Site",
        "Prediction / Start Date", "QC Backlog at Start", "Planned Release Date",
        "Actual Release Date", "Recent OOS Count - Prior 90 Days",
        "QC Sample Count", "Testing Type / Complexity"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing workbook columns: {missing}")

    for c in ["Prediction / Start Date", "Planned Release Date", "Actual Release Date"]:
        df[c] = as_date(df[c])
    for c in ["Batch ID", "Material", "CMO Site", "Supplier Name"]:
        df[c] = df[c].astype(str).str.strip()
    df["Supplier Number"] = pd.to_numeric(df["Supplier Number"], errors="coerce").astype("Int64").astype(str)
    for c in ["QC Backlog at Start", "Recent OOS Count - Prior 90 Days", "QC Sample Count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    split = df["Testing Type / Complexity"].astype(str).str.split("/", n=1, expand=True)
    df["Testing Type"] = split[0].str.strip()
    df["Complexity"] = split[1].str.strip() if split.shape[1] > 1 else "Unknown"
    df["Start Month"] = df["Prediction / Start Date"].dt.month.astype("Int64").astype(str)
    df["Release Lead Time (Days)"] = (
        df["Actual Release Date"] - df["Prediction / Start Date"]
    ).dt.days
    df = df.dropna(subset=["Prediction / Start Date", "Actual Release Date", "Release Lead Time (Days)"])
    return df.sort_values("Prediction / Start Date").reset_index(drop=True)


@st.cache_resource
def train_model(df):
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut], df.iloc[cut:]
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), CAT),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUM)
    ])
    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    model.fit(train[FEATURES], train["Release Lead Time (Days)"])
    holdout_pred = model.predict(test[FEATURES])
    metrics = {
        "MAE": mean_absolute_error(test["Release Lead Time (Days)"], holdout_pred),
        "RMSE": mean_squared_error(test["Release Lead Time (Days)"], holdout_pred) ** 0.5,
        "R2": r2_score(test["Release Lead Time (Days)"], holdout_pred)
    }
    return model, metrics


def driver_impacts(model, row):
    transformed = model.named_steps["prep"].transform(row[FEATURES])
    names = model.named_steps["prep"].get_feature_names_out()
    values = shap.TreeExplainer(model.named_steps["model"])(transformed).values[0]
    grouped = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, values):
        clean = name.split("__", 1)[-1]
        for feature in FEATURES:
            if clean == feature or clean.startswith(feature + "_"):
                grouped[feature] += float(value)
                break
    friendly = {
        "Material":"Material history", "Supplier Number":"Supplier history",
        "CMO Site":"CMO site history", "QC Backlog at Start":"QC backlog",
        "Recent OOS Count - Prior 90 Days":"Recent OOS history",
        "QC Sample Count":"QC sample count", "Testing Type":"Testing type",
        "Complexity":"Testing complexity", "Start Month":"Start-month pattern"
    }
    return pd.Series({friendly[k]: v for k, v in grouped.items()})


def all_batch_chart(chart_df):
    plot_df = chart_df.sort_values(["Predicted Release Date", "Batch ID"]).reset_index(drop=True)
    hover = (
        "<b>Batch:</b> " + plot_df["Batch ID"] +
        "<br><b>Supplier:</b> " + plot_df["Supplier Number"] + " | " + plot_df["Supplier Name"] +
        "<br><b>Material:</b> " + plot_df["Material"] +
        "<br><b>CMO site:</b> " + plot_df["CMO Site"] +
        "<br><b>Predicted release:</b> " + plot_df["Predicted Release Date"].dt.strftime("%d %b %Y") +
        "<br><b>Planned release:</b> " + plot_df["Planned Release Date"].dt.strftime("%d %b %Y") +
        "<br><b>Predicted lead time:</b> " + plot_df["Predicted Lead Time (Days)"].round(1).astype(str) + " days"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Batch ID"], y=plot_df["Predicted Release Date"],
        mode="lines+markers", name="Predicted release date",
        line=dict(color="#A6006D", width=1.5), marker=dict(size=5, color="#D4147A"),
        text=hover, hovertemplate="%{text}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Batch ID"], y=plot_df["Planned Release Date"],
        mode="lines", name="Planned release date",
        line=dict(color="#5577AA", width=1, dash="dot"),
        hovertemplate="Batch %{x}<br>Planned %{y|%d %b %Y}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Predicted Release-Date Spikes Across All {len(plot_df):,} Batches",
        xaxis_title="Batch ID", yaxis_title="Release date", hovermode="closest",
        height=650, margin=dict(l=40, r=25, t=70, b=170),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(
            type="category", tickmode="array",
            tickvals=plot_df["Batch ID"].tolist(),
            ticktext=plot_df["Batch ID"].tolist(),
            tickangle=-90, tickfont=dict(size=7),
            rangeslider=dict(visible=True, thickness=0.08)
        ),
        yaxis=dict(tickformat="%d %b %Y"),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig


def render_selected_batch(row, impacts, mae):
    target = row["Planned Release Date"]
    predicted = row["Predicted Release Date"]
    delta = int((predicted - target).days)
    if delta >= 5:
        risk, color = "HIGH RISK", "#dc0073"
    elif delta >= 2:
        risk, color = "MEDIUM RISK", "#f59e0b"
    else:
        risk, color = "LOW RISK", "#16866b"

    left, right = st.columns([0.40, 1], gap="medium")
    with left:
        st.markdown(f"""
        <div class="snap"><div class="eyebrow">BATCH SNAPSHOT</div>
        <div class="batchid">{row['Batch ID']}</div>
        <div class="sub">{row['CMO Site']} | {row['Material']}</div><div class="rule"></div>
        <div class="kv"><span>Supplier</span><b>{row['Supplier Number']}<br>{row['Supplier Name']}</b></div>
        <div class="kv"><span>Target release</span><b>{target:%d %b %Y}</b></div>
        <div class="kv"><span>Predicted release</span><b>{predicted:%d %b %Y}</b></div>
        <div class="risk" style="background:{color}">{delta:+d} DAYS | {risk}</div>
        <div class="kv"><span>Predicted lead time</span><b>{row['Predicted Lead Time (Days)']:.1f} days</b></div>
        <div class="kv"><span>Model MAE</span><b>± {mae:.1f} days</b></div></div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="driverhead">DRIVER VIEW</div>', unsafe_allow_html=True)
        impacts = impacts.reindex(impacts.abs().sort_values(ascending=False).index).head(7)
        p = impacts.iloc[::-1]
        max_abs = max(1.0, float(p.abs().max()))
        fig, ax = plt.subplots(figsize=(8.8, 4.5))
        colors = ["#319474" if v < 0 else "#CB3E55" for v in p]
        ax.barh(p.index, p.values, color=colors)
        ax.axvline(0, color="#999", lw=1)
        ax.set_xlim(-max_abs * 1.55, max_abs * 1.55)
        for i, value in enumerate(p.values):
            ax.text(value + (0.05 * max_abs if value >= 0 else -0.05 * max_abs), i,
                    f"{value:+.1f} days", va="center",
                    ha="left" if value >= 0 else "right", fontweight="bold", fontsize=9)
        ax.set_xlabel("SHAP impact on predicted lead time (days)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        plt.close(fig)
        st.markdown(
            f'<div class="impact-note">Displayed-driver net impact: {impacts.sum():+.1f} days versus model baseline</div>',
            unsafe_allow_html=True
        )


try:
    data = load_data()
    model, metrics = train_model(data)
except Exception as exc:
    st.error(f"Unable to initialize app: {exc}")
    st.stop()

data = data.copy()
data["Predicted Lead Time (Days)"] = model.predict(data[FEATURES])
data["Predicted Release Date"] = data["Prediction / Start Date"] + pd.to_timedelta(
    np.rint(data["Predicted Lead Time (Days)"]).astype(int), unit="D"
)
data["Predicted Delay vs Plan (Days)"] = (
    data["Predicted Release Date"] - data["Planned Release Date"]
).dt.days

st.title("EQ Predictive Release Lead Time")
st.caption("All-batch release-date outlook with supplier filtering and batch-level SHAP explainability")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Batches", f"{len(data):,}")
m2.metric("Holdout MAE", f"{metrics['MAE']:.1f} days")
m3.metric("Holdout RMSE", f"{metrics['RMSE']:.1f} days")
m4.metric("Holdout R²", f"{metrics['R2']:.2f}")

st.subheader("All-batch predicted release-date line chart")
chart_scope = st.radio(
    "Chart scope", ["All batches", "Selected supplier"], horizontal=True,
    help="All batches displays every Batch ID. Selected supplier makes dense labels easier to inspect."
)

supplier_table = data[["Supplier Number", "Supplier Name"]].drop_duplicates().sort_values(["Supplier Name", "Supplier Number"])
supplier_options = ["All Suppliers"] + [f"{r['Supplier Number']} | {r['Supplier Name']}" for _, r in supplier_table.iterrows()]
filters1, filters2 = st.columns(2)
with filters1:
    supplier_choice = st.selectbox("Supplier ID | Supplier Name", supplier_options)

supplier_df = data if supplier_choice == "All Suppliers" else data[data["Supplier Number"] == supplier_choice.split(" | ", 1)[0]]
chart_df = data if chart_scope == "All batches" else supplier_df
st.plotly_chart(all_batch_chart(chart_df), use_container_width=True)
st.info("Every Batch ID is included on the x-axis. Because 600 labels are dense, use the range slider below the chart or drag to zoom. Hover over any spike to see the Batch ID, supplier, material, planned date and predicted date.")

with st.expander("View and download all predicted batch release dates"):
    display_cols = [
        "Batch ID", "Supplier Number", "Supplier Name", "Material", "CMO Site",
        "Prediction / Start Date", "Planned Release Date", "Predicted Release Date",
        "Predicted Lead Time (Days)", "Predicted Delay vs Plan (Days)"
    ]
    full_output = chart_df[display_cols].sort_values("Predicted Release Date")
    st.dataframe(full_output, hide_index=True, use_container_width=True, height=420)
    st.download_button(
        "Download predicted release dates (CSV)",
        full_output.to_csv(index=False).encode("utf-8"),
        "EQ_All_Batch_Predicted_Release_Dates.csv", "text/csv"
    )

st.divider()
st.subheader("Selected batch prediction and driver explanation")
batch_options = supplier_df["Batch ID"].drop_duplicates().sort_values().tolist()
with filters2:
    selected_batch = st.selectbox("Batch ID", batch_options)
selected = supplier_df[supplier_df["Batch ID"] == selected_batch].iloc[[0]]
render_selected_batch(selected.iloc[0], driver_impacts(model, selected), metrics["MAE"])

with st.expander("Point-in-time model inputs"):
    st.dataframe(selected[["Batch ID", "Supplier Number", "Supplier Name"] + FEATURES], hide_index=True, use_container_width=True)
