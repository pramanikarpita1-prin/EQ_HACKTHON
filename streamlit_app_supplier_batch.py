import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EQ Predictive Release Lead Time", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1220px;}
.snap {background:#06466e;border-radius:10px;padding:30px 24px;color:white;min-height:500px;}
.eyebrow {color:#62c8dc;font-weight:800;letter-spacing:2px;font-size:14px;}
.batchid {font-size:38px;font-weight:800;margin:15px 0 5px;}
.sub {font-size:16px;margin-bottom:18px;}
.rule {border-top:1px solid #5f88a1;margin:18px 0;}
.kv {display:flex;justify-content:space-between;gap:10px;margin:19px 0;color:#d3e4ef;}
.kv b {color:white;font-size:17px;text-align:right;}
.risk {border-radius:6px;padding:20px 16px;margin:30px 0 26px;font-weight:800;text-align:center;color:white;}
.driver-title {font-size:18px;font-weight:700;color:#2d2030;margin:8px 0 2px;}
.driverhead {color:#06466e;font-weight:800;letter-spacing:2px;font-size:14px;}
.impact-note {background:#eaf3f7;color:#06466e;font-weight:700;text-align:center;padding:9px;border-radius:4px;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DEMO DATA
# Replace this dataframe with your real model prediction dataframe.
# Required columns:
# Batch ID, Supplier Number, Supplier Name, CMO Site, Material,
# Planned Release Date, Predicted Release Date
# ------------------------------------------------------------------
predictions = pd.DataFrame([
    {"Batch ID":"B-4533","Supplier Number":"SUP-1001","Supplier Name":"Supplier Alpha","CMO Site":"CMO Beta","Material":"Drug Y","Planned Release Date":"2026-07-23","Predicted Release Date":"2026-07-28"},
    {"Batch ID":"B-5022","Supplier Number":"SUP-1002","Supplier Name":"Supplier Beta","CMO Site":"CMO Alpha","Material":"Drug X","Planned Release Date":"2026-08-04","Predicted Release Date":"2026-08-06"},
    {"Batch ID":"B-4768","Supplier Number":"SUP-1001","Supplier Name":"Supplier Alpha","CMO Site":"CMO Gamma","Material":"Drug Z","Planned Release Date":"2026-08-12","Predicted Release Date":"2026-08-11"},
    {"Batch ID":"B-5180","Supplier Number":"SUP-1003","Supplier Name":"Supplier Gamma","CMO Site":"CMO Delta","Material":"Drug P","Planned Release Date":"2026-08-19","Predicted Release Date":"2026-08-25"},
])

# Replace with actual per-batch aggregated SHAP impacts.
driver_map = {
    "B-4533": {"Site QC backlog":2.1,"Recent OOS history":1.5,"High test complexity":0.9,"Strong product history":-0.6},
    "B-5022": {"Site QC backlog":0.8,"Recent OOS history":0.3,"Testing complexity":0.7,"Supplier history":-0.4},
    "B-4768": {"Site QC backlog":-0.2,"Recent OOS history":0.1,"Testing complexity":-0.5,"Product history":-0.7},
    "B-5180": {"Site QC backlog":2.4,"Recent OOS history":1.1,"Testing complexity":1.3,"Supplier history":0.5,"Product history":-0.3},
}

MODEL_MAE_DAYS = 3.3


def render_batch_driver_view(row, driver_impacts, model_mae_days):
    target = pd.to_datetime(row["Planned Release Date"])
    predicted = pd.to_datetime(row["Predicted Release Date"])
    delta = int((predicted - target).days)

    if delta >= 5:
        risk_label, risk_color = "HIGH RISK", "#dc0073"
    elif delta >= 2:
        risk_label, risk_color = "MEDIUM RISK", "#f59e0b"
    else:
        risk_label, risk_color = "LOW RISK", "#16866b"

    batch = str(row["Batch ID"])
    site = str(row.get("CMO Site", ""))
    product = str(row.get("Material", ""))
    supplier_id = str(row.get("Supplier Number", ""))
    supplier_name = str(row.get("Supplier Name", ""))

    c1, c2 = st.columns([0.39, 1.0], gap="medium")

    with c1:
        st.markdown(f"""
        <div class="snap">
          <div class="eyebrow">BATCH SNAPSHOT</div>
          <div class="batchid">{batch}</div>
          <div class="sub">{site} &nbsp;|&nbsp; {product}</div>
          <div class="rule"></div>
          <div class="kv"><span>Supplier</span><b>{supplier_id}<br>{supplier_name}</b></div>
          <div class="kv"><span>Target release</span><b>{target.strftime('%d %b')}</b></div>
          <div class="kv"><span>Predicted release</span><b>{predicted.strftime('%d %b')}</b></div>
          <div class="risk" style="background:{risk_color}">{delta:+d} DAYS &nbsp;|&nbsp; {risk_label}</div>
          <div class="kv"><span>Model MAE</span><b>± {model_mae_days:.1f} days</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="driverhead">DRIVER VIEW</div><div class="driver-title">Impact on predicted release lead time</div>',
            unsafe_allow_html=True,
        )

        impacts = pd.Series(driver_impacts, dtype=float).dropna()
        if impacts.empty:
            st.info("No driver explanation is available for the selected batch.")
            return

        impacts = impacts.reindex(impacts.abs().sort_values(ascending=False).index).head(6)
        plot_impacts = impacts.iloc[::-1]
        bar_colors = ["#319474" if v < 0 else "#cb3e55" for v in plot_impacts.values]
        strongest = impacts.abs().idxmax()
        strongest_pos = list(plot_impacts.index).index(strongest)
        bar_colors[strongest_pos] = "#e00073" if impacts[strongest] >= 0 else "#16866b"

        fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor="white")
        ax.set_facecolor("white")
        y = np.arange(len(plot_impacts))
        ax.barh(y, plot_impacts.values, color=bar_colors, height=0.55)
        ax.axvline(0, color="#9aa5ad", linewidth=1.2)
        max_abs = max(1.0, float(plot_impacts.abs().max()))
        ax.set_xlim(-max_abs * 1.55, max_abs * 1.55)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_impacts.index, fontsize=10, color="#2d2030")

        for i, value in enumerate(plot_impacts.values):
            offset = max_abs * 0.05
            ax.text(
                value + (offset if value >= 0 else -offset), i,
                f"{value:+.1f} days", va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9, fontweight="bold",
                color="#cb3e55" if value >= 0 else "#25876d",
            )

        ax.text(0.02, 1.03, "Earlier release", transform=ax.transAxes, color="#25876d", fontsize=9)
        ax.text(0.98, 1.03, "Later release", transform=ax.transAxes, color="#cb3e55", fontsize=9, ha="right")
        ax.set_xlabel("SHAP impact on predicted lead time (days)", fontsize=9)
        ax.set_ylabel("")
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        fig.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.18)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        plt.close(fig)

        net_impact = float(impacts.sum())
        st.markdown(
            f'<div class="impact-note">Net displayed-driver impact: approximately {net_impact:+.1f} days versus the model baseline</div>',
            unsafe_allow_html=True,
        )


st.title("EQ Predictive Release Lead Time")
st.caption("Prediction + batch-level explainability")

# ---------------------- FILTERS ----------------------
# One supplier dropdown displays BOTH Supplier ID and Supplier Name.
supplier_lookup = (
    predictions[["Supplier Number", "Supplier Name"]]
    .drop_duplicates()
    .sort_values(["Supplier Name", "Supplier Number"])
    .reset_index(drop=True)
)

supplier_options = ["All Suppliers"] + [
    f"{r['Supplier Number']} | {r['Supplier Name']}"
    for _, r in supplier_lookup.iterrows()
]

f1, f2 = st.columns(2)
with f1:
    selected_supplier = st.selectbox(
        "Select Supplier ID | Supplier Name",
        supplier_options,
        index=0,
        key="selected_supplier",
    )

if selected_supplier == "All Suppliers":
    filtered = predictions.copy()
else:
    selected_supplier_id = selected_supplier.split(" | ", 1)[0]
    filtered = predictions[
        predictions["Supplier Number"].astype(str) == selected_supplier_id
    ].copy()

batch_options = filtered["Batch ID"].dropna().astype(str).unique().tolist()
if not batch_options:
    st.warning("No Batch IDs are available for the selected supplier.")
    st.stop()

with f2:
    selected_batch = st.selectbox(
        "Select Batch ID",
        batch_options,
        index=0,
        key="selected_batch_id",
    )

selected_row = filtered.loc[
    filtered["Batch ID"].astype(str) == selected_batch
].iloc[0]
selected_drivers = driver_map.get(selected_batch, {})

render_batch_driver_view(selected_row, selected_drivers, MODEL_MAE_DAYS)
