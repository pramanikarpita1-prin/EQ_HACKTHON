import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='EQ Predictive Release Lead Time', layout='wide')

# Expected integration contract:
# prediction_row: one row/Series containing Batch ID, CMO Site, Material/Supplier,
#                 Planned Release Date, Predicted Release Date, Prediction / Start Date
# driver_impacts: dict {business_driver_name: shap_impact_days}
# model_mae_days: numeric validation MAE (displayed as model uncertainty context)

def render_batch_driver_view(prediction_row, driver_impacts, model_mae_days):
    target = pd.to_datetime(prediction_row['Planned Release Date'])
    predicted = pd.to_datetime(prediction_row['Predicted Release Date'])
    delta = int((predicted - target).days)
    if delta >= 5:
        risk_label, risk_color = 'HIGH RISK', '#dc0073'
    elif delta >= 2:
        risk_label, risk_color = 'MEDIUM RISK', '#f59e0b'
    else:
        risk_label, risk_color = 'LOW RISK', '#16866b'

    batch = str(prediction_row['Batch ID'])
    site = str(prediction_row.get('CMO Site', ''))
    product = str(prediction_row.get('Material', prediction_row.get('Supplier Name', '')))

    st.markdown('''
    <style>
      .block-container {padding-top: 1.4rem; max-width: 1220px;}
      .snap {background:#06466e;border-radius:10px;padding:30px 24px;color:white;min-height:495px;}
      .eyebrow {color:#62c8dc;font-weight:800;letter-spacing:2px;font-size:14px;}
      .batchid {font-size:38px;font-weight:800;margin:15px 0 5px;}
      .sub {font-size:16px;margin-bottom:18px;}
      .rule {border-top:1px solid #5f88a1;margin:18px 0;}
      .kv {display:flex;justify-content:space-between;gap:10px;margin:21px 0;color:#d3e4ef;}
      .kv b {color:white;font-size:19px;}
      .risk {border-radius:6px;padding:20px 16px;margin:36px 0 30px;font-weight:800;text-align:center;color:white;}
      .driverbox {border:1px solid #ccd6de;border-radius:9px;padding:27px 28px;min-height:495px;background:white;}
      .driverhead {color:#06466e;font-weight:800;letter-spacing:2px;font-size:14px;}
      .driver-title {font-size:18px;font-weight:700;color:#2d2030;margin:12px 0 4px;}
      .impact-note {background:#eaf3f7;color:#06466e;font-weight:700;text-align:center;padding:9px;border-radius:4px;margin-top:8px;}
    </style>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns([0.39, 1], gap='medium')
    with c1:
        st.markdown(f'''<div class="snap">
        <div class="eyebrow">BATCH SNAPSHOT</div><div class="batchid">{batch}</div>
        <div class="sub">{site} &nbsp;|&nbsp; {product}</div><div class="rule"></div>
        <div class="kv"><span>Target release</span><b>{target.strftime('%d %b')}</b></div>
        <div class="kv"><span>Predicted release</span><b>{predicted.strftime('%d %b')}</b></div>
        <div class="risk" style="background:{risk_color}">{delta:+d} DAYS &nbsp;|&nbsp; {risk_label}</div>
        <div class="kv"><span>Model MAE</span><b>± {model_mae_days:.1f} days</b></div>
        </div>''', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="driverbox"><div class="driverhead">DRIVER VIEW</div><div class="driver-title">Impact on predicted release lead time</div>', unsafe_allow_html=True)
        impacts = pd.Series(driver_impacts, dtype=float)
        impacts = impacts.reindex(impacts.abs().sort_values(ascending=False).index).head(6)
        impacts = impacts.sort_values()
        colors = ['#319474' if v < 0 else ('#e00073' if v == impacts.max() else '#cb3e55') for v in impacts.values]
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        ax.barh(impacts.index, impacts.values, color=colors, height=.55)
        ax.axvline(0, color='#9aa5ad', lw=1.2)
        max_abs = max(1.0, float(impacts.abs().max()))
        ax.set_xlim(-max_abs*1.55, max_abs*1.55)
        for i,v in enumerate(impacts.values):
            ax.text(v + (0.05*max_abs if v>=0 else -0.05*max_abs), i, f'{v:+.1f} days', va='center',
                    ha='left' if v>=0 else 'right', color='#cb3e55' if v>=0 else '#25876d', fontweight='bold', fontsize=9)
        ax.set_xlabel('SHAP impact on predicted lead time (days)')
        ax.set_ylabel('')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(False)
        st.pyplot(fig, use_container_width=True)
        net=float(impacts.sum())
        st.markdown(f'<div class="impact-note">Net displayed-driver impact: approximately {net:+.1f} days versus the model baseline</div></div>', unsafe_allow_html=True)

# ---------------- Example wiring ----------------
# Replace this sample block with your existing selected-batch/model outputs.
if __name__ == '__main__':
    st.title('EQ Predictive Release Lead Time')
    st.caption('Prediction + explainability view')
    demo = pd.Series({
        'Batch ID':'B-4533','CMO Site':'CMO Beta','Material':'Drug Y',
        'Planned Release Date':'2026-07-23','Predicted Release Date':'2026-07-28'
    })
    demo_drivers = {
        'Site QC backlog': 2.1,
        'Recent OOS history': 1.5,
        'High test complexity': 0.9,
        'Strong product history': -0.6,
    }
    render_batch_driver_view(demo, demo_drivers, 3.3)
