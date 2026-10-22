import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='EQ Predictive Release Lead-Time', page_icon='📦', layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.5rem}.hero{background:linear-gradient(110deg,#7f005d,#d4147a);padding:22px 28px;border-radius:16px;color:white;margin-bottom:18px}
.kpi{background:#fff5fb;border:1px solid #eed8e6;border-radius:14px;padding:16px}.small{color:#77536c;font-size:.85rem}
</style>''', unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>EQ Predictive Release Lead-Time</h1><div>Interactive point-in-time prediction and SHAP explainability</div></div>", unsafe_allow_html=True)

DATA=Path(__file__).with_name('Expanded_Synthetic_Historical_Batch_Data_with_Suppliers 2 (1).xlsx')

@st.cache_data
def load_data():
    b=pd.read_excel(DATA,sheet_name='Historical Batch Data',header=3,engine='openpyxl')
    o=pd.read_excel(DATA,sheet_name='OOS Events',header=0,engine='openpyxl')
    for c in ['Prediction / Start Date','Planned Release Date','Actual Release Date']: b[c]=pd.to_datetime(b[c],errors='coerce')
    for c in ['OOS Event Date','Closure Date']: o[c]=pd.to_datetime(o[c],errors='coerce')
    def c90(r):
        m=(o.Material.astype(str)==str(r.Material))&(o['CMO Site'].astype(str)==str(r['CMO Site']))&(o['OOS Event Date']>=r['Prediction / Start Date']-pd.Timedelta(days=90))&(o['OOS Event Date']<r['Prediction / Start Date'])
        return int(m.sum())
    b['OOS Count Previous 90 Days']=b.apply(c90,axis=1)
    b['Release Lead Time (Days)']=(b['Actual Release Date']-b['Prediction / Start Date']).dt.days
    sp=b['Testing Type / Complexity'].astype(str).str.split('/',n=1,expand=True)
    b['Testing Type']=sp[0].str.strip(); b['Complexity']=sp[1].str.strip(); b['Start Month']=b['Prediction / Start Date'].dt.month.astype(str)
    return b.sort_values('Prediction / Start Date').reset_index(drop=True)

FEATURES=['Material','Supplier Number','CMO Site','QC Backlog at Start','OOS Count Previous 90 Days','QC Sample Count','Testing Type','Complexity','Start Month']
CAT=['Material','Supplier Number','CMO Site','Testing Type','Complexity','Start Month']
NUM=['QC Backlog at Start','OOS Count Previous 90 Days','QC Sample Count']

@st.cache_resource
def train_model(df):
    cut=int(len(df)*.8); tr=df.iloc[:cut]
    pre=ColumnTransformer([('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT),('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM)])
    pipe=Pipeline([('prep',pre),('model',RandomForestRegressor(n_estimators=500,min_samples_leaf=3,random_state=42))])
    pipe.fit(tr[FEATURES],tr['Release Lead Time (Days)'])
    return pipe

df=load_data(); model=train_model(df)
with st.sidebar:
    st.header('Find a batch')
    batch=st.selectbox('Batch ID',df['Batch ID'].astype(str).tolist(),index=0)
    st.caption('Select or type a Batch ID to view the model prediction.')

row=df[df['Batch ID'].astype(str)==str(batch)].iloc[[0]].copy()
pred_days=float(model.predict(row[FEATURES])[0]); pred_date=row['Prediction / Start Date'].iloc[0]+pd.to_timedelta(round(pred_days),unit='D')
actual_date=row['Actual Release Date'].iloc[0]; planned=row['Planned Release Date'].iloc[0]

st.subheader(f'Batch {batch}')
c1,c2,c3,c4=st.columns(4)
c1.metric('Predicted release date',pred_date.strftime('%d %b %Y'))
c2.metric('Predicted lead time',f'{pred_days:.1f} days')
c3.metric('QC backlog at start',int(row['QC Backlog at Start'].iloc[0]))
c4.metric('Prior 90-day OOS',int(row['OOS Count Previous 90 Days'].iloc[0]))

st.caption('Prediction uses the point-in-time fields available at QC entry. Actual release is displayed only for historical comparison and is not a model input.')
left,right=st.columns([1,1])
with left:
    st.markdown('### Batch context')
    context=pd.DataFrame({'Field':['Material','Supplier','CMO Site','Prediction / Start Date','Planned Release Date','Actual Release Date (historical)'], 'Value':[str(row.Material.iloc[0]),str(row['Supplier Name'].iloc[0]),str(row['CMO Site'].iloc[0]),row['Prediction / Start Date'].iloc[0].strftime('%d %b %Y'),planned.strftime('%d %b %Y'),actual_date.strftime('%d %b %Y')]})
    st.dataframe(context,hide_index=True,use_container_width=True)
with right:
    st.markdown('### Release-date comparison')
    dates=pd.DataFrame({'Date':[planned,pred_date,actual_date]},index=['Planned','Predicted','Actual'])
    base=row['Prediction / Start Date'].iloc[0]
    days=(dates['Date']-base).dt.days
    fig,ax=plt.subplots(figsize=(6,3.4)); colors=['#b7a7b1','#a6006d','#54213f']; ax.barh(days.index,days.values,color=colors); ax.set_xlabel('Days from QC entry'); ax.set_title('Lead-time comparison'); st.pyplot(fig,use_container_width=True)

st.markdown('### Why did the model predict this date?')
Xt=model.named_steps['prep'].transform(row[FEATURES])
fn=model.named_steps['prep'].get_feature_names_out()
rf=model.named_steps['model']; ex=shap.TreeExplainer(rf); sv=ex(Xt)
vals=sv.values[0]
# aggregate one-hot encoded SHAP values back to business feature groups
agg=[]
for original in FEATURES:
    if original in NUM:
        idx=[i for i,n in enumerate(fn) if n==f'num__{original}']
    else:
        idx=[i for i,n in enumerate(fn) if n.startswith(f'cat__{original}_')]
    agg.append((original,float(vals[idx].sum()) if idx else 0.0))
drivers=pd.DataFrame(agg,columns=['Driver','SHAP impact (days)']); drivers['Abs']=drivers['SHAP impact (days)'].abs(); drivers=drivers.sort_values('Abs',ascending=False).head(7)
fig,ax=plt.subplots(figsize=(10,4)); dd=drivers.sort_values('SHAP impact (days)'); ax.barh(dd.Driver,dd['SHAP impact (days)'],color=np.where(dd['SHAP impact (days)']>=0,'#d4147a','#5577aa')); ax.axvline(0,color='#555',lw=.8); ax.set_xlabel('Impact on predicted lead time (days)'); ax.set_title('SHAP drivers for selected batch'); st.pyplot(fig,use_container_width=True)
st.dataframe(drivers[['Driver','SHAP impact (days)']].style.format({'SHAP impact (days)':'{:+.2f}'}),hide_index=True,use_container_width=True)
st.info('Positive SHAP values push predicted lead time later; negative values pull it earlier. SHAP explains the model prediction, not a causal reason for a quality delay.')

with st.expander('Model input used for this prediction'):
    st.dataframe(row[FEATURES],hide_index=True,use_container_width=True)
with st.expander('About this prototype'):
    st.write('Hackathon proof of concept using the supplied expanded synthetic historical dataset. Model training uses the earlier 80% of batches by prediction date; the current-batch OOS outcome is excluded from predictors to preserve point-in-time logic.')
