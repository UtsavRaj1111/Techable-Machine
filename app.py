import streamlit as st
import pandas as pd, numpy as np, io, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, r2_score, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.cluster import KMeans

# ─── 1 ▸ global page settings & custom CSS ────────────────────────── #
st.set_page_config(page_title="AI Sensei", page_icon="🤖", layout="wide")

CSS = r"""
<style>
/* Solid background color */
html, body, [data-testid="stApp"]{
  height:100%;
  background: #black;
}


.app-name {
  font-size: 2rem;
  font-weight: 800;
  text-align: center;
  margin-bottom: 1.5rem;
  background: linear-gradient(90deg,#7cfdfd,#ff6fd8,#fcff66);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  filter: drop-shadow(0 0 6px rgba(255,255,255,.35));
}


h1{
  font-weight:800!important;
  background:linear-gradient(90deg,#7cfdfd,#ff6fd8,#fcff66);
  background-clip:text;-webkit-background-clip:text;color:transparent;
  filter:drop-shadow(0 0 6px rgba(255,255,255,.35));
}



section[data-testid="stFileUploader"]{
  background:rgba(255,255,255,.25);
  border:2px dashed rgba(255,255,255,.4);
  border-radius:16px;
  padding:1.4rem;
  transition:border-color .3s ease;
}
section[data-testid="stFileUploader"]:hover{
  border-color:#7d5bff;
}


div.streamlit-expander, select, input[type=range]{
  color:#000 !important;
  font-weight:600;
}


div.stButton>button{
  background:linear-gradient(135deg,#5562ff 0%,#ab37ff 100%);
  color:#fff;border:none;border-radius:40px;font-weight:600;
  padding:.55rem 1.6rem;box-shadow:0 4px 15px rgba(123,97,255,.4);
  transition:all .25s ease;
}
div.stButton>button:hover{
  transform:scale(1.05) translateY(-2px);
  box-shadow:0 8px 22px rgba(123,97,255,.65);
}
div.stButton>button:focus:not(:active){border:none;}

[data-testid="baseButton-secondary"]{
  border-radius:40px;font-weight:600;
}


div[data-testid="stDataFrame"]{
  background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.2);
  border-radius:12px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── 2 ▸ training helper utils (unchanged) ───────────────────────── #
def split_X_y(df_, tgt):
    X = df_.drop(columns=tgt)
    y = df_[tgt]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat

def make_pre(num, cat):
    tr = []
    if num: tr.append(("num", StandardScaler(), num))
    if cat: tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(tr)

def train_classify(df_, tgt, test_size):
    X, y, num, cat = split_X_y(df_, tgt)
    Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=test_size,stratify=y,random_state=42)
    pipe = Pipeline([("pre",make_pre(num,cat)),
                     ("rf",RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=42))])
    pipe.fit(Xtr,ytr)
    preds = pipe.predict(Xts)
    metrics = {"Accuracy":accuracy_score(yts,preds),
               "F1":f1_score(yts,preds,average="weighted")}
    if len(np.unique(y))==2:
        metrics["ROC‑AUC"] = roc_auc_score(yts,pipe.predict_proba(Xts)[:,1])
    return pipe,metrics

def train_regress(df_, tgt, test_size):
    X, y, num, cat = split_X_y(df_, tgt)
    Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=test_size,random_state=42)
    pipe = Pipeline([("pre",make_pre(num,cat)),
                     ("rf",RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=42))])
    pipe.fit(Xtr,ytr)
    preds = pipe.predict(Xts)
    metrics = {"MAE":mean_absolute_error(yts,preds),
               "R²":r2_score(yts,preds)}
    return pipe,metrics

def train_cluster(df_, k):
    num_df = df_.select_dtypes(include="number")
    model = KMeans(n_clusters=k,random_state=42)
    labels = model.fit_predict(num_df)
    score = silhouette_score(num_df,labels)
    df_["cluster"] = labels
    return model, {"Silhouette":score}, df_

def dl_btn(model, fname):
    buf = io.BytesIO(); joblib.dump(model, buf); buf.seek(0)
    st.download_button("💾 Download model", buf, file_name=fname, mime="application/octet-stream")

# ─── 3 ▸ layout: 40 / 60 columns ────────────────────────────────── #
left, right = st.columns([4,6], gap="large")   # 40 %  /  60 %

# ─── 3‑A ▸ control panel (left column) ──────────────────────────── #
with left:
    with st.container():       # wrapper so CSS target uses .control-card
        st.markdown('<div class="control-card">', unsafe_allow_html=True)

        # Added app name section
        st.markdown('<div class="app-name">AI Sensei</div>', unsafe_allow_html=True)

        st.header("Upload & Configure")
        data_file = st.file_uploader("CSV file", type="csv")

        task = st.selectbox("Task", ("Classification", "Regression", "Clustering"))

        target_col, k_clusters = None, None
        if data_file is not None:
            df = pd.read_csv(data_file)
            if task != "Clustering":
                target_col = st.selectbox("Target column", df.columns)
            else:
                k_clusters = st.slider("Clusters (k)", 2, 10, 3)

        test_size = st.slider("Test‑size split", 0.1, 0.4, 0.2, 0.05,
                              help="Proportion of data held for testing")

        train_click = st.button("🚀 Train Model", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close control-card

# ─── 3‑B ▸ output / results (right column) ──────────────────────── #
with right:
    st.subheader("Results & Preview")

    if train_click and data_file is not None:
        with st.spinner("Training…"):
            if task == "Classification":
                model, metrics = train_classify(df, target_col, test_size)
            elif task == "Regression":
                model, metrics = train_regress(df, target_col, test_size)
            else:
                model, metrics, df = train_cluster(df, k_clusters)

        st.success("Model trained!")
        st.json({k: f"{v:.4f}" for k, v in metrics.items()})

        dl_btn(model, f"{task.lower()}_model.joblib")

        # Quick inference (supervised only)
        if task in ("Classification", "Regression"):
            st.divider()
            st.markdown("### Quick inference")
            sample = {}
            for col in df.columns:
                if col == target_col: continue
                if df[col].dtype == "object":
                    sample[col] = st.text_input(col, value=str(df[col].mode()[0]))
                else:
                    sample[col] = st.number_input(col, value=float(df[col].median()))
            if st.button("Predict", key="pred_btn"):
                pred = model.predict(pd.DataFrame([sample]))[0]
                st.info(f"*Prediction → {pred}*")
        else:
            st.divider()
            st.markdown("### Cluster preview")
            st.dataframe(df.head())