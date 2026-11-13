import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="GDSC Drug Sensitivity", page_icon="🧬", layout="wide")

HERE = Path(__file__).resolve().parent
SEARCH_DIRS = [HERE / "artifacts", HERE, Path.cwd() / "artifacts", Path.cwd()]


def find(name):
    """Locate an artifact file: in artifacts/ next to the app, or flat next to the app.
    Also accepts browser-style duplicate names such as 'model_comparison (1).csv'."""
    stem, suffix = Path(name).stem, Path(name).suffix
    for folder in SEARCH_DIRS:
        if not folder.is_dir():
            continue
        if (folder / name).exists():
            return folder / name
        hits = sorted(folder.glob(f"{stem} (*){suffix}"))
        if hits:
            return hits[0]
    return None


REQUIRED = ["best_model.joblib", "model_comparison.csv", "confusion_matrices.json",
            "meta.json", "drug_info.csv", "eda_sample.csv"]
missing = [n for n in REQUIRED if find(n) is None]
if find("cell_info.csv") is None and find("options.json") is None:
    missing.append("cell_info.csv (or options.json)")
if missing:
    st.error("Some files from the training notebook are missing from the GitHub repo: " + ", ".join(missing))
    st.write("Upload them into a folder named `artifacts` next to the app file. "
             "Files the app can currently see next to it:")
    st.code("\n".join(sorted(p.name for p in HERE.iterdir())))
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(find("best_model.joblib"))


@st.cache_data
def load_all():
    cell_path, opt_path = find("cell_info.csv"), find("options.json")
    return dict(
        results=pd.read_csv(find("model_comparison.csv")),
        cms=json.load(open(find("confusion_matrices.json"))),
        meta=json.load(open(find("meta.json"))),
        cells=pd.read_csv(cell_path).fillna("Unknown") if cell_path else None,
        opts=json.load(open(opt_path)) if opt_path else None,
        drugs=pd.read_csv(find("drug_info.csv")).fillna("Unknown"),
        eda=pd.read_csv(find("eda_sample.csv")),
    )


d = load_all()
meta, results = d["meta"], d["results"]
labels = ["Resistant", "Sensitive"]

st.title("🧬 Genomics of Drug Sensitivity in Cancer (GDSC)")
st.caption("Predict whether a cancer cell line is Sensitive or Resistant to a drug")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🏆 Model comparison", "🧩 Confusion matrix", "🔮 Predict"])

with tab1:
    eda = d["eda"]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(eda["LN_IC50"], bins=50, kde=True, ax=ax)
        ax.set_title("LN_IC50 distribution")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.heatmap(eda[["LN_IC50", "AUC", "Z_SCORE"]].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation heatmap")
        st.pyplot(fig)
    top_p = eda["TARGET_PATHWAY"].value_counts().head(12).index
    top_t = eda["GDSC Tissue descriptor 1"].value_counts().head(12).index
    piv = eda[eda["TARGET_PATHWAY"].isin(top_p) & eda["GDSC Tissue descriptor 1"].isin(top_t)].pivot_table(
        index="GDSC Tissue descriptor 1", columns="TARGET_PATHWAY", values="LN_IC50", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(piv, cmap="RdYlBu", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Mean LN_IC50: tissue x pathway (lower = more sensitive)")
    st.pyplot(fig)

with tab2:
    st.subheader("Accuracy and metrics for all 4 models")
    st.dataframe(results.style.format({c: "{:.4f}" for c in results.columns[1:]})
                 .background_gradient(cmap="Greens", subset=results.columns[1:]), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(data=results, x="Model", y="Accuracy", hue="Model", palette="viridis", legend=False, ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.3f")
        ax.set_ylim(0.5, 1.0)
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.heatmap(results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
                    annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    st.success(f"Best model: **{meta['best_model']}**")
    if "accuracy_unseen_cell_lines" in meta:
        st.info(f"Accuracy on random test split: {meta['accuracy_random_split']:.3f} | "
                f"on completely unseen cell lines: {meta['accuracy_unseen_cell_lines']:.3f}")

with tab3:
    choice = st.selectbox("Model", list(d["cms"].keys()),
                          index=list(d["cms"].keys()).index(meta["best_model"]))
    cm = np.array(d["cms"][choice])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {choice}")
    st.pyplot(fig)

with tab4:
    st.subheader(f"Prediction with {meta['best_model']}")
    features = meta["features"]
    drugs = d["drugs"]
    row = None

    if d["cells"] is not None and "CELL_LINE_NAME" in features:
        # newer notebook: pick a cell line and a drug
        cells = d["cells"]
        c1, c2 = st.columns(2)
        cell = c1.selectbox("Cell line", sorted(cells["CELL_LINE_NAME"].unique()))
        drug = c2.selectbox("Drug", sorted(drugs["DRUG_NAME"].unique()))
        crow = cells[cells["CELL_LINE_NAME"] == cell].iloc[0]
        drow = drugs[drugs["DRUG_NAME"] == drug].iloc[0]
        c1.caption(f"Tissue: {crow['GDSC Tissue descriptor 1']} / {crow['GDSC Tissue descriptor 2']} | TCGA: {crow['TCGA_DESC']}")
        c2.caption(f"Target: {drow['TARGET']} | Pathway: {drow['TARGET_PATHWAY']}")
        row = {**crow.to_dict(), **drow.to_dict()}
        row["DRUG_x_TISSUE"] = f"{row['DRUG_NAME']}|{row['GDSC Tissue descriptor 1']}"
        row["DRUG_x_TCGA"] = f"{row['DRUG_NAME']}|{row['TCGA_DESC']}"
        row["CELL_x_PATHWAY"] = f"{row['CELL_LINE_NAME']}|{row['TARGET_PATHWAY']}"
        row["CELL_x_TARGET"] = f"{row['CELL_LINE_NAME']}|{row['TARGET']}"
    elif d["opts"] is not None:
        # first notebook version: choose drug and cell-line attributes
        opts, defaults = d["opts"], d["opts"]["_defaults"]
        c1, c2, c3 = st.columns(3)
        drug = c1.selectbox("Drug", opts["DRUG_NAME"])
        drow = drugs[drugs["DRUG_NAME"] == drug].iloc[0]
        c1.caption(f"Target: {drow['TARGET']} | Pathway: {drow['TARGET_PATHWAY']}")
        tissue1 = c2.selectbox("Tissue (1)", opts["GDSC Tissue descriptor 1"])
        tissue2 = c2.selectbox("Tissue (2)", opts["GDSC Tissue descriptor 2"])
        tcga = c3.selectbox("TCGA description", opts["TCGA_DESC"])
        msi = c3.selectbox("MSI status", opts["Microsatellite instability Status (MSI)"])
        growth = c1.selectbox("Growth properties", opts["Growth Properties"])
        medium = c2.selectbox("Screen medium", opts["Screen Medium"])
        row = dict(defaults)
        row.update({"DRUG_NAME": drug, "TARGET": drow["TARGET"], "TARGET_PATHWAY": drow["TARGET_PATHWAY"],
                    "GDSC Tissue descriptor 1": tissue1, "GDSC Tissue descriptor 2": tissue2,
                    "TCGA_DESC": tcga, "Microsatellite instability Status (MSI)": msi,
                    "Growth Properties": growth, "Screen Medium": medium})
    else:
        st.warning("The saved artifacts do not match this app version. Re-run the notebook and upload the new artifacts folder.")

    if row is not None and st.button("Predict", type="primary"):
        try:
            X = pd.DataFrame([row])[features]
            p = load_model().predict_proba(X)[0, 1]
        except Exception as e:  # e.g. scikit-learn version mismatch between Colab and Streamlit Cloud
            st.error(f"Could not run the model: {e}")
            st.info("If this mentions a scikit-learn version, pin the same version in requirements.txt "
                    "that Colab used (run `!pip show scikit-learn` in Colab).")
        else:
            st.metric("Probability Sensitive", f"{p:.1%}")
            st.progress(float(p))
            st.write("Prediction: **Sensitive**" if p >= 0.5 else "Prediction: **Resistant**")
