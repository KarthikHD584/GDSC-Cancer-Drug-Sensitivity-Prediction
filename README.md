# GDSC Drug Sensitivity Classifier

Predicts whether a cancer cell line is **Sensitive** or **Resistant** to a drug using the Genomics of Drug Sensitivity in Cancer (GDSC) data.
Sensitive = LN_IC50 below the dataset median. LN_IC50, AUC and Z_SCORE are not used as features (they define the label).

## Files
| File | Purpose |
|---|---|
| `GDSC_Drug_Sensitivity_v2.ipynb` | Ready-to-upload Colab notebook (same code as the script) |
| `gdsc_colab_pipeline.py` | Colab script: load, clean, EDA, transform, train 4 models, compare, evaluate, save artifacts |
| `app.py` | Streamlit app (EDA, model comparison, confusion matrix, prediction) |
| `requirements.txt` | Python dependencies |
| `artifacts/` | Created by the pipeline (model, metrics, options, figures) |

## 1. Run in Google Colab
1. Make sure your Drive has the folder `Genomics of Drug Sensitivity in Cancer (GDSC)` containing `GDSC_DATASET.csv` and `Cell_Lines_Details.xlsx`.
2. Open a new Colab notebook and paste `gdsc_colab_pipeline.py`, one `# %%` block per cell.
3. If your folder path differs, edit `DATA_DIR` in cell 1. Set `SAMPLE_ROWS = None` to use all rows.
4. Run all cells. The last cell prints the model table and the best model. Artifacts are saved to `/content/artifacts` and copied to `<DATA_DIR>/artifacts` in Drive.

## 2. Launch Streamlit from Colab (localtunnel)
Download `artifacts/` from Drive into `/content/`, upload `app.py` next to it, then:

```
!pip -q install streamlit
!streamlit run app.py &>/content/logs.txt &
!curl ipv4.icanhazip.com
!npx localtunnel --port 8501
```
Open the printed URL and enter the IP address from `icanhazip` as the tunnel password.

## 3. Deploy on streamlit.io (Community Cloud)
1. Create a GitHub repo containing `app.py`, `requirements.txt` and the `artifacts/` folder.
   - Keep `best_model.joblib` under GitHub's 100 MB file limit. If it is too big, lower `n_estimators` or `max_depth` for Random Forest and rerun.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose the repo and branch, and set the main file to `app.py`.
4. Click **Deploy**.

## Notes
- Random seed is 42 everywhere.
- Features are categorical metadata plus cell-line identity and drug/cell-line interactions, encoded with leakage-safe target encoding. No genomic measurements (mutations, expression) are used.
- The random-split accuracy applies to cell lines and drugs seen in training. The notebook also reports accuracy on completely unseen cell lines, which is lower and more realistic for new samples.
- `lightgbm` is required in `requirements.txt` because the saved model may be a LightGBM pipeline.
