# =====================================================================
# GDSC (Genomics of Drug Sensitivity in Cancer) - Drug Sensitivity Classifier
# Run in Google Colab. Each "# %%" block = one Colab cell.
# Task: predict whether a (cell line, drug) pair is SENSITIVE or RESISTANT
# =====================================================================

# %% 1. Setup + mount Google Drive
!pip -q install seaborn joblib lightgbm
from google.colab import drive
drive.mount('/content/drive')

import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import joblib
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# >>> EDIT THIS PATH if your Drive folder is somewhere else <<<
DATA_DIR = '/content/drive/MyDrive/Genomics of Drug Sensitivity in Cancer (GDSC)'
DATA_FILE = f'{DATA_DIR}/GDSC_DATASET.csv'
ART = '/content/artifacts'              # outputs for the Streamlit app
os.makedirs(ART, exist_ok=True)
SAMPLE_ROWS = None                      # None = all rows (best accuracy); e.g. 120_000 for a quick run
RANDOM_STATE = 42

# %% 2. INPUT DATA / COLLECT DATA
df_raw = pd.read_csv(DATA_FILE)
print('Shape:', df_raw.shape)
display(df_raw.head())
df_raw.info()

# Optional: cell-line details sheet (collected from the same Drive folder)
cells = pd.read_excel(f'{DATA_DIR}/Cell_Lines_Details.xlsx', header=1)
print('Cell line details:', cells.shape)

# %% 3. PRE-PROCESSING
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
print('Missing values:\n', df.isna().sum()[df.isna().sum() > 0])
print('Duplicates:', df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna(subset=['LN_IC50'])                      # target source must exist
cat_cols = ['TCGA_DESC', 'DRUG_NAME', 'GDSC Tissue descriptor 1',
            'GDSC Tissue descriptor 2', 'Cancer Type (matching TCGA label)',
            'Microsatellite instability Status (MSI)', 'Screen Medium',
            'Growth Properties', 'CNA', 'Gene Expression', 'Methylation',
            'TARGET', 'TARGET_PATHWAY']
for c in cat_cols:
    df[c] = df[c].fillna('Unknown').astype(str).str.strip()
print('After cleaning:', df.shape)
print('Missing values AFTER cleaning (total):', int(df.isna().sum().sum()))
print('Dtypes:\n', df.dtypes)

# %% 4. EDA
print(df[['LN_IC50', 'AUC', 'Z_SCORE']].describe())

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
sns.histplot(df['LN_IC50'], bins=60, kde=True, ax=ax[0]); ax[0].set_title('LN_IC50 distribution')
sns.histplot(df['AUC'], bins=60, kde=True, color='orange', ax=ax[1]); ax[1].set_title('AUC distribution')
df['GDSC Tissue descriptor 1'].value_counts().head(15).plot.barh(ax=ax[2]); ax[2].set_title('Top tissues')
plt.tight_layout(); plt.savefig(f'{ART}/eda_distributions.png', dpi=130); plt.show()

# Heatmap 1: correlation of numeric columns
plt.figure(figsize=(5, 4))
sns.heatmap(df[['LN_IC50', 'AUC', 'Z_SCORE']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation heatmap'); plt.tight_layout()
plt.savefig(f'{ART}/heatmap_correlation.png', dpi=130); plt.show()

# Heatmap 2: mean LN_IC50 - tissue x pathway
top_p = df['TARGET_PATHWAY'].value_counts().head(12).index
top_t = df['GDSC Tissue descriptor 1'].value_counts().head(12).index
piv = (df[df['TARGET_PATHWAY'].isin(top_p) & df['GDSC Tissue descriptor 1'].isin(top_t)]
       .pivot_table(index='GDSC Tissue descriptor 1', columns='TARGET_PATHWAY',
                    values='LN_IC50', aggfunc='mean'))
plt.figure(figsize=(12, 6))
sns.heatmap(piv, cmap='RdYlBu', annot=True, fmt='.1f')
plt.title('Mean LN_IC50 by tissue x target pathway (lower = more sensitive)')
plt.tight_layout(); plt.savefig(f'{ART}/heatmap_tissue_pathway.png', dpi=130); plt.show()

# %% 5. DATA TRANSFORMATION (feature engineering + target encoding)
# Target: 1 = Sensitive (LN_IC50 below median), 0 = Resistant
median_ic50 = df['LN_IC50'].median()
df['SENSITIVE'] = (df['LN_IC50'] < median_ic50).astype(int)
print('Median LN_IC50:', round(median_ic50, 3)); print(df['SENSITIVE'].value_counts())
plt.figure(figsize=(4, 3))
sns.countplot(x=df['SENSITIVE'].map({0: 'Resistant', 1: 'Sensitive'}))
plt.title('Class balance'); plt.tight_layout()
plt.savefig(f'{ART}/class_balance.png', dpi=130); plt.show()

# NEW: cell-line identity + interaction features (the biggest accuracy gain).
# Sensitivity depends on WHICH cell line meets WHICH drug / pathway.
df['CELL_LINE_NAME'] = df['CELL_LINE_NAME'].fillna('Unknown').astype(str).str.strip()

def add_interactions(d):
    d = d.copy()
    d['DRUG_x_TISSUE']  = d['DRUG_NAME']      + '|' + d['GDSC Tissue descriptor 1']
    d['DRUG_x_TCGA']    = d['DRUG_NAME']      + '|' + d['TCGA_DESC']
    d['CELL_x_PATHWAY'] = d['CELL_LINE_NAME'] + '|' + d['TARGET_PATHWAY']
    d['CELL_x_TARGET']  = d['CELL_LINE_NAME'] + '|' + d['TARGET']
    return d

df = add_interactions(df)
FEATURES = cat_cols + ['CELL_LINE_NAME', 'DRUG_x_TISSUE', 'DRUG_x_TCGA', 'CELL_x_PATHWAY', 'CELL_x_TARGET']
# AUC / Z_SCORE / LN_IC50 are EXCLUDED on purpose (they define the label -> leakage)
print(len(FEATURES), 'features:', FEATURES)

data = df.sample(SAMPLE_ROWS, random_state=RANDOM_STATE) if SAMPLE_ROWS and len(df) > SAMPLE_ROWS else df
X, y = data[FEATURES], data['SENSITIVE']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print('Train:', X_train.shape, 'Test:', X_test.shape)

# TargetEncoder turns each category into its (smoothed) sensitivity rate.
# fit_transform uses internal 5-fold cross-fitting, so the training rows never see their own label -> no leakage.
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

def make_pre():
    return ColumnTransformer([('te', TargetEncoder(target_type='binary', cv=5,
                                                   random_state=RANDOM_STATE), FEATURES)])

# %% 6. MACHINE LEARNING - TRAIN 4 TUNED MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
import time

models = {
    'Logistic Regression': Pipeline([('sc', StandardScaler()),
                                     ('lr', LogisticRegression(max_iter=1000))]),
    'Random Forest':       RandomForestClassifier(n_estimators=250, max_depth=18, min_samples_leaf=5,
                                                  max_features=0.5, n_jobs=-1, random_state=RANDOM_STATE),
    'Gradient Boosting':   HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                                          l2_regularization=1.0, early_stopping=True,
                                                          validation_fraction=0.1, n_iter_no_change=30,
                                                          random_state=RANDOM_STATE),
    'LightGBM':            LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                                          random_state=RANDOM_STATE, verbose=-1),
}

fitted, preds, probs = {}, {}, {}
for name, clf in models.items():
    t0 = time.time()
    pipe = Pipeline([('pre', make_pre()), ('clf', clone(clf))])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    preds[name] = pipe.predict(X_test)
    probs[name] = pipe.predict_proba(X_test)[:, 1]
    print(f'{name:20s} trained in {time.time()-t0:6.1f}s')

# %% 7. MODEL COMPARISON + EVALUATION (accuracy, precision, recall, F1, ROC-AUC)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

rows = []
for name in models:
    rows.append({'Model': name,
                 'Accuracy':  accuracy_score(y_test, preds[name]),
                 'Precision': precision_score(y_test, preds[name]),
                 'Recall':    recall_score(y_test, preds[name]),
                 'F1':        f1_score(y_test, preds[name]),
                 'ROC_AUC':   roc_auc_score(y_test, probs[name])})
results = pd.DataFrame(rows).sort_values('Accuracy', ascending=False).reset_index(drop=True)

# 3-fold cross-validated accuracy on a 50k training subsample (stability check, kept fast)
from sklearn.model_selection import cross_val_score
cv_idx = X_train.sample(min(50_000, len(X_train)), random_state=RANDOM_STATE).index
cv = {n: cross_val_score(Pipeline([('pre', make_pre()), ('clf', clone(models[n]))]),
                         X_train.loc[cv_idx], y_train.loc[cv_idx], cv=3, scoring='accuracy').mean()
      for n in models}
results['CV_Accuracy'] = results['Model'].map(cv)
display(results.style.format({c: '{:.4f}' for c in results.columns[1:]}).background_gradient(cmap='Greens'))
results.to_csv(f'{ART}/model_comparison.csv', index=False)

plt.figure(figsize=(8, 4))
ax = sns.barplot(data=results, x='Model', y='Accuracy', palette='viridis')
ax.bar_label(ax.containers[0], fmt='%.3f'); plt.ylim(0.5, 1.0)
plt.title('Model accuracy comparison'); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(f'{ART}/accuracy_comparison.png', dpi=130); plt.show()

plt.figure(figsize=(9, 4))
sns.heatmap(results.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']],
            annot=True, fmt='.3f', cmap='YlGnBu')
plt.title('Model comparison heatmap'); plt.tight_layout()
plt.savefig(f'{ART}/heatmap_model_comparison.png', dpi=130); plt.show()

plt.figure(figsize=(6, 5))
for n in models:
    fpr, tpr, _ = roc_curve(y_test, probs[n]); plt.plot(fpr, tpr, label=n)
plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC curves')
plt.tight_layout(); plt.savefig(f'{ART}/roc_curves.png', dpi=130); plt.show()

# %% 8. BEST MODEL: CONFUSION MATRIX HEATMAP
best_name = results.loc[0, 'Model']
print('BEST MODEL:', best_name)
print(classification_report(y_test, preds[best_name], target_names=['Resistant', 'Sensitive']))

cm = confusion_matrix(y_test, preds[best_name])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Resistant', 'Sensitive'], yticklabels=['Resistant', 'Sensitive'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion matrix - {best_name}')
plt.tight_layout(); plt.savefig(f'{ART}/confusion_matrix_best.png', dpi=130); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, n in zip(axes, models):
    sns.heatmap(confusion_matrix(y_test, preds[n]), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(n); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig(f'{ART}/confusion_matrices_all.png', dpi=130); plt.show()

# %% 9. HONEST CHECK: accuracy on cell lines the model has NEVER seen
# The split above is random, so test rows share cell lines/drugs with training rows.
# This grouped split holds out whole cell lines to show real-world generalization.
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_i, te_i = next(gss.split(X, y, groups=data['CELL_LINE_NAME']))
p_unseen = Pipeline([('pre', make_pre()), ('clf', clone(models[best_name]))]).fit(X.iloc[tr_i], y.iloc[tr_i])
acc_unseen = accuracy_score(y.iloc[te_i], p_unseen.predict(X.iloc[te_i]))
print(f'{best_name} accuracy on UNSEEN cell lines: {acc_unseen:.4f}')

# %% 10. SAVE ARTIFACTS FOR STREAMLIT
DEPLOY_MODEL = None   # e.g. 'LightGBM' to deploy a smaller model instead of the best one
deploy_name = DEPLOY_MODEL or best_name
joblib.dump(fitted[deploy_name], f'{ART}/best_model.joblib', compress=3)
print('Saved model:', deploy_name, '| size MB:', round(os.path.getsize(f'{ART}/best_model.joblib')/1e6, 1))

json.dump({n: confusion_matrix(y_test, preds[n]).tolist() for n in models},
          open(f'{ART}/confusion_matrices.json', 'w'))
json.dump({'best_model': deploy_name, 'median_ln_ic50': float(median_ic50), 'features': FEATURES,
           'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
           'accuracy_random_split': float(results.loc[results['Model'] == deploy_name, 'Accuracy'].iloc[0]),
           'accuracy_unseen_cell_lines': float(acc_unseen)},
          open(f'{ART}/meta.json', 'w'))
df[['DRUG_NAME', 'TARGET', 'TARGET_PATHWAY']].drop_duplicates('DRUG_NAME').to_csv(f'{ART}/drug_info.csv', index=False)
df[['CELL_LINE_NAME', 'TCGA_DESC', 'GDSC Tissue descriptor 1', 'GDSC Tissue descriptor 2',
    'Cancer Type (matching TCGA label)', 'Microsatellite instability Status (MSI)', 'Screen Medium',
    'Growth Properties', 'CNA', 'Gene Expression', 'Methylation']] \
    .drop_duplicates('CELL_LINE_NAME').to_csv(f'{ART}/cell_info.csv', index=False)
df[['LN_IC50', 'AUC', 'Z_SCORE', 'GDSC Tissue descriptor 1', 'TARGET_PATHWAY', 'DRUG_NAME']] \
    .sample(min(20000, len(df)), random_state=1).to_csv(f'{ART}/eda_sample.csv', index=False)

!mkdir -p "{DATA_DIR}/artifacts" && cp -r {ART}/* "{DATA_DIR}/artifacts/"
print(os.listdir(ART))

# %% 11. FINAL SUMMARY
print('\n=== MODEL SUMMARY (sorted by accuracy) ===')
print(results.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print(f'\nBEST MODEL: {best_name}  (accuracy = {results.loc[0, "Accuracy"]:.4f})')
print(f'Accuracy on unseen cell lines: {acc_unseen:.4f}')
