import difflib
import json
import random
import re
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_ace import st_ace

st.set_page_config(
    page_title="🧠 Python ML Reps",
    layout="wide",
    page_icon="🧠",
)

st.markdown("""
<style>
.block-container { padding-top: 0.6rem !important; padding-bottom: 0rem !important; }
[data-testid="baseButton-primary"][id*="nav_"],
[data-testid="baseButton-secondary"][id*="nav_"] { font-size: 0.55rem !important; padding: 0.2rem 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# QUESTION BANK  (10 topics)
# ──────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": "kmeans",
        "title": "K-Means Clustering",
        "category": "ML/Statistics",
        "prompt": """\
**Task:** Cluster the Iris dataset into 3 groups using K-Means.

1. Load `iris` from `sklearn.datasets`
2. Create a `KMeans(n_clusters=3, random_state=42)` model
3. Fit on `iris.data`
4. Print the cluster labels
5. Print the inertia (within-cluster sum of squares)
""",
        "workspace_tip": (
            "Try `model.labels_` for assignments and `model.inertia_` for the "
            "within-cluster sum of squares. Plot the first two features colored by label."
        ),
        "hint": """\
```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
model = KMeans(n_clusters=3, random_state=42)
model.fit(iris.data)
print(model.labels_)
print(model.inertia_)
```""",
        "solution": """\
```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print("Cluster labels:", model.labels_)
print("Inertia:", model.inertia_)
print("Cluster centers shape:", model.cluster_centers_.shape)

# Optional: scatter of first 2 features
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering on Iris")
plt.show()
```""",
    },
    {
        "id": "logreg",
        "title": "Logistic Regression",
        "category": "ML/Statistics",
        "prompt": """\
**Task:** Train a Logistic Regression classifier on Iris.

1. Load `iris` from `sklearn.datasets`
2. Split 80/20 with `random_state=42`
3. Fit `LogisticRegression(max_iter=200)` on the training set
4. Print test accuracy
5. Print the first 5 predictions vs actual labels
""",
        "workspace_tip": (
            "Remember the order: `fit` on X_train/y_train, `predict` on X_test, "
            "then compare with `accuracy_score(y_test, preds)`."
        ),
        "hint": """\
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```""",
        "solution": """\
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("First 5 predictions:", preds[:5])
print("First 5 actual:     ", y_test[:5])
```""",
    },
    {
        "id": "pca",
        "title": "PCA — Dimensionality Reduction",
        "category": "ML/Statistics",
        "prompt": """\
**Task:** Reduce Iris (4 features → 2 components) using PCA.

1. Load `iris` from `sklearn.datasets`
2. Standardize with `StandardScaler`
3. Apply `PCA(n_components=2)`
4. Print explained variance ratio
5. Plot the 2-D result colored by class
""",
        "workspace_tip": (
            "**Always scale before PCA.** "
            "Use `pca.explained_variance_ratio_` to see how much variance each PC captures. "
            "Sum should be ~0.73 for 2 components on Iris."
        ),
        "hint": """\
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```""",
        "solution": """\
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained:", pca.explained_variance_ratio_.sum().round(3))

df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df["species"] = iris.target

for label, name in enumerate(iris.target_names):
    mask = df["species"] == label
    plt.scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"], label=name)

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA of Iris"); plt.legend(); plt.show()
```""",
    },
    {
        "id": "train_test",
        "title": "Train/Test Split + Pipeline",
        "category": "ML/Statistics",
        "prompt": """\
**Task:** Build a full ML pipeline on the wine dataset.

1. Load `wine` from `sklearn.datasets`
2. Split 70/30, stratified, `random_state=0`
3. Chain `StandardScaler` → `RandomForestClassifier(n_estimators=100)` in a `Pipeline`
4. Fit and print a full `classification_report`
""",
        "workspace_tip": (
            "`Pipeline([('step_name', transformer), ...])` lets you chain steps cleanly. "
            "`fit` on train, `predict` on test, then `classification_report(y_test, preds)`."
        ),
        "hint": """\
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3,
    stratify=wine.target, random_state=0
)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=0)),
])
pipe.fit(X_train, y_train)
```""",
        "solution": """\
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3,
    stratify=wine.target, random_state=0
)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=0)),
])
pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
print(classification_report(y_test, preds, target_names=wine.target_names))
```""",
    },
    {
        "id": "groupby",
        "title": "Pandas GroupBy & Aggregation",
        "category": "Pandas/EDA",
        "prompt": """\
**Task:** Explore the Titanic dataset with groupby.

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

1. Group by `sex` + `pclass`, compute mean `survived` and mean `age`
2. Find the `pclass` with the highest survival rate
3. Count passengers per `embark_town`
4. Return the top-3 ports by count
""",
        "workspace_tip": (
            "Chain `.groupby().agg({'col': 'func'})` for multiple aggregations. "
            "Named aggregations look like `.agg(avg_age=('age', 'mean'))`."
        ),
        "hint": """\
```python
import seaborn as sns
df = sns.load_dataset("titanic")

result = df.groupby(["sex", "pclass"]).agg(
    avg_survived=("survived", "mean"),
    avg_age=("age", "mean"),
).round(2)
print(result)
```""",
        "solution": """\
```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")

# 1. sex × pclass
result = df.groupby(["sex", "pclass"]).agg(
    avg_survived=("survived", "mean"),
    avg_age=("age", "mean"),
).round(2)
print(result)

# 2. Best pclass
print("\\nSurvival by pclass:")
print(df.groupby("pclass")["survived"].mean().sort_values(ascending=False))

# 3. Count per embark_town
print("\\nPassengers per embark_town:")
print(df.groupby("embark_town").size())

# 4. Top 3 ports
print("\\nTop 3:", df["embark_town"].value_counts().head(3))
```""",
    },
    {
        "id": "merge",
        "title": "Merging Tables (SQL-style Joins)",
        "category": "Pandas/EDA",
        "prompt": """\
**Task:** Practice all common join types.

```python
import pandas as pd
orders    = pd.DataFrame({"order_id": [1,2,3,4], "customer_id": [10,10,20,30], "amount": [50,30,80,20]})
customers = pd.DataFrame({"customer_id": [10,20,40], "name": ["Alice","Bob","Charlie"]})
```

1. Inner join — matched rows only
2. Left join — all orders, fill missing names with `"Unknown"`
3. Identify customers who have **never** placed an order
""",
        "workspace_tip": (
            "`pd.merge(left, right, on='key', how='inner|left|right|outer')` — "
            "think SQL JOIN. Outer + filter NaN on a left-key finds unmatched rows."
        ),
        "hint": """\
```python
# Inner
inner = pd.merge(orders, customers, on="customer_id", how="inner")

# Left
left = pd.merge(orders, customers, on="customer_id", how="left")
left["name"] = left["name"].fillna("Unknown")

# Customers with no orders
outer = pd.merge(orders, customers, on="customer_id", how="outer")
no_orders = outer[outer["order_id"].isna()]
```""",
        "solution": """\
```python
import pandas as pd

orders    = pd.DataFrame({"order_id": [1,2,3,4], "customer_id": [10,10,20,30], "amount": [50,30,80,20]})
customers = pd.DataFrame({"customer_id": [10,20,40], "name": ["Alice","Bob","Charlie"]})

# 1. Inner join
inner = pd.merge(orders, customers, on="customer_id", how="inner")
print("Inner join:"); print(inner)

# 2. Left join + fill
left = pd.merge(orders, customers, on="customer_id", how="left")
left["name"] = left["name"].fillna("Unknown")
print("\\nLeft join:"); print(left)

# 3. Customers with no orders
outer = pd.merge(orders, customers, on="customer_id", how="outer")
no_orders = outer[outer["order_id"].isna()][["customer_id", "name"]]
print("\\nNo orders:"); print(no_orders)
```""",
    },
    {
        "id": "missing",
        "title": "Handling Missing Values",
        "category": "Pandas/EDA",
        "prompt": """\
**Task:** Clean the Titanic dataset's missing data.

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

1. Print null counts per column
2. Fill missing `age` with the **median**
3. Fill missing `embark_town` with the **mode**
4. Drop rows where `deck` is null
5. Verify — confirm `age` and `embark_town` have 0 nulls
""",
        "workspace_tip": (
            "`df.isnull().sum()` shows counts. "
            "`.fillna(df['col'].median())` fills numeric. "
            "`.fillna(df['col'].mode()[0])` fills categorical. "
            "`.dropna(subset=['col'])` drops only when that column is null."
        ),
        "hint": """\
```python
import seaborn as sns
df = sns.load_dataset("titanic")

print(df.isnull().sum())
df["age"]         = df["age"].fillna(df["age"].median())
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df = df.dropna(subset=["deck"])
```""",
        "solution": """\
```python
import seaborn as sns

df = sns.load_dataset("titanic")

# 1. Null counts
print("Nulls:\\n", df.isnull().sum())

# 2. Fill age with median
df["age"] = df["age"].fillna(df["age"].median())

# 3. Fill embark_town with mode
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# 4. Drop rows with null deck
df = df.dropna(subset=["deck"])

# 5. Verify
print("\\nAfter cleaning:")
print(df[["age", "embark_town", "deck"]].isnull().sum())
print(f"Rows remaining: {len(df)}")
```""",
    },
    {
        "id": "plotting",
        "title": "Plotting with Matplotlib & Seaborn",
        "category": "Pandas/EDA",
        "prompt": """\
**Task:** Build a 2×2 figure from the Titanic dataset.

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

1. `countplot` — survival count by sex
2. `histplot` — age distribution colored by survived (with KDE)
3. `boxplot` — fare by passenger class
4. `heatmap` — correlation matrix of numeric columns (annotated)
""",
        "workspace_tip": (
            "`fig, axes = plt.subplots(2, 2, figsize=(14, 10))` then pass `ax=axes[row, col]` "
            "to each seaborn call. Finish with `plt.tight_layout()`."
        ),
        "hint": """\
```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.countplot(data=df, x="sex", hue="survived",            ax=axes[0, 0])
sns.histplot( data=df, x="age", hue="survived", kde=True,  ax=axes[0, 1])
sns.boxplot(  data=df, x="pclass", y="fare",               ax=axes[1, 0])
sns.heatmap(df.select_dtypes("number").corr(), annot=True,  ax=axes[1, 1])

plt.tight_layout(); plt.show()
```""",
        "solution": """\
```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. countplot
sns.countplot(data=df, x="sex", hue="survived", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("Survival by Sex")
axes[0, 0].legend(title="Survived", labels=["No", "Yes"])

# 2. histplot
sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Age Distribution by Survival")

# 3. boxplot
sns.boxplot(data=df, x="pclass", y="fare", palette="coolwarm", ax=axes[1, 0])
axes[1, 0].set_title("Fare by Passenger Class")

# 4. heatmap
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Matrix")

plt.tight_layout(); plt.show()
```""",
    },
    {
        "id": "metrics",
        "title": "Simple ML Metrics",
        "category": "ML/Statistics",
        "prompt": """\
**Task:** Evaluate a classifier and interpret every metric.

Using the breast cancer dataset:

1. Train `LogisticRegression` (80/20 split, `random_state=42`)
2. Print accuracy, precision, recall, F1 (macro avg)
3. Plot the **confusion matrix** as a heatmap and explain each cell
4. Plot the **ROC curve** and print the AUC score
""",
        "workspace_tip": (
            "Import `classification_report`, `confusion_matrix`, `roc_curve`, `auc` "
            "from `sklearn.metrics`. Use `predict_proba(X_test)[:, 1]` for ROC."
        ),
        "hint": """\
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    bc.data, bc.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```""",
        "solution": """\
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    bc.data, bc.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds, target_names=bc.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=bc.target_names, yticklabels=bc.target_names)
plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.show()

# ROC Curve
proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.show()
```""",
    },
    {
        "id": "messy_files",
        "title": "Parsing Messy Files",
        "category": "Python Basics",
        "prompt": """\
**Task:** Parse a real-world messy CSV string.

```python
import pandas as pd
from io import StringIO   # ← Python 3: StringIO lives in 'io', not a top-level module

raw = \"\"\"
name,age,salary,joined
Alice, 30, 75000, 2020-01-15
Bob, , 55000, 2019-03-22
Carol,28,, 2021-07-01
Dave,35,90000,bad-date
  Eve ,25,62000,2022-11-30
\"\"\"
```

1. Read it with `pd.read_csv(StringIO(raw))`
2. Strip all string-column whitespace
3. Fill missing `age` with median; missing `salary` with `0`
4. Parse `joined` as datetime — turn bad dates into `NaT`
5. Print the cleaned DataFrame and its dtypes
""",
        "workspace_tip": (
            "⚠️ **Python 3:** `StringIO` is `from io import StringIO` — "
            "there is no top-level `StringIO` module (that was Python 2). "
            "`pd.to_datetime(df['col'], errors='coerce')` converts bad values to `NaT`. "
            "`df.select_dtypes('object')` grabs all string columns for bulk `.str.strip()`."
        ),
        "hint": """\
```python
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO(raw))
df.columns = df.columns.str.strip()
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()
df["age"]    = df["age"].fillna(df["age"].median())
df["salary"] = df["salary"].fillna(0)
df["joined"] = pd.to_datetime(df["joined"], errors="coerce")
```""",
        "solution": """\
```python
import pandas as pd
from io import StringIO

raw = \"\"\"
name,age,salary,joined
Alice, 30, 75000, 2020-01-15
Bob, , 55000, 2019-03-22
Carol,28,, 2021-07-01
Dave,35,90000,bad-date
  Eve ,25,62000,2022-11-30
\"\"\"

df = pd.read_csv(StringIO(raw))

# Strip column names and all string values
df.columns = df.columns.str.strip()
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# Coerce numerics first (they were read as mixed strings)
df["age"]    = pd.to_numeric(df["age"],    errors="coerce")
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

# Fill missing
df["age"]    = df["age"].fillna(df["age"].median())
df["salary"] = df["salary"].fillna(0)

# Parse dates, bad → NaT
df["joined"] = pd.to_datetime(df["joined"], errors="coerce")

print(df)
print("\\nDtypes:\\n", df.dtypes)
print("\\nNulls:\\n", df.isnull().sum())
```""",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# HELIX INTERVIEW QUESTIONS  (bioinformatics + algorithms + data structures)
# ──────────────────────────────────────────────────────────────────────────────
HELIX_QUESTIONS = [
    # ── Strings / Bioinformatics ─────────────────────────────────────────────
    {
        "id": "rev_complement",
        "title": "Reverse Complement DNA",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Given a DNA string, return its reverse complement.

Rules: A↔T, C↔G — complement each base, then reverse the whole string.

```python
seq = "ATCGGCTA"
# Expected: "TAGCCGAT"
```

Implement `rev_complement(seq: str) -> str` with no imports.
""",
        "workspace_tip": (
            "Build a complement dict `{'A':'T','T':'A','C':'G','G':'C'}`, "
            "map each base, then reverse with `[::-1]` or `reversed()`."
        ),
        "hint": """\
```python
def rev_complement(seq):
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[b] for b in reversed(seq))
```""",
        "solution": """\
```python
def rev_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[b] for b in reversed(seq))

print(rev_complement("ATCGGCTA"))    # TAGCCGAT
print(rev_complement("AAAAACCCCC"))  # GGGGGTTTT
print(rev_complement("GCGC"))        # GCGC
```""",
    },
    {
        "id": "count_kmers",
        "title": "Count k-mers",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Count all k-mers (substrings of length k) in a DNA sequence.

```python
seq = "ATCGATCG"
k   = 3
# Expected (sorted by count desc): {'ATC': 2, 'TCG': 2, 'CGA': 1, 'GAT': 1}
```

Return a `dict` sorted by count descending.
""",
        "workspace_tip": (
            "Slide a window of size k: `seq[i:i+k]` for `i in range(len(seq)-k+1)`. "
            "Use `collections.Counter`, then sort by `-count`."
        ),
        "hint": """\
```python
from collections import Counter

def count_kmers(seq, k):
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
```""",
        "solution": """\
```python
from collections import Counter

def count_kmers(seq: str, k: int) -> dict:
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

print(count_kmers("ATCGATCG", 3))
print(count_kmers("AAABBBCCC", 2))
```""",
    },
    {
        "id": "find_mutations",
        "title": "Find Mutations (ref vs query)",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Given a reference and a query string of equal length, find all positions where they differ.

Return a list of `(position, ref_base, query_base)` tuples (0-indexed).

```python
ref   = "ATCGATCG"
query = "ATCTATGG"
# Expected: [(3, 'G', 'T'), (6, 'C', 'G')]
```
""",
        "workspace_tip": (
            "Use `enumerate(zip(ref, query))` to iterate both simultaneously. "
            "Filter where `r != q`."
        ),
        "hint": """\
```python
def find_mutations(ref, query):
    return [(i, r, q) for i, (r, q) in enumerate(zip(ref, query)) if r != q]
```""",
        "solution": """\
```python
def find_mutations(ref: str, query: str) -> list:
    return [(i, r, q) for i, (r, q) in enumerate(zip(ref, query)) if r != q]

ref   = "ATCGATCG"
query = "ATCTATGG"
print(find_mutations(ref, query))  # [(3, 'G', 'T'), (6, 'C', 'G')]

print(find_mutations("AAAA", "AATA"))  # [(2, 'A', 'T')]
```""",
    },
    {
        "id": "parse_fasta",
        "title": "Parse FASTA Text",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Parse a FASTA-formatted string into `{header: sequence}`.

```python
fasta = \"""
>gene1
ATCGATCG
GCTAGCTA
>gene2
TTTTAAAA
\"""
# Expected: {'gene1': 'ATCGATCGGCTAGCTA', 'gene2': 'TTTTAAAA'}
```

Strip `>` from headers; concatenate multi-line sequences.
""",
        "workspace_tip": (
            "Iterate lines. Lines starting with `>` open a new record. "
            "All other lines append to the current sequence."
        ),
        "hint": """\
```python
def parse_fasta(text):
    records, cur = {}, None
    for line in text.strip().splitlines():
        if line.startswith(">"):
            cur = line[1:].strip()
            records[cur] = ""
        else:
            records[cur] += line.strip()
    return records
```""",
        "solution": """\
```python
def parse_fasta(text: str) -> dict:
    records, cur = {}, None
    for line in text.strip().splitlines():
        if line.startswith(">"):
            cur = line[1:].strip()
            records[cur] = ""
        else:
            records[cur] += line.strip()
    return records

fasta = \"""
>gene1
ATCGATCG
GCTAGCTA
>gene2
TTTTAAAA
\"""
print(parse_fasta(fasta))
# {'gene1': 'ATCGATCGGCTAGCTA', 'gene2': 'TTTTAAAA'}
```""",
    },
    # ── Hash Maps / Dicts ────────────────────────────────────────────────────
    {
        "id": "count_variants_by_gene",
        "title": "Count Variants by Gene",
        "category": "Hash Maps",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Count how many variants each gene has.

```python
variants = [
    {"gene": "BRCA1", "var": "c.123A>T"},
    {"gene": "TP53",  "var": "c.456C>G"},
    {"gene": "BRCA1", "var": "c.789G>A"},
    {"gene": "EGFR",  "var": "c.111T>C"},
    {"gene": "TP53",  "var": "c.222A>G"},
    {"gene": "TP53",  "var": "c.333G>T"},
]
# Expected: {'TP53': 3, 'BRCA1': 2, 'EGFR': 1}
```

Solve it two ways: with `Counter` AND with a plain `dict`.
""",
        "workspace_tip": (
            "`Counter(v['gene'] for v in variants)` is the fastest. "
            "For plain dict: `d[key] = d.get(key, 0) + 1`."
        ),
        "hint": """\
```python
from collections import Counter

counts = Counter(v["gene"] for v in variants)
print(dict(counts))
```""",
        "solution": """\
```python
from collections import Counter

variants = [
    {"gene": "BRCA1", "var": "c.123A>T"},
    {"gene": "TP53",  "var": "c.456C>G"},
    {"gene": "BRCA1", "var": "c.789G>A"},
    {"gene": "EGFR",  "var": "c.111T>C"},
    {"gene": "TP53",  "var": "c.222A>G"},
    {"gene": "TP53",  "var": "c.333G>T"},
]

# Counter approach
counts = Counter(v["gene"] for v in variants)
print(dict(counts.most_common()))

# Plain dict approach
plain = {}
for v in variants:
    plain[v["gene"]] = plain.get(v["gene"], 0) + 1
print(plain)
```""",
    },
    {
        "id": "group_by_sample",
        "title": "Group Records by Sample ID",
        "category": "Hash Maps",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Group variant records by `sample_id`.

```python
variants = [
    {"sample_id": "S1", "gene": "BRCA1", "vaf": 0.35},
    {"sample_id": "S2", "gene": "TP53",  "vaf": 0.50},
    {"sample_id": "S1", "gene": "EGFR",  "vaf": 0.12},
    {"sample_id": "S3", "gene": "BRCA2", "vaf": 0.80},
    {"sample_id": "S2", "gene": "KRAS",  "vaf": 0.45},
]
```

Return a `dict` mapping each `sample_id` → list of its records.
""",
        "workspace_tip": (
            "Use `collections.defaultdict(list)` and `.append()`. "
            "Equivalent with plain dict: `d.setdefault(key, []).append(v)`."
        ),
        "hint": """\
```python
from collections import defaultdict

grouped = defaultdict(list)
for v in variants:
    grouped[v["sample_id"]].append(v)
print(dict(grouped))
```""",
        "solution": """\
```python
from collections import defaultdict

variants = [
    {"sample_id": "S1", "gene": "BRCA1", "vaf": 0.35},
    {"sample_id": "S2", "gene": "TP53",  "vaf": 0.50},
    {"sample_id": "S1", "gene": "EGFR",  "vaf": 0.12},
    {"sample_id": "S3", "gene": "BRCA2", "vaf": 0.80},
    {"sample_id": "S2", "gene": "KRAS",  "vaf": 0.45},
]

grouped = defaultdict(list)
for v in variants:
    grouped[v["sample_id"]].append(v)

for sid, records in grouped.items():
    print(f"{sid}: {[r['gene'] for r in records]}")
```""",
    },
    {
        "id": "deduplicate_variants",
        "title": "Deduplicate Variant Records",
        "category": "Hash Maps",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Remove duplicate variants. A duplicate shares the same `(chrom, pos, ref, alt)`. Keep the first occurrence.

```python
variants = [
    {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "T", "sample": "S1"},
    {"chrom": "chr1", "pos": 200, "ref": "G", "alt": "C", "sample": "S2"},
    {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "T", "sample": "S3"},  # dup
    {"chrom": "chr2", "pos": 300, "ref": "C", "alt": "A", "sample": "S1"},
]
# Expected: 3 unique records
```
""",
        "workspace_tip": (
            "Maintain a `seen` set of `(chrom, pos, ref, alt)` tuples. "
            "Skip a record if its key is already in `seen`."
        ),
        "hint": """\
```python
def dedup(variants):
    seen, result = set(), []
    for v in variants:
        key = (v["chrom"], v["pos"], v["ref"], v["alt"])
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result
```""",
        "solution": """\
```python
def dedup(variants):
    seen, result = set(), []
    for v in variants:
        key = (v["chrom"], v["pos"], v["ref"], v["alt"])
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result

variants = [
    {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "T", "sample": "S1"},
    {"chrom": "chr1", "pos": 200, "ref": "G", "alt": "C", "sample": "S2"},
    {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "T", "sample": "S3"},
    {"chrom": "chr2", "pos": 300, "ref": "C", "alt": "A", "sample": "S1"},
]
result = dedup(variants)
print(f"{len(result)} unique records")
for r in result:
    print(r)
```""",
    },
    {
        "id": "top_n_frequent",
        "title": "Top N Frequent Items",
        "category": "Hash Maps",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Given a list of gene names with repeats, return the top N most frequent.

```python
genes = ["TP53","BRCA1","TP53","EGFR","TP53","BRCA1","KRAS","EGFR","EGFR","KRAS","KRAS","KRAS"]
n = 2
# Expected: [('KRAS', 4), ('TP53', 3)]
```
""",
        "workspace_tip": (
            "`Counter.most_common(n)` returns `[(item, count), ...]` sorted by count desc. "
            "No sorting needed — it's built in."
        ),
        "hint": """\
```python
from collections import Counter

def top_n(items, n):
    return Counter(items).most_common(n)
```""",
        "solution": """\
```python
from collections import Counter

def top_n(items, n):
    return Counter(items).most_common(n)

genes = ["TP53","BRCA1","TP53","EGFR","TP53","BRCA1","KRAS","EGFR","EGFR","KRAS","KRAS","KRAS"]
print(top_n(genes, 2))   # [('KRAS', 4), ('TP53', 3)]
print(top_n(genes, 3))   # [('KRAS', 4), ('TP53', 3), ('EGFR', 3)]
```""",
    },
    # ── Lists / Sorting ──────────────────────────────────────────────────────
    {
        "id": "sort_by_vaf",
        "title": "Sort Variants by VAF",
        "category": "Lists/Sorting",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Sort a list of variant dicts by `vaf` (variant allele frequency) descending.

```python
variants = [
    {"gene": "TP53",  "vaf": 0.45},
    {"gene": "BRCA1", "vaf": 0.12},
    {"gene": "EGFR",  "vaf": 0.78},
    {"gene": "KRAS",  "vaf": 0.33},
]
# Expected order: EGFR(0.78) → TP53(0.45) → KRAS(0.33) → BRCA1(0.12)
```
""",
        "workspace_tip": (
            "`sorted(variants, key=lambda v: v['vaf'], reverse=True)` returns a new list. "
            "`variants.sort(key=...)` sorts in place."
        ),
        "hint": """\
```python
sorted_v = sorted(variants, key=lambda v: v["vaf"], reverse=True)
```""",
        "solution": """\
```python
variants = [
    {"gene": "TP53",  "vaf": 0.45},
    {"gene": "BRCA1", "vaf": 0.12},
    {"gene": "EGFR",  "vaf": 0.78},
    {"gene": "KRAS",  "vaf": 0.33},
]

sorted_v = sorted(variants, key=lambda v: v["vaf"], reverse=True)
for v in sorted_v:
    print(f"{v['gene']}: {v['vaf']:.2f}")
```""",
    },
    {
        "id": "filter_variants",
        "title": "Filter Variants (depth, VAF, consequence)",
        "category": "Lists/Sorting",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Keep only variants where `depth >= 20` AND `vaf >= 0.1` AND `consequence == "missense"`.

```python
variants = [
    {"gene": "TP53",  "depth": 50, "vaf": 0.35, "consequence": "missense"},
    {"gene": "BRCA1", "depth": 10, "vaf": 0.20, "consequence": "missense"},    # depth too low
    {"gene": "EGFR",  "depth": 30, "vaf": 0.05, "consequence": "missense"},    # vaf too low
    {"gene": "KRAS",  "depth": 25, "vaf": 0.45, "consequence": "synonymous"},  # wrong csq
    {"gene": "BRCA2", "depth": 40, "vaf": 0.60, "consequence": "missense"},
]
# Expected: TP53, BRCA2
```
""",
        "workspace_tip": (
            "A list comprehension with multiple `and` conditions is cleaner than nested `filter()`. "
            "Consider writing a reusable `passes_qc(v, min_depth, min_vaf)` function."
        ),
        "hint": """\
```python
filtered = [
    v for v in variants
    if v["depth"] >= 20 and v["vaf"] >= 0.1 and v["consequence"] == "missense"
]
```""",
        "solution": """\
```python
variants = [
    {"gene": "TP53",  "depth": 50, "vaf": 0.35, "consequence": "missense"},
    {"gene": "BRCA1", "depth": 10, "vaf": 0.20, "consequence": "missense"},
    {"gene": "EGFR",  "depth": 30, "vaf": 0.05, "consequence": "missense"},
    {"gene": "KRAS",  "depth": 25, "vaf": 0.45, "consequence": "synonymous"},
    {"gene": "BRCA2", "depth": 40, "vaf": 0.60, "consequence": "missense"},
]

filtered = [
    v for v in variants
    if v["depth"] >= 20 and v["vaf"] >= 0.1 and v["consequence"] == "missense"
]

for v in filtered:
    print(f"{v['gene']}: depth={v['depth']}, vaf={v['vaf']}")
```""",
    },
    {
        "id": "merge_intervals",
        "title": "Merge Genomic Intervals",
        "category": "Lists/Sorting",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Merge overlapping genomic intervals.

```python
intervals = [[1, 3], [2, 6], [8, 10], [15, 18], [9, 12]]
# Expected: [[1, 6], [8, 12], [15, 18]]
```

Sort by start position, then greedily merge.
""",
        "workspace_tip": (
            "Sort by `x[0]`. Walk through: if current start ≤ last merged end, "
            "extend with `max(last_end, current_end)`. Otherwise append new interval."
        ),
        "hint": """\
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```""",
        "solution": """\
```python
def merge(intervals: list) -> list:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

print(merge([[1,3],[2,6],[8,10],[15,18],[9,12]]))  # [[1,6],[8,12],[15,18]]
print(merge([[1,4],[4,5]]))                         # [[1,5]]
print(merge([[1,4],[2,3]]))                         # [[1,4]]
```""",
    },
    # ── Algorithms ───────────────────────────────────────────────────────────
    {
        "id": "two_sum",
        "title": "Two Sum",
        "category": "Algorithms",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Given a list of numbers and a target, return the indices of the two numbers that add up to the target. Exactly one solution exists.

```python
nums, target = [2, 7, 11, 15], 9   → [0, 1]
nums, target = [3, 2, 4],       6   → [1, 2]
```

O(n) solution using a hash map.
""",
        "workspace_tip": (
            "Store `{value: index}` as you walk forward. "
            "For each number, check if `target - num` is already in the map before inserting."
        ),
        "hint": """\
```python
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```""",
        "solution": """\
```python
def two_sum(nums: list, target: int) -> list:
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
    return []

print(two_sum([2, 7, 11, 15], 9))   # [0, 1]
print(two_sum([3, 2, 4], 6))        # [1, 2]
print(two_sum([3, 3], 6))           # [0, 1]
```""",
    },
    {
        "id": "longest_no_repeat",
        "title": "Longest Substring Without Repeating",
        "category": "Algorithms",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Find the length of the longest substring without repeating characters.

```python
"abcabcbb"  → 3   ("abc")
"bbbbb"     → 1   ("b")
"pwwkew"    → 3   ("wke")
```

O(n) sliding window.
""",
        "workspace_tip": (
            "Maintain a `left` pointer and a `{char: last_index}` map. "
            "When a repeat is found at `right`, move `left` to `seen[c] + 1` (only if that index ≥ left)."
        ),
        "hint": """\
```python
def length_of_longest(s):
    seen, left, best = {}, 0, 0
    for right, c in enumerate(s):
        if c in seen and seen[c] >= left:
            left = seen[c] + 1
        seen[c] = right
        best = max(best, right - left + 1)
    return best
```""",
        "solution": """\
```python
def length_of_longest(s: str) -> int:
    seen, left, best = {}, 0, 0
    for right, c in enumerate(s):
        if c in seen and seen[c] >= left:
            left = seen[c] + 1
        seen[c] = right
        best = max(best, right - left + 1)
    return best

print(length_of_longest("abcabcbb"))  # 3
print(length_of_longest("bbbbb"))     # 1
print(length_of_longest("pwwkew"))    # 3
print(length_of_longest(""))          # 0
```""",
    },
    # ── Genomic File Formats ──────────────────────────────────────────────────
    {
        "id": "parse_vcf",
        "title": "Parse VCF Format",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Parse a VCF (Variant Call Format) string.

VCF structure:
- `##` lines → meta-information (skip)
- `#CHROM ...` line → column headers (strip the `#`)
- Data lines → tab-separated variant records

```
##fileformat=VCFv4.2
#CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO
chr1    100  .   A    T    50    PASS    DP=30;VAF=0.35;GENE=TP53
chr1    200  .   G    C    30    LowQual DP=10;VAF=0.20;GENE=BRCA1
chr2    300  .   C    A    80    PASS    DP=45;VAF=0.60;GENE=EGFR
```

1. Parse into a list of dicts using the `#CHROM` line as headers
2. Filter to `FILTER == "PASS"` only
3. For each passing record, parse `INFO` into a nested `{key: value}` dict
4. Print the gene and VAF for each passing variant
""",
        "workspace_tip": (
            "Split INFO on `;`, then split each token on `=` to build a nested dict. "
            "The header line starts with `#` — use `.lstrip('#')` before `.split('\\t')`."
        ),
        "hint": """\
```python
def parse_info(info_str):
    return dict(kv.split("=") for kv in info_str.split(";"))

def parse_vcf(text):
    headers, records = None, []
    for line in text.strip().splitlines():
        if line.startswith("##"):
            continue
        elif line.startswith("#"):
            headers = line.lstrip("#").split("\t")
        else:
            record = dict(zip(headers, line.split("\t")))
            record["INFO"] = parse_info(record["INFO"])
            records.append(record)
    return [r for r in records if r["FILTER"] == "PASS"]
```""",
        "solution": """\
```python
def parse_info(info_str):
    return dict(kv.split("=") for kv in info_str.split(";"))

def parse_vcf(text):
    headers, records = None, []
    for line in text.strip().splitlines():
        if line.startswith("##"):
            continue
        elif line.startswith("#"):
            headers = line.lstrip("#").split("\t")
        else:
            record = dict(zip(headers, line.split("\t")))
            record["INFO"] = parse_info(record["INFO"])
            records.append(record)
    return [r for r in records if r["FILTER"] == "PASS"]

vcf_lines = [
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    "chr1\t100\t.\tA\tT\t50\tPASS\tDP=30;VAF=0.35;GENE=TP53",
    "chr1\t200\t.\tG\tC\t30\tLowQual\tDP=10;VAF=0.20;GENE=BRCA1",
    "chr2\t300\t.\tC\tA\t80\tPASS\tDP=45;VAF=0.60;GENE=EGFR",
]
vcf_text = "\n".join(vcf_lines)

for v in parse_vcf(vcf_text):
    print(f"{v['CHROM']}:{v['POS']}  gene={v['INFO']['GENE']}  VAF={v['INFO']['VAF']}")
```""",
    },
    {
        "id": "parse_fastq",
        "title": "Parse FASTQ Reads",
        "category": "Strings",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Parse a FASTQ string and compute per-read quality statistics.

FASTQ records are exactly 4 lines:
```
@read_name
SEQUENCE
+
QUALITY  (Phred+33 ASCII encoding)
```

Phred score formula: `ord(char) - 33`

1. Parse all reads into `(name, seq, qual_scores)` — `qual_scores` is a list of ints
2. Compute mean quality per read
3. Print each read's name, length, and mean quality
4. Flag reads with mean quality **< 20** as LOW
""",
        "workspace_tip": (
            "Step through lines in chunks of 4: `for i in range(0, len(lines), 4)`. "
            "Convert quality string: `[ord(c) - 33 for c in qual_str]`."
        ),
        "hint": """\
```python
def parse_fastq(text):
    lines = text.strip().splitlines()
    reads = []
    for i in range(0, len(lines), 4):
        name = lines[i][1:]
        seq  = lines[i+1]
        qual = [ord(c) - 33 for c in lines[i+3]]
        reads.append((name, seq, qual))
    return reads
```""",
        "solution": """\
```python
def parse_fastq(text):
    lines = text.strip().splitlines()
    reads = []
    for i in range(0, len(lines), 4):
        name = lines[i][1:]
        seq  = lines[i+1]
        qual = [ord(c) - 33 for c in lines[i+3]]
        reads.append((name, seq, qual))
    return reads

fastq = "\n".join([
    "@read1", "ATCGATCGAT", "+", "IIIIIIIII!",
    "@read2", "GCTAGCTAGC", "+", "!!!!!!!!!!",
    "@read3", "TTTTAAAACG", "+", "IIIIIIIIII",
])

for name, seq, qual in parse_fastq(fastq):
    mean_q = sum(qual) / len(qual)
    flag   = "LOW" if mean_q < 20 else "OK"
    print(f"{name}: len={len(seq)}  mean_qual={mean_q:.1f}  [{flag}]")
```""",
    },
    {
        "id": "bed_operations",
        "title": "BED Interval Operations",
        "category": "Lists/Sorting",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Work with BED-format genomic intervals.

BED columns: `chrom`, `start`, `end`, `name` (0-based, half-open `[start, end)`)

```
chr1  100  300  exon1
chr1  250  450  exon2
chr1  600  800  exon3
chr2  100  200  exon4
```

1. **Find overlapping pairs** on the same chrom (`a.start < b.end and b.start < a.end`)
2. **Merge** overlapping intervals per chrom (sort by chrom+start, greedy merge)
3. **Total coverage** = sum of `end - start` across all merged intervals
""",
        "workspace_tip": (
            "Two half-open intervals [a,b) and [c,d) overlap iff `a < d and c < b`. "
            "Sort by `(chrom, start)` first; then walk and extend or append."
        ),
        "hint": """\
```python
def find_overlaps(bed):
    return [
        (bed[i], bed[j])
        for i in range(len(bed))
        for j in range(i+1, len(bed))
        if bed[i][0] == bed[j][0] and bed[i][1] < bed[j][2] and bed[j][1] < bed[i][2]
    ]

def merge_bed(intervals):
    ivs = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(ivs[0][:3])]
    for chrom, start, end, *_ in ivs[1:]:
        if chrom == merged[-1][0] and start < merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], end)
        else:
            merged.append([chrom, start, end])
    return merged
```""",
        "solution": """\
```python
bed = [
    ("chr1", 100, 300, "exon1"),
    ("chr1", 250, 450, "exon2"),
    ("chr1", 600, 800, "exon3"),
    ("chr2", 100, 200, "exon4"),
]

# 1. Overlapping pairs
def find_overlaps(intervals):
    return [
        (intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i+1, len(intervals))
        if intervals[i][0] == intervals[j][0]
        and intervals[i][1] < intervals[j][2]
        and intervals[j][1] < intervals[i][2]
    ]

for a, b in find_overlaps(bed):
    print(f"Overlap: {a[3]} [{a[1]}-{a[2]}) & {b[3]} [{b[1]}-{b[2]})")

# 2. Merge
def merge_bed(intervals):
    ivs = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(ivs[0][:3])]
    for chrom, start, end, *_ in ivs[1:]:
        if chrom == merged[-1][0] and start < merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], end)
        else:
            merged.append([chrom, start, end])
    return merged

merged = merge_bed(bed)
print("\nMerged:", merged)

# 3. Coverage
coverage = sum(end - start for _, start, end in merged)
print(f"Total coverage: {coverage} bp")
```""",
    },
    {
        "id": "pandas_variants",
        "title": "Pandas Variant Table Analysis",
        "category": "Pandas/EDA",
        "bucket": "Bioinformatics Engineer",
        "prompt": """\
**Task:** Analyze a variant table with pandas.

```python
import pandas as pd

df = pd.DataFrame({
    "gene":        ["TP53","BRCA1","TP53","EGFR","KRAS","BRCA1","TP53"],
    "sample":      ["S1",  "S1",   "S2",  "S2",  "S1",  "S2",   "S1"],
    "vaf":         [0.45,  0.12,   0.33,  0.78,  0.55,  0.60,   0.28],
    "depth":       [50,    10,     30,    45,    25,    40,     60],
    "consequence": ["missense","missense","synonymous","missense","missense","missense","missense"],
    "filter":      ["PASS","LowDepth","PASS","PASS","PASS","PASS","PASS"],
})
```

1. Filter to `filter == "PASS"` AND `depth >= 20` AND `consequence == "missense"`
2. Per gene (filtered): mean VAF and variant count — sort by count desc
3. Which sample has the most variants after filtering?
4. Pivot table: rows = genes, columns = samples, values = variant count
""",
        "workspace_tip": (
            "Use `&` (not `and`) between DataFrame filter conditions. "
            "`groupby().agg(count=(...,'count'), mean_vaf=(...,'mean'))` for step 2. "
            "`pd.crosstab(df['gene'], df['sample'])` for the pivot."
        ),
        "hint": """\
```python
mask = (df["filter"] == "PASS") & (df["depth"] >= 20) & (df["consequence"] == "missense")
filt = df[mask]

gene_stats = filt.groupby("gene").agg(
    count=("gene", "count"),
    mean_vaf=("vaf", "mean"),
).sort_values("count", ascending=False)

top_sample = filt["sample"].value_counts().idxmax()
pivot = pd.crosstab(filt["gene"], filt["sample"])
```""",
        "solution": """\
```python
import pandas as pd

df = pd.DataFrame({
    "gene":        ["TP53","BRCA1","TP53","EGFR","KRAS","BRCA1","TP53"],
    "sample":      ["S1","S1","S2","S2","S1","S2","S1"],
    "vaf":         [0.45, 0.12, 0.33, 0.78, 0.55, 0.60, 0.28],
    "depth":       [50, 10, 30, 45, 25, 40, 60],
    "consequence": ["missense","missense","synonymous","missense","missense","missense","missense"],
    "filter":      ["PASS","LowDepth","PASS","PASS","PASS","PASS","PASS"],
})

# 1. Filter
mask = (df["filter"] == "PASS") & (df["depth"] >= 20) & (df["consequence"] == "missense")
filt = df[mask]
print(f"Passing variants: {len(filt)}")
print(filt[["gene","sample","vaf","depth"]])

# 2. Per-gene stats
gene_stats = filt.groupby("gene").agg(
    count=("gene", "count"),
    mean_vaf=("vaf", "mean"),
).sort_values("count", ascending=False).round(2)
print("\nPer-gene stats:")
print(gene_stats)

# 3. Top sample
top = filt["sample"].value_counts()
print(f"\nMost variants: {top.idxmax()} ({top.max()} variants)")

# 4. Pivot
pivot = pd.crosstab(filt["gene"], filt["sample"])
print("\nGene x Sample pivot:")
print(pivot)
```""",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# CLINICAL QUESTIONS  (pandas from memory · sklearn · survival analysis)
# ──────────────────────────────────────────────────────────────────────────────
CLINICAL_QUESTIONS = [
    # ── Pandas from Memory ────────────────────────────────────────────────────
    {
        "id": "clinical_eda",
        "title": "Clinical DataFrame EDA",
        "category": "Pandas/EDA",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Perform a quick EDA on a clinical trial DataFrame.

Columns: `patient_id`, `age`, `sex`, `treatment_arm`, `os_months`, `event`.

From memory, write code to:
1. Print shape, dtypes, and null counts
2. Describe all numeric columns
3. Value counts for `sex` and `treatment_arm`
4. Event rate (fraction where `event == 1`)
""",
        "workspace_tip": (
            "5-line EDA sequence: `.shape`, `.dtypes`, `.isnull().sum()`, `.describe()`, "
            "`.value_counts()`. Event rate = `df['event'].mean()`."
        ),
        "hint": """\
```python
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df['sex'].value_counts())
print(df['treatment_arm'].value_counts())
print(f"Event rate: {df['event'].mean():.1%}")
```""",
        "solution": """\
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 120
df = pd.DataFrame({
    'patient_id':    range(1, n+1),
    'age':           np.random.randint(40, 80, n).astype(float),
    'sex':           np.random.choice(['M', 'F'], n),
    'treatment_arm': np.random.choice(['A', 'B'], n),
    'os_months':     np.random.exponential(18, n).clip(1, 60).round(1),
    'event':         np.random.binomial(1, 0.6, n),
})
df.loc[[5, 20, 47], 'age'] = np.nan

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNulls:\n", df.isnull().sum())
print("\nDescribe:\n", df.describe().round(2))
print("\nSex:\n", df['sex'].value_counts())
print("\nArm:\n", df['treatment_arm'].value_counts())
print(f"\nEvent rate: {df['event'].mean():.1%}")
```""",
    },
    {
        "id": "clinical_groupby",
        "title": "GroupBy Clinical Outcomes",
        "category": "Pandas/EDA",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Summarize a clinical dataset by treatment arm.

From memory, using named `.agg()`:
1. GroupBy `treatment_arm`: patient count, median OS, mean age, event rate
2. GroupBy `event` (0 vs 1): mean age and median OS
3. Sort arm table by median OS descending
4. What % of each arm had an event?
""",
        "workspace_tip": (
            "`groupby().agg(name=('col', 'func'))` for named aggregations. "
            "`.mul(100).round(1)` to convert event rate to a readable percentage."
        ),
        "hint": """\
```python
arm_stats = df.groupby('treatment_arm').agg(
    n=('patient_id', 'count'),
    median_os=('os_months', 'median'),
    mean_age=('age', 'mean'),
    event_rate=('event', 'mean'),
).sort_values('median_os', ascending=False).round(2)
print(arm_stats)
```""",
        "solution": """\
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 120
df = pd.DataFrame({
    'patient_id':    range(1, n+1),
    'age':           np.random.randint(40, 80, n),
    'treatment_arm': np.random.choice(['A', 'B'], n),
    'os_months':     np.random.exponential(18, n).clip(1, 60).round(1),
    'event':         np.random.binomial(1, 0.6, n),
})

arm_stats = df.groupby('treatment_arm').agg(
    n=('patient_id', 'count'),
    median_os=('os_months', 'median'),
    mean_age=('age', 'mean'),
    event_rate=('event', 'mean'),
).sort_values('median_os', ascending=False).round(2)
print("By arm:\n", arm_stats)

event_stats = df.groupby('event').agg(
    mean_age=('age', 'mean'),
    median_os=('os_months', 'median'),
).round(2)
print("\nBy event status:\n", event_stats)

pct = df.groupby('treatment_arm')['event'].mean().mul(100).round(1)
print("\nEvent % by arm:\n", pct)
```""",
    },
    {
        "id": "clinical_merge",
        "title": "Merge Patient Tables",
        "category": "Pandas/EDA",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Join two clinical tables.

```
patients  → patient_id, age, sex          (5 rows)
outcomes  → patient_id, os_months, event  (4 rows — patient 4 missing)
```

1. Inner join — how many rows survive?
2. Left join — which patient is missing? Fill `event` nulls with 0
3. Event rate from the left-joined (complete) table
""",
        "workspace_tip": (
            "`pd.merge(left, right, on='key', how='inner'/'left')`. "
            "After left join: `merged[merged['os_months'].isna()]` finds unmatched rows."
        ),
        "hint": """\
```python
inner = pd.merge(patients, outcomes, on='patient_id', how='inner')
left  = pd.merge(patients, outcomes, on='patient_id', how='left')
left['event'] = left['event'].fillna(0).astype(int)
print(left)
```""",
        "solution": """\
```python
import pandas as pd

patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5],
    'age': [55, 62, 48, 71, 39],
    'sex': ['M', 'F', 'M', 'F', 'M'],
})
outcomes = pd.DataFrame({
    'patient_id': [1, 2, 3, 5],
    'os_months':  [24.0, 8.5, 36.2, 12.1],
    'event':      [1, 1, 0, 1],
})

inner = pd.merge(patients, outcomes, on='patient_id', how='inner')
print(f"Inner join: {len(inner)} rows\n", inner)

left = pd.merge(patients, outcomes, on='patient_id', how='left')
missing = left[left['os_months'].isna()]
print(f"\nMissing outcomes: patient_id = {missing['patient_id'].tolist()}")
left['event'] = left['event'].fillna(0).astype(int)
print(left)

print(f"\nEvent rate (left join, censored=0): {left['event'].mean():.1%}")
```""",
    },
    {
        "id": "clinical_derive",
        "title": "Derived Columns on Clinical Data",
        "category": "Pandas/EDA",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Create derived columns on a clinical DataFrame.

From memory:
1. `os_years` = `os_months / 12`, rounded to 2 d.p.
2. `age_group` = `<50`, `50-65`, `>65` via `pd.cut`
3. `long_survivor` = True if `os_months > 24` **AND** `event == 1`
4. BMI from `weight_kg` / `height_m²`; flag `"obese"` (BMI ≥ 30) vs `"normal"` with `np.where`
""",
        "workspace_tip": (
            "`pd.cut(df['age'], bins=[0,50,65,120], labels=['<50','50-65','>65'])`. "
            "Boolean masks combine with `&` (not `and`). "
            "`np.where(condition, 'true_val', 'false_val')` for string flags."
        ),
        "hint": """\
```python
df['os_years']      = (df['os_months'] / 12).round(2)
df['age_group']     = pd.cut(df['age'], bins=[0,50,65,120], labels=['<50','50-65','>65'])
df['long_survivor'] = (df['os_months'] > 24) & (df['event'] == 1)
df['bmi']           = df['weight_kg'] / df['height_m']**2
df['bmi_flag']      = np.where(df['bmi'] >= 30, 'obese', 'normal')
```""",
        "solution": """\
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 60
df = pd.DataFrame({
    'age':       np.random.randint(35, 80, n),
    'os_months': np.random.exponential(18, n).clip(1, 60).round(1),
    'event':     np.random.binomial(1, 0.6, n),
    'weight_kg': np.random.normal(80, 15, n).clip(45, 140).round(1),
    'height_m':  np.random.normal(1.70, 0.10, n).clip(1.45, 2.00).round(2),
})

df['os_years']      = (df['os_months'] / 12).round(2)
df['age_group']     = pd.cut(df['age'], bins=[0, 50, 65, 120], labels=['<50', '50-65', '>65'])
df['long_survivor'] = (df['os_months'] > 24) & (df['event'] == 1)
df['bmi']           = (df['weight_kg'] / df['height_m']**2).round(1)
df['bmi_flag']      = np.where(df['bmi'] >= 30, 'obese', 'normal')

print(df[['age','age_group','os_months','os_years','long_survivor','bmi','bmi_flag']].head(10))
print("\nAge groups:\n", df['age_group'].value_counts())
print(f"Long survivors: {df['long_survivor'].sum()} / {len(df)}")
print("BMI flags:\n", df['bmi_flag'].value_counts())
```""",
    },
    # ── scikit-learn ─────────────────────────────────────────────────────────
    {
        "id": "sklearn_logreg_clinical",
        "title": "Logistic Regression: Fit → Predict → Report",
        "category": "ML/Statistics",
        "bucket": "Clinical",
        "prompt": """\
**Task:** End-to-end binary classification from memory.

Features: `age`, `tumor_size`, `treatment` (0/1). Target: `event`.

1. `train_test_split` — 80/20, stratified, `random_state=42`
2. `StandardScaler` — fit on train only, transform both splits
3. `LogisticRegression(max_iter=1000)`
4. Accuracy + `classification_report`
""",
        "workspace_tip": (
            "Fit the scaler on `X_train` only (`fit_transform`), then `transform(X_test)`. "
            "Fitting on the full dataset leaks test information."
        ),
        "hint": """\
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc  = sc.transform(X_te)
model = LogisticRegression(max_iter=1000).fit(X_tr_sc, y_tr)
print(accuracy_score(y_te, model.predict(X_te_sc)))
print(classification_report(y_te, model.predict(X_te_sc)))
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':        np.random.randint(40, 80, n),
    'tumor_size': np.random.normal(3.5, 1.5, n).clip(0.5, 10),
    'treatment':  np.random.binomial(1, 0.5, n),
    'event':      np.random.binomial(1, 0.55, n),
})
X = df[['age', 'tumor_size', 'treatment']].values
y = df['event'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc  = sc.transform(X_te)

model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_sc, y_tr)
y_pred = model.predict(X_te_sc)

print(f"Accuracy: {accuracy_score(y_te, y_pred):.3f}")
print(classification_report(y_te, y_pred, target_names=['censored', 'event']))
```""",
    },
    {
        "id": "sklearn_roc_auc",
        "title": "ROC Curve + AUC",
        "category": "ML/Statistics",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Plot a ROC curve and compute AUC from memory.

After fitting a logistic regression model:
1. `predict_proba(X_test)[:, 1]` → positive-class probabilities
2. `roc_curve(y_true, y_prob)` → FPR, TPR, thresholds
3. `roc_auc_score(y_true, y_prob)`
4. Plot FPR vs TPR, add grey diagonal (random baseline), annotate AUC in legend
""",
        "workspace_tip": (
            "Use `predict_proba` (probabilities), not `predict` (labels), for ROC. "
            "Diagonal: `plt.plot([0,1],[0,1],'--',color='gray')`. "
            "AUC 0.5 = random; >0.7 is generally useful clinically."
        ),
        "hint": """\
```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_te_sc)[:, 1]
fpr, tpr, _ = roc_curve(y_te, y_prob)
auc = roc_auc_score(y_te, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':        np.random.randint(40, 80, n),
    'tumor_size': np.random.normal(3.5, 1.5, n).clip(0.5, 10),
    'treatment':  np.random.binomial(1, 0.5, n),
    'event':      np.random.binomial(1, 0.55, n),
})
X = df[['age', 'tumor_size', 'treatment']].values
y = df['event'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
model = LogisticRegression(max_iter=1000, random_state=42).fit(sc.fit_transform(X_tr), y_tr)

y_prob = model.predict_proba(sc.transform(X_te))[:, 1]
fpr, tpr, _ = roc_curve(y_te, y_prob)
auc = roc_auc_score(y_te, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f'LogReg (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random (0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
print(f"AUC: {auc:.3f}")
```""",
    },
    {
        "id": "sklearn_pipeline_clinical",
        "title": "sklearn Pipeline",
        "category": "ML/Statistics",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Build a clean `Pipeline` that handles missing values.

Steps (in order):
1. `SimpleImputer(strategy='median')`
2. `StandardScaler()`
3. `LogisticRegression(max_iter=1000)`

Use the same `.fit()` / `.predict()` / `.score()` API as a single estimator.
Access a fitted step with `pipe.named_steps['step_name']`.
""",
        "workspace_tip": (
            "`Pipeline([('name', estimator), ...])` — the last step is the model; "
            "all prior steps must implement `transform`. "
            "Fit once: `pipe.fit(X_train, y_train)` applies all transforms automatically."
        ),
        "hint": """\
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
print(pipe.named_steps['model'].coef_)
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':        np.random.randint(40, 80, n).astype(float),
    'tumor_size': np.random.normal(3.5, 1.5, n).clip(0.5, 10),
    'treatment':  np.random.binomial(1, 0.5, n).astype(float),
    'event':      np.random.binomial(1, 0.55, n),
})
idx_age   = np.random.choice(n, 20, replace=False)
idx_tumor = np.random.choice(n, 15, replace=False)
df.loc[idx_age,   'age']        = np.nan
df.loc[idx_tumor, 'tumor_size'] = np.nan

X = df[['age', 'tumor_size', 'treatment']].values
y = df['event'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000, random_state=42)),
])
pipe.fit(X_tr, y_tr)

print(f"Accuracy: {pipe.score(X_te, y_te):.3f}")
print(classification_report(y_te, pipe.predict(X_te)))
print("Coefficients:", pipe.named_steps['model'].coef_)
```""",
    },
    {
        "id": "sklearn_crossval_clinical",
        "title": "Cross-Validation AUC",
        "category": "ML/Statistics",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Evaluate a pipeline with stratified CV — from memory.

1. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
2. Manual fold loop: fit pipeline, compute AUC per fold
3. `cross_val_score(scoring='roc_auc')` shorthand — should match
4. Print `mean ± std` AUC for both methods

Key rule: **fit scaler inside the fold** (the pipeline handles this automatically).
""",
        "workspace_tip": (
            "`cross_val_score(estimator, X, y, cv=skf, scoring='roc_auc')` — "
            "if estimator is a Pipeline, transforms are safely applied per fold. "
            "Manual loop: iterate `skf.split(X, y)` for `train_idx, val_idx`."
        ),
        "hint": """\
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
print(f"AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':        np.random.randint(40, 80, n).astype(float),
    'tumor_size': np.random.normal(3.5, 1.5, n).clip(0.5, 10),
    'treatment':  np.random.binomial(1, 0.5, n).astype(float),
    'event':      np.random.binomial(1, 0.55, n),
})
X = df[['age', 'tumor_size', 'treatment']].values
y = df['event'].values

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000, random_state=42)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Manual loop
manual_aucs = []
for tr, va in skf.split(X, y):
    pipe.fit(X[tr], y[tr])
    prob = pipe.predict_proba(X[va])[:, 1]
    manual_aucs.append(roc_auc_score(y[va], prob))
print(f"Manual CV AUC:   {np.mean(manual_aucs):.3f} ± {np.std(manual_aucs):.3f}")

# Shorthand
cv_aucs = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
print(f"cross_val_score: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}")
```""",
    },
    # ── Survival Analysis (lifelines) ─────────────────────────────────────────
    {
        "id": "km_basic",
        "title": "Kaplan-Meier Curve (Basic)",
        "category": "Survival Analysis",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Fit and plot a basic Kaplan-Meier curve from memory.

```python
from lifelines import KaplanMeierFitter
```

Given `os_months` (duration) and `event` (1 = event, 0 = censored):
1. Fit a `KaplanMeierFitter`
2. Plot survival function with confidence intervals
3. Print median survival time
4. Survival probability at 24 months?
""",
        "workspace_tip": (
            "`kmf.fit(durations, event_observed=..., label='...')`. "
            "`kmf.median_survival_time_`. "
            "`kmf.survival_function_at_times([24]).values[0][0]` for a point estimate."
        ),
        "hint": """\
```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

kmf = KaplanMeierFitter()
kmf.fit(df['os_months'], event_observed=df['event'], label='All patients')
ax = kmf.plot_survival_function(ci_show=True)
ax.set_xlabel('Months'); ax.set_ylabel('Survival')
plt.show()
print("Median OS:", kmf.median_survival_time_)
print("Surv@24m:", kmf.survival_function_at_times([24]).values[0][0])
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

np.random.seed(42)
n = 150
df = pd.DataFrame({
    'os_months': np.random.exponential(20, n).clip(1, 60).round(1),
    'event':     np.random.binomial(1, 0.65, n),
})

kmf = KaplanMeierFitter()
kmf.fit(df['os_months'], event_observed=df['event'], label='All patients')

fig, ax = plt.subplots(figsize=(7, 5))
kmf.plot_survival_function(ax=ax, ci_show=True)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival probability')
ax.set_title('Kaplan-Meier Survival Curve')
plt.tight_layout()
plt.show()

print(f"Median survival: {kmf.median_survival_time_:.1f} months")
print(f"Surv at 24m:    {kmf.survival_function_at_times([24]).values[0][0]:.3f}")
print(f"Events: {df['event'].sum()} / {len(df)}")
```""",
    },
    {
        "id": "km_groups",
        "title": "KM Curves by Treatment Arm",
        "category": "Survival Analysis",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Overlay KM curves for two arms on one plot.

1. Fit a separate `KaplanMeierFitter` per arm; use `groupby` to loop
2. Pass `ax=ax` to each `.plot_survival_function()` call to overlay
3. Add a vertical dashed line at 24 months
4. Print median OS for each arm
""",
        "workspace_tip": (
            "`for arm, grp in df.groupby('arm'):` — reuse the same `ax` each iteration. "
            "`ax.axvline(24, ls='--', color='gray')` for the time marker."
        ),
        "hint": """\
```python
fig, ax = plt.subplots()
for arm, grp in df.groupby('arm'):
    kmf = KaplanMeierFitter()
    kmf.fit(grp['os_months'], event_observed=grp['event'], label=f'Arm {arm}')
    kmf.plot_survival_function(ax=ax, ci_show=True)
    print(f"Arm {arm} median: {kmf.median_survival_time_:.1f}m")
ax.axvline(24, ls='--', color='gray')
plt.show()
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

np.random.seed(42)
n = 160
os_a = np.random.exponential(24, n//2).clip(1, 60).round(1)
os_b = np.random.exponential(14, n//2).clip(1, 60).round(1)
ev   = np.random.binomial(1, 0.65, n)

df = pd.DataFrame({
    'arm':       ['A'] * (n//2) + ['B'] * (n//2),
    'os_months': np.concatenate([os_a, os_b]),
    'event':     ev,
})

fig, ax = plt.subplots(figsize=(7, 5))
for arm, grp in df.groupby('arm'):
    kmf = KaplanMeierFitter()
    kmf.fit(grp['os_months'], event_observed=grp['event'], label=f'Arm {arm} (n={len(grp)})')
    kmf.plot_survival_function(ax=ax, ci_show=True)
    print(f"Arm {arm} median OS: {kmf.median_survival_time_:.1f} months")

ax.axvline(24, ls='--', color='gray', alpha=0.7, label='24 months')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival probability')
ax.set_title('KM Curves by Treatment Arm')
ax.legend()
plt.tight_layout()
plt.show()
```""",
    },
    {
        "id": "logrank_test",
        "title": "Log-Rank Test",
        "category": "Survival Analysis",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Test whether survival differs significantly between groups.

```python
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
```

1. `logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)`
2. Print p-value and whether it's significant at α = 0.05
3. Extend to 3 arms with `multivariate_logrank_test(durations, groups, event_observed=events)`
""",
        "workspace_tip": (
            "`results.p_value` and `results.test_statistic`. "
            "Subset by arm: `df.loc[df['arm']=='A', 'os_months']`. "
            "For multivariate, pass the group column directly — no need to subset."
        ),
        "hint": """\
```python
from lifelines.statistics import logrank_test

T1 = df.loc[df['arm']=='A', 'os_months']
E1 = df.loc[df['arm']=='A', 'event']
T2 = df.loc[df['arm']=='B', 'os_months']
E2 = df.loc[df['arm']=='B', 'event']

res = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
print(f"p = {res.p_value:.4f}  significant: {res.p_value < 0.05}")
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test, multivariate_logrank_test

np.random.seed(42)
n = 180
arm   = np.random.choice(['A', 'B', 'C'], n)
scale = np.where(arm == 'A', 24, np.where(arm == 'B', 16, 10))
os    = np.random.exponential(scale).clip(1, 60).round(1)
ev    = np.random.binomial(1, 0.65, n)
df    = pd.DataFrame({'arm': arm, 'os_months': os, 'event': ev})

# Pairwise log-rank tests
for g1, g2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
    T1 = df.loc[df['arm']==g1, 'os_months']
    E1 = df.loc[df['arm']==g1, 'event']
    T2 = df.loc[df['arm']==g2, 'os_months']
    E2 = df.loc[df['arm']==g2, 'event']
    res = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    sig = '* p<0.05' if res.p_value < 0.05 else ''
    print(f"{g1} vs {g2}: p={res.p_value:.4f}  stat={res.test_statistic:.2f} {sig}")

# Multivariate (all 3 groups simultaneously)
mv = multivariate_logrank_test(df['os_months'], df['arm'], event_observed=df['event'])
print(f"\nMultivariate log-rank p = {mv.p_value:.4f}")
```""",
    },
    {
        "id": "cox_ph",
        "title": "Cox Proportional Hazards Model",
        "category": "Survival Analysis",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Fit a multivariate Cox PH model from memory.

```python
from lifelines import CoxPHFitter
```

1. `cph.fit(df, duration_col='os_months', event_col='event')`
2. `cph.print_summary()` — read hazard ratios and p-values
3. `cph.concordance_index_` — C-index (survival analogue of AUC)
4. Plot the baseline survival function
""",
        "workspace_tip": (
            "HR > 1 = increased hazard (worse). HR < 1 = protective. "
            "C-index: 0.5 = random, >0.65 is meaningful. "
            "Pass only the columns needed: `cph.fit(df[cols], duration_col=..., event_col=...)`."
        ),
        "hint": """\
```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df[['os_months','event','age','treatment','tumor_size']],
        duration_col='os_months', event_col='event')
cph.print_summary()
print("C-index:", cph.concordance_index_)
cph.baseline_survival_.plot()
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

np.random.seed(42)
n = 200
treatment = np.random.binomial(1, 0.5, n)
age       = np.random.randint(40, 80, n)
tumor     = np.random.normal(3.5, 1.5, n).clip(0.5, 10)
hazard    = np.exp(0.02*(age - 60) + 0.15*tumor - 0.5*treatment)
os_months = np.random.exponential(20 / hazard).clip(1, 60).round(1)
event     = np.random.binomial(1, 0.65, n)

df = pd.DataFrame({
    'os_months':  os_months,
    'event':      event,
    'age':        age,
    'treatment':  treatment,
    'tumor_size': tumor.round(2),
})

cph = CoxPHFitter()
cph.fit(df, duration_col='os_months', event_col='event')
cph.print_summary(decimals=3)
print(f"\nC-index: {cph.concordance_index_:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
cph.baseline_survival_.plot(ax=ax)
ax.set_title('Cox PH Baseline Survival')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Baseline survival')
plt.tight_layout()
plt.show()
```""",
    },
    {
        "id": "clinical_integrated_drill",
        "title": "Integrated Clinical Genomics Drill",
        "category": "Survival Analysis",
        "bucket": "Clinical",
        "prompt": """\
**Task:** Complete one end-to-end interview drill without context switching.

You must cover all steps in one script:

1. **Clinical + VCF parsing**
   - Build a clinical DataFrame (`patient_id`, `age`, `stage`, `treatment_arm`, `cancer_type`, `survival_time`, `event`)
   - Parse VCF text into a variant DataFrame with `patient_id`, `gene`, `depth`, `af`, `pathogenic`
2. **Pandas operations**
   - Filter to LUAD and summarize mean survival by treatment arm
   - Aggregate variant features to patient-level
   - Merge clinical + variant features
3. **Basic sklearn**
   - Define binary target: event within 12 months
   - Train/test split + `RandomForestClassifier`
   - Report ROC-AUC
4. **Survival analysis**
   - KM curves by treatment arm
   - Log-rank p-value for A vs B
   - Cox PH with clinical + variant covariates; print hazard ratios and p-values
""",
        "workspace_tip": (
            "Treat this as a 45-minute capstone. Parse first, aggregate second, "
            "model third, survival last. Keep patient-level merge keys explicit and "
            "avoid leakage by defining the binary target from survival outcomes only."
        ),
        "hint": """\
```python
# 1) parse VCF -> var_df
# 2) var_df.groupby('patient_id').agg(...) -> patient variant features
# 3) merged = clinical.merge(features, on='patient_id', how='left').fillna(0)
# 4) y = ((merged['survival_time'] <= 12) & (merged['event'] == 1)).astype(int)
# 5) RandomForest + roc_auc_score
# 6) KaplanMeierFitter curves, logrank_test, CoxPHFitter
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

np.random.seed(42)

# -----------------------------
# 1) Build clinical table
# -----------------------------
n = 140
patient_ids = [f"P{i:03d}" for i in range(1, n + 1)]

stage = np.random.choice(["I", "II", "III", "IV"], size=n, p=[0.2, 0.3, 0.3, 0.2])
treatment_arm = np.random.choice(["A", "B"], size=n)
cancer_type = np.random.choice(["LUAD", "LUSC", "BRCA"], size=n, p=[0.5, 0.3, 0.2])
age = np.random.randint(40, 82, size=n)

stage_risk = pd.Series(stage).map({"I": 0.0, "II": 0.35, "III": 0.8, "IV": 1.2}).values
treat_effect = np.where(treatment_arm == "B", -0.35, 0.0)
linpred = 0.02 * (age - 60) + stage_risk + treat_effect
hazard = np.exp(linpred)

survival_time = np.random.exponential(scale=22 / hazard).clip(1, 60).round(1)
event_prob = np.clip(0.40 + 0.22 * (hazard / np.percentile(hazard, 75)), 0.25, 0.9)
event = np.random.binomial(1, event_prob)

clinical = pd.DataFrame(
    {
        "patient_id": patient_ids,
        "age": age,
        "stage": stage,
        "treatment_arm": treatment_arm,
        "cancer_type": cancer_type,
        "survival_time": survival_time,
        "event": event,
    }
)

# -----------------------------
# 2) Generate + parse VCF text
# -----------------------------
genes = ["TP53", "EGFR", "KRAS", "ALK", "PIK3CA"]
consequences = ["missense", "nonsense", "frameshift", "synonymous"]

vcf_lines = [
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
]

for pid in patient_ids:
    n_var = np.random.poisson(2)
    for idx in range(n_var):
        chrom = np.random.choice(["1", "7", "12", "17"])
        pos = np.random.randint(10_000, 900_000)
        ref, alt = np.random.choice(["A", "C", "G", "T"], 2, replace=False)
        qual = np.random.randint(35, 99)
        gene = np.random.choice(genes)
        csq = np.random.choice(consequences)
        pathogenic = int(gene in {"TP53", "EGFR", "KRAS"} and np.random.rand() < 0.45)
        dp = np.random.randint(20, 260)
        af = np.random.uniform(0.03, 0.85)
        var_id = f"{pid}_v{idx + 1}"
        info = f"GENE={gene};CSQ={csq};PATH={pathogenic}"
        sample = f"0/1:{dp}:{af:.3f}"
        vcf_lines.append(
            f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t{qual}\tPASS\t{info}\tGT:DP:AF\t{sample}"
        )

vcf_text = "\n".join(vcf_lines)

records = []
for line in vcf_text.splitlines():
    if not line or line.startswith("#"):
        continue
    chrom, pos, vid, ref, alt, qual, filt, info, fmt, sample = line.split("\t")
    info_map = dict(item.split("=", 1) for item in info.split(";"))
    fmt_keys = fmt.split(":")
    fmt_vals = sample.split(":")
    sample_map = dict(zip(fmt_keys, fmt_vals))

    records.append(
        {
            "patient_id": vid.split("_")[0],
            "gene": info_map["GENE"],
            "depth": int(sample_map["DP"]),
            "af": float(sample_map["AF"]),
            "pathogenic": int(info_map["PATH"]),
        }
    )

var_df = pd.DataFrame(records)

# -----------------------------
# 3) Pandas filtering/grouping
# -----------------------------
luad = clinical[clinical["cancer_type"] == "LUAD"]
print("LUAD mean survival by arm:")
print(luad.groupby("treatment_arm")["survival_time"].mean().round(2))

if var_df.empty:
    variant_feats = pd.DataFrame(
        {
            "patient_id": clinical["patient_id"],
            "variant_count": 0,
            "pathogenic_variant_count": 0,
            "mean_af": 0.0,
            "max_af": 0.0,
            "has_tp53": 0,
            "has_egfr": 0,
            "has_kras": 0,
        }
    )
else:
    variant_feats = var_df.groupby("patient_id").agg(
        variant_count=("gene", "size"),
        pathogenic_variant_count=("pathogenic", "sum"),
        mean_af=("af", "mean"),
        max_af=("af", "max"),
        has_tp53=("gene", lambda s: int((s == "TP53").any())),
        has_egfr=("gene", lambda s: int((s == "EGFR").any())),
        has_kras=("gene", lambda s: int((s == "KRAS").any())),
    ).reset_index()

merged = clinical.merge(variant_feats, on="patient_id", how="left")
fill_cols = [
    "variant_count",
    "pathogenic_variant_count",
    "mean_af",
    "max_af",
    "has_tp53",
    "has_egfr",
    "has_kras",
]
merged[fill_cols] = merged[fill_cols].fillna(0)

# -----------------------------
# 4) Basic ML: RF + AUC
# -----------------------------
merged["event_12m"] = ((merged["survival_time"] <= 12) & (merged["event"] == 1)).astype(int)

X = pd.get_dummies(
    merged[
        [
            "age",
            "variant_count",
            "pathogenic_variant_count",
            "mean_af",
            "max_af",
            "has_tp53",
            "has_egfr",
            "has_kras",
            "stage",
            "treatment_arm",
            "cancer_type",
        ]
    ],
    drop_first=True,
)
y = merged["event_12m"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=250, random_state=42)
rf.fit(X_train, y_train)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"\nRandomForest ROC-AUC (12m event): {auc:.3f}")

# -----------------------------
# 5) Survival: KM + log-rank
# -----------------------------
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 5))
for arm, grp in merged.groupby("treatment_arm"):
    kmf.fit(grp["survival_time"], event_observed=grp["event"], label=f"Arm {arm}")
    kmf.plot_survival_function(ax=ax, ci_show=True)
ax.set_title("Kaplan-Meier by Treatment Arm")
ax.set_xlabel("Months")
ax.set_ylabel("Survival probability")
plt.tight_layout()
plt.show()

gA = merged[merged["treatment_arm"] == "A"]
gB = merged[merged["treatment_arm"] == "B"]
lr = logrank_test(
    gA["survival_time"],
    gB["survival_time"],
    event_observed_A=gA["event"],
    event_observed_B=gB["event"],
)
print(f"Log-rank A vs B p-value: {lr.p_value:.4f}")

# -----------------------------
# 6) Cox PH: adjusted effects
# -----------------------------
cox_df = pd.get_dummies(
    merged[
        [
            "survival_time",
            "event",
            "age",
            "pathogenic_variant_count",
            "has_tp53",
            "has_egfr",
            "has_kras",
            "stage",
            "treatment_arm",
        ]
    ],
    drop_first=True,
)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="survival_time", event_col="event")

summary = cph.summary[["coef", "exp(coef)", "p"]].round(4)
print("\nCox PH summary (coef, HR, p):")
print(summary)
```
""",
    },
]

INTEGRATED_QUESTIONS = [
    {
        "id": "integrated_vcf_eda_pipeline",
        "title": "Integrated VCF Pipeline: Parse → Clean → EDA → Merge → Plot",
        "category": "Integrated Drill",
        "bucket": "Integrated",
        "prompt": """\
    **Task:** Build one realistic mini variant-analysis pipeline from a VCF file.

You should include all steps below in a single script:

    1. **Read / parse `data/sample.vcf` with line parsing** (skip `##` meta lines, keep header row)
2. Build a tidy variant table with useful columns (e.g., `chrom`, `pos`, `gene`, `sample`, `dp`, `af`, `impact`)
3. **Preprocess missing values** (drop or impute where reasonable)
4. **EDA**: shape, dtypes, null counts, basic numeric summary
5. **GroupBy summaries** (for example by `gene` and `sample`)
6. **Merge tables** (e.g., with sample-level clinical metadata)
7. Plot distributions using **both seaborn and matplotlib** (hist / box / count plots)

Output should make it easy to reason about variant burden and quality.
""",
        "workspace_tip": (
            "Interview-friendly pattern: loop through lines, split by tab, parse INFO with `split(';')` and `split('=', 1)`, "
            "append dict rows, then `pd.DataFrame(rows)`. Convert `.` to NaN, coerce numeric columns, do simple "
            "groupby summaries, then merge with sample metadata on `sample`."
        ),
        "hint": """\
```python
import numpy as np
import pandas as pd

# Parse with line-by-line logic (easy to explain)
rows = []
with open("data/sample.vcf", "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            continue
        c = line.split("\t")
        info = {}
        for item in c[7].split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                info[k] = v
        rows.append({
            "chrom": c[0], "pos": c[1], "qual": c[5], "filter": c[6], "sample": c[9],
            "gene": info.get("GENE"), "dp": info.get("DP"), "af": info.get("AF"), "impact": info.get("IMPACT")
        })

df = pd.DataFrame(rows).replace(".", np.nan)
df["dp"] = pd.to_numeric(df["dp"], errors="coerce")
df["af"] = pd.to_numeric(df["af"], errors="coerce")
```""",
        "solution": """\
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# 2) Parse VCF file line-by-line into row dicts
# -------------------------------------------------
rows = []
with open("data/sample.vcf", "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            continue

        c = line.split("\t")
        info_map = {}
        for item in c[7].split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                info_map[k] = v

        rows.append({
            "chrom": c[0],
            "pos": c[1],
            "qual": c[5],
            "filter": c[6],
            "sample": c[9],
            "gene": info_map.get("GENE"),
            "dp": info_map.get("DP"),
            "af": info_map.get("AF"),
            "impact": info_map.get("IMPACT"),
        })

df = pd.DataFrame(rows)

# --------------------------------------------------
# 3) Preprocess: missing values + type conversions
# --------------------------------------------------
df = df.replace(".", np.nan)
df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
df["qual"] = pd.to_numeric(df["qual"], errors="coerce")
df["dp"] = pd.to_numeric(df["dp"], errors="coerce")
df["af"] = pd.to_numeric(df["af"], errors="coerce")

# Keep PASS rows, fill dp with median, and drop rows missing af/qual
df_clean = df[df["filter"] == "PASS"].copy()
df_clean["dp"] = df_clean["dp"].fillna(df_clean["dp"].median())
df_clean = df_clean.dropna(subset=["af", "qual"])

# --------------------------------
# 4) EDA: structure + quick checks
# --------------------------------
print("Raw shape:", df.shape)
print("Clean shape:", df_clean.shape)
print("\nDtypes:\n", df_clean[["chrom", "pos", "qual", "dp", "af", "gene", "sample", "impact"]].dtypes)
print("\nNulls in clean table:\n", df_clean[["qual", "dp", "af", "gene", "sample"]].isna().sum())
print("\nNumeric summary:\n", df_clean[["qual", "dp", "af"]].describe().round(3))

# --------------------------------
# 5) GroupBy summaries
# --------------------------------
gene_summary = df_clean.groupby("gene").agg(
    n_variants=("gene", "size"),
    mean_af=("af", "mean"),
    median_dp=("dp", "median"),
).sort_values(["n_variants", "mean_af"], ascending=[False, False]).round(3)

sample_summary = df_clean.groupby("sample").agg(
    n_variants=("sample", "size"),
    mean_af=("af", "mean"),
    mean_qual=("qual", "mean"),
).sort_values("n_variants", ascending=False).round(3)

print("\nBy gene:\n", gene_summary)
print("\nBy sample:\n", sample_summary)

# --------------------------------
# 6) Merge with sample metadata
# --------------------------------
sample_meta = pd.DataFrame({
    "sample": ["S01", "S02", "S03", "S04", "S05"],
    "cohort": ["Lung", "Lung", "Breast", "Melanoma", "Breast"],
    "age": [61, 67, np.nan, 58, 49],
    "response": ["PR", "SD", "PD", "PR", "CR"],
})

merged = sample_summary.reset_index().merge(sample_meta, on="sample", how="left")
merged["age"] = merged["age"].fillna(merged["age"].median())

cohort_summary = merged.groupby("cohort").agg(
    samples=("sample", "nunique"),
    total_variants=("n_variants", "sum"),
    mean_af=("mean_af", "mean"),
).sort_values("total_variants", ascending=False).round(3)

print("\nMerged sample-level table:\n", merged)
print("\nCohort summary:\n", cohort_summary)

# --------------------------------
# 7) Plot distributions (sns + plt)
# --------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Seaborn histogram
sns.histplot(df_clean["af"], bins=8, kde=True, ax=axes[0], color="#2C7FB8")
axes[0].set_title("AF Distribution")
axes[0].set_xlabel("Allele Frequency")

# Seaborn boxplot
sns.boxplot(data=df_clean, x="impact", y="dp", ax=axes[1], palette="Set2")
axes[1].set_title("Depth by Impact")
axes[1].set_xlabel("Impact")
axes[1].set_ylabel("Depth (DP)")

# Matplotlib bar plot (from value_counts)
gene_counts = df_clean["gene"].value_counts()
axes[2].bar(gene_counts.index, gene_counts.values, color="#7FC97F")
axes[2].set_title("Variant Count by Gene")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
```""",
    }
]

ALL_QUESTIONS = QUESTIONS + HELIX_QUESTIONS + CLINICAL_QUESTIONS + INTEGRATED_QUESTIONS

CATEGORY_ICON = {
    "ML/Statistics":    "🔵",
    "Pandas/EDA":       "🟢",
    "Python Basics":    "🟡",
    "Strings":          "🧬",
    "Hash Maps":        "🗂️",
    "Lists/Sorting":    "📋",
    "Algorithms":       "⚙️",
    "Survival Analysis": "📈",
    "Integrated Drill": "🧩",
}

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
_defaults = {
    "current_q_id":  QUESTIONS[0]["id"],
    "active_bucket": "Basic",
    "show_hint":     False,
    "show_solution": False,
    "difficulties":  {},   # q_id → "easy" | "hard"
    "miss_counts":   {},   # q_id → int
    "hard_only":     False,
    "timer_start":   time.time(),
    "user_code":     {},   # q_id → str  (persists per question)
    "editor_mode":   {},   # q_id → "mine" | "solution"
    "notes":         {},   # q_id → str  (personal notes per question)
    "custom_solutions": {},  # q_id → str  (user-promoted tested solution)
    "last_run_ok":   {},   # q_id → bool (last execution status)
    "last_run_code": {},   # q_id → str  (code that was last executed)
    "run_output":    "",
    "run_error":     "",
    "run_figures":   [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_PROGRESS_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "practice_progress.json"
_PERSIST_KEYS = [
    "user_code",
    "notes",
    "editor_mode",
    "difficulties",
    "miss_counts",
    "custom_solutions",
]


def _load_progress_from_disk() -> dict:
    if not _PROGRESS_PATH.exists():
        return {}
    try:
        data = json.loads(_PROGRESS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_progress_to_disk():
    data = {k: st.session_state.get(k, {}) for k in _PERSIST_KEYS}
    try:
        _PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROGRESS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


if not st.session_state.get("_progress_loaded", False):
    _persisted = _load_progress_from_disk()
    for _k in _PERSIST_KEYS:
        _v = _persisted.get(_k)
        if isinstance(st.session_state.get(_k), dict) and isinstance(_v, dict):
            st.session_state[_k].update(_v)
    st.session_state._progress_loaded = True

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _active_questions() -> list[dict]:
    bucket = st.session_state.active_bucket
    return [q for q in ALL_QUESTIONS if q.get("bucket", "Basic") == bucket]


def _pool() -> list[str]:
    qs = _active_questions()
    if st.session_state.hard_only:
        hard_ids = {qid for qid, d in st.session_state.difficulties.items() if d == "hard"}
        hard_in_bucket = [q["id"] for q in qs if q["id"] in hard_ids]
        return hard_in_bucket if hard_in_bucket else [q["id"] for q in qs]
    return [q["id"] for q in qs]


def _current() -> dict:
    return next(q for q in ALL_QUESTIONS if q["id"] == st.session_state.current_q_id)


def _go_next():
    pool   = _pool()
    others = [qid for qid in pool if qid != st.session_state.current_q_id]
    if others:
        def _w(qid):
            d = st.session_state.difficulties.get(qid)
            if d == "easy":
                return 1
            if d == "hard":
                return 4 + st.session_state.miss_counts.get(qid, 0)
            return 2  # unrated: medium priority
        st.session_state.current_q_id = random.choices(
            others, weights=[_w(i) for i in others], k=1
        )[0]
    st.session_state.show_hint     = False
    st.session_state.show_solution = False
    st.session_state.run_output    = ""
    st.session_state.run_error     = ""
    st.session_state.run_figures   = []
    st.session_state.timer_start   = time.time()


def _extract_solution_code(solution: str) -> str:
    """Strip markdown fences from a solution string, return bare Python."""
    m = re.search(r"```python\n(.*?)```", solution, re.DOTALL)
    return m.group(1).rstrip() if m else solution.strip()


def _is_valid_python(code: str) -> bool:
    try:
        compile(code, "<solution>", "exec")
        return True
    except Exception:
        return False


def _run_code(qid: str, code: str):
    """Execute user code, capture stdout + matplotlib figures."""
    import io, contextlib, traceback, matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")          # non-interactive backend, safe for Streamlit
    plt.close("all")               # clear any leftover figures
    buf = io.StringIO()
    try:
        st.session_state.last_run_code[qid] = code
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<editor>", "exec"), {})  # isolated namespace
        st.session_state.run_output  = buf.getvalue() or ""
        st.session_state.run_error   = ""
        st.session_state.last_run_ok[qid] = True
        # Grab every figure matplotlib created during exec
        st.session_state.run_figures = [plt.figure(n) for n in plt.get_fignums()]
    except Exception:
        st.session_state.run_output  = buf.getvalue()
        st.session_state.run_error   = traceback.format_exc()
        st.session_state.last_run_ok[qid] = False
        st.session_state.run_figures = []


def _elapsed() -> str:
    secs = int(time.time() - st.session_state.timer_start)
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
_aq_h   = _active_questions()
_total  = len(_aq_h)
_easy_n = sum(1 for tq in _aq_h if st.session_state.difficulties.get(tq["id"]) == "easy")
_hard_n = sum(1 for tq in _aq_h if st.session_state.difficulties.get(tq["id"]) == "hard")
_done_n = sum(1 for tq in _aq_h if tq["id"] in st.session_state.difficulties)

_pct = _easy_n / _total if _total else 0.0
_prog_text = (
    f"🎯 {_easy_n}/{_total} mastered ({_pct:.0%})"
    + (" · 🏆 Bucket complete!" if _easy_n == _total else f" · {_hard_n} to drill")
)

hc1, hc2, hc3, hc4, hc5 = st.columns([3, 1, 1, 1, 4])
with hc1:
    st.markdown("**🧠 Python ML Reps**")
with hc2:
    st.markdown(f"✅ **{_easy_n}**&nbsp;Easy")
with hc3:
    st.markdown(f"🔴 **{_hard_n}**&nbsp;Hard")
with hc4:
    st.markdown(f"📝 **{_done_n}/{_total}**")
with hc5:
    st.progress(_pct, text=_prog_text)

st.divider()

# Bucket selector
_bucket_choice = st.radio(
    "Bucket",
    options=["Basic", "Bioinformatics Engineer", "Clinical", "Integrated"],
    format_func=lambda b: {
        "Basic":    "🔵 Basic (ML / Python)",
        "Bioinformatics Engineer": "🧬 Bioinformatics Engineer",
        "Clinical": "🏥 Clinical DS",
        "Integrated": "🧩 Integrated Drills",
    }[b],
    horizontal=True,
    index=["Basic", "Bioinformatics Engineer", "Clinical", "Integrated"].index(st.session_state.active_bucket),
    label_visibility="collapsed",
)
if _bucket_choice != st.session_state.active_bucket:
    st.session_state.active_bucket = _bucket_choice
    st.session_state.current_q_id  = _active_questions()[0]["id"]
    st.session_state.show_hint     = False
    st.session_state.show_solution = False
    st.session_state.run_output    = ""
    st.session_state.run_error     = ""
    st.session_state.run_figures   = []
    st.session_state.timer_start   = time.time()
    st.rerun()

q = _current()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT  ·  LEFT · CENTER · RIGHT
# ──────────────────────────────────────────────────────────────────────────────
left, center, right = st.columns([1.33, 4.9, 3.77])

# ── LEFT: question prompt ─────────────────────────────────────────────────────
with left:
    cat_icon = CATEGORY_ICON.get(q["category"], "⚪")
    diff     = st.session_state.difficulties.get(q["id"])
    diff_badge = {"easy": " ✅", "hard": " 🔴"}.get(diff, "")

    st.markdown(f"**{cat_icon} {q['category']}**&nbsp;&nbsp; ⏱️ `{_elapsed()}`")
    st.markdown(f"## {q['title']}{diff_badge}")
    st.markdown(q["prompt"])

    st.divider()
    st.markdown("<small>🗂️ <b>Topics</b></small>", unsafe_allow_html=True)
    _topics = _active_questions()
    _topic_ids = [tq["id"] for tq in _topics]
    _topic_index = _topic_ids.index(q["id"]) if q["id"] in _topic_ids else 0

    def _topic_label(qid: str) -> str:
        tq = next(t for t in _topics if t["id"] == qid)
        short_title = tq["title"].split("—")[0].split("(")[0].strip()
        level = st.session_state.difficulties.get(qid)
        if level == "easy":
            return f"{short_title} [easy]"
        if level == "hard":
            return f"{short_title} [hard]"
        return short_title

    _selected_topic = st.radio(
        "Topics",
        options=_topic_ids,
        index=_topic_index,
        format_func=_topic_label,
        label_visibility="collapsed",
        key=f"topic_nav_{st.session_state.active_bucket}",
    )

    if _selected_topic != q["id"]:
        st.session_state.current_q_id  = _selected_topic
        st.session_state.show_hint     = False
        st.session_state.show_solution = False
        st.session_state.run_output    = ""
        st.session_state.run_error     = ""
        st.session_state.run_figures   = []
        st.session_state.timer_start   = time.time()
        st.rerun()

# ── CENTER: workspace + tracker ───────────────────────────────────────────────
with center:
    # ── Code editor ──────────────────────────────────────────────────────────
    _default_code = f"# {q['title']}\n# Write your solution here\n\n"
    _mode = st.session_state.editor_mode.get(q["id"], "mine")
    _base_solution_code = _extract_solution_code(q["solution"])
    _custom_solution_code = st.session_state.custom_solutions.get(q["id"])
    _invalid_custom_solution = False
    if _custom_solution_code is not None and not _is_valid_python(_custom_solution_code):
        _invalid_custom_solution = True
        st.session_state.custom_solutions.pop(q["id"], None)
        _custom_solution_code = None
    _solution_code = _custom_solution_code or _base_solution_code

    # Pick content based on mode; key change forces editor remount with correct value
    current_code = (
        _solution_code
        if _mode == "solution"
        else st.session_state.user_code.get(q["id"], _default_code)
    )
    _editor_key = f"editor_{q['id']}_{_mode}"
    _live_editor_code = st.session_state.get(_editor_key, current_code)

    _mine_code = st.session_state.user_code.get(q["id"], _default_code)
    _can_promote_solution = (
        st.session_state.last_run_ok.get(q["id"], False)
        and st.session_state.last_run_code.get(q["id"]) == _mine_code
    )

    # Mode toggle + action buttons on one row
    _tc1, _tc2, _tc3, _tc4, _tc5 = st.columns([2, 2, 2, 2, 3])
    with _tc1:
        if st.button(
            "📝 My Code",
            type="primary" if _mode == "mine" else "secondary",
            use_container_width=True,
            help="Switch to your work-in-progress",
        ):
            st.session_state.editor_mode[q["id"]] = "mine"
            st.rerun()
    with _tc2:
        if st.button(
            "✅ Solution",
            type="primary" if _mode == "solution" else "secondary",
            use_container_width=True,
            help="Load the full solution — your WIP is preserved",
        ):
            st.session_state.editor_mode[q["id"]] = "solution"
            st.rerun()
    with _tc3:
        if st.button("▶ Run Code", type="primary", use_container_width=True):
            _run_code(q["id"], _live_editor_code)
    with _tc4:
        if st.button("🧪 Run Solution", use_container_width=True):
            if _mode == "solution":
                _run_code(q["id"], _live_editor_code)
            else:
                _run_code(q["id"], _solution_code)
    with _tc5:
        if st.button("🗑️ Clear Editor", use_container_width=True):
            st.session_state.user_code[q["id"]] = _default_code
            st.session_state.editor_mode[q["id"]] = "mine"
            st.session_state.run_output  = ""
            st.session_state.run_error   = ""
            st.session_state.run_figures = []
            st.rerun()

    _pc1, _pc2, _pc3 = st.columns([2, 2, 4])
    with _pc1:
        if st.button(
            "⭐ Use My Code as Solution",
            use_container_width=True,
            disabled=not _can_promote_solution,
            help="Enabled only when this exact My Code content just ran successfully.",
        ):
            st.session_state.custom_solutions[q["id"]] = _mine_code
            st.session_state.editor_mode[q["id"]] = "solution"
            st.rerun()
    with _pc2:
        if st.button(
            "↩ Restore Original Solution",
            use_container_width=True,
            disabled=q["id"] not in st.session_state.custom_solutions,
        ):
            st.session_state.custom_solutions.pop(q["id"], None)
            st.rerun()
    with _pc3:
        if _invalid_custom_solution:
            st.caption("Invalid saved custom solution was detected and replaced with original.")
        else:
            st.caption("Autosave on: progress reloads across app restarts.")

    # Ctrl+S / Cmd+S — toggle My Code ↔ Solution
    # Ctrl+Enter / Cmd+Enter / Shift+Enter — run current code
    # Must attach to EVERY iframe (Ace editor lives in its own iframe and
    # swallows keydown events before they reach window.parent.document).
    components.html("""
    <script>
    (function() {
        var par = window.parent;

        function clickToggle() {
            var btns = par.document.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {
                var txt = (btns[i].innerText || btns[i].textContent || '').trim();
                var tid = btns[i].getAttribute('data-testid') || '';
                if ((txt.includes('My Code') || txt.includes('Solution')) &&
                        tid === 'baseButton-secondary') {
                    btns[i].click();
                    return;
                }
            }
        }

        function clickRun() {
            var btns = par.document.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {
                var txt = (btns[i].innerText || btns[i].textContent || '').trim();
                if (txt.includes('Run Code')) {
                    btns[i].click();
                    return;
                }
            }
        }

        function onKey(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                e.stopImmediatePropagation();
                clickToggle();
                return;
            }
            if (!e.ctrlKey && !e.metaKey && e.shiftKey && e.key === 'Enter') {
                e.preventDefault();
                e.stopImmediatePropagation();
                clickRun();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                e.stopImmediatePropagation();
                clickRun();
            }
        }

        // Parent document — remove old listener then re-add fresh one
        if (par._editorShortcut) {
            par.document.removeEventListener('keydown', par._editorShortcut, true);
        }
        par._editorShortcut = onKey;
        par.document.addEventListener('keydown', onKey, true);

        // All iframes (Ace editor, other components) — attach once per doc
        function attachToFrames() {
            var frames = par.document.querySelectorAll('iframe');
            for (var i = 0; i < frames.length; i++) {
                try {
                    var doc = frames[i].contentDocument;
                    if (doc && !doc._shortcutBound) {
                        doc._shortcutBound = true;
                        doc.addEventListener('keydown', onKey, true);
                    }
                } catch(ex) {}
            }
        }

        attachToFrames();
        // Poll for newly mounted iframes (e.g. Ace editor remount after toggle)
        if (par._shortcutPoll) clearInterval(par._shortcutPoll);
        par._shortcutPoll = setInterval(attachToFrames, 800);
    })();
    </script>
    """, height=0)

    user_code = st_ace(
        value=current_code,
        language="python",
        theme="monokai",
        key=_editor_key,
        height=500,
        tab_size=4,
        font_size=14,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
    )
    # Persist WIP only in "mine" mode; solution view stays pristine
    if user_code is not None:
        if _mode == "mine":
            st.session_state.user_code[q["id"]] = user_code
    else:
        user_code = current_code

    if st.session_state.run_error:
        st.error("**Runtime error:**")
        st.code(st.session_state.run_error, language="python")
    else:
        if st.session_state.run_output:
            st.success("**Output:**")
            st.code(st.session_state.run_output, language="")
        for fig in st.session_state.run_figures:
            st.pyplot(fig)

    st.caption(
        "💡 `print()` output appears above · matplotlib plots render inline · "
        "use Jupyter for interactive plots"
    )

    st.divider()
    st.markdown("**📌 My Notes**")
    _note = st.text_area(
        "notes",
        value=st.session_state.notes.get(q["id"], ""),
        height=90,
        placeholder="Key insight, what you missed, mnemonic to remember...",
        key=f"note_{q['id']}",
        label_visibility="collapsed",
    )
    st.session_state.notes[q["id"]] = _note

    # Miss tracker
    if st.session_state.miss_counts:
        st.divider()
        st.markdown("#### 📊 Miss Tracker — Top 5")
        sorted_misses = sorted(
            st.session_state.miss_counts.items(), key=lambda x: x[1], reverse=True
        )
        for qid, cnt in sorted_misses[:5]:
            name = next((tq["title"] for tq in ALL_QUESTIONS if tq["id"] == qid), qid)
            dots = "🔴" * min(cnt, 6)
            st.markdown(f"- **{name}** {dots} ×{cnt}")


# ── RIGHT: reference panel ────────────────────────────────────────────────────
with right:
    # ── Code diff ────────────────────────────────────────────────────────────
    _wip_diff = st.session_state.user_code.get(q["id"], "").strip()
    _hide_diff = st.session_state.get("diff_hard_mode", False)
    with st.expander("🔍 Diff: My Code vs Solution", expanded=not _hide_diff):
        if not _wip_diff:
            st.info("✏️ Write some code above — diff will appear here.")
        else:
            def _norm(text, strip_boilerplate=False):
                """Normalise lines for diff comparison."""
                _boiler = {f"# {q['title']}", "# Write your solution here"}
                lines = text.splitlines()
                if strip_boilerplate:
                    lines = [ln for ln in lines if ln.strip() not in _boiler]
                    while lines and not lines[0].strip():
                        lines.pop(0)
                out = []
                for ln in lines:
                    stripped = ln.strip()
                    if stripped.startswith("#"):
                        continue
                    if stripped and stripped[0] in ")]}":
                        ln = stripped
                    out.append(re.sub(r"  +", " ", ln.rstrip()))
                return out
            _diff_lines = list(difflib.unified_diff(
                _norm(_wip_diff, strip_boilerplate=True),
                _norm(_solution_code),
                lineterm="",
                fromfile="My Code",
                tofile="Solution",
                n=3,
            ))
            if _diff_lines:
                st.code("\n".join(_diff_lines), language="diff")
            else:
                st.success("✅ Matches solution!")

    _hide_diff = st.toggle(
        "🧠 Hard mode (hide diff)",
        value=st.session_state.get("diff_hard_mode", False),
        key="diff_hard_mode_toggle",
        help="Hard: diff collapsed — think before you peek",
    )
    st.session_state.diff_hard_mode = _hide_diff

    st.divider()
    st.markdown("### 📋 Reference Panel")

    # Hard-only toggle
    new_hard_only = st.toggle(
        "🔴 Hard Questions Only",
        value=st.session_state.hard_only,
        key="hard_only_toggle",
    )
    if new_hard_only != st.session_state.hard_only:
        st.session_state.hard_only = new_hard_only
        if new_hard_only:
            pool = _pool()
            if pool:
                st.session_state.current_q_id = random.choice(pool)
        st.session_state.show_hint     = False
        st.session_state.show_solution = False
        st.rerun()

    st.divider()

    # Mark Easy / Hard
    st.markdown("**Rate this question:**")
    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button(
            "✅ Easy" if diff == "easy" else "Mark Easy",
            use_container_width=True,
            type="primary" if diff == "easy" else "secondary",
        ):
            st.session_state.difficulties[q["id"]] = "easy"
            st.rerun()
    with mc2:
        if st.button(
            "🔴 Hard" if diff == "hard" else "Mark Hard",
            use_container_width=True,
            type="primary" if diff == "hard" else "secondary",
        ):
            st.session_state.difficulties[q["id"]] = "hard"
            st.session_state.miss_counts[q["id"]] = (
                st.session_state.miss_counts.get(q["id"], 0) + 1
            )
            st.rerun()

    if diff:
        st.caption(f"Marked as: **{diff}**")

    st.divider()

    # Hint toggle
    hint_label = "💡 Hide Hint" if st.session_state.show_hint else "💡 Show Hint"
    if st.button(hint_label, use_container_width=True):
        st.session_state.show_hint     = not st.session_state.show_hint
        st.session_state.show_solution = False
        st.rerun()

    if st.session_state.show_hint:
        st.markdown("**💡 Hint:**")
        st.markdown(q["hint"])

    st.divider()

    # Reset progress
    if st.button("🗑️ Reset Progress", use_container_width=True):
        st.session_state.difficulties = {}
        st.session_state.miss_counts  = {}
        st.rerun()

# Persist progress at the end of each rerun.
_save_progress_to_disk()
