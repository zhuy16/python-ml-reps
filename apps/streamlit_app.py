import streamlit as st
import random
import time
from streamlit_ace import st_ace

st.set_page_config(
    page_title="🧠 Python ML Reps",
    layout="wide",
    page_icon="🧠",
)

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

CATEGORY_ICON = {
    "ML/Statistics": "🔵",
    "Pandas/EDA":    "🟢",
    "Python Basics": "🟡",
}

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
_defaults = {
    "current_q_id":  QUESTIONS[0]["id"],
    "show_hint":     False,
    "show_solution": False,
    "difficulties":  {},   # q_id → "easy" | "hard"
    "miss_counts":   {},   # q_id → int
    "hard_only":     False,
    "timer_start":   time.time(),
    "user_code":     {},   # q_id → str  (persists per question)
    "run_output":    "",
    "run_error":     "",
    "run_figures":   [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _pool() -> list[str]:
    if st.session_state.hard_only:
        hard = [qid for qid, d in st.session_state.difficulties.items() if d == "hard"]
        return hard if hard else [q["id"] for q in QUESTIONS]
    return [q["id"] for q in QUESTIONS]


def _current() -> dict:
    return next(q for q in QUESTIONS if q["id"] == st.session_state.current_q_id)


def _go_next():
    pool = _pool()
    others = [qid for qid in pool if qid != st.session_state.current_q_id]
    st.session_state.current_q_id = random.choice(others) if others else st.session_state.current_q_id
    st.session_state.show_hint     = False
    st.session_state.show_solution = False
    st.session_state.run_output    = ""
    st.session_state.run_error     = ""
    st.session_state.run_figures   = []
    st.session_state.timer_start   = time.time()


def _run_code(code: str):
    """Execute user code, capture stdout + matplotlib figures."""
    import io, contextlib, traceback, matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")          # non-interactive backend, safe for Streamlit
    plt.close("all")               # clear any leftover figures
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<editor>", "exec"), {})  # isolated namespace
        st.session_state.run_output  = buf.getvalue() or ""
        st.session_state.run_error   = ""
        # Grab every figure matplotlib created during exec
        st.session_state.run_figures = [plt.figure(n) for n in plt.get_fignums()]
    except Exception:
        st.session_state.run_output  = buf.getvalue()
        st.session_state.run_error   = traceback.format_exc()
        st.session_state.run_figures = []


def _elapsed() -> str:
    secs = int(time.time() - st.session_state.timer_start)
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
hc1, hc2, hc3, hc4 = st.columns([4, 1, 1, 1])
with hc1:
    st.title("🧠 Python ML Reps")
    st.caption("Active-recall training · 10 topics · ML, Pandas, Python")
with hc2:
    easy_n = sum(1 for v in st.session_state.difficulties.values() if v == "easy")
    st.metric("✅ Easy", easy_n)
with hc3:
    hard_n = sum(1 for v in st.session_state.difficulties.values() if v == "hard")
    st.metric("🔴 Hard", hard_n)
with hc4:
    done_n = len(st.session_state.difficulties)
    st.metric("📝 Tagged", f"{done_n}/{len(QUESTIONS)}")

st.divider()

q = _current()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT  ·  LEFT · CENTER · RIGHT
# ──────────────────────────────────────────────────────────────────────────────
left, center, right = st.columns([2, 3, 2])

# ── LEFT: question prompt ─────────────────────────────────────────────────────
with left:
    cat_icon = CATEGORY_ICON.get(q["category"], "⚪")
    diff     = st.session_state.difficulties.get(q["id"])
    diff_badge = {"easy": " ✅", "hard": " 🔴"}.get(diff, "")

    st.markdown(f"**{cat_icon} {q['category']}**&nbsp;&nbsp; ⏱️ `{_elapsed()}`")
    st.markdown(f"## {q['title']}{diff_badge}")
    st.markdown(q["prompt"])

    st.divider()

    if st.button("🎲 Next Random", use_container_width=True, type="primary"):
        _go_next()
        st.rerun()

    st.divider()
    st.markdown("**📌 Jump to a topic:**")
    topic_labels = [
        f"{CATEGORY_ICON.get(tq['category'], '⚪')} {tq['title']}"
        + ({"easy": " ✅", "hard": " 🔴"}.get(st.session_state.difficulties.get(tq["id"]), ""))
        for tq in QUESTIONS
    ]
    current_idx = next(i for i, tq in enumerate(QUESTIONS) if tq["id"] == q["id"])
    chosen_idx = st.selectbox(
        label="Select question",
        options=range(len(QUESTIONS)),
        format_func=lambda i: topic_labels[i],
        index=current_idx,
        label_visibility="collapsed",
    )
    if chosen_idx != current_idx:
        st.session_state.current_q_id  = QUESTIONS[chosen_idx]["id"]
        st.session_state.show_hint     = False
        st.session_state.show_solution = False
        st.session_state.run_output    = ""
        st.session_state.run_error     = ""
        st.session_state.run_figures   = []
        st.session_state.timer_start   = time.time()
        st.rerun()

# ── CENTER: workspace + tracker ───────────────────────────────────────────────
with center:
    st.markdown("### 💻 Your Workspace")
    st.info(q["workspace_tip"])

    # ── Code editor ──────────────────────────────────────────────────────────
    _default_code = f"# {q['title']}\n# Write your solution here\n\n"
    current_code = st.session_state.user_code.get(q["id"], _default_code)

    st.markdown("**✏️ Code Editor**")
    user_code = st_ace(
        value=current_code,
        language="python",
        theme="monokai",
        key=f"editor_{q['id']}",
        height=300,
        tab_size=4,
        font_size=14,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
    )
    # Persist per question
    if user_code is not None:
        st.session_state.user_code[q["id"]] = user_code
    else:
        user_code = current_code

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        if st.button("▶ Run Code", type="primary", use_container_width=True):
            _run_code(user_code)
    with rc2:
        if st.button("🗑️ Clear Editor", use_container_width=True):
            st.session_state.user_code[q["id"]] = _default_code
            st.session_state.run_output  = ""
            st.session_state.run_error   = ""
            st.session_state.run_figures = []
            st.rerun()

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

    # Miss tracker
    if st.session_state.miss_counts:
        st.divider()
        st.markdown("#### 📊 Miss Tracker — Top 5")
        sorted_misses = sorted(
            st.session_state.miss_counts.items(), key=lambda x: x[1], reverse=True
        )
        for qid, cnt in sorted_misses[:5]:
            name = next((tq["title"] for tq in QUESTIONS if tq["id"] == qid), qid)
            dots = "🔴" * min(cnt, 6)
            st.markdown(f"- **{name}** {dots} ×{cnt}")

    # Topic overview with click-to-jump buttons (3-column grid)
    st.divider()
    st.markdown("**🗂️ Topics**")
    grid = st.columns(3)
    for i, tq in enumerate(QUESTIONS):
        icon  = CATEGORY_ICON.get(tq["category"], "⚪")
        badge = {"easy": "✅", "hard": "🔴"}.get(
            st.session_state.difficulties.get(tq["id"]), ""
        )
        is_current = tq["id"] == q["id"]
        # Short label: just the icon + abbreviated title + badge
        short_title = tq["title"].split("—")[0].split("(")[0].strip()
        label = f"{icon} {short_title} {badge}{'▶' if is_current else ''}"
        btn_type = "primary" if is_current else "secondary"
        with grid[i % 3]:
            if st.button(label, key=f"nav_{tq['id']}", use_container_width=True, type=btn_type):
                if not is_current:
                    st.session_state.current_q_id  = tq["id"]
                    st.session_state.show_hint     = False
                    st.session_state.show_solution = False
                    st.session_state.run_output    = ""
                    st.session_state.run_error     = ""
                    st.session_state.run_figures   = []
                    st.session_state.timer_start   = time.time()
                    st.rerun()

# ── RIGHT: reference panel ────────────────────────────────────────────────────
with right:
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

    # Solution toggle
    sol_label = "🔍 Hide Solution" if st.session_state.show_solution else "🔍 Show Solution"
    if st.button(sol_label, use_container_width=True):
        st.session_state.show_solution = not st.session_state.show_solution
        st.rerun()

    if st.session_state.show_solution:
        st.markdown("**🔍 Full Solution:**")
        st.markdown(q["solution"])

    st.divider()

    # Reset progress
    if st.button("🗑️ Reset Progress", use_container_width=True):
        st.session_state.difficulties = {}
        st.session_state.miss_counts  = {}
        st.rerun()
