"""Basic data science workflow tests."""

from pathlib import Path
import sys

import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Ensure src-layout package imports in environments without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import python_reps


def _parse_sample_vcf(vcf_path: Path) -> pd.DataFrame:
    """Stream-parse VCF rows and return a typed DataFrame for downstream checks."""
    rows = []
    with vcf_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue

            cols = line.split("\t")
            info_map = {}
            for item in cols[7].split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    info_map[key] = value

            rows.append(
                {
                    "chrom": cols[0],
                    "pos": cols[1],
                    "qual": cols[5],
                    "filter": cols[6],
                    "sample": cols[9],
                    "gene": info_map.get("GENE"),
                    "dp": info_map.get("DP"),
                    "af": info_map.get("AF"),
                    "impact": info_map.get("IMPACT"),
                }
            )

    df = pd.DataFrame(rows).replace(".", pd.NA)
    for col in ["pos", "qual", "dp", "af"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean_vcf(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same interview pipeline cleaning as the app solution."""
    df_clean = df[df["filter"] == "PASS"].copy()
    df_clean["dp"] = df_clean["dp"].fillna(df_clean["dp"].median())
    return df_clean.dropna(subset=["af", "qual"])


def test_import() -> None:
    """Verify the package can be imported."""
    assert python_reps


def test_loading_and_inspecting_data(tmp_path) -> None:
    """Load a CSV and validate head/shape/dtypes style checks."""
    source = pd.DataFrame(
        {
            "cancer_type": ["LUAD", "BRCA", "LUAD"],
            "treatment": ["A", "B", "A"],
            "survival_time": [12, 18, 9],
            "time": [12, 18, 9],
            "event": [1, 0, 1],
            "response": [1, 0, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    source.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    assert df.head().shape[0] == 3
    assert df.shape == (3, 6)
    assert str(df.dtypes["survival_time"]).startswith("int")


def test_filtering_and_grouping() -> None:
    """Filter by cancer type and compute grouped mean survival time."""
    df = pd.DataFrame(
        {
            "cancer_type": ["LUAD", "LUAD", "BRCA", "BRCA"],
            "treatment": ["A", "B", "A", "B"],
            "survival_time": [10, 20, 30, 40],
        }
    )

    luad = df[df["cancer_type"] == "LUAD"]
    means = df.groupby("treatment")["survival_time"].mean()

    assert luad.shape[0] == 2
    assert means.loc["A"] == 20
    assert means.loc["B"] == 30


def test_basic_ml_random_forest_auc() -> None:
    """Train/test a RandomForestClassifier and compute ROC-AUC."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    assert 0 <= auc <= 1


def test_survival_analysis_kaplan_meier() -> None:
    """Fit a Kaplan-Meier model and validate survival output exists."""
    df = pd.DataFrame(
        {
            "time": [5, 6, 6, 2, 4, 4],
            "event": [1, 0, 1, 1, 0, 1],
        }
    )

    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], df["event"])
    ax = kmf.plot_survival_function()

    assert not kmf.survival_function_.empty
    assert kmf.survival_function_.iloc[0, 0] == 1.0
    assert ax is not None
    plt.close(ax.figure)


def test_line_parse_sample_vcf_file() -> None:
    """Parse the sample VCF file line-by-line and validate pipeline-ready outputs."""
    vcf_path = Path(__file__).resolve().parents[1] / "data" / "sample.vcf"
    assert vcf_path.exists()

    expectations = {
        "min_rows": 300,
        "min_unique_samples_after_clean": 15,
        "min_unique_genes_after_clean": 8,
    }

    df = _parse_sample_vcf(vcf_path)

    assert df.shape[0] >= expectations["min_rows"]
    assert {"chrom", "pos", "gene", "sample", "dp", "af", "impact"}.issubset(df.columns)
    assert int(df["dp"].isna().sum()) > 0
    assert int(df["af"].isna().sum()) > 0
    assert int(df["qual"].isna().sum()) > 0

    df_clean = _clean_vcf(df)

    assert not df_clean.empty
    assert df_clean["sample"].nunique() >= expectations["min_unique_samples_after_clean"]
    assert df_clean["gene"].nunique() >= expectations["min_unique_genes_after_clean"]


def test_sample_vcf_groupby_and_merge_pipeline() -> None:
    """Validate the integrated-style groupby and merge outputs are well-formed."""
    vcf_path = Path(__file__).resolve().parents[1] / "data" / "sample.vcf"
    df = _parse_sample_vcf(vcf_path)
    df_clean = _clean_vcf(df)

    gene_summary = (
        df_clean.groupby("gene")
        .agg(
            n_variants=("gene", "size"),
            mean_af=("af", "mean"),
            median_dp=("dp", "median"),
        )
        .sort_values(["n_variants", "mean_af"], ascending=[False, False])
    )

    sample_summary = (
        df_clean.groupby("sample")
        .agg(
            n_variants=("sample", "size"),
            mean_af=("af", "mean"),
            mean_qual=("qual", "mean"),
        )
        .sort_values("n_variants", ascending=False)
    )

    sample_meta = pd.DataFrame(
        {
            "sample": [f"S{i:02d}" for i in range(1, 21)],
            "cohort": ["Lung", "Lung", "Breast", "Melanoma", "Breast"] * 4,
            "age": [61, 67, pd.NA, 58, 49] * 4,
            "response": ["PR", "SD", "PD", "PR", "CR"] * 4,
        }
    )

    merged = sample_summary.reset_index().merge(sample_meta, on="sample", how="left")
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged["age"] = merged["age"].fillna(merged["age"].median())

    cohort_summary = (
        merged.groupby("cohort")
        .agg(
            samples=("sample", "nunique"),
            total_variants=("n_variants", "sum"),
            mean_af=("mean_af", "mean"),
        )
        .sort_values("total_variants", ascending=False)
    )

    assert not gene_summary.empty
    assert not sample_summary.empty
    assert gene_summary["n_variants"].sum() == len(df_clean)
    assert sample_summary["n_variants"].sum() == len(df_clean)
    assert merged["cohort"].notna().all()
    assert not cohort_summary.empty
