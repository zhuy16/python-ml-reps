"""Basic data science workflow tests."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

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


# ---------------------------------------------------------------------------
# Efficiency patterns
# ---------------------------------------------------------------------------


def test_dict_lookup_faster_than_nested_loop() -> None:
    """O(1) dict membership check outperforms O(n²) nested-loop for gene matching."""
    genes_a = [f"GENE_{i}" for i in range(500)]
    genes_b = [f"GENE_{i}" for i in range(250, 750)]

    # O(n²) nested loop
    t0 = time.perf_counter()
    nested_matches = [g for g in genes_a for g2 in genes_b if g == g2]
    nested_time = time.perf_counter() - t0

    # O(n) dict / set lookup
    t0 = time.perf_counter()
    lookup = set(genes_b)
    dict_matches = [g for g in genes_a if g in lookup]
    dict_time = time.perf_counter() - t0

    assert sorted(nested_matches) == sorted(dict_matches)
    assert dict_time < nested_time


def test_generator_lower_memory_than_list() -> None:
    """Generator expression does not materialise all items; list does."""
    import sys as _sys

    n = 10_000

    list_obj = [x * 2 for x in range(n)]
    gen_obj = (x * 2 for x in range(n))

    # A generator has a tiny, fixed footprint regardless of n
    assert _sys.getsizeof(gen_obj) < _sys.getsizeof(list_obj)

    # Both produce the same values
    assert list(gen_obj) == list_obj


def test_list_comprehension_filter() -> None:
    """List-comprehension filter mirrors dict-based QC threshold filtering."""
    qc_dict = {
        f"cell_{i}": {"n_genes": 200 + i * 3, "pct_mt": 5 + (i % 30)}
        for i in range(100)
    }

    high_quality = [
        cell
        for cell, qc in qc_dict.items()
        if qc["n_genes"] > 200 and qc["pct_mt"] < 20
    ]

    assert isinstance(high_quality, list)
    assert all(qc_dict[c]["n_genes"] > 200 for c in high_quality)
    assert all(qc_dict[c]["pct_mt"] < 20 for c in high_quality)


# ---------------------------------------------------------------------------
# Threading patterns
# ---------------------------------------------------------------------------


def test_threadpoolexecutor_parallel_file_parse(tmp_path) -> None:
    """ThreadPoolExecutor processes multiple VCF-style files concurrently."""
    vcf_src = Path(__file__).resolve().parents[1] / "data" / "sample.vcf"
    assert vcf_src.exists()

    # Create a few copies to simulate multiple files
    copies = []
    for i in range(3):
        dest = tmp_path / f"sample_{i}.vcf"
        dest.write_bytes(vcf_src.read_bytes())
        copies.append(dest)

    def process_one(path: Path) -> tuple[Path, int]:
        df = _parse_sample_vcf(path)
        return path, len(df)

    results: dict[Path, int] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_one, p): p for p in copies}
        for future in as_completed(futures):
            path, count = future.result()
            results[path] = count

    assert len(results) == len(copies)
    assert all(count > 0 for count in results.values())
    # All copies of the same file must parse to the same row count
    counts = list(results.values())
    assert counts[0] == counts[1] == counts[2]


def test_threadpoolexecutor_exception_handling(tmp_path) -> None:
    """Exceptions from worker threads are re-raised via future.result()."""
    bad_path = tmp_path / "nonexistent.vcf"
    good_path = Path(__file__).resolve().parents[1] / "data" / "sample.vcf"

    def process_one(path: Path) -> tuple[Path, int]:
        df = _parse_sample_vcf(path)
        return path, len(df)

    errors = []
    results: dict[Path, int] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_one, p): p for p in [good_path, bad_path]}
        for future in as_completed(futures):
            try:
                path, count = future.result()
                results[path] = count
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    assert len(errors) == 1
    assert len(results) == 1
