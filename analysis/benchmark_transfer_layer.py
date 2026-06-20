from __future__ import annotations

from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"
OUT_DIR = PROJECT / "outputs"

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

PATHWAY_SUFFIX = {
    "Theis confined": "theis",
    "Hantush-Jacob leaky": "hantush",
    "Lagging Darcy no-leakage": "lag0",
    "Lagging Darcy with leakage": "lagleak",
}

CLASS_ORDER = [
    "homogeneous confined",
    "vertical leakage",
    "finite no-flow boundary",
    "constant head boundary",
    "heterogeneous K",
    "anisotropic heterogeneous K",
    "heterogeneous storage",
    "heterogeneous leakage",
    "heterogeneous boundary",
    "combined leakage boundary",
]

CLASS_SHORT = {
    "homogeneous confined": "homogeneous",
    "vertical leakage": "leakage",
    "finite no-flow boundary": "no-flow",
    "constant head boundary": "constant-head",
    "heterogeneous K": "het K",
    "anisotropic heterogeneous K": "anisotropic K",
    "heterogeneous storage": "het S",
    "heterogeneous leakage": "het leakage",
    "heterogeneous boundary": "het boundary",
    "combined leakage boundary": "combined",
}

CLASS_TO_GROUP = {
    "homogeneous confined": "baseline",
    "vertical leakage": "leakage",
    "finite no-flow boundary": "boundary",
    "constant head boundary": "boundary",
    "heterogeneous K": "heterogeneity",
    "anisotropic heterogeneous K": "heterogeneity",
    "heterogeneous storage": "heterogeneity",
    "heterogeneous leakage": "leakage",
    "heterogeneous boundary": "boundary",
    "combined leakage boundary": "combined",
}

GROUP_ORDER = ["baseline", "leakage", "boundary", "heterogeneity", "combined"]

GROUP_SHORT = {
    "baseline": "baseline",
    "leakage": "leakage",
    "boundary": "boundary",
    "heterogeneity": "heterogeneity",
    "combined": "combined",
}


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, name)(sample))


CLASS_COLORS = {
    label: cmc_color("batlow", sample)
    for label, sample in zip(CLASS_ORDER, np.linspace(0.08, 0.92, len(CLASS_ORDER)))
}
GROUP_COLORS = {
    label: cmc_color("batlow", sample)
    for label, sample in zip(GROUP_ORDER, np.linspace(0.10, 0.88, len(GROUP_ORDER)))
}
FIELD_COLORS = {
    "Massachusetts": cmc_color("batlow", 0.28),
    "Lovelock Valley": cmc_color("batlow", 0.74),
}
NEUTRAL_DARK = cmc_color("grayC", 0.15)
NEUTRAL_MID = cmc_color("grayC", 0.45)
NEUTRAL_LIGHT = cmc_color("grayC", 0.90)


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, text: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=7.5,
    )


def export(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for base in [FIG_DIR, PROJECT]:
        for suffix in ["pdf", "png", "svg"]:
            kwargs = {"dpi": 450} if suffix == "png" else {}
            fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_benchmark_features() -> pd.DataFrame:
    cols = [
        "scenario_id",
        "scenario_class",
        "well",
        "pathway",
        "sd_eta",
        "rmse_m",
        "bic",
        "delta_bic",
        "boundary_hit",
    ]
    fits = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_fits.csv", usecols=cols)
    fits["boundary_hit"] = fits["boundary_hit"].astype(bool)

    summary = (
        fits.groupby(["scenario_id", "scenario_class", "pathway"], observed=True)
        .agg(
            median_sd_eta=("sd_eta", "median"),
            median_rmse_m=("rmse_m", "median"),
            median_delta_bic=("delta_bic", "median"),
            boundary_hit_fraction=("boundary_hit", "mean"),
            well_count=("well", "nunique"),
        )
        .reset_index()
    )

    blocks: list[pd.DataFrame] = []
    for value_col in ["median_sd_eta", "median_rmse_m", "median_delta_bic", "boundary_hit_fraction"]:
        pivot = summary.pivot_table(
            index=["scenario_id", "scenario_class"],
            columns="pathway",
            values=value_col,
            aggfunc="first",
        )
        pivot = pivot.rename(columns={p: f"{value_col}_{PATHWAY_SUFFIX[p]}" for p in PATHWAYS})
        blocks.append(pivot)

    features = pd.concat(blocks, axis=1).reset_index()

    idx = fits.groupby(["scenario_id", "well"], observed=True)["bic"].idxmin()
    best = fits.loc[idx, ["scenario_id", "pathway"]].copy()
    pref = (
        best.assign(count=1)
        .pivot_table(index="scenario_id", columns="pathway", values="count", aggfunc="sum", fill_value=0)
        .reindex(columns=PATHWAYS, fill_value=0)
    )
    pref = pref.div(pref.sum(axis=1), axis=0)
    pref = pref.rename(columns={p: f"pref_frac_{PATHWAY_SUFFIX[p]}" for p in PATHWAYS}).reset_index()

    features = features.merge(pref, on="scenario_id", how="left")

    cases = pd.read_csv(
        TABLE_DIR / "mf6_mega_benchmark_cases.csv",
        usecols=[
            "scenario_id",
            "well_count",
            "observation_count",
            "max_actual_radius_m",
            "duration_days",
            "noise_sd_m",
        ],
    )
    features = features.merge(cases, on="scenario_id", how="left")

    features["sd_gain_lag0_vs_theis"] = _safe_ratio(
        features["median_sd_eta_theis"] - features["median_sd_eta_lag0"],
        features["median_sd_eta_theis"],
    )
    features["sd_gain_lagleak_vs_theis"] = _safe_ratio(
        features["median_sd_eta_theis"] - features["median_sd_eta_lagleak"],
        features["median_sd_eta_theis"],
    )
    features["sd_gain_lagleak_vs_lag0"] = _safe_ratio(
        features["median_sd_eta_lag0"] - features["median_sd_eta_lagleak"],
        features["median_sd_eta_lag0"],
    )
    features["best_sd_eta"] = features[
        [f"median_sd_eta_{PATHWAY_SUFFIX[p]}" for p in PATHWAYS]
    ].min(axis=1)
    features["sd_eta_range"] = (
        features[[f"median_sd_eta_{PATHWAY_SUFFIX[p]}" for p in PATHWAYS]].max(axis=1)
        - features[[f"median_sd_eta_{PATHWAY_SUFFIX[p]}" for p in PATHWAYS]].min(axis=1)
    )
    features["benchmark_group"] = features["scenario_class"].map(CLASS_TO_GROUP)
    return features


def _field_summary_to_features(case_name: str, summary: pd.DataFrame, well_count: int, observation_count: int) -> pd.DataFrame:
    row = {"field_case": case_name, "well_count": float(well_count), "observation_count": float(observation_count)}
    for pathway in PATHWAYS:
        suffix = PATHWAY_SUFFIX[pathway]
        sub = summary.loc[summary["pathway"].eq(pathway)]
        if sub.empty:
            raise ValueError(f"Missing {pathway} in {case_name}")
        record = sub.iloc[0]
        row[f"median_sd_eta_{suffix}"] = float(record.get("pooled_sd_eta", record.get("median_sd_eta")))
        row[f"median_rmse_m_{suffix}"] = float(record.get("median_rmse_m", np.nan))
        row[f"median_delta_bic_{suffix}"] = float(record.get("median_abs_delta_bic_minus_theis", np.nan))
        row[f"boundary_hit_fraction_{suffix}"] = float(record.get("boundary_hit_count", 0.0)) / max(well_count, 1)
        row[f"pref_frac_{suffix}"] = float(record.get("bic_preferred_wells", 0.0)) / max(well_count, 1)
    row["max_actual_radius_m"] = np.nan
    row["duration_days"] = np.nan
    row["noise_sd_m"] = np.nan
    row["sd_gain_lag0_vs_theis"] = (
        row["median_sd_eta_theis"] - row["median_sd_eta_lag0"]
    ) / max(row["median_sd_eta_theis"], 1e-12)
    row["sd_gain_lagleak_vs_theis"] = (
        row["median_sd_eta_theis"] - row["median_sd_eta_lagleak"]
    ) / max(row["median_sd_eta_theis"], 1e-12)
    row["sd_gain_lagleak_vs_lag0"] = (
        row["median_sd_eta_lag0"] - row["median_sd_eta_lagleak"]
    ) / max(row["median_sd_eta_lag0"], 1e-12)
    sd_values = [row[f"median_sd_eta_{PATHWAY_SUFFIX[p]}"] for p in PATHWAYS]
    row["best_sd_eta"] = min(sd_values)
    row["sd_eta_range"] = max(sd_values) - min(sd_values)
    return pd.DataFrame([row])


def build_field_features() -> pd.DataFrame:
    ma = pd.read_csv(TABLE_DIR / "field_pathway_summary.csv")
    lv = pd.read_csv(TABLE_DIR / "lovelock_pathway_summary.csv")
    return pd.concat(
        [
            _field_summary_to_features("Massachusetts", ma, well_count=12, observation_count=420),
            _field_summary_to_features("Lovelock Valley", lv, well_count=7, observation_count=520),
        ],
        ignore_index=True,
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"scenario_id", "scenario_class", "benchmark_group", "field_case"}
    unavailable_or_noncomparable = {
        "max_actual_radius_m",
        "duration_days",
        "noise_sd_m",
    }
    prefixes_to_exclude = ("median_delta_bic_", "median_rmse_m_")
    columns = []
    for col in frame.columns:
        if col in excluded or col in unavailable_or_noncomparable:
            continue
        if col.startswith(prefixes_to_exclude):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            columns.append(col)
    return columns


def train_transfer_model(features: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, list[str]]:
    columns = feature_columns(features)
    X = features[columns]
    y = features["benchmark_group"].astype(str)
    classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=20260618,
        n_jobs=-1,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260618)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    classes = np.array(sorted(y.unique()))
    # cross_val_predict preserves estimator class ordering; sorted labels match scikit-learn ordering for strings.
    predicted = classes[np.argmax(probabilities, axis=1)]
    top3 = []
    for true_label, prob in zip(y.to_numpy(), probabilities):
        top3_labels = classes[np.argsort(prob)[-3:]]
        top3.append(true_label in set(top3_labels))
    validation = pd.DataFrame(
        {
            "metric": ["top1_accuracy", "top3_accuracy", "multiclass_log_loss", "scenario_count"],
            "value": [
                accuracy_score(y, predicted),
                float(np.mean(top3)),
                log_loss(y, probabilities, labels=classes),
                float(len(y)),
            ],
        }
    )
    probability_frame = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])
    probability_frame.insert(0, "predicted_group", predicted)
    probability_frame.insert(0, "benchmark_group", y.to_numpy())
    probability_frame.insert(0, "scenario_class", features["scenario_class"].to_numpy())
    probability_frame.insert(0, "scenario_id", features["scenario_id"].to_numpy())
    model.fit(X, y)

    importances = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return model, validation, importances, columns


def field_probabilities(model: Pipeline, field: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    proba = model.predict_proba(field[columns])
    classes = model.named_steps["classifier"].classes_
    rows = []
    for i, field_case in enumerate(field["field_case"]):
        order = np.argsort(proba[i])[::-1]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "field_case": field_case,
                    "rank": rank,
                    "benchmark_group": classes[idx],
                    "transfer_probability": float(proba[i, idx]),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    benchmark: pd.DataFrame,
    field: pd.DataFrame,
    model: Pipeline,
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    probabilities: pd.DataFrame,
    columns: list[str],
) -> None:
    lock_style()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(benchmark[columns]))
    field_x = scaler.transform(imputer.transform(field[columns]))
    pca = PCA(n_components=2, random_state=20260618)
    coords = pca.fit_transform(X)
    field_coords = pca.transform(field_x)
    rng = np.random.default_rng(20260618)
    sample_idx = []
    for group in GROUP_ORDER:
        idx = np.where(benchmark["benchmark_group"].to_numpy() == group)[0]
        take = min(130, idx.size)
        sample_idx.extend(rng.choice(idx, size=take, replace=False).tolist())
    sample_idx = np.array(sample_idx, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.93, wspace=0.34, hspace=0.42)

    ax = axes[0, 0]
    panel_label(ax, "a", "benchmark diagnostic space")
    for group in GROUP_ORDER:
        mask = benchmark.iloc[sample_idx]["benchmark_group"].eq(group).to_numpy()
        idx = sample_idx[mask]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=7, alpha=0.38, color=GROUP_COLORS[group], linewidths=0)
    for i, field_case in enumerate(field["field_case"]):
        ax.scatter(
            field_coords[i, 0],
            field_coords[i, 1],
            marker="*",
            s=110,
            color=FIELD_COLORS[field_case],
            edgecolor=NEUTRAL_DARK,
            linewidth=0.45,
            zorder=5,
        )
        ax.text(field_coords[i, 0], field_coords[i, 1], f" {field_case.split()[0]}", va="center", fontsize=6.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% variance)")

    ax = axes[0, 1]
    panel_label(ax, "b", "scenario-level validation")
    top1 = validation.loc[validation["metric"].eq("top1_accuracy"), "value"].iloc[0]
    top3 = validation.loc[validation["metric"].eq("top3_accuracy"), "value"].iloc[0]
    logloss = validation.loc[validation["metric"].eq("multiclass_log_loss"), "value"].iloc[0]
    ax.plot([1, 3], [top1, top3], marker="o", color=cmc_color("batlow", 0.63), linewidth=1.2)
    ax.scatter([1, 3], [top1, top3], s=35, color=cmc_color("batlow", 0.63), edgecolor=NEUTRAL_DARK, linewidth=0.4)
    ax.set_xlim(0.7, 3.3)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 3])
    ax.set_xticklabels(["top-1", "top-3"])
    ax.set_ylabel("cross-validated accuracy")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.5)
    ax.text(0.04, 0.10, f"log loss = {logloss:.2f}\nscenario split = 5 folds", transform=ax.transAxes, fontsize=6.5)

    ax = axes[1, 0]
    panel_label(ax, "c", "diagnostic variables used for transfer")
    imp = importances.head(8).iloc[::-1]
    ax.hlines(imp["feature"], 0, imp["importance"], color=NEUTRAL_MID, linewidth=0.8)
    ax.scatter(imp["importance"], imp["feature"], s=24, color=cmc_color("batlow", 0.55), edgecolor=NEUTRAL_DARK, linewidth=0.4)
    ax.set_xlabel("random-forest importance")
    ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.5)

    ax = axes[1, 1]
    panel_label(ax, "d", "field-to-benchmark transfer")
    top_probs = probabilities.loc[probabilities["rank"] <= 4].copy()
    y_positions = {"Massachusetts": 1.0, "Lovelock Valley": 0.0}
    for field_case, group in top_probs.groupby("field_case", sort=False):
        y0 = y_positions[field_case]
        group = group.sort_values("rank")
        x = np.arange(group.shape[0])
        ax.plot(x, group["transfer_probability"], color=FIELD_COLORS[field_case], linewidth=1.1)
        ax.scatter(x, group["transfer_probability"], s=28, color=FIELD_COLORS[field_case], edgecolor=NEUTRAL_DARK, linewidth=0.4)
        for xi, (_, row) in zip(x, group.iterrows()):
            ax.text(xi, row["transfer_probability"] + 0.025, GROUP_SHORT[row["benchmark_group"]], ha="center", va="bottom", fontsize=5.9, rotation=15)
        ax.text(-0.55, max(0.02, group["transfer_probability"].iloc[0]), field_case, ha="right", va="center", fontsize=6.4, color=FIELD_COLORS[field_case])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["rank 1", "rank 2", "rank 3", "rank 4"])
    ax.set_ylabel("transfer probability")
    ax.set_ylim(0, max(0.42, float(top_probs["transfer_probability"].max()) * 1.23))
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.5)

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[group], markeredgecolor="none", markersize=4.8, label=GROUP_SHORT[group])
        for group in GROUP_ORDER
    ]
    axes[0, 0].legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, -0.23), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.3)
    export(fig, "fig05_benchmark_transfer_layer")


def write_report(validation: pd.DataFrame, probabilities: pd.DataFrame, importances: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top = probabilities.loc[probabilities["rank"].le(4)].copy()
    lines = [
        "# Benchmark-to-field transfer layer",
        "",
        "This analysis trains an interpretable, shallow random-forest classifier on one row per numerical benchmark scenario.",
        "Inputs are response diagnostics that can also be computed from field aquifer tests: pathway-specific log model factor dispersion, BIC preference fractions, leakage-boundary hit fractions, and observation counts.",
        "The target is a broad process family rather than a unique numerical scenario class, because field diagnostics alone do not identify a unique generating aquifer condition.",
        "The classifier is not a drawdown surrogate and is not used as a fifth interpretation model.",
        "",
        "## Scenario-level validation",
        validation.to_markdown(index=False),
        "",
        "## Field transfer probabilities",
        top.to_markdown(index=False),
        "",
        "## Main diagnostic variables",
        importances.head(12).to_markdown(index=False),
        "",
        "Claim boundary: the learned transfer rule supports benchmark-class screening only. It does not recover true field transmissivity, storage, leakage, or response-time factors.",
    ]
    (OUT_DIR / "benchmark_transfer_layer_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    benchmark = build_benchmark_features()
    field = build_field_features()
    model, validation, importances, columns = train_transfer_model(benchmark)
    probabilities = field_probabilities(model, field, columns)

    benchmark.to_csv(TABLE_DIR / "benchmark_transfer_benchmark_features.csv", index=False)
    field.to_csv(TABLE_DIR / "benchmark_transfer_field_features.csv", index=False)
    validation.to_csv(TABLE_DIR / "benchmark_transfer_validation.csv", index=False)
    importances.to_csv(TABLE_DIR / "benchmark_transfer_feature_importance.csv", index=False)
    probabilities.to_csv(TABLE_DIR / "benchmark_field_transfer_probabilities.csv", index=False)

    make_figure(benchmark, field, model, validation, importances, probabilities, columns)
    write_report(validation, probabilities, importances)


if __name__ == "__main__":
    main()

