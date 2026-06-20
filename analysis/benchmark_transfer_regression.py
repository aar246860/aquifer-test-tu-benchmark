from __future__ import annotations

from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
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
TARGETS = {
    "p90_abs_lnM_T": "T",
    "p90_abs_lnM_S": "S",
    "p90_abs_lnM_response_time": "response time",
}


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, name)(sample))


GROUP_COLORS = {
    group: cmc_color("batlow", sample)
    for group, sample in zip(GROUP_ORDER, np.linspace(0.10, 0.88, len(GROUP_ORDER)))
}
TARGET_COLORS = {
    "p90_abs_lnM_T": cmc_color("batlow", 0.20),
    "p90_abs_lnM_S": cmc_color("batlow", 0.55),
    "p90_abs_lnM_response_time": cmc_color("batlow", 0.85),
}
FIELD_COLORS = {
    "Massachusetts": cmc_color("batlow", 0.28),
    "Lovelock Valley": cmc_color("batlow", 0.74),
}
NEUTRAL_DARK = cmc_color("grayC", 0.16)
NEUTRAL_MID = cmc_color("grayC", 0.47)
NEUTRAL_LIGHT = cmc_color("grayC", 0.90)
DIAGNOSTIC_LABELS = {
    "median_sd_eta_theis": "Theis residual dispersion",
    "median_sd_eta_hantush": "Hantush-Jacob residual dispersion",
    "median_sd_eta_lag0": "Lagging Darcy residual dispersion",
    "median_sd_eta_lagleak": "Lagging leaky residual dispersion",
    "boundary_hit_fraction_theis": "Theis boundary-limit fraction",
    "boundary_hit_fraction_hantush": "Hantush-Jacob boundary-limit fraction",
    "boundary_hit_fraction_lag0": "Lagging Darcy boundary-limit fraction",
    "boundary_hit_fraction_lagleak": "Lagging leaky boundary-limit fraction",
    "pref_frac_theis": "Theis BIC preference",
    "pref_frac_hantush": "Hantush-Jacob BIC preference",
    "pref_frac_lag0": "Lagging Darcy BIC preference",
    "pref_frac_lagleak": "Lagging leaky BIC preference",
    "sd_gain_lag0_vs_theis": "Residual gain: lagging vs. Theis",
    "sd_gain_lagleak_vs_theis": "Residual gain: lagging leaky vs. Theis",
    "sd_gain_lagleak_vs_lag0": "Residual gain: leakage with lagging",
    "best_sd_eta": "Lowest residual dispersion",
    "sd_eta_range": "Residual-dispersion contrast",
    "well_count": "Number of observation wells",
    "observation_count": "Number of response observations",
}


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"({label}) {title}",
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
    return numerator / denominator.replace(0, np.nan)


def _q90_abs(series: pd.Series) -> float:
    values = np.abs(pd.to_numeric(series, errors="coerce").dropna().to_numpy(float))
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 0.90))


def build_benchmark_data() -> pd.DataFrame:
    usecols = [
        "scenario_id",
        "scenario_class",
        "well",
        "pathway",
        "sd_eta",
        "bic",
        "boundary_hit",
        "quality_gate",
        "lnM_T",
        "lnM_S",
        "lnM_response_time",
    ]
    fits = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_fits.csv", usecols=usecols)
    fits["boundary_hit"] = fits["boundary_hit"].astype(bool)
    fits["quality_gate"] = fits["quality_gate"].astype(bool)

    grouped = (
        fits.groupby(["scenario_id", "scenario_class", "pathway"], observed=True)
        .agg(
            median_sd_eta=("sd_eta", "median"),
            boundary_hit_fraction=("boundary_hit", "mean"),
        )
        .reset_index()
    )

    pivots = []
    for value in ["median_sd_eta", "boundary_hit_fraction"]:
        pivot = grouped.pivot_table(
            index=["scenario_id", "scenario_class"],
            columns="pathway",
            values=value,
            aggfunc="first",
        )
        pivot = pivot.rename(columns={path: f"{value}_{PATHWAY_SUFFIX[path]}" for path in PATHWAYS})
        pivots.append(pivot)
    features = pd.concat(pivots, axis=1).reset_index()

    idx = fits.groupby(["scenario_id", "well"], observed=True)["bic"].idxmin()
    best = fits.loc[idx, ["scenario_id", "pathway"]]
    pref = (
        best.assign(count=1)
        .pivot_table(index="scenario_id", columns="pathway", values="count", aggfunc="sum", fill_value=0)
        .reindex(columns=PATHWAYS, fill_value=0)
    )
    pref = pref.div(pref.sum(axis=1), axis=0)
    pref = pref.rename(columns={path: f"pref_frac_{PATHWAY_SUFFIX[path]}" for path in PATHWAYS}).reset_index()
    features = features.merge(pref, on="scenario_id", how="left")

    sd_cols = [f"median_sd_eta_{PATHWAY_SUFFIX[path]}" for path in PATHWAYS]
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
    features["best_sd_eta"] = features[sd_cols].min(axis=1)
    features["sd_eta_range"] = features[sd_cols].max(axis=1) - features[sd_cols].min(axis=1)
    features["benchmark_group"] = features["scenario_class"].map(CLASS_TO_GROUP)

    gated = fits[fits["quality_gate"]].copy()
    targets = (
        gated.groupby("scenario_id", observed=True)
        .agg(
            p90_abs_lnM_T=("lnM_T", _q90_abs),
            p90_abs_lnM_S=("lnM_S", _q90_abs),
            p90_abs_lnM_response_time=("lnM_response_time", _q90_abs),
        )
        .reset_index()
    )
    return features.merge(targets, on="scenario_id", how="inner")


def _field_summary_to_features(case_name: str, summary: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"field_case": case_name}
    well_count = float(summary["bic_preferred_wells"].sum())
    for pathway in PATHWAYS:
        suffix = PATHWAY_SUFFIX[pathway]
        rec = summary.loc[summary["pathway"].eq(pathway)].iloc[0]
        sd_value = float(rec.get("pooled_sd_eta", rec.get("median_sd_eta")))
        row[f"median_sd_eta_{suffix}"] = sd_value
        row[f"boundary_hit_fraction_{suffix}"] = float(rec.get("boundary_hit_count", 0.0)) / max(well_count, 1.0)
        row[f"pref_frac_{suffix}"] = float(rec.get("bic_preferred_wells", 0.0)) / max(well_count, 1.0)
    row["sd_gain_lag0_vs_theis"] = (
        float(row["median_sd_eta_theis"]) - float(row["median_sd_eta_lag0"])
    ) / max(float(row["median_sd_eta_theis"]), 1e-12)
    row["sd_gain_lagleak_vs_theis"] = (
        float(row["median_sd_eta_theis"]) - float(row["median_sd_eta_lagleak"])
    ) / max(float(row["median_sd_eta_theis"]), 1e-12)
    row["sd_gain_lagleak_vs_lag0"] = (
        float(row["median_sd_eta_lag0"]) - float(row["median_sd_eta_lagleak"])
    ) / max(float(row["median_sd_eta_lag0"]), 1e-12)
    sd_values = [float(row[f"median_sd_eta_{PATHWAY_SUFFIX[path]}"]) for path in PATHWAYS]
    row["best_sd_eta"] = min(sd_values)
    row["sd_eta_range"] = max(sd_values) - min(sd_values)
    return row


def build_field_data() -> pd.DataFrame:
    ma = pd.read_csv(TABLE_DIR / "field_pathway_summary.csv")
    lv = pd.read_csv(TABLE_DIR / "lovelock_pathway_summary.csv")
    return pd.DataFrame(
        [
            _field_summary_to_features("Massachusetts", ma),
            _field_summary_to_features("Lovelock Valley", lv),
        ]
    )


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {"scenario_id", "scenario_class", "benchmark_group", "field_case", *TARGETS.keys()}
    return [col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])]


def make_regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=9,
                    min_samples_leaf=8,
                    random_state=20260618,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_transfer_models(data: pd.DataFrame, columns: list[str]) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = data[columns]
    cv = KFold(n_splits=5, shuffle=True, random_state=20260618)
    models: dict[str, Pipeline] = {}
    validation_rows = []
    cv_rows = []
    importance_frames = []
    for target in TARGETS:
        y = data[target]
        model = make_regressor()
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        validation_rows.append(
            {
                "target": target,
                "label": TARGETS[target],
                "r2": r2_score(y, pred),
                "mae_abs_log": mean_absolute_error(y, pred),
                "n_scenarios": len(y),
            }
        )
        cv_rows.append(pd.DataFrame({"scenario_id": data["scenario_id"], "target": target, "observed": y, "predicted": pred}))
        model.fit(X, y)
        models[target] = model
        imp = pd.DataFrame(
            {
                "feature": columns,
                "importance": model.named_steps["regressor"].feature_importances_,
                "target": target,
            }
        )
        importance_frames.append(imp)
    validation = pd.DataFrame(validation_rows)
    cv_pred = pd.concat(cv_rows, ignore_index=True)
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    return models, validation, cv_pred, importance


def equivalent_cov(p90_abs_log: float) -> float:
    sigma = max(float(p90_abs_log), 0.0) / 1.645
    return float(np.sqrt(np.exp(sigma * sigma) - 1.0))


def predict_field(models: dict[str, Pipeline], field: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for _, field_row in field.iterrows():
        one = field_row.to_frame().T
        for target, label in TARGETS.items():
            p90_abs_log = float(models[target].predict(one[columns])[0])
            rows.append(
                {
                    "field_case": field_row["field_case"],
                    "quantity": label,
                    "target": target,
                    "predicted_p90_abs_log_model_factor": p90_abs_log,
                    "p90_equivalent_lognormal_cov": equivalent_cov(p90_abs_log),
                }
            )
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame, field: pd.DataFrame, columns: list[str], validation: pd.DataFrame, cv_pred: pd.DataFrame, importance: pd.DataFrame, field_pred: pd.DataFrame) -> None:
    lock_style()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(data[columns]))
    field_x = scaler.transform(imputer.transform(field[columns]))
    pca = PCA(n_components=2, random_state=20260618)
    coords = pca.fit_transform(X)
    field_coords = pca.transform(field_x)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.25), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.93, wspace=0.34, hspace=0.48)

    ax = axes[0, 0]
    panel_label(ax, "a", "benchmark diagnostic space")
    rng = np.random.default_rng(20260618)
    for group in GROUP_ORDER:
        idx = np.where(data["benchmark_group"].to_numpy() == group)[0]
        idx = rng.choice(idx, size=min(220, idx.size), replace=False)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=7, alpha=0.32, color=GROUP_COLORS[group], linewidths=0, label=group)
    for i, field_case in enumerate(field["field_case"]):
        ax.scatter(field_coords[i, 0], field_coords[i, 1], marker="*", s=110, color=FIELD_COLORS[field_case], edgecolor=NEUTRAL_DARK, linewidth=0.45, zorder=5)
        ax.text(field_coords[i, 0], field_coords[i, 1], f" {field_case.split()[0]}", va="center", fontsize=6.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% variance)")
    ax.legend(loc="lower left", frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.8)

    ax = axes[0, 1]
    panel_label(ax, "b", "scenario-wise validation")
    sample = pd.concat(
        [
            group.sample(min(650, len(group)), random_state=20260618)
            for _, group in cv_pred.groupby("target", sort=False)
        ],
        ignore_index=True,
    )
    for target, label in TARGETS.items():
        sub = sample[sample["target"].eq(target)]
        ax.scatter(sub["observed"], sub["predicted"], s=5, alpha=0.22, color=TARGET_COLORS[target], linewidths=0, label=label)
    lim = float(np.nanquantile(pd.concat([cv_pred["observed"], cv_pred["predicted"]]), 0.995))
    ax.plot([0, lim], [0, lim], color=NEUTRAL_DARK, linewidth=0.8, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("observed p90 |ln M|")
    ax.set_ylabel("withheld-scenario estimate")
    ax.legend(frameon=False, loc="upper left")
    txt = "\n".join(f"{row.label}: $R^2$={row.r2:.2f}" for row in validation.itertuples())
    ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)

    ax = axes[1, 0]
    panel_label(ax, "c", "diagnostic variables")
    imp = importance.head(8).copy()
    imp["label"] = imp["feature"].map(DIAGNOSTIC_LABELS).fillna(imp["feature"]).astype(str)
    imp = imp.iloc[::-1]
    ax.hlines(imp["label"], 0, imp["importance"], color=NEUTRAL_MID, linewidth=0.8)
    ax.scatter(imp["importance"], imp["label"], s=24, color=cmc_color("batlow", 0.55), edgecolor=NEUTRAL_DARK, linewidth=0.4)
    ax.set_xlabel("diagnostic contribution")
    ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.5)

    ax = axes[1, 1]
    panel_label(ax, "d", "field transferred model-factor scale")
    quantities = ["T", "S", "response time"]
    y_positions = np.arange(len(quantities))
    offsets = {"Massachusetts": 0.09, "Lovelock Valley": -0.09}
    for field_case, sub in field_pred.groupby("field_case", sort=False):
        ordered = sub.set_index("quantity").reindex(quantities)
        y = y_positions + offsets[field_case]
        ax.plot(ordered["p90_equivalent_lognormal_cov"], y, color=FIELD_COLORS[field_case], linewidth=1.0, alpha=0.85)
        ax.scatter(ordered["p90_equivalent_lognormal_cov"], y, s=30, color=FIELD_COLORS[field_case], edgecolor=NEUTRAL_DARK, linewidth=0.4, label=field_case)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(quantities)
    ax.set_xlabel("p90-equivalent model-factor COV")
    ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")

    export(fig, "fig05_benchmark_transfer_layer")


def write_report(validation: pd.DataFrame, field_pred: pd.DataFrame, importance: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_report = validation.rename(
        columns={
            "target": "benchmark_quantity",
            "label": "quantity",
        }
    )
    field_report = field_pred.rename(columns={"target": "benchmark_quantity"})
    importance_report = importance.copy()
    importance_report.insert(
        0,
        "diagnostic_variable",
        importance_report["feature"].map(DIAGNOSTIC_LABELS).fillna(importance_report["feature"]),
    )
    lines = [
        "# Benchmark-to-field transfer analysis",
        "",
        "The benchmark-transfer analysis estimates the magnitude of numerical benchmark model factors from diagnostics that can also be computed for field aquifer tests.",
        "It reports benchmark-conditioned model-factor magnitude while the numerical responses and four analytical interpretations retain separate roles.",
        "The estimated quantity is the 90th percentile absolute log model factor, converted to a p90-equivalent lognormal COV for reporting.",
        "",
        "## Scenario-wise validation",
        validation_report.to_markdown(index=False),
        "",
        "## Field transferred model-factor scale",
        field_report.to_markdown(index=False),
        "",
        "## Main diagnostic variables",
        importance_report[["diagnostic_variable", "importance"]].head(12).to_markdown(index=False),
        "",
        "The transfer estimates are benchmark-conditioned sensitivity brackets for field diagnostics represented by the numerical benchmark.",
    ]
    (OUT_DIR / "benchmark_transfer_regression_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = build_benchmark_data()
    field = build_field_data()
    columns = feature_columns(data)
    models, validation, cv_pred, importance = train_transfer_models(data, columns)
    field_pred = predict_field(models, field, columns)

    data.to_csv(TABLE_DIR / "benchmark_transfer_regression_benchmark.csv", index=False)
    field.to_csv(TABLE_DIR / "benchmark_transfer_regression_field_features.csv", index=False)
    validation.to_csv(TABLE_DIR / "benchmark_transfer_regression_validation.csv", index=False)
    cv_pred.to_csv(TABLE_DIR / "benchmark_transfer_regression_cv_predictions.csv", index=False)
    importance.to_csv(TABLE_DIR / "benchmark_transfer_regression_feature_importance.csv", index=False)
    field_pred.to_csv(TABLE_DIR / "benchmark_field_transfer_regression_predictions.csv", index=False)

    make_figure(data, field, columns, validation, cv_pred, importance, field_pred)
    write_report(validation, field_pred, importance)


if __name__ == "__main__":
    main()

