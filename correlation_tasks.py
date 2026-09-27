"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(
        data.csv_path,
        sep=r"\s+",
        na_values="NA",
    )

    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    women_mri_correlation = {
        feature: float(women[feature].corr(women["MRI_Count"], method="pearson"))
        for feature in features
    }

    men_mri_correlation = {
        feature: float(men[feature].corr(men["MRI_Count"], method="pearson"))
        for feature in features
    }

    all_correlations = {
        **{f"women_{feature}": value for feature, value in women_mri_correlation.items()},
        **{f"men_{feature}": value for feature, value in men_mri_correlation.items()},
    }

    strongest_key = max(
        all_correlations,
        key=lambda key: abs(all_correlations[key]),
    )

    strongest_mri_feature = strongest_key.split("_", 1)[1]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )