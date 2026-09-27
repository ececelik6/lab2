"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    row_count = len(df)

    missing_by_column = df.isna().sum().astype(int).to_dict()

    adults_over_30_count = int((df["Age"] > 30).sum())

    mean_age_by_pclass = (
        df.groupby("Pclass")["Age"]
        .mean()
        .to_dict()
    )

    survival_rate_by_pclass = (
        df.groupby("Pclass")["Survived"]
        .mean()
        .to_dict()
    )

    highest_fares = (
        df["Fare"]
        .nlargest(5)
        .tolist()
    )

    mean_age_by_pclass = {
        int(key): float(value)
        for key, value in mean_age_by_pclass.items()
    }

    survival_rate_by_pclass = {
        int(key): float(value)
        for key, value in survival_rate_by_pclass.items()
    }

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )