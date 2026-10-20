"""Funcoes auxiliares do LAB de Random Forest (Titanic e Vega Datasets).

Preserva copias locais dos conjuntos de dados do Titanic (train.csv, test.csv e
gender_submission.csv) e centraliza o pre-processamento, as estrategias de
imputacao estatistica de idade e a preparacao do dataset cars do vega_datasets
para que o notebook seja 100% reproduzivel offline.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from vega_datasets import data


BASE_DIR = Path(__file__).resolve().parent
TITANIC_DIR = BASE_DIR / "titanic"

TITANIC_URLS = {
    "train.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "test.csv": "https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv",
}

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

TITLE_CODE = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
EMBARKED_CODE = {"S": 1, "C": 2, "Q": 3}
SEX_CODE = {"male": 100, "female": 200}


def ensure_titanic_data(output_dir: str | Path = TITANIC_DIR) -> Path:
    """Garante que train.csv, test.csv e gender_submission.csv existam localmente."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in TITANIC_URLS.items():
        filepath = target_dir / filename
        if not filepath.exists() or filepath.stat().st_size < 100:
            urllib.request.urlretrieve(url, filepath)

    gender_sub_path = target_dir / "gender_submission.csv"
    if not gender_sub_path.exists() or gender_sub_path.stat().st_size < 100:
        test_df = pd.read_csv(target_dir / "test.csv")
        sub = pd.DataFrame(
            {
                "PassengerId": test_df["PassengerId"],
                "Survived": (test_df["Sex"] == "female").astype(int),
            }
        )
        sub.to_csv(gender_sub_path, index=False)
    return target_dir


def impute_age_series(df: pd.DataFrame, strategy: str, random_state: int = 42) -> pd.Series:
    """Preenche valores ausentes na coluna Age segundo a estrategia informada."""

    age = df["Age"].copy()
    if strategy == "media":
        return age.fillna(age.mean())
    if strategy == "mediana":
        return age.fillna(age.median())
    if strategy == "moda":
        return age.fillna(age.mode().iloc[0])
    if strategy == "media_por_sexo":
        return df.groupby("Sex")["Age"].transform(lambda s: s.fillna(s.mean()))
    if strategy == "mediana_sexo_classe":
        return df.groupby(["Sex", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
    if strategy == "interpolacao_linear":
        return age.interpolate(method="linear", limit_direction="both")
    if strategy == "normal_media_std":
        rng = np.random.default_rng(random_state)
        missing_mask = age.isna()
        mean_val = age.mean()
        std_val = age.std()
        samples = rng.normal(loc=mean_val, scale=std_val, size=int(missing_mask.sum()))
        age.loc[missing_mask] = np.clip(samples, 0.5, 80.0)
        return age
    raise ValueError(f"Estrategia desconhecida para Age: {strategy!r}")


def evaluate_age_imputations(train_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Compara estrategias de preenchimento de Age usando validacao cruzada."""

    strategies = [
        ("Media global", "media"),
        ("Mediana global", "mediana"),
        ("Moda global", "moda"),
        ("Media por sexo", "media_por_sexo"),
        ("Mediana por sexo e classe", "mediana_sexo_classe"),
        ("Interpolacao linear", "interpolacao_linear"),
        ("Amostragem Normal (media ± std)", "normal_media_std"),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y = train_df["Survived"].to_numpy()
    rows = []

    for label, key in strategies:
        temp = train_df.copy()
        temp["Age_imputed"] = impute_age_series(temp, key, random_state=random_state)
        temp["SexNumber"] = temp["Sex"].map(SEX_CODE)
        X = temp[["Pclass", "SexNumber", "Age_imputed"]]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
        )
        probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)

        rows.append(
            {
                "estrategia": label,
                "idade_media_pos": float(temp["Age_imputed"].mean()),
                "idade_std_pos": float(temp["Age_imputed"].std()),
                "acuracia_cv": float(accuracy_score(y, preds)),
                "f1_cv": float(f1_score(y, preds)),
                "auc_cv": float(roc_auc_score(y, probs)),
            }
        )

    return pd.DataFrame(rows).sort_values("auc_cv", ascending=False).reset_index(drop=True)


def extract_title(name_series: pd.Series) -> pd.Series:
    """Extrai o titulo social do nome do passageiro e agrupa titulos raros."""

    raw_title = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    return raw_title.map(lambda t: TITLE_MAP.get(t, "Rare"))


def clean_titanic_data(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Limpa e enriquece o dataframe do Titanic preservando idades reais (< 1 ano)."""

    ref = reference_df if reference_df is not None else df
    out = df.copy()

    out["Title"] = extract_title(out["Name"])
    ref_with_title = ref.copy()
    ref_with_title["Title"] = extract_title(ref_with_title["Name"])

    # Mediana de idade por (Sex, Pclass, Title) calculada na base de referencia
    group_medians = (
        ref_with_title.groupby(["Sex", "Pclass", "Title"])["Age"]
        .median()
        .to_dict()
    )
    fallback_medians = ref_with_title.groupby(["Sex", "Pclass"])["Age"].median().to_dict()
    global_median = float(ref_with_title["Age"].median())

    def _fill_age(row: pd.Series) -> float:
        if pd.notna(row["Age"]):
            return float(row["Age"])
        val = group_medians.get((row["Sex"], row["Pclass"], row["Title"]), np.nan)
        if pd.isna(val):
            val = fallback_medians.get((row["Sex"], row["Pclass"]), global_median)
        return float(val)

    out["Age"] = out.apply(_fill_age, axis=1)

    # Fare: mediana por Pclass
    fare_medians = ref.groupby("Pclass")["Fare"].median().to_dict()
    out["Fare"] = out.apply(
        lambda r: float(r["Fare"])
        if pd.notna(r["Fare"])
        else float(fare_medians.get(r["Pclass"], ref["Fare"].median())),
        axis=1,
    )
    out["LogFare"] = np.log1p(out["Fare"])

    # Embarked: moda da referencia
    embarked_mode = str(ref["Embarked"].mode().iloc[0])
    out["Embarked"] = out["Embarked"].fillna(embarked_mode)

    # Engenharia de atributos familiares
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Codificacoes numericas
    out["SexNumber"] = out["Sex"].map(SEX_CODE).astype(int)
    out["EmbarkedCode"] = out["Embarked"].map(EMBARKED_CODE).fillna(1).astype(int)
    out["TitleCode"] = out["Title"].map(TITLE_CODE).fillna(5).astype(int)

    return out


CLEAN_FEATURES = [
    "Pclass",
    "SexNumber",
    "Age",
    "SibSp",
    "Parch",
    "LogFare",
    "FamilySize",
    "IsAlone",
    "EmbarkedCode",
    "TitleCode",
]


def load_and_prepare_cars() -> pd.DataFrame:
    """Carrega e limpa o dataset cars do vega_datasets para classificacao e regressao."""

    cars = data.cars().copy()
    cars["YearNum"] = pd.to_datetime(cars["Year"]).dt.year
    # Imputa Horsepower pela mediana de cilindros e descarta linhas sem consumo MPG
    cars["Horsepower"] = cars.groupby("Cylinders")["Horsepower"].transform(
        lambda s: s.fillna(s.median())
    )
    cars = cars.dropna(subset=["Miles_per_Gallon", "Horsepower", "Weight_in_lbs", "Origin"]).copy()
    cars["OriginCode"] = cars["Origin"].map({"USA": 0, "Europe": 1, "Japan": 2}).astype(int)
    return cars.reset_index(drop=True)
