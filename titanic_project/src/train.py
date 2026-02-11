"""
모델 학습 및 5-Fold 교차검증 모듈

모델 비교: Logistic Regression, Random Forest, XGBoost
5-Fold Cross-Validation으로 평균 Accuracy 비교
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb


RANDOM_STATE = 42

# 학습에 사용할 feature 컬럼
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FareBin",
    "Sex_encoded",
    "Embarked_encoded",
    "Title_encoded",
]


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """비교할 3개 모델 반환"""
    return {
        "LogisticRegression": LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, random_state=random_state, eval_metric="logloss"
        ),
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    5-Fold 교차검증으로 모델 비교
    
    Returns:
        DataFrame: 모델별 평균 Accuracy 및 표준편차
    """
    models = get_models(random_state)
    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=n_folds, scoring="accuracy"
        )
        results.append({
            "Model": name,
            "Mean_Accuracy": scores.mean(),
            "Std_Accuracy": scores.std(),
        })
        print(f"{name}: Mean Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return pd.DataFrame(results)


def train_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "RandomForest",
    random_state: int = RANDOM_STATE,
):
    """
    최고 성능 모델로 학습
    
    Returns:
        fitted model
    """
    models = get_models(random_state)
    model = models[model_name]
    model.fit(X, y)
    return model


def save_model(model, path: str) -> None:
    """모델 저장"""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """모델 로드"""
    with open(path, "rb") as f:
        return pickle.load(f)
