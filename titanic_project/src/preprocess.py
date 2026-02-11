"""
결측치 처리 모듈

결측치 처리 전략:
1. Age: Pclass, Sex 그룹 기반 중앙값 보간
2. Embarked: 최빈값 처리
3. Cabin: 결측률 77%로 정보 활용 어려움 -> drop
4. Fare: 중앙값 보간 (test set에 결측 존재)
"""
import pandas as pd
import numpy as np


def impute_age(df: pd.DataFrame, age_median_by_group: pd.Series = None) -> tuple:
    """
    Age 결측치: Pclass, Sex 그룹 기반 중앙값 보간
    
    Returns:
        tuple: (처리된 DataFrame, 그룹별 중앙값 Series - test용)
    """
    if age_median_by_group is None:
        # Train: 그룹별 중앙값 계산
        age_median_by_group = df.groupby(["Pclass", "Sex"])["Age"].median()
    
    def fill_age(row):
        if pd.isna(row["Age"]):
            return age_median_by_group.loc[(row["Pclass"], row["Sex"])]
        return row["Age"]
    
    df = df.copy()
    df["Age"] = df.apply(fill_age, axis=1)
    return df, age_median_by_group


def impute_embarked(df: pd.DataFrame, mode_value: str = None) -> tuple:
    """
    Embarked 결측치: 최빈값 처리
    
    Returns:
        tuple: (처리된 DataFrame, 최빈값 - test용)
    """
    if mode_value is None:
        mode_value = df["Embarked"].mode()[0]
    
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna(mode_value)
    return df, mode_value


def impute_fare(df: pd.DataFrame, fare_median: float = None) -> tuple:
    """
    Fare 결측치: 중앙값 보간
    
    Returns:
        tuple: (처리된 DataFrame, 중앙값 - test용)
    """
    if fare_median is None:
        fare_median = df["Fare"].median()
    
    df = df.copy()
    df["Fare"] = df["Fare"].fillna(fare_median)
    return df, fare_median


def handle_cabin(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Cabin 처리: 결측률 77%로 정보 활용 어려움 판단 -> drop
    
    strategy: 'drop' (컬럼 제거)
    """
    df = df.copy()
    if strategy == "drop":
        df = df.drop(columns=["Cabin"], errors="ignore")
    return df


def preprocess_missing_values(
    df: pd.DataFrame,
    is_train: bool = True,
    age_median_by_group: pd.Series = None,
    embarked_mode: str = None,
    fare_median: float = None,
    cabin_strategy: str = "drop",
) -> tuple:
    """
    전체 결측치 처리 파이프라인
    
    Returns:
        tuple: (처리된 DataFrame, imputation 파라미터 dict)
    """
    imputation_params = {}
    
    # 1. Age: 그룹 기반 중앙값 보간
    df, age_median_by_group = impute_age(df, age_median_by_group)
    imputation_params["age_median_by_group"] = age_median_by_group
    
    # 2. Embarked: 최빈값 처리
    df, embarked_mode = impute_embarked(df, embarked_mode)
    imputation_params["embarked_mode"] = embarked_mode
    
    # 3. Fare: 중앙값 보간
    df, fare_median = impute_fare(df, fare_median)
    imputation_params["fare_median"] = fare_median
    
    # 4. Cabin: drop (정보 활용 X)
    df = handle_cabin(df, cabin_strategy)
    
    return df, imputation_params
