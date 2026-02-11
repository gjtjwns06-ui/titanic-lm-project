"""
Feature Engineering 모듈

1. FamilySize = SibSp + Parch + 1
2. IsAlone 변수 생성
3. Name에서 Title 추출 후 인코딩
4. Fare 구간화
5. 범주형 변수 인코딩: Sex, Embarked, Title
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """FamilySize = SibSp + Parch + 1"""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """IsAlone: FamilySize == 1이면 1, 아니면 0"""
    df = df.copy()
    if "FamilySize" not in df.columns:
        df = create_family_size(df)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Name에서 Title 추출
    Mr, Mrs, Miss, Master 등 추출 후 그룹화
    """
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    
    # 희귀 Title을 그룹화
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(title_mapping).fillna("Rare")
    return df


def bin_fare(df: pd.DataFrame, n_bins: int = 5, fare_bins: np.ndarray = None) -> tuple:
    """
    Fare 구간화 (분위수 기반)
    
    Returns:
        tuple: (처리된 DataFrame, bins - test용)
    """
    df = df.copy()
    if fare_bins is None:
        _, fare_bins = pd.qcut(df["Fare"], q=n_bins, retbins=True, duplicates="drop")
        fare_bins = np.array(fare_bins)
        fare_bins[0] = -np.inf
        fare_bins[-1] = np.inf
    
    df["FareBin"] = pd.cut(df["Fare"], bins=fare_bins, labels=False, include_lowest=True)
    df["FareBin"] = df["FareBin"].fillna(-1).astype(int)
    return df, fare_bins


def encode_categorical(
    df: pd.DataFrame,
    columns: list = None,
    encoders: dict = None,
) -> tuple:
    """
    범주형 변수 Label Encoding: Sex, Embarked, Title
    
    test에서 train에 없는 범주는 최빈값(0번 클래스)으로 대체
    
    Returns:
        tuple: (처리된 DataFrame, encoders dict - test용)
    """
    if columns is None:
        columns = ["Sex", "Embarked", "Title"]
    
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        if col not in encoders:
            encoders[col] = LabelEncoder()
            encoders[col].fit(df[col].astype(str))
        
        # test에서 unseen label은 classes_[0]으로 대체
        ser = df[col].astype(str)
        unseen = ~ser.isin(encoders[col].classes_)
        ser = ser.where(~unseen, encoders[col].classes_[0])
        df[f"{col}_encoded"] = encoders[col].transform(ser)
    
    return df, encoders


def apply_feature_engineering(
    df: pd.DataFrame,
    is_train: bool = True,
    fare_bins: list = None,
    encoders: dict = None,
    n_fare_bins: int = 5,
) -> tuple:
    """
    전체 Feature Engineering 파이프라인
    
    Returns:
        tuple: (처리된 DataFrame, 파라미터 dict)
    """
    params = {}
    
    # 1. FamilySize
    df = create_family_size(df)
    
    # 2. IsAlone
    df = create_is_alone(df)
    
    # 3. Title 추출
    df = extract_title(df)
    
    # 4. Fare 구간화
    df, fare_bins = bin_fare(df, n_bins=n_fare_bins, fare_bins=fare_bins)
    params["fare_bins"] = fare_bins
    
    # 5. 범주형 인코딩
    df, encoders = encode_categorical(df, encoders=encoders)
    params["encoders"] = encoders
    
    return df, params
