"""
Titanic - Machine Learning from Disaster
메인 실행 스크립트

실행: python main.py
"""
import matplotlib.pyplot as plt
import os
import sys
import shutil
import pandas as pd
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_project_root, ensure_dirs
from src.preprocess import preprocess_missing_values
from src.feature_engineering import apply_feature_engineering
from src.train import (
    compare_models,
    train_best_model,
    save_model,
    FEATURE_COLUMNS,
)
from src.predict import predict, create_submission


def setup_data(config: dict) -> None:
    """data/raw에 원본 데이터 복사 (없을 경우)"""
    root = get_project_root()
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # titanic 폴더(상위)에서 raw로 복사
    parent_data = os.path.dirname(root)
    for fname in ["train.csv", "test.csv", "gender_submission.csv"]:
        src = os.path.join(parent_data, fname)
        dst = os.path.join(raw_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {fname} to data/raw/")
        elif os.path.exists(dst):
            print(f"{fname} already in data/raw/")


def run_pipeline():
    """전체 파이프라인 실행"""
    root = get_project_root()
    os.chdir(root)

    config = load_config()
    random_state = config["random_state"]
    n_folds = config["n_folds"]
    paths = config["paths"]
    n_fare_bins = config["feature_engineering"].get("fare_bins", 5)

    # 1. 데이터 준비
    print("\n=== 1. 데이터 로드 ===")
    setup_data(config)

    train_path = os.path.join(root, paths["raw_train"])
    test_path = os.path.join(root, paths["raw_test"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # 2. 결측치 처리
    print("\n=== 2. 결측치 처리 ===")
    print("  - Age: Pclass, Sex 그룹 기반 중앙값 보간")
    print("  - Embarked: 최빈값 처리")
    print("  - Fare: 중앙값 보간")
    print("  - Cabin: drop (결측률 높음, 정보 활용 X)")

    train_df, impute_params = preprocess_missing_values(train_df, is_train=True)
    test_df, _ = preprocess_missing_values(
        test_df,
        is_train=False,
        age_median_by_group=impute_params["age_median_by_group"],
        embarked_mode=impute_params["embarked_mode"],
        fare_median=impute_params["fare_median"],
    )

    # 3. Feature Engineering
    print("\n=== 3. Feature Engineering ===")
    print("  - FamilySize = SibSp + Parch + 1")
    print("  - IsAlone")
    print("  - Title 추출 및 인코딩")
    print("  - Fare 구간화")
    print("  - Sex, Embarked, Title 범주형 인코딩")

    train_df, fe_params = apply_feature_engineering(
        train_df, is_train=True, n_fare_bins=n_fare_bins
    )
    test_df, _ = apply_feature_engineering(
        test_df,
        is_train=False,
        fare_bins=fe_params["fare_bins"],
        encoders=fe_params["encoders"],
        n_fare_bins=n_fare_bins,
    )

    # 4. 모델 비교 (5-Fold CV)
    print("\n=== 4. 5-Fold 교차검증 모델 비교 ===")
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Survived"]

    results_df = compare_models(
        X_train, y_train, n_folds=n_folds, random_state=random_state
    )
    print("\n모델 비교 결과:")
    print(results_df.to_string(index=False))

    # 5. 최고 모델 선택 및 학습
    best_model_name = results_df.loc[
        results_df["Mean_Accuracy"].idxmax(), "Model"
    ]
    print(f"\n=== 5. 최고 모델 선택: {best_model_name} ===")
    
    model = train_best_model(
    X_train, y_train, model_name=best_model_name, random_state=random_state
    )

    # ===== Feature Importance =====


    importances = model.feature_importances_

    fi = pd.Series(importances, index=X_train.columns)
    fi = fi.sort_values(ascending=False)

    plt.figure()
    fi.head(15).plot(kind="bar")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


    # 6. 모델 저장
    model_path = os.path.join(root, paths["model"])
    ensure_dirs(model_path)
    save_model(model, model_path)
    print(f"모델 저장: {model_path}")

    # 7. test 예측 및 submission 생성
    print("\n=== 6. Test 예측 및 Submission 생성 ===")
    predictions = predict(model, test_df)
    submission_path = os.path.join(root, paths["submission"])
    ensure_dirs(submission_path)
    submission = create_submission(
        test_df["PassengerId"], predictions, submission_path
    )
    print(f"Submission 저장: {submission_path}")
    print(f"예측 분포 - Survived 0: {(predictions == 0).sum()}, 1: {(predictions == 1).sum()}")

    # 8. processed data 저장 (선택)
    processed_path = os.path.join(root, paths["processed_data"])
    ensure_dirs(processed_path)
    train_df.to_csv(processed_path, index=False)
    print(f"\n처리된 데이터 저장: {processed_path}")

    print("\n=== 완료 ===")
    return submission, results_df


if __name__ == "__main__":
    run_pipeline()
