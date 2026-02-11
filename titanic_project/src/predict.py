"""예측 모듈"""
import pandas as pd
from .train import load_model, FEATURE_COLUMNS


def predict(model, X: pd.DataFrame) -> pd.Series:
    """모델로 예측 수행"""
    # 학습 시 사용한 feature만 선택
    available_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    X_pred = X[available_cols]
    return pd.Series(model.predict(X_pred), index=X.index)


def create_submission(
    passenger_ids: pd.Series,
    predictions: pd.Series,
    output_path: str,
) -> pd.DataFrame:
    """
    gender_submission.csv 형식으로 제출 파일 생성
    
    Format:
        PassengerId,Survived
        892,0
        893,1
        ...
    """
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int),
    })
    submission.to_csv(output_path, index=False)
    return submission
