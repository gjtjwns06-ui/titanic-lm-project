# Titanic - Machine Learning from Disaster

Kaggle Titanic 예측 프로젝트. train.csv로 학습하고 test.csv를 예측하여 제출 파일을 생성합니다.

## 요구사항

- Python 3.8+
- requirements.txt 의존성 설치

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 전체 파이프라인 실행 (학습 + 예측 + 제출 파일 생성)
python main.py
```

## 프로젝트 구조

```
titanic_project/
├── data/
│   ├── raw/           # train.csv, test.csv, gender_submission.csv
│   ├── processed/     # 전처리된 데이터
│   └── submission/    # submission.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocess.py       # 결측치 처리
│   ├── feature_engineering.py
│   ├── train.py            # 모델 비교 및 학습
│   ├── predict.py
│   └── utils.py
├── models/
│   └── model.pkl
├── config/
│   └── config.yaml
├── requirements.txt
└── main.py
```

## 전처리 및 Feature Engineering

### 결측치 처리
- **Age**: Pclass, Sex 그룹 기반 중앙값 보간
- **Embarked**: 최빈값 처리
- **Cabin**: 결측률 77%로 drop
- **Fare**: 중앙값 보간

### Feature Engineering
- FamilySize = SibSp + Parch + 1
- IsAlone
- Name에서 Title 추출 및 인코딩
- Fare 구간화

### 모델 비교 (5-Fold CV)
- Logistic Regression
- Random Forest
- XGBoost

`random_state=42`로 재현 가능하게 설정되어 있습니다.
