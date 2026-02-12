# titanic-lm-project
# Titanic Survival Prediction

## 📌 프로젝트 개요
Kaggle Titanic 데이터를 활용하여 승객의 생존 여부를 예측하는 머신러닝 프로젝트입니다.

데이터 전처리, Feature Engineering, 모델 비교, 교차 검증을 통해 최적 모델을 선정하고 제출 파일을 생성하는 파이프라인을 구축했습니다.

---

## 🛠 사용 기술

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

---

## 📊 Feature Engineering

모델 성능 향상을 위해 다음 Feature를 생성했습니다.

### FamilySize
- SibSp + Parch + 1
- 가족 동반 여부가 생존률에 영향

### IsAlone
- 혼자 탑승 여부

### Title
- 이름에서 사회적 신분 추출 (Mr, Mrs, Miss 등)

---

## 🤖 모델 비교 결과

| Model | Accuracy |
|--------|------------|
| Logistic Regression | 0.800 |
| Random Forest | 0.802 |
| XGBoost | 0.820 |

👉 최종 모델: XGBoost

---

## 📈 Feature Importance

모델 분석 결과 주요 변수:

![Feature Importance](reports/feature_importance.png)

---

## 🚀 실행 방법
pip install -r requirements.txt
python main.py


---

## 📁 프로젝트 구조
titanic_project/
├─ config/
├─ notebooks/
├─ src/
├─ data/
│ ├─ raw/ # train.csv, test.csv, gender_submission.csv
│ ├─ processed/ # processed_data.csv
│ └─ submission/ # submission.csv (Kaggle 제출용)
├─ models/ # model.pkl
├─ main.py
├─ requirements.txt
└─ README.md


---

## 📚 배운 점

- 데이터 전처리 중요성
- Feature Engineering이 모델 성능에 미치는 영향
- 교차 검증을 통한 모델 평가 방법




