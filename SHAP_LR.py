import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def analyze_shap_logistic(model_path, data_path, exclude_cols, target_col):
    
    # 1. 저장된 모델(GridSearchCV) 불러오기
    print(f"📂 모델 로딩 중... ({model_path})")
    try:
        loaded_object = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다.")
        return

    # 2. 최적의 파이프라인 추출 (GridSearchCV -> Pipeline)
    # 저장된 객체가 GridSearch라면 .best_estimator_를 꺼내야 진짜 모델입니다.
    if hasattr(loaded_object, 'best_estimator_'):
        best_pipeline = loaded_object.best_estimator_
    else:
        best_pipeline = loaded_object # GridSearch 안 썼으면 바로 파이프라인

    # 3. 파이프라인에서 '스케일러'와 '로지스틱모델' 분리
    # 파이프라인 단계 이름: [('scaler', StandardScaler), ('clf', LogisticRegression)]
    scaler = best_pipeline.named_steps['scaler']
    model = best_pipeline.named_steps['clf']
    
    print("✅ 모델과 스케일러 분리 완료!")

    # 4. 데이터 로드 및 전처리 (학습 때와 똑같이!)
    df = pd.read_csv(data_path)
    
    # 제외 열 삭제
    existing_exclude = [c for c in exclude_cols if c in df.columns]
    df = df.drop(columns=existing_exclude)
    
    # X, y 분리
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df # 타겟이 없는 새 데이터일 경우
    
    # 숫자 데이터만 남기기
    X = X.select_dtypes(include=['number'])

    # 5. [핵심] SHAP을 위해 데이터를 스케일링 (Transform)
    # 모델이 학습할 때 정규화된 데이터를 봤기 때문에, 설명할 때도 정규화해서 줘야 함
    print("🤖 데이터를 스케일링 변환 중...")
    X_scaled = scaler.transform(X)
    
    # 스케일링하면 컬럼 이름이 사라지므로 다시 데이터프레임으로 복구 (그래프에 이름 띄우기 위함)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 6. SHAP 값 계산
    print("📊 SHAP 값 계산 시작 (시간이 걸릴 수 있습니다)...")
    
    # 로지스틱 회귀는 LinearExplainer가 가장 빠르고 정확함
    explainer = shap.LinearExplainer(model, X_scaled_df, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled_df)

    # 7. 시각화 1: 요약 차트 (Beeswarm Plot) - 가장 많이 씀
    print("\n📈 [1] SHAP 요약 차트 (Summary Plot)")
    plt.figure()
    shap.summary_plot(shap_values, X_scaled_df, show=False)
    plt.title("SHAP Summary Plot (Feature Importance)", fontsize=15)
    plt.tight_layout()
    plt.show()

    # 8. 시각화 2: 막대 차트 (절대적 중요도 순위)
    print("\n📊 [2] 중요도 순위 차트 (Bar Plot)")
    plt.figure()
    shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", show=False)
    plt.title("Feature Importance Ranking", fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================
# 실행 설정
# ==========================================

# 저장했던 모델 파일명
saved_model = 'LR_model_1.pkl'

# 데이터 파일
data_file = 'mimic_ver_1_0.csv'

# 제외할 컬럼 & 타겟 (학습 때랑 똑같이)
ex_cols = ['subject_id', 'hadm_id', 'stay_id']
target = 'outcome_icu_exit_3d'

# 실행
analyze_shap_logistic(saved_model, data_file, ex_cols, target)