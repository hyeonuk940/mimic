import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm  # ⭐ 진행바 라이브러리 추가

# 경고 무시
warnings.filterwarnings('ignore')

# 모델 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def run_unified_shap_with_progress(model_path, data_path, exclude_cols, target_col):
    print(f"\n📂 [1] 모델 불러오는 중... ({model_path})")
    try:
        loaded_object = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다.")
        return

    # 1. 모델 추출
    if hasattr(loaded_object, 'best_estimator_'):
        main_model = loaded_object.best_estimator_
    else:
        main_model = loaded_object

    # 2. 데이터 로드
    print(f"📂 [2] 데이터 로드 중... ({data_path})")
    df = pd.read_csv(data_path)
    
    existing_exclude = [c for c in exclude_cols if c in df.columns]
    df = df.drop(columns=existing_exclude)
    
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    
    # 전처리
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X = X.select_dtypes(include=['number'])
    
    # 3. 파이프라인 처리
    final_estimator = main_model
    X_transformed = X

    if isinstance(main_model, Pipeline):
        print("   ℹ️ Pipeline 감지 -> 스케일러 적용 중...")
        final_estimator = main_model.steps[-1][1] 
        preprocessor = Pipeline(main_model.steps[:-1])
        X_transformed_array = preprocessor.transform(X)
        X_transformed = pd.DataFrame(X_transformed_array, columns=X.columns)
    
    # 4. SHAP 분석 (진행바 추가됨)
    print(f"🤖 [3] 감지된 모델 타입: {type(final_estimator).__name__}")
    print("📊 SHAP 값 계산 시작 (진행바가 표시됩니다)...")

    explainer = None
    shap_values_list = [] # 결과를 모을 리스트

    try:
        # A. 트리 모델 (RF, XGB, LGBM)
        if isinstance(final_estimator, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
            # 속도를 위해 데이터가 너무 많으면 2000개만 샘플링 (원하면 주석 처리 가능)
            if len(X_transformed) > 2000:
                print("   ⚡ (속도 최적화) 데이터 2,000개만 샘플링하여 계산합니다.")
                X_transformed = X_transformed.sample(4000, random_state=42)

            explainer = shap.TreeExplainer(final_estimator)
            
            # ⭐ [핵심] 데이터를 100개씩 쪼개서(Batch) 계산하며 진행바 표시
            batch_size = 100
            # 데이터를 100개 단위로 나눔
            batches = [X_transformed[i:i + batch_size] for i in range(0, X_transformed.shape[0], batch_size)]
            
            print(f"   🚀 총 {len(batches)}개의 배치를 처리합니다.")
            
            for batch in tqdm(batches, desc="SHAP 계산 중"):
                # 부분 계산
                batch_shap = explainer.shap_values(batch, check_additivity=False)
                
                # 결과가 리스트(이진분류)면 1번 클래스만 가져옴
                if isinstance(batch_shap, list) and len(batch_shap) == 2:
                    batch_shap = batch_shap[1]
                
                shap_values_list.append(batch_shap)

        # B. 선형 모델 (워낙 빨라서 배치 불필요하지만 구조 통일)
        elif isinstance(final_estimator, (LogisticRegression, LinearSVC)):
            explainer = shap.LinearExplainer(final_estimator, X_transformed, feature_perturbation="interventional")
            
            # 한방에 계산 (워낙 빠름)
            print("   🚀 선형 모델은 순식간에 계산됩니다.")
            batch_shap = explainer.shap_values(X_transformed)
            shap_values_list.append(batch_shap)

        # C. SVM 및 기타 (KernelExplainer)
        else:
            print("   ⚠️ KernelExplainer 진입")
            if hasattr(final_estimator, "predict_proba"):
                pred_func = final_estimator.predict_proba
                link_type = "identity"
            else:
                pred_func = final_estimator.decision_function
                link_type = "identity"

            # KernelExplainer는 너무 느려서 50개만 샘플링 (진행바는 내부적으로 지원 안됨)
            print("   ⚠️ SVM은 진행바 표시가 어렵습니다. (50개 샘플링 계산 중...)")
            X_summary = shap.sample(X_transformed, 50) 
            explainer = shap.KernelExplainer(pred_func, X_summary, link=link_type)
            
            # 전체 데이터 계산
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            shap_values_list.append(shap_values)

        # 쪼개서 계산한 결과 합치기
        if shap_values_list:
            shap_values = np.vstack(shap_values_list)
        else:
            return

    except Exception as e:
        print(f"❌ SHAP 계산 중 오류 발생: {e}")
        return

    # 5. 시각화
    print("\n📈 [1] SHAP 요약 차트 (Summary Plot)")
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.title(f"SHAP Summary: {type(final_estimator).__name__}", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n📊 [2] 중요도 순위 차트 (Bar Plot)")
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    plt.title(f"Feature Importance: {type(final_estimator).__name__}", fontsize=12)
    plt.tight_layout()
    plt.show()

# ==========================================
# ⚙️ 실행 설정
# ==========================================

target_model_file = 'XGBoost_model_1.pkl' 
input_file = 'mimic_ver_1_0.csv' 
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
target_column = 'outcome_icu_exit_3d' 

# 실행
run_unified_shap_with_progress(target_model_file, input_file, columns_to_exclude, target_column)