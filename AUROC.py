import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline

def evaluate_model_auroc(model_path, data_path, exclude_cols, target_col):
    print(f"\n📂 [1] 모델 로딩 중... ({model_path})")
    try:
        loaded_object = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return

    # -------------------------------------------------------
    # 1. 모델 추출 (GridSearch -> Best Estimator)
    # -------------------------------------------------------
    if hasattr(loaded_object, 'best_estimator_'):
        print("   ℹ️ GridSearchCV 객체 감지 -> 최적 모델 추출")
        model = loaded_object.best_estimator_
    else:
        model = loaded_object

    # -------------------------------------------------------
    # 2. 데이터 로드 및 전처리 (학습 때와 동일하게)
    # -------------------------------------------------------
    print(f"📂 [2] 데이터 로드 및 전처리... ({data_path})")
    df = pd.read_csv(data_path)
    
    # 제외 열 삭제
    existing_exclude = [c for c in exclude_cols if c in df.columns]
    df = df.drop(columns=existing_exclude)
    
    # X, y 분리
    if target_col not in df.columns:
        print(f"❌ 데이터에 타겟 열 '{target_col}'이 없습니다.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 특수문자 제거 (LGBM/XGB 대비)
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X = X.select_dtypes(include=['number'])

    # -------------------------------------------------------
    # 3. 테스트 데이터 분리 (검증은 Test Set으로 해야 함)
    # -------------------------------------------------------
    # 주의: 학습 때 random_state=42를 썼다면 여기서도 똑같이 써야
    # 학습에 안 쓴 데이터를 정확히 나눌 수 있습니다.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------------------------------
    # 4. 확률(Probability) 예측
    # -------------------------------------------------------
    print("🤖 [3] 예측 확률 계산 중...")
    
    y_pred_proba = None

    # SVM 등 일부 모델은 predict_proba를 지원 안 할 수도 있음
    try:
        # [Case 1] 확률 예측 기능이 있는 경우 (대부분의 모델)
        if hasattr(model, "predict_proba"):
            # [:, 1]은 '1(양성)' 클래스일 확률만 가져온다는 뜻
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # [Case 2] SVM인데 probability=True 설정을 안 했을 경우
        elif hasattr(model, "decision_function"):
            print("   ⚠️ (참고) 확률 대신 decision_function(거리값)을 사용합니다.")
            y_pred_proba = model.decision_function(X_test)
            
        else:
            # 파이프라인 안쪽 깊숙이 있는 경우 처리
            if isinstance(model, Pipeline):
                final_step = model.steps[-1][1]
                if hasattr(final_step, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(final_step, "decision_function"):
                    y_pred_proba = model.decision_function(X_test)

    except Exception as e:
        print(f"❌ 예측 중 에러 발생: {e}")
        return

    if y_pred_proba is None:
        print("❌ 이 모델은 확률 값을 출력할 수 없습니다.")
        return

    # -------------------------------------------------------
    # 5. AUROC 계산 및 시각화
    # -------------------------------------------------------
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    print("\n" + "="*40)
    print(f"🏆 AUROC 점수: {auc_score:.4f}")
    print("="*40)

    # 그래프 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve\nModel: {model_path}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# ⚙️ 실행 설정 (여기서 파일명만 바꿔가며 쓰세요)
# ==========================================

# 평가할 모델 파일명

target_model_file = 'LR_model_2.pkl'

input_file = 'mimic_ver_1_0.csv' 
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
target_column = 'outcome_icu_exit_3d' 

# 실행
evaluate_model_auroc(target_model_file, input_file, columns_to_exclude, target_column)