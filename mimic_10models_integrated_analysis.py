import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, roc_curve, roc_auc_score)

# 부드러운 곡선 처리를 위한 라이브러리
from scipy.interpolate import make_interp_spline

# 모델 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 출력 경고 무시
warnings.filterwarnings('ignore')


def run_comprehensive_analysis(input_file, exclude_cols, target_col,
                               run_external=False, external_file=None,
                               selected_models=None):
    """
    10개 모델 통합 학습, 그리드 서치, 내부/외부 검증 및 시각화 수행
    """

    # 1. 데이터 로드 및 분할
    try:
        df = pd.read_csv(input_file)
        print(f"✅ 데이터 로드 성공: {input_file} (총 {len(df)}행)")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return None, None, None

    X = df.drop(columns=[col for col in exclude_cols if col in df.columns] + [target_col])
    y = df[target_col]

    # 내부 학습/테스트 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 전체 모델 후보군 및 하이퍼파라미터 정의
    full_model_dict = {
        'LR': (LogisticRegression(max_iter=5000),
               {'clf__C': [1]}),
        'RF': (RandomForestClassifier(random_state=42),
               {'clf__n_estimators': [300], 'clf__max_depth': [20]}),
        'DT': (DecisionTreeClassifier(random_state=42),
               {'clf__max_depth': [5], 'clf__min_samples_leaf': [4]}),
        'KNN': (KNeighborsClassifier(),
                {'clf__n_neighbors': [21], 'clf__weights': ['distance']}),
        'MLP': (MLPClassifier(max_iter=1000, random_state=42),
                {'clf__hidden_layer_sizes': [(100,)], 'clf__alpha': [0.0001]}),
        'AdaBoost': (AdaBoostClassifier(random_state=42),
                     {'clf__n_estimators': [100], 'clf__learning_rate': [1.0]}),
        'SVM': (SVC(probability=True, random_state=42),
                {'clf__C': [1], 'clf__gamma': ['scale']}),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'),
                    {
                        'clf__n_estimators': [500],  # 더 많이 학습
                        'clf__learning_rate': [0.01],  # 학습률을 낮춰 정밀도 향상
                        'clf__max_depth': [5],  # 복잡도를 살짝 올림
                        'clf__min_child_weight': [5],  # 과적합 방지용 (중요)
                        'clf__subsample': [0.8]  # 데이터 샘플링 비율
                    }
                    ),
        'LightGBM': (LGBMClassifier(random_state=42),
                     {
                         'clf__n_estimators': [500],  # 현재 300이 부족해 보임
                         'clf__learning_rate': [0.005],  # 더 잘게 쪼개서 학습
                         'clf__num_leaves': [63],  # 트리 노드 개수를 늘려 복잡한 패턴 학습
                         'clf__feature_fraction': [0.8],  # 변수 샘플링 (과적합 방지)
                         'clf__min_child_samples': [30]  # 리프 노드의 최소 데이터 수
                     }
                     ),
        'CatBoost': (CatBoostClassifier(random_state=42, verbose=0),
                     {'clf__iterations': [300], 'clf__depth': [4], 'clf__learning_rate': [0.1]})
    }

    # 모델 필터링
    if selected_models is not None:
        model_dict = {k: v for k, v in full_model_dict.items() if k in selected_models}
        if not model_dict:
            print("⚠️ 선택된 모델이 유효하지 않아 전체 모델을 실행합니다.")
            model_dict = full_model_dict
    else:
        model_dict = full_model_dict

    internal_results = []
    external_results = []
    trained_models = {}

    # 그래프 스타일 설정
    plt.figure(figsize=(12, 9))
    sns.set_style("whitegrid")
    # 모델 개수에 맞는 색상 팔레트
    colors = sns.color_palette("husl", len(model_dict))

    print(f"🔎 [내부 학습 시작] 총 {len(model_dict)}개 모델 최적화 중...")

    for i, (name, (clf, params)) in enumerate(model_dict.items()):
        # Pipeline 구축
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

        # Grid Search 실행 (cv=3)
        grid = GridSearchCV(pipe, params, cv=10, n_jobs=-1, scoring='roc_auc')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        trained_models[name] = best_model
        best_params = grid.best_params_

        print(f"✅ {name} 학습 완료! [Best: {best_params}]")

        # --- 내부 검증 데이터 성능 평가 ---
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        auc_score = roc_auc_score(y_test, y_prob)

        internal_results.append({
            'Model': name,
            'In_AUC': auc_score,
            'In_F1': f1_score(y_test, y_pred),
            'In_Sens': recall_score(y_test, y_pred),
            'In_Spec': tn / (tn + fp),
            'Best_Params': str(best_params)
        })

        # --- 부드러운 ROC 곡선 시각화 (Spline Interpolation) ---
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # 보간을 위해 중복값 제거
        fpr_unique, indices = np.unique(fpr, return_index=True)
        tpr_unique = tpr[indices]

        # 60개의 포인트로 흐름 단순화
        fpr_new = np.linspace(0, 1, 60)
        spl = make_interp_spline(fpr_unique, tpr_unique, k=3)
        tpr_smooth = np.clip(spl(fpr_new), 0, 1)

        plt.plot(fpr_new, tpr_smooth,
                 label=f'{name:<10} (AUC: {auc_score:.3f})',
                 color=colors[i], linewidth=2.0, alpha=0.8)

        # --- 외부 검증 (스위치 ON 시 실행) ---
        if run_external and external_file:
            try:
                ext_df = pd.read_csv(external_file)
                X_ext = ext_df.drop(columns=[col for col in exclude_cols if col in ext_df.columns] + [target_col])
                y_ext = ext_df[target_col]

                y_ext_prob = best_model.predict_proba(X_ext)[:, 1]
                y_ext_pred = best_model.predict(X_ext)
                tn_e, fp_e, fn_e, tp_e = confusion_matrix(y_ext, y_ext_pred).ravel()

                external_results.append({
                    'Model': name,
                    'Ext_AUC': roc_auc_score(y_ext, y_ext_prob),
                    'Ext_F1': f1_score(y_ext, y_ext_pred),
                    'Ext_Sens': recall_score(y_ext, y_ext_pred),
                    'Ext_Spec': tn_e / (tn_e + fp_e)
                })
            except Exception as e:
                print(f"⚠️ {name} 외부 검증 중 오류 발생: {e}")

    # 3. 데이터프레임 통합 및 리포트 출력
    df_in = pd.DataFrame(internal_results)
    if run_external and external_results:
        df_ext = pd.DataFrame(external_results)
        final_report = pd.merge(df_in, df_ext, on='Model')
    else:
        final_report = df_in

    print("\n" + "=" * 110)
    print(f"🏆 최종 모델 성능 통합 보고서 (AUC 기준 정렬)")
    pd.set_option('display.max_colwidth', None)
    print(final_report.sort_values(by='In_AUC', ascending=False).to_string(index=False))
    print("=" * 110)

    # 4. 그래프 레이아웃 마무리
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5)
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Comparison of 10 Models: Smoothed ROC Curves', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return trained_models, X_test, final_report


def run_shap_analysis(model, X_test, model_name):
    """
    최적 모델에 대한 SHAP Feature Importance 시각화
    """
    print(f"\n💡 {model_name} 모델 SHAP 가중치 분석 시작...")

    # 파이프라인 내부의 스케일러로 변환
    X_test_scaled = model.named_steps['scaler'].transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 모델별 Explainer 선택
    try:
        if model_name in ['RF', 'DT', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost']:
            explainer = shap.TreeExplainer(model.named_steps['clf'])
            shap_values = explainer.shap_values(X_test_scaled_df)
        elif model_name == 'LR':
            explainer = shap.LinearExplainer(model.named_steps['clf'], X_test_scaled_df)
            shap_values = explainer.shap_values(X_test_scaled_df)
        else:
            # 속도를 위해 50개 샘플만 사용
            explainer = shap.KernelExplainer(model.named_steps['clf'].predict_proba, shap.sample(X_test_scaled_df, 50))
            shap_values = explainer.shap_values(shap.sample(X_test_scaled_df, 50))
            X_test_scaled_df = shap.sample(X_test_scaled_df, 50)

        # 결과 차원 보정
        if isinstance(shap_values, list):
            shap_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_to_plot = shap_values[:, :, 1]
        else:
            shap_to_plot = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_to_plot, X_test_scaled_df, plot_type="bar", show=False)
        plt.title(f"Feature Importance (SHAP) - {model_name}")
        plt.show()
    except Exception as e:
        print(f"⚠️ SHAP 분석 중 오류 발생: {e}")


# ==========================================
# ⚙️ 메인 실행 설정 (사용자 맞춤 수정 가능)
# ==========================================
# 1. 파일 및 컬럼 설정
DATA_PATH = 'mimic_ver_1_0.csv'
EXT_PATH = 'external_mimic_test.csv'  # 외부 파일이 있을 경우 지정
EXCLUDE_LIST = ['subject_id', 'hadm_id', 'stay_id']
TARGET_COL = 'outcome_icu_exit_3d'

# 2. 실행 모델 선택 (원하는 모델만 리스트에 넣으세요. 전부 실행하려면 None)
MY_MODELS = ['LR', 'RF', 'DT', 'KNN', 'MLP', 'AdaBoost', 'SVM', 'XGBoost', 'LightGBM', 'CatBoost']
# MY_MODELS = ['XGBoost', 'LightGBM']

# 3. 옵션 제어
USE_EXTERNAL = False  # 외부 검증 수행 여부

# 4. 전체 프로세스 실행
models, x_test_data, report_df = run_comprehensive_analysis(
    DATA_PATH, EXCLUDE_LIST, TARGET_COL,
    run_external=USE_EXTERNAL,
    external_file=EXT_PATH,
    selected_models=MY_MODELS
)

# 5. 분석이 성공했다면 성능 1위 모델에 대해 SHAP 분석 수행
if report_df is not None and not report_df.empty:
    top_model_name = report_df.sort_values(by='In_AUC', ascending=False).iloc[0]['Model']
    run_shap_analysis(models[top_model_name], x_test_data, top_model_name)