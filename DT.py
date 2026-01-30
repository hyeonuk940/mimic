import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run_decision_tree_advanced(file_path, exclude_cols, target_col, use_grid_search=False, is_save=False,
                               save_path='dt_model.pkl'):
    # 1. 데이터 불러오기
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 성공! 총 {len(df)}개의 행이 있습니다.")
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 2. 불필요한 열 제거
    existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
    if existing_exclude_cols:
        df = df.drop(columns=existing_exclude_cols)

    # 3. 데이터 분리
    if target_col not in df.columns:
        print(f"❌ 오류: '{target_col}' 열이 없습니다.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 숫자 데이터 확인
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"❌ [주의] 숫자가 아닌 열이 포함됨: {list(non_numeric_cols)}")
        return

    # 4. Train/Test 분리 (동일한 데이터 분할을 위해 random_state=42 유지)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==================================================
    # [1] 학습 및 최적화 (Pipeline 적용)
    # ==================================================
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # DT는 스케일링이 필수는 아니지만 구조 유지를 위해 포함
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    model = None

    if use_grid_search:
        print("\n🌳 [모드: 그리드 서치 ON] 결정 트리 최적의 깊이(Depth) 탐색 중...")

        param_grid = {
            'clf__criterion': ['gini', 'entropy'],  # 불순도 측정 지표
            'clf__max_depth': [None, 5, 10, 15, 20],  # 나무의 최대 깊이 (과적합 방지 핵심)
            'clf__min_samples_split': [2, 5, 10],  # 노드를 분할하기 위한 최소 샘플 수
            'clf__class_weight': [None, 'balanced']  # 데이터 불균형 대응
        }

        # cv=5로 K-fold 적용
        grid = GridSearchCV(pipe, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)

        model = grid
        print(f"\n🎉 찾은 최적 파라미터: {grid.best_params_}")
        print(f"   (최고 점수: {grid.best_score_:.4f})")

    else:
        print("\n🚀 [모드: 그리드 서치 OFF] 기본 설정(Depth 제한 없음)으로 실행...")
        model = pipe
        model.fit(X_train, y_train)

    # 5. 예측 및 평가
    y_pred = model.predict(X_test)

    print("\n" + "=" * 40)
    print(f"🏆 DT 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("=" * 40)
    print("\n[분류 보고서]")
    print(classification_report(y_test, y_pred))

    print("\n[혼동 행렬 (Confusion Matrix)]")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # 저장
    if is_save:
        joblib.dump(model, save_path)
        print(f"\n💾 [저장 완료] '{save_path}'")

    return model


# ==========================================
# ⚙️ 실행 설정
# ==========================================
input_file = 'mimic_ver_1_0.csv'
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id']
target_column = 'outcome_icu_exit_3d'

run_decision_tree_advanced(
    input_file,
    columns_to_exclude,
    target_column,
    use_grid_search=True,
    is_save=True,
    save_path='DT_model_1.pkl'
)