import pandas as pd
import joblib  # 모델 저장용 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

def run_lightgbm(file_path, exclude_cols, target_col, use_grid_search=False, is_save=False, save_path='model.pkl'):
    """
    Args:
        use_grid_search (bool): 최적화(GridSearch) 사용 여부
        is_save (bool): 모델 저장 여부 (True면 저장함)
        save_path (str): 저장할 파일 이름 (예: 'my_model.pkl')
    """
    
    # 1. 데이터 불러오기
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 성공! (총 {len(df)}행)")
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 2. 불필요한 열 제거
    existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
    if existing_exclude_cols:
        df = df.drop(columns=existing_exclude_cols)
    
    # 3. 데이터 분리 및 유효성 검사
    if target_col not in df.columns:
        print(f"❌ 오류: '{target_col}' 열이 없습니다.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # LightGBM용 특수문자 처리 (컬럼명에 특수문자 제거)
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    # 숫자 데이터 확인
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"❌ [주의] 숫자가 아닌 열이 포함됨: {list(non_numeric_cols)}")
        return

    # 4. 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==================================================
    # [1] 학습 모드 선택 (일반 vs 그리드 서치)
    # ==================================================
    model = None

    if use_grid_search:
        print("\n⚡ [모드: 그리드 서치 ON] 최적의 파라미터 탐색 중...")
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 60]
        }
        lgbm = LGBMClassifier(random_state=42, verbose=-1)
        grid = GridSearchCV(lgbm, param_grid, cv=3, verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid
        print(f"🎉 찾은 최적 파라미터: {grid.best_params_}")
        
    else:
        print("\n🚀 [모드: 그리드 서치 OFF] 기본 설정으로 빠르게 실행합니다.")
        model = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

    # 5. 예측 및 평가
    y_pred = model.predict(X_test)

    print("\n" + "="*40)
    print(f"🏆 모델 정확도: {accuracy_score(y_test, y_pred):.4f}")
    print("="*40)
    print(classification_report(y_test, y_pred))

    # ✅ [추가됨] 혼동 행렬 출력 부분
    print("\n[혼동 행렬 (Confusion Matrix)]")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    # ==================================================
    # [2] 저장 기능 (True일 때만 저장)
    # ==================================================
    if is_save:
        joblib.dump(model, save_path)
        print(f"\n💾 [저장 완료] 모델이 '{save_path}' 이름으로 저장되었습니다!")
    else:
        print("\n📢 [알림] 모델이 저장되지 않았습니다. (저장을 원하면 is_save_model=True 로 설정하세요)")
    
    return model

# ==========================================
# ⚙️ 실행 설정 (여기서 모든 걸 조절하세요)
# ==========================================

# 1. 파일 및 컬럼 설정
input_file = 'mimic_ver_1_0.csv' 
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
target_column = 'outcome_icu_exit_3d' 

# 2. 기능 스위치
is_grid_search_on = True

# 3. 저장 설정
is_save_model = False      
save_file_name = 'LightGBM_model_1.pkl' 

# 실행
run_lightgbm(
    input_file, 
    columns_to_exclude, 
    target_column, 
    use_grid_search=is_grid_search_on,
    is_save=is_save_model,
    save_path=save_file_name
)