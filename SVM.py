import pandas as pd
import joblib  # ⭐ 모델 저장용 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  # ✅ 추가: 정규화 도구
from sklearn.pipeline import Pipeline             # ✅ 추가: 파이프라인

def run_svm_prediction(file_path, exclude_cols, target_col, use_grid_search=False, is_save=False, save_path='svm_model.pkl'):
    # ... (데이터 로드 및 전처리 부분은 위와 동일하므로 생략) ...
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

    # 숫자 데이터 확인
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"❌ [주의] 숫자가 아닌 열이 포함됨: {list(non_numeric_cols)}")
        return

    # 4. 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ==================================================
    # [1] 학습 모드 선택 (Pipeline 적용)
    # ==================================================
    model = None 

    if use_grid_search:
        print("\n🔍 [모드: 그리드 서치 ON] 정규화 포함 SVM 최적 파라미터 탐색...")
        
        # ✅ [수정 포인트 1] probability=True 추가
        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('svc', SVC(probability=True)) 
        ])
        
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': [1, 0.1, 0.01, 0.001],
            'svc__kernel': ['rbf', 'linear']
        }
        
        grid = GridSearchCV(pipe, param_grid, refit=True, verbose=2, cv=3)
        grid.fit(X_train, y_train)
        
        model = grid
        print(f"🎉 찾은 최적 파라미터: {grid.best_params_}")
        
    else:
        print("\n🚀 [모드: 그리드 서치 OFF] 정규화 적용 후 기본 설정 실행...")
        
        # ✅ [수정 포인트 2] probability=True 추가
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('svc', SVC(kernel='rbf', C=1.0, random_state=42, probability=True))
        ])
        model.fit(X_train, y_train)

    # 5. 예측 및 평가
    y_pred = model.predict(X_test)

    print("\n" + "="*40)
    print(f"🏆 SVM 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("="*40)
    
    print("\n[분류 보고서]")
    print(classification_report(y_test, y_pred))
    
    print("\n[혼동 행렬 (Confusion Matrix)]")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # 저장
    if is_save:
        joblib.dump(model, save_path)
        print(f"\n💾 [저장 완료] 확률 기능이 포함된 모델이 '{save_path}'에 저장되었습니다!")
    else:
        print("\n📢 [알림] 모델이 저장되지 않았습니다.")
    
    return model

# ... (실행 설정 부분은 그대로 유지) ...
input_file = 'mimic_ver_1_0.csv' 
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
target_column = 'outcome_icu_exit_3d' 
is_grid_search_on = False 
is_save_model = False      
save_file_name = 'SVM_model_2.pkl' 

run_svm_prediction(input_file, columns_to_exclude, target_column, use_grid_search=is_grid_search_on, is_save=is_save_model, save_path=save_file_name)