# import pandas as pd
# import joblib  # ⭐ 모델 저장용 라이브러리
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler  # ✅ 추가: 정규화 도구
# from sklearn.pipeline import Pipeline             # ✅ 추가: 파이프라인

# def run_logistic_regression(file_path, exclude_cols, target_col, use_grid_search=False, is_save=False, save_path='log_reg_model.pkl'):
#     """
#     Args:
#         use_grid_search (bool): True면 최적 파라미터 탐색, False면 기본 실행
#         is_save (bool): 모델 저장 여부 (True면 저장함)
#         save_path (str): 저장할 파일 이름
#     """
    
#     # 1. 데이터 불러오기
#     try:
#         df = pd.read_csv(file_path)
#         print(f"✅ 데이터 로드 성공! 총 {len(df)}개의 행이 있습니다.")
#     except FileNotFoundError:
#         print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
#         return

#     # 2. 불필요한 열 제거
#     existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
#     if existing_exclude_cols:
#         df = df.drop(columns=existing_exclude_cols)
#         print(f"ℹ️ 제외된 열: {existing_exclude_cols}")
    
#     # 3. 특성(X)과 타겟(y) 분리
#     if target_col not in df.columns:
#         print(f"❌ 오류: 예측할 열 '{target_col}'이(가) 데이터에 없습니다.")
#         return

#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     # [안전장치] 숫자 데이터 확인
#     non_numeric_cols = X.select_dtypes(exclude=['number']).columns
#     if len(non_numeric_cols) > 0:
#         print(f"❌ [주의] 숫자가 아닌 열이 포함됨: {list(non_numeric_cols)}")
#         return

#     # 4. 데이터 분리
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # ==================================================
#     # [1] 학습 모드 선택 (Pipeline 적용)
#     # ==================================================
#     model = None
    
    

#     if use_grid_search:
#         print("\n🔍 [모드: 그리드 서치 ON] 정규화 포함 최적 파라미터 탐색 중...")
        
#         # ✅ 파이프라인 구축 (스케일러 -> 로지스틱 회귀)
#         pipe = Pipeline([
#             ('scaler', StandardScaler()),       # 1단계: 정규화
#             ('clf', LogisticRegression(max_iter=1000)) # 2단계: 모델
#         ])

#         # ✅ 파이프라인 내부 파라미터 접근법: '이름__파라미터' (언더바 2개)
#         param_grid = {
#             'clf__C': [0.01, 0.1, 1, 10, 100], 
#             'clf__solver': ['lbfgs', 'liblinear'] 
#         }

#         # Grid Search 실행
#         grid = GridSearchCV(pipe, param_grid, cv=3, verbose=1)
#         grid.fit(X_train, y_train)
        
#         model = grid
#         print(f"🎉 찾은 최적 파라미터: {grid.best_params_}")
        
#     else:
#         print("\n🚀 [모드: 그리드 서치 OFF] 정규화 적용 후 기본 설정 실행...")
        
#         # ✅ 기본 실행도 파이프라인으로 묶음
#         model = Pipeline([
#             ('scaler', StandardScaler()), 
#             ('clf', LogisticRegression(max_iter=1000))
#         ])
#         model.fit(X_train, y_train)

#     # 5. 예측 및 평가
#     # (모델 안에 스케일러가 들어있어서, X_test를 그대로 넣어도 알아서 정규화 후 예측함)
#     y_pred = model.predict(X_test)

#     # 결과 출력
#     print("\n" + "="*40)
#     print(f"🏆 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
#     print("="*40)
#     print("\n[분류 보고서]")
#     print(classification_report(y_test, y_pred))
    
#     print("\n[혼동 행렬 (Confusion Matrix)]")
#     cm = confusion_matrix(y_test, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

#     # ==================================================
#     # [2] 저장 기능 (True일 때만 저장)
#     # ==================================================
#     if is_save:
#         joblib.dump(model, save_path)
#         print(f"\n💾 [저장 완료] 정규화 기능이 포함된 모델이 '{save_path}'에 저장되었습니다!")
#     else:
#         print("\n📢 [알림] 모델이 저장되지 않았습니다. (저장을 원하면 is_save=True 로 설정하세요)")
    
#     return model

# # ==========================================
# # ⚙️ 실행 설정
# # ==========================================

# # 1. 파일 및 컬럼 설정
# input_file = 'mimic_ver_1_0.csv' 
# columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
# target_column = 'outcome_icu_exit_3d' 

# # 2. 기능 스위치
# is_grid_search_on = True

# # 3. 저장 설정
# is_save_model = False       
# save_file_name = 'LR_model_1.pkl' 

# # 실행
# run_logistic_regression(
#     input_file, 
#     columns_to_exclude, 
#     target_column, 
#     use_grid_search=is_grid_search_on,
#     is_save=is_save_model,
#     save_path=save_file_name
# )

import pandas as pd
import joblib  # ⭐ 모델 저장용 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_logistic_regression_advanced(file_path, exclude_cols, target_col, use_grid_search=False, is_save=False, save_path='log_reg_model.pkl'):
    
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

    # 4. Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==================================================
    # [1] 학습 및 최적화 (Pipeline 적용)
    # ==================================================
    
    # 기본 파이프라인 (스케일러 + 로지스틱)
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', LogisticRegression(max_iter=5000)) # 반복 횟수 넉넉히
    ])

    model = None

    if use_grid_search:
        print("\n🔥 [모드: 강력한 그리드 서치 ON] 최적의 파라미터(L1, L2, 가중치) 탐색 중...")
        
        # 💡 [핵심] 파라미터 조합을 리스트로 분리하여 '되는 조합'끼리만 묶음
        param_grid = [
            # Case 1: L2 규제 (일반적인 Ridge) - lbfgs, liblinear 둘 다 가능
            {
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs', 'liblinear'],
                'clf__C': [0.01, 0.1, 1, 10, 100],  # 규제 강도
                'clf__class_weight': [None, 'balanced'] # ⭐ 데이터 불균형 해결의 핵심
            },
            # Case 2: L1 규제 (변수 선택 기능 Lasso) - liblinear만 가능
            {
                'clf__penalty': ['l1'],
                'clf__solver': ['liblinear'], 
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__class_weight': [None, 'balanced']
            }
        ]

        # n_jobs=-1: 컴퓨터의 모든 CPU 코어를 써서 속도 향상
        grid = GridSearchCV(pipe, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        model = grid
        print(f"\n🎉 찾은 최적 파라미터: {grid.best_params_}")
        print(f"   (최고 점수: {grid.best_score_:.4f})")
        
    else:
        print("\n🚀 [모드: 그리드 서치 OFF] 'balanced' 모드로 기본 실행...")
        
        # 기본 실행이어도 성능을 위해 class_weight='balanced'는 켜줍니다.
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        model.fit(X_train, y_train)

    # 5. 예측 및 평가
    y_pred = model.predict(X_test)

    print("\n" + "="*40)
    print(f"🏆 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
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
        print(f"\n💾 [저장 완료] '{save_path}'")
    
    return model

# ==========================================
# ⚙️ 실행 설정
# ==========================================

input_file = 'mimic_ver_1_0.csv' 
columns_to_exclude = ['subject_id', 'hadm_id', 'stay_id'] 
target_column = 'outcome_icu_exit_3d' 

# True로 설정하여 강력한 탐색 시작!
run_logistic_regression_advanced(
    input_file, 
    columns_to_exclude, 
    target_column, 
    use_grid_search=True, 
    is_save=True,
    save_path='LR_model_2.pkl'
)
