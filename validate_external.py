import pandas as pd
import joblib  # 모델 불러오기용
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def validate_external_data(model_path, data_path, exclude_cols, target_col, save_result_to=None):
    """
    Args:
        model_path (str): 저장된 모델 파일 경로 (.pkl)
        data_path (str): 검증할 외부 데이터 파일 경로 (.csv)
        exclude_cols (list): 학습 때 제외했던 열 이름 리스트 (똑같이 맞춰야 함)
        target_col (str): 정답(0, 1)이 들어있는 열 이름
        save_result_to (str): 예측 결과를 저장할 파일명 (None이면 저장 안 함)
    """
    
    print(f"\n📂 [1] 모델 불러오는 중... ({model_path})")
    try:
        model = joblib.load(model_path)
        print("   ✅ 모델 로드 완료!")
    except FileNotFoundError:
        print("   ❌ 모델 파일을 찾을 수 없습니다.")
        return

    print(f"\n📂 [2] 검증 데이터 불러오는 중... ({data_path})")
    try:
        df = pd.read_csv(data_path)
        print(f"   ✅ 데이터 로드 완료! (총 {len(df)}행)")
    except FileNotFoundError:
        print("   ❌ 데이터 파일을 찾을 수 없습니다.")
        return

    # 3. 전처리 (학습 때와 동일하게 불필요한 열 제거)
    existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
    if existing_exclude_cols:
        df_processed = df.drop(columns=existing_exclude_cols)
    else:
        df_processed = df.copy() # 원본 보존

    # 4. 특성(X)과 정답(y) 분리
    if target_col not in df_processed.columns:
        print(f"❌ 오류: 검증 데이터에 정답 열 '{target_col}'이 없습니다.")
        return

    X_val = df_processed.drop(columns=[target_col])
    y_val = df_processed[target_col] # 실제 정답

    # LightGBM 등을 위해 특수문자 제거 (필요시 사용)
    import re
    X_val = X_val.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    # 데이터 타입 확인 (숫자가 아닌 열이 있으면 에러)
    non_numeric = X_val.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        print(f"❌ [주의] 데이터에 숫자가 아닌 열이 있습니다: {list(non_numeric)}")
        print("   학습 데이터와 형식이 똑같은지 확인해주세요.")
        return

    # 5. 모델로 예측 실행
    print("\n🤖 [3] 예측 수행 중...")
    try:
        y_pred = model.predict(X_val)
    except Exception as e:
        print(f"❌ 예측 중 에러 발생: {e}")
        print("   (힌트: 학습 데이터와 검증 데이터의 컬럼 순서나 개수가 다를 수 있습니다.)")
        return

    # 6. 결과 평가 (채점)
    print("\n" + "="*40)
    print(f"🎯 검증 데이터 정확도 (Accuracy): {accuracy_score(y_val, y_pred):.4f}")
    print("="*40)
    
    print("\n[상세 리포트]")
    print(classification_report(y_val, y_pred))

    print("\n[혼동 행렬 (오답 분석)]")
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN(정답-0): {tn}개 | FP(오답-0을1로): {fp}개")
    print(f"FN(오답-1을0으로): {fn}개 | TP(정답-1): {tp}개")

    # 7. 결과 파일로 저장 (옵션)
    if save_result_to:
        # 원본 데이터 옆에 'Predicted' 컬럼을 붙여서 저장
        result_df = df.copy()
        result_df['Predicted'] = y_pred
        
        # 틀린 것만 따로 보기 쉽게 표시 (Correct: True/False)
        result_df['Is_Correct'] = (result_df[target_col] == result_df['Predicted'])
        
        result_df.to_csv(save_result_to, index=False)
        print(f"\n💾 [4] 검증 결과가 '{save_result_to}' 파일로 저장되었습니다.")
        print("   (열어보시면 원본 데이터 옆에 예측값과 정답 여부가 표시되어 있습니다)")

# ==========================================
# ⚙️ 검증 실행 설정
# ==========================================

# 1. 불러올 모델 파일 (아까 저장한 파일명)
saved_model_file = 'best_rf_model.pkl'  

# 2. 검증할 외부 데이터 파일
validation_data_file = 'validation_data.csv' 

# 3. 학습 때 제외했던 열 (똑같이 적어야 함)
columns_to_exclude = ['ID', 'Name', 'Date']

# 4. 정답이 들어있는 열
target_column = 'Survived'

# 5. 결과를 저장할 파일명 (저장 안 하려면 None)
result_save_file = 'validation_result.csv'

# 실행
# validate_external_data(
#     saved_model_file, 
#     validation_data_file, 
#     columns_to_exclude, 
#     target_column, 
#     save_result_to=result_save_file
# )