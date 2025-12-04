# ML_Project
"model_colab_cpu.ipynb"는 "model_local_CUDA_GPU" 기반으로 수정한것이라 오류가 있을 수 있습니다.
# Local CUDA GPU
RTX3080 기반으로 cuRF, XGBOOST, LIGHTGBM 3가지 모델을 GPU로 돌렸으며 
CPU 코드와 차이점
Randomforest -> cuRF
XGBOOST, LIGHTGBM -> 학습시 "devices = 'CUDA'" 부분

# 진행상황
배경 제거 -> 데이터 로드시 전처리 단계에서 배경 색 자체를 검정색으로 통일
그레이 스케일 -> 그레이 스케일 파라미터 RGB값을 이용하여 계산시 np.mean 이 아닌 np.max 이용하여 고대비 이미지 생성
이진화 -> 그레이스케일 이미지의 픽셀 밝기 분포를 통해 임계점을 확인하여 임계점 이상의 그레이 픽셀을 백색으로 전환
HOG -> 28*28 픽셀이므로 28의 인수 2, 4, 7 의 값을 ppc로 설정하여 4가 최적의 값임을 입증

따라서 현재 배경제거 후 그레이스케일을 적용하고 이진화를 하여 HOG feature를 추출하여 학습하였다.
그 결과 3가지 모델 모두 acc 0.99에 내외로 나온다.

Robustness 검증을 위하여 Rotate, Scale, Shear, Shift 4가지 특성을 반영한 검증 데이터 셋을 생성 후 검증
-> Shift를 제외한 나머지에서 acc 0.9 이상 나오며, Shift는 acc 0.5 정도 나온다.
-> Shift 내성을 가지기 위하여 전처리 단계에서 무게중심에 따른 Centering이 필요하다 생각이 든다. -> 효과 입증

Hu moments 추가 적용시
Robustness에 좋은 내성을 보여주지만 마찬가지로 Shift 에 취약하다.
-> 원본 데이터에도 좋은 성능이 나오지 않으므로 Hu moments 적용만으로 개선이 어렵다.

# 2025.12.04 10:20 Update
1. 기존 약점이던 Shift는 이진화 까지 진행된 후 Centering을 하고 HOG 추출을 하니 Shift 부분에 성능 개선이 매우 뚜렷하다.
2. CUDA를 이용하여 GPU로 돌려보았다.
3. model_final_GPU 가 digit 학습 최종버전
