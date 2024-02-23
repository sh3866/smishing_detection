# Leveraging Pretrained Language Models for Effective Smishing Detection
효과적인 스미싱 탐지를 위한 사전 학습된 언어 모델의 활용

## 🖥️ 프로젝트 설명
In-Context learning 및 Fine-Tuning 기법을 활용한 Smishing Detection 모델 개발

## ⏰ 개발 기간
24.02.18 ~ 24.02.23

## ⚙️ 개발 환경
- python
- pytorch
- Model : llama-2-7b-hf

## 기존 코드의 성능을 향상하기 위해 고안한 방안
- 데이터 전처리
    데이터에 있는 uni-8 이외의 단어 제거.
- 데이터 증강
    - 문자 치환
- prompt의 demonstrate에 들어가는 데이터 불균형 문제를 해결
    들어가는 ham, smishing, spam 비율을 4:3:3으로 투입
- prompt 방식 추가. [사이트][https://www.promptingguide.ai/] 
    - few-shot (현)
    - ...
- 사전 학습 모델 변경
    - [KT-AI Mi:dm][https://huggingface.co/KT-AI/midm-bitext-S-7B-inst-v1]
    - [Gemma][https://github.com/google-deepmind/gemma]

## 2월 23일 결과
### hyperparameter
- seed 42
- fewshot 32
- CoT True
- Imbalance Dataset 고려

### 가장 성능이 높았던 실험 결과: 
- accuracy: 	94.1471%
- F1 score: 	79.6678%
- Recall :	    80.3096%
- Precision:	79.6918%

### 프로젝트의 한계
1. 가장 점수가 높은 임의의 seed을 결정
    - seed에 따라 달라지는 결과를 통합하기 위해서는 평균값을 사용해야 함
2. CoT와 Directional Stimulus Prompting 기법의 미숙
    - CoT로 "Let's think step by step ..." 만을 사용
    - Directional Stimulus Prompting으로 Hint만 제공

### 앞으로의 방향
1. seed 값이 변함에 따라 결과가 달라지는 것을 방지하고자 seed의 평균 사용
2. test를 정확하게 맞추기 위한 최적의 demonstrate 예제 추출 방안 고려
    - 예를 들어 들어온 test text와 가장 유사한 text를 train data에서 뽑아와 해당 train text를 demonstrate에 삽입
3. smishing&spam 탐지에 가장 적합한 prompt 방식 모색
    - 위에서 사용한 CoT와 Directionla Stimulus Prompting은 논리적은 흐름의 예제에 적합한 것으로 보이고, 우리 Task에서는 논리적인 흐름이 없는 classification에 맞는 다른 prompt 방식을 모색해야 함