# Leveraging Pretrained Language Models for Effective Smishing Detection
효과적인 스미싱 탐지를 위한 사전 학습된 언어 모델의 활용

## 🖥️ 프로젝트 설명
In-Context learning 및 Fine-Tuning 기법을 활용한 Smishing Detection 모델 개발

## ⏰ 개발 기간
24.02.18 ~ 24.02.28

## ⚙️ 개발 환경
- python
- pytorch
- Model : Bert

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