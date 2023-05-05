# [level1-nlp-07] level1_semantictextsimilarity_nlp
level1_semantictextsimilarity_nlp-level1-nlp-07 created by GitHub Classroom

# 팀 소개

### 

|김광연|김민준|김병준|김상혁|서재명|
| :-: | :-: | :-: | :-: | :-: |
|![광연님](https://user-images.githubusercontent.com/59431433/217448461-bb7a37d4-f5d4-418b-a1b9-583b561b5733.png)|![민준님](https://user-images.githubusercontent.com/59431433/217448432-a3d093c4-0145-4846-a775-00650198fc2f.png)|![병준님](https://user-images.githubusercontent.com/59431433/217448424-11666f05-dda6-406d-95e8-47b3bab7c2f6.png)|![상혁2](https://user-images.githubusercontent.com/59431433/217448849-758c8e25-87db-4902-ab06-0aa8c359500c.png)|![재명님](https://user-images.githubusercontent.com/59431433/217448416-b2ba2070-6cfb-4829-a3bd-861f526cb74a.png)|

## 프로젝트 주제

- 문맥적 유사도 측정(semantic text similarity, STS)

## 프로젝트 개요

- 주어진 데이터 셋에 두 문장이 문맥적으로 얼마나 유사한지 0과 5 사이의 유사도점수를 예측하는 모델을 설계하였다.

## 활용 장비 및 재료

- ai stage GPU server 활용
    - GPU: V 100

## 데이터 셋 구조

train data 개수: 9,324

test data 개수: 1,100

dev data 개수: 550

Label 점수: 0 ~ 5사이의 실수

- 5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
- 4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
- 3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
- 2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
- 1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
- 0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음