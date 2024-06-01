import os
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# .env 파일 로드
load_dotenv()

# OpenAI API 키 환경 변수 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# 하드코딩된 ADHD 체크리스트 질문
questions = [
    "어떤 일의 어려운 부분은 끝내 놓고, 그 일을 마무리를 짓지 못해 곤란을 겪은 적이 있습니까?",
    "체계가 필요한 일을 해야 할 때 순서대로 진행하기 어려운 경우가 있습니까?",
    "약속이나 해야 할 일을 잊어버려 곤란을 겪은 적이 있습니까?",
    "골치 아픈 일은 피하거나 미루는 경우가 있습니까?",
    "오래 앉아 있을 때, 손을 만지작거리거나 발을 꼼지락거리는 경우가 있습니까?",
    "마치 모터가 달린 것처럼, 과도하게 혹은 멈출 수 없이 활동을 하는 경우가 있습니까?"
]

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"
)

# 인사와 안내 메시지
print("안녕하세요! ADHD 증상 체크를 도와드리겠습니다. 몇 가지 질문을 드릴게요.")
time.sleep(0.5)
print("각 질문에 대해 지난 6개월 동안 어떻게 느끼고 행동하였는지 잘 나타내는 것에 다음과 같이 응답해주세요:")
time.sleep(0.5)
print("1: 전혀 그렇지 않다")
print("2: 거의 그렇지 않다")
print("3: 약간 혹은 가끔 그렇다")
print("4: 자주 그렇다")
print("5: 매우 자주 그렇다")

# 질문과 답변을 기록할 리스트
user_answers = []

# 대화 루프
current_question_index = 0
while current_question_index < len(questions):
    current_question = questions[current_question_index]
    user_input = input(current_question + "\n")
    
    # 사용자 답변 기록
    try:
        user_response = int(user_input)
        if user_response < 1 or user_response > 5:
            raise ValueError
        user_answers.append((current_question, user_response))
    except ValueError:
        print("유효한 숫자를 입력해주세요 (1-5).")
        continue
    
    # 다음 질문으로 이동
    current_question_index += 1

# 응답 평가
count = 0
for i, (_, answer) in enumerate(user_answers):
    if (i < 3 and answer >= 3) or (i >= 3 and answer >= 4):
        count += 1

# ADHD 평가
if count >= 4:
    diagnosis = "ADHD의 가능성이 높습니다."
else:
    diagnosis = "ADHD의 가능성이 낮습니다."

# 결과 출력
print("\n사용자의 응답을 바탕으로 한 평가:")
for question, answer in user_answers:
    print(f"Q: {question}\nA: {answer}")
print(f"\n결과: {diagnosis}")

