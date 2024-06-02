import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import time

# .env 파일 로드
load_dotenv()

# OpenAI API 키 환경 변수 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo-0125"
)

# 사용자 메시지 생성 함수
def create_user_message(content):
    return HumanMessage(content=content)

# 챗봇 응답 메시지 생성 함수
def create_assistant_message(content):
    return AIMessage(content=content)

# 대화 함수
def conduct_conversation(user_message_content):
    # 사용자 메시지 생성
    user_message = create_user_message(user_message_content)

    # 대화를 챗봇에 전달하고 응답 받기
    response = chat.invoke([user_message])

    # 챗봇의 응답 반환
    return response.content

questions = [
    "어떤 일의 어려운 부분은 끝내 놓고, 그 일을 마무리를 짓지 못해 곤란을 겪은 적이 있습니까?",
    "체계가 필요한 일을 해야 할 때 순서대로 진행하기 어려운 경우가 있습니까?",
    "약속이나 해야 할 일을 잊어버려 곤란을 겪은 적이 있습니까?",
    "골치 아픈 일은 피하거나 미루는 경우가 있습니까?",
    "오래 앉아 있을 때, 손을 만지작거리거나 발을 꼼지락거리는 경우가 있습니까?",
    "마치 모터가 달린 것처럼, 과도하게 혹은 멈출 수 없이 활동을 하는 경우가 있습니까?"
]

# 대화 루프
print("안녕하세요! ADHD 증상 체크를 도와드리겠습니다. 몇 가지 질문을 드릴게요.")
print("각 질문에 대해 다음과 같이 응답해주세요:")
time.sleep(1)
print("1: 전혀 그렇지 않다")
print("2: 거의 그렇지 않다")
print("3: 약간 혹은 가끔 그렇다")
print("4: 자주 그렇다")
print("5: 매우 자주 그렇다")

user_answers = []

current_question_index = 0
while current_question_index < len(questions):
    current_question = questions[current_question_index]
    user_input = input(current_question + "\n")
    
    try:
        user_response = int(user_input)
        if user_response < 1 or user_response > 5:
            raise ValueError
        user_answers.append((current_question, user_response))
    except ValueError:
        print("유효한 숫자를 입력해주세요 (1-5).")
        continue
    
    current_question_index += 1

count = 0
for i, (_, answer) in enumerate(user_answers):
    if (i < 3 and answer >= 3) or (i >= 3 and answer >= 4):
        count += 1

if count >= 4:
    diagnosis = "ADHD의 가능성이 높습니다."
else:
    diagnosis = "ADHD의 가능성이 낮습니다."

# 진단 결과 출력
time.sleep(1)
print(f"\n진단 결과: {diagnosis}")

# 진단 결과에 따른 전문적인 대화 시작
if count >= 4:
    # 진단 결과를 포맷팅하여 전문적인 대화 템플릿에 적용
    system_message_content = "대화를 시작합니다."
    system_message = SystemMessage(content=system_message_content)

    message_content = """
    당신은 ADHD 진단을 받았습니다. 이것은 당황스러울 수 있지만, 치료 옵션이 많다는 점을 기억하는 것이 중요합니다. 다음은 몇 가지 단계입니다:

    1. **전문가의 도움을 받으세요**: ADHD를 전문으로 하는 정신과 의사나 심리학자와 상담 일정을 잡으세요. 그들은 포괄적인 평가를 제공하고 당신의 필요에 맞는 치료 옵션을 논의할 수 있습니다.

    2. **치료 옵션 탐색**: ADHD 치료는 종종 약물, 치료 및 생활 방식 변화를 결합합니다. 당신의 건강 관리 제공자가 이러한 옵션을 탐색하고 맞춤형 치료 계획을 개발하는 데 도움을 줄 수 있습니다.

    3. **자신에 대해 교육하세요**: ADHD가 당신에게 어떤 영향을 미치는지 더 많이 배우세요. 자신의 상태를 이해하면 치료 및 자기 관리에 대한 정보를 바탕으로 결정을 내릴 수 있습니다.

    4. **지원과 연결**: 친구, 가족 또는 ADHD를 가진 개인의 지원 그룹에서 지원을 찾으세요. 당신의 경험을 공유하고 이해하는 사람들로부터 지원을 받는 것이 매우 중요할 수 있습니다.

    5. **자기 관리를 우선시하세요**: 운동, 충분한 수면 및 스트레스 관리를 포함한 자기 관리 활동을 우선시하세요. 전반적인 건강을 관리하는 것은 ADHD 증상을 관리하는 데 도움이 될 수 있습니다.

    6. **긍정적인 태도를 유지하세요**: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.

    질문이 있거나 추가 논의가 필요하다면 언제든지 물어보세요. 우리는 당신의 ADHD 관리 여정에 함께할 것입니다.
    """
    
    human_message = HumanMessage(content=message_content)
    response = chat.invoke([system_message, human_message])
    
    print("\n전문적인 대화 내용:")
    print(response.content)

    # ChatPromptTemplate를 사용하여 진단 후 대화 템플릿 생성
    diagnosis_prompt_template = ChatPromptTemplate(
        input_variables=["diagnosis"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['input'],
                    template="""
                You have received a diagnosis of ADHD. This can feel overwhelming, but it's important to remember that there are many treatment options available. Here are some steps you can take:

                1. **Seek Professional Help**: Schedule an appointment with a healthcare provider, such as a psychiatrist or psychologist, who specializes in ADHD. They can provide a comprehensive evaluation and discuss treatment options tailored to your needs.

                2. **Explore Treatment Options**: Treatment for ADHD often involves a combination of medication, therapy, and lifestyle changes. Your healthcare provider can help you explore these options and develop a personalized treatment plan.

                3. **Educate Yourself**: Learn more about ADHD and how it affects you. Understanding your condition can empower you to make informed decisions about your treatment and self-care.

                4. **Connect with Support**: Seek support from friends, family, or support groups for individuals with ADHD. Sharing your experiences and receiving support from others who understand can be invaluable.

                5. **Take Care of Yourself**: Prioritize self-care activities such as exercise, adequate sleep, and stress management. Taking care of your overall health can improve your ability to manage ADHD symptoms.

                6. **Stay Positive**: ADHD is a manageable condition, and many people lead successful and fulfilling lives with proper treatment and support. Stay hopeful and remember that seeking help is the first step towards better managing your symptoms.

                Feel free to ask any questions or discuss your concerns further. We're here to support you on your journey to managing ADHD.
                """
                )
            )
        ]
    )

    # 진단 메시지를 생성하고 챗봇에 전달
    diagnosis_message = diagnosis_prompt_template.format(diagnosis=diagnosis)
    diagnosis_response = chat.invoke([diagnosis_message])
    
    print("\n진단 후 대화 내용:")
    print(diagnosis_response.content)

# 상호작용 루프
while True:
    user_message_content = input("추가 질문이나 논의하고 싶은 내용이 있으면 입력해주세요 (종료하려면 'exit' 입력):\n")
    
    if user_message_content.lower() == 'exit':
        print("대화를 종료합니다. 감사합니다!")
        break

    # 대화 진행
    response_content = conduct_conversation(user_message_content)

    # 챗봇 응답 출력
    print("\n챗봇 응답:")
    print(response_content)

