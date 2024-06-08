import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate,SystemMessagePromptTemplate, AIMessagePromptTemplate,PromptTemplate
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

# 진단 결과에 따른 답변 생성
if count >= 4:
    # 진단 메시지
    message_content = """
    ADHD와 관련된 다음과 같은 두가지 치료법을 무조건 먼저 얘기하세요 : ADHD는 전통적인 치료법의 약물치료와 대체 치료가 있습니다. 이 둘 중에서 선호하는 것을 알아낼 수 있게 무조건 먼저 말할 수 있습니다.
    
    자신에 대해 교육하세요: ADHD가 당신에게 어떤 영향을 미치는지 더 많이 배우세요. 자신의 상태를 이해하면 치료 및 자기 관리에 대한 정보를 바탕으로 결정을 내릴 수 있습니다.

    지원과 연결: 친구, 가족 또는 ADHD를 가진 개인의 지원 그룹에서 지원을 찾으세요. 당신의 경험을 공유하고 이해하는 사람들로부터 지원을 받는 것이 매우 중요할 수 있습니다.

    긍정적인 태도를 유지하세요: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.
    
    대화의 마지막에는 다시 한번 다음의 두 가지 치료법에 대해서 결정할 수 있게 도와주세요 : 전통적인 치료법의 약물치료와 대체 치료에 대해서 선택하라고 알려주세요.
    
    최대 문장의 길이는 3문장으로 해주세요.
    """
    
    systemMessage = SystemMessage(content=message_content)
    response = chat.invoke([systemMessage])
    print(response.content)

else:
    message_content_N = """
        지금 이 사용자는 ADHD 진단 결과에서 ADHD 판정을 받지 않았습니다: 이를 숙지하고 대화하세요.
    
        자신에 대해 교육하세요: ADHD가 당신에게 어떤 영향을 미치는지 더 많이 배우세요. 자신의 상태를 이해하면 치료 및 자기 관리에 대한 정보를 바탕으로 결정을 내릴 수 있습니다.
        
        사용자에게 ADHD가 흔하다는 말을 해주세요 : ADHD는 바쁜 현대인들에게 충분히 생길 수 있는 말입니다.       
        """

    human_message = HumanMessage(content=message_content_N)
    response = chat.invoke([human_message])
    print(response.content)


# 사용자 메시지 생성 함수
def create_user_message(content):
    return HumanMessage(content=content)

# 챗봇 응답 메시지 생성 함수
def create_assistant_message(content):
    return AIMessage(content=content)

# # 대화 함수
# cnt = 0
# while True:
#     if cnt
#     user_message_content = input("질문이나 논의하고 싶은 내용이 있으면 입력해주세요 (종료를 원하면 '종료'라고 입력하세요): ")
#     if user_message_content.lower() == '종료':
#         break
#
#     # 사용자 메시지 생성
#     user_message = create_user_message(user_message_content)
#
#     # 챗봇 응답 메시지 생성
#     assistant_message_content = "도움이 되셨길 바랍니다! ADHD 관련하여 추천드릴 만한 자료나 지원 그룹을 찾고 계신다면, CHADD(Children and Adults with Attention-Deficit/Hyperactivity Disorder), ADDitude Magazine, 그리고 National Resource Center on ADHD를 확인해보세요. 또한, 온라인 커뮤니티와 포럼도 비슷한 경험을 공유하는 사람들과 연결될 수 있는 공간을 제공합니다. 추가적인 질문이 있거나 도움이 필요하다면 언제든지 물어보세요."
#     assistant_message = create_assistant_message(assistant_message_content)
#
#     # 대화를 챗봇에 전달하고 응답 받기
#     response = chat.invoke([user_message, assistant_message])
#
#     # 챗봇의 응답 출력
#     print(response.content)

while True:
    user_message_content = input("질문이나 논의하고 싶은 내용이 있으면 입력해주세요 (종료를 원하면 '종료'라고 입력하세요): ")
    if user_message_content.lower() == '종료':
        break
    user_message = create_user_message(user_message_content)
    # 대화의 흐름에 따른 프롬프트 생성
    prompt_content = """
    사용자가 ADHD 관련해서 병원이나 약물 치료를 원할 경우, 주변의 병원을 추천하고 그 병원의 위치와 오픈 시간, 가는 방법 등의 정보를 제공하세요.

    사용자가 병원 추천을 원하지 않으면 ADHD 관련 일반적인 도움말과 지원 그룹 정보를 제공하세요.
    """
    system_content = '''
    당신은 ADHD환자가 믿고 의지할 수 있는 전문가입니다.

    당신은 ADHD 환자에게 도움을 줄 수 있습니다.

    긍정적인 태도를 유지하세요: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.

    병원과 관련된 질문을 받아주세요 : 약물치료는 병원에서 이루어집니다. 약물 치료와 관련된 얘기가 나오면 병원 이야기를 해주세요.

    *필수 기능입니다. 병원에 대해서 모든 것을 말해주세요 : 병원의 정보에 대해서 궁금한 사용자에게 위치를 물어보고 그 주변 병원의 이름, 정보, 가는 방법 등 가능한 방법을 알려주세요.
    '''

    # 프롬프트 생성
    prompt = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system_content),
        SystemMessagePromptTemplate.from_template(prompt_content),
        HumanMessagePromptTemplate.from_template("나는 약물치료를 받고싶어."),
        AIMessagePromptTemplate.from_template("약물치료를 받고 싶으시군요. 약물치료는 병원에서 받으실 수 있습니다. 위치를 알려주시면 주변 병원을 추천해드릴게요."),
    ])

    # 사용자 메시지와 생성된 프롬프트로 응답 생성
    response = chat.invoke([user_message, prompt])

    # 응답 출력
    print(response.content)