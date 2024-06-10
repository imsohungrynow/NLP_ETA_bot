import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# .env 파일 로드
load_dotenv()

# OpenAI API 키 환경 변수 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo-0125"
)

# ADHD 증상 체크 질문 리스트
questions = [
    "어떤 일의 어려운 부분은 끝내 놓고, 그 일을 마무리를 짓지 못해 곤란을 겪은 적이 있습니까?",
    "체계가 필요한 일을 해야 할 때 순서대로 진행하기 어려운 경우가 있습니까?",
    "약속이나 해야 할 일을 잊어버려 곤란을 겪은 적이 있습니까?",
    "골치 아픈 일은 피하거나 미루는 경우가 있습니까?",
    "오래 앉아 있을 때, 손을 만지작거리거나 발을 꼼지락거리는 경우가 있습니까?",
    "마치 모터가 달린 것처럼, 과도하게 혹은 멈출 수 없이 활동을 하는 경우가 있습니까?"
]

# 병원 데이터 로드
hospital_data = pd.read_csv("chech_adhd/data/정신건강관련기관정보.csv", encoding="cp949")

# 대화 히스토리를 저장하기 위한 메모리 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 시스템 메시지와 사용자 프롬프트를 설정합니다
system_content = """
당신은 ADHD환자가 믿고 의지할 수 있는 전문가입니다.

당신은 ADHD 환자에게 도움을 줄 수 있습니다.

긍정적인 태도를 유지하세요: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.

사용자가 ADHD 관련해서 병원이나 약물 치료를 원할 경우, 주변의 병원을 추천하고 그 병원의 위치와 오픈 시간, 가는 방법 등의 정보를 제공하세요.

사용자가 병원 추천을 원하지 않으면 ADHD 관련 일반적인 도움말과 지원 그룹 정보를 제공하세요.

약물 치료 이야기가 나오면 병원과 관련된 질문을 유도해주세요: 약물치료는 병원에서 이루어집니다. 약물 치료와 관련된 얘기가 나오면 병원 이야기를 해주세요.

병원에 대해서 궁금한 사용자에게 위치를 물어보세요.

위치를 물어볼때는 예를들어 서울특별시, 부산광역시 등 정확한 대한민국 행정 구역을 입력하라고 무조건 얘기하세요.

당신은 각 병원의 기관명, 기관구분, 주소, 홈페이지를 알고 있습니다. csv파일을 근거로 정확하게 대답하세요.

사용자가 병원에 대해 궁금해하면 병원의 정보를 알려줄 수 있다는 말을 하세요.

사용자에게 위치를 물어보면 해당 위치에 대한 병원 정보를 제공하세요. 이때 주소 열에 있는 단어와 무조건 같지 않아도 됩니다.

병원에 대해서 모든 것을 말해주세요: 병원의 정보에 대해서 궁금한 사용자에게 위치를 물어보고 그 주변 병원의 이름, 정보, 가는 방법 등 가능한 방법을 알려주세요.
"""

# 프롬프트 템플릿을 설정합니다
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_content),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# 대화 체인 설정
conversation = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=False,
    memory=memory
)

# 에이전트 생성
agent = create_pandas_dataframe_agent(
    chat,
    hospital_data,
    verbose=False,
    prompt=prompt,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

# Gradio 인터페이스 설정
def chat_interface(user_input, chat_history=[]):
    # 초기 인사 메시지 및 첫 질문
    if not chat_history:
        chat_history = [
            ("system", "안녕하세요! ADHD 증상 체크를 도와드리겠습니다. 몇 가지 질문을 드릴게요.각 질문에 대해 다음과 같이 응답해주세요\n1: 전혀 그렇지 않다\n2: 거의 그렇지 않다\n3: 약간 혹은 가끔 그렇다\n4: 자주 그렇다\n5: 매우 자주 그렇다"),
        ]
        current_question = questions[0]
        chat_history.append(("system", current_question))
        return "", chat_history

    # 사용자 입력 처리
    memory.chat_memory.add_message(HumanMessage(content=user_input))  # 사용자 입력을 메모리에 추가

    # ADHD 증상 체크
    current_question_index = len([msg for msg in chat_history if msg[0] == 'system' and msg[1] in questions])
    if current_question_index < len(questions):
        current_question = questions[current_question_index]
        chat_history.append(("system", current_question))
        return "", chat_history

    # 사용자 입력 처리 후 마지막 질문에 대한 응답 저장
    try:
        user_responses = [int(msg[1]) for msg in chat_history if msg[0] == 'user']
    except ValueError:
        user_responses = []

    # ADHD 여부 판단
    count = 0
    for i, answer in enumerate(user_responses):
        if (i < 3 and answer >= 3) or (i >= 3 and answer >= 4):
            count += 1

    if count >= 4:
        # ADHD 진단 메시지 추가
        message_content = """
        당신은 상담가이자 전문가입니다.

        사용자는 ADHD 증상의 의심됩니다. 먼저, 사용자에게 ADHD증상이 의심된다는 말을 전달하세요.

        친절하고 부드럽게 대화를 이어나가세요 : 사용자는 ADHD 판정을 받았습니다. 당황스러워할 수 있습니다.

        사용자는 ADHD 진단을 받았기 때문에 치료를 권장하세요.

        긍정적인 태도를 유지하세요: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.

        필수사항입니다. 대화의 마지막에는 다시 한번 다음의 두 가지 치료법에 대해서 결정할 수 있게 도와주세요 : 전통적인 치료법의 약물치료와 대체 치료에 대해서 선택하라고 알려주세요.

        최대 문장의 길이는 3문장으로 해주세요.
        """
        system_message = SystemMessage(content=message_content)
        response = chat.invoke([system_message])
        response_content = response.content
        memory.chat_memory.messages.append(SystemMessage(content=response.content))  # 시스템 응답을 메모리에 추가
    else:
        diagnosis = "ADHD의 가능성이 낮습니다."
        chat_history.append(("system", diagnosis))  # ADHD 진단 메시지 추가
        message_content_N = """
        당신은 상담가이자 전문가입니다.

        사용자는 ADHD 증상 의심이 가지 않습니다. 먼저, 사용자에게 ADHD증상이 의심되지 않는 말을 전달하세요.

        지금 이 사용자는 ADHD 진단 결과에서 ADHD 판정을 받지 않았습니다: 이를 숙지하고 대화하세요.

        자신에 대해 교육하세요: ADHD가 당신에게 어떤 영향을 미치는지 더 많이 배우세요. 자신의 상태를 이해하면 치료 및 자기 관리에 대한 정보를 바탕으로 결정을 내릴 수 있습니다.

        사용자에게 ADHD가 흔하다는 말을 해주세요: ADHD는 바쁜 현대인들에게 충분히 생길 수 있는 말입니다.
        """
        system_message = SystemMessage(content=message_content_N)
        response = chat.invoke([system_message])
        response_content = response.content
        memory.chat_memory.messages.append(SystemMessage(content=system_message.content))  # 응답을 메모리에 추가

    if "병원" in user_input:
        hospital_response = agent.run(user_input)
        memory.chat_memory.messages.append(SystemMessage(content=hospital_response))
        response_content = hospital_response

    else:
        result = conversation.invoke({"question": user_input})
        chat_history.append(("system", result["text"]))  # 수정된 부분
        response_content = result["text"]


    return response_content, chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="ADHD 챗봇")
    with gr.Row():
        message_input = gr.Textbox(label="입력")
        submit_button = gr.Button("전송")
    submit_button.click(chat_interface, [message_input, chatbot], [message_input, chatbot])
    message_input.value = ''  # 사용자 입력란 비우기

# # Gradio 앱 설정
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(label="ADHD 챗봇")
#     with gr.Row():
#         message_input = gr.Textbox(label="입력")
#         submit_button = gr.Button("전송")
#     submit_button.click(chat_interface, [message_input, chatbot], [message_input, chatbot])


# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(label="ADHD 챗봇")
#     with gr.Row():
#         message_input = gr.Textbox(label="입력")
#         submit_button = gr.Button("전송")
#     def submit_action():
#         user_input = message_input.value
#         response, chat_history = chat_interface(user_input, chatbot.messages)
#         chatbot.messages = chat_history
#         chat_history.append(("user", user_input))  # 사용자 입력을 메시지에 추가
#         chat_history.append(("system", response))  # 챗봇 응답을 메시지에 추가
#         chatbot.update_chat()
#     submit_button.click(submit_action)


# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(chat_interface, inputs="textbox", outputs="textbox", label="ADHD 챗봇")

# Gradio 앱 실행
demo.launch()
