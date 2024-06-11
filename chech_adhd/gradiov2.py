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

# Load hospital data
hospital_data = pd.read_csv("chech_adhd/data/정신건강관련기관정보.csv", encoding="cp949")

# Load environment variables
load_dotenv()

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create ChatOpenAI instance
chat = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo-0125"
)

questions = [
    "어떤 일의 어려운 부분은 끝내 놓고, 그 일을 마무리를 짓지 못해 곤란을 겪은 적이 있습니까?",
    "체계가 필요한 일을 해야 할 때 순서대로 진행하기 어려운 경우가 있습니까?",
    "약속이나 해야 할 일을 잊어버려 곤란을 겪은 적이 있습니까?",
    "골치 아픈 일은 피하거나 미루는 경우가 있습니까?",
    "오래 앉아 있을 때, 손을 만지작거리거나 발을 꼼지락거리는 경우가 있습니까?",
    "마치 모터가 달린 것처럼, 과도하게 혹은 멈출 수 없이 활동을 하는 경우가 있습니까?"
]

# ADHD Self-diagnosis function
def adhd_self_diagnosis(user_answers):
    count = 0
    for i, answer in enumerate(user_answers):
        if (i < 3 and answer >= 3) or (i >= 3 and answer >= 4):
            count += 1

    if count >= 4:
        diagnosis = "ADHD의 가능성이 높습니다."
    else:
        diagnosis = "ADHD의 가능성이 낮습니다."
    return diagnosis

# Create prompt templates and memory for conversation
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

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_content),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

conversation = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
    memory=memory
)

agent = create_pandas_dataframe_agent(
    chat,
    hospital_data,
    verbose=True,
    prompt=prompt,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Main function for Gradio interface
def gradio_adhd_assessment(*user_answers):
    user_answers = list(map(int, user_answers))
    diagnosis = adhd_self_diagnosis(user_answers)
    message_content = """
        당신은 상담가이자 전문가입니다.

        친절하고 부드럽게 대화를 이어나가세요 : 사용자는 ADHD 판정을 받았습니다. 당황스러워할 수 있습니다.

        사용자는 ADHD 진단을 받았기 때문에 치료를 권장하세요.

        긍정적인 태도를 유지하세요: ADHD는 관리 가능한 상태이며, 적절한 치료와 지원을 통해 많은 사람들이 성공적이고 만족스러운 삶을 살아갑니다. 희망을 유지하고 도움을 구하는 것이 증상을 더 잘 관리하는 첫 걸음임을 기억하세요.

        필수사항입니다. 대화의 마지막에는 다시 한번 다음의 두 가지 치료법에 대해서 결정할 수 있게 도와주세요 : 전통적인 치료법의 약물치료와 대체 치료에 대해서 선택하라고 알려주세요.

        최대 문장의 길이는 3문장으로 해주세요.
        """
    system_message = SystemMessage(content=message_content + diagnosis)
    response = chat.invoke([system_message])
    diagnosis = response.content
    return diagnosis

def gradio_conversation(user_message_content):
    if "병원" in user_message_content:
        hospital_response = agent.run(user_message_content)
        return hospital_response
    else:
        result = conversation.invoke({"question": user_message_content})
        return result["text"]

# Gradio interface
iface = gr.Interface(
    fn=gradio_adhd_assessment,
    inputs=[gr.Radio(label=q, choices=["1", "2", "3", "4", "5"], type="index") for q in questions],
    outputs="text",
    title="ADHD 자가 진단",
    description="아래 질문에 답해주세요."
)

iface_conversation = gr.Interface(
    fn=gradio_conversation,
    inputs="text",
    outputs="text",
    title="ADHD 관련 대화",
    description="질문이나 논의하고 싶은 내용을 입력해주세요."
)

# Combine interfaces
demo = gr.TabbedInterface(
    [iface, iface_conversation],
    ["ADHD 자가 진단", "ADHD 관련 대화"]
)


demo.launch()
