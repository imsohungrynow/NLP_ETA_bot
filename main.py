import gradio as gr
import os
from dotenv import load_dotenv
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import numpy as np
import scipy.io.wavfile
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import time
import pandas as pd
import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# GPU 사용 여부 확인 및 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-0125", temperature=0)
# Create ChatOpenAI instance
chat = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo-0125"
)


# 병원 데이터 로드
hospital_data = pd.read_csv("./chech_adhd/data/정신건강관련기관정보.csv", encoding="cp949")


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



# 음악 설명 생성
def generate_response(mood, genre, elements, tempo):
    description = f"현재 기분: {mood}, 선호하는 음악 장르: {genre}, 음악 요소: {elements}, 분위기/속도: {tempo}"
    full_prompt = f"""
    당신은 ADHD 환자를 위한 음악 설명을 작성하는 어시스턴트입니다. 
    사용자의 입력을 바탕으로 음악 설명을 작성해야 합니다.
    사용자의 기분, 상황, 선호 장르, 분위기/속도에 맞춰 집중력을 향상시키고 스트레스를 감소시키는 음악의 설명을 작성해주세요.
    음악 설명은 다음의 예시의 형식대로 작성해야 합니다. 

    ## 예시
    이 음악은 차분하고 느린 템포의 클래식 음악으로, 바쁜 일상 속에서 집중력을 높이고 스트레스를 줄이는 데 도움을 줄 수 있습니다. 
    피아노와 바이올린의 부드러운 선율이 조화를 이루며, 마음을 편안하게 해줍니다. 
    특히 바흐의 '골드베르크 변주곡'처럼 복잡하지 않으면서도 집중력을 향상시키는 음악이 이상적입니다. 
    이 음악을 들으면 중요한 프로젝트를 완성하는 동안 마음의 안정을 찾고, 차분하게 일에 몰두할 수 있을 것입니다.

    {description}
    """
    result = llm([HumanMessage(content=full_prompt)])
    
    if hasattr(result, 'content'):
        return result.content
    elif hasattr(result, 'text'):
        return result.text
    else:
        return "No text content found in the response."

# 음악 생성 모델을 위한 음악 설명 요약 및 번역
def music_features(response):
    summary_prompt = f"""
    요약: 이 음악 설명에서 음악의 특징적인 부분만을 30자 이내로 간략하게 서술해주세요. 
    음악을 통해 얻을 수 있는 효과는 제거하고, 음악적 특성만을 서술해주세요. 
    답변은 영어로 출력합니다. 
    아래의 형식처럼 음악의 특성만을 고려하여 작성하고, 한 줄로 출력해주세요.

    출력 형식:
    a catchy beat for a podcast intro
    
    예시: 
    1. a catchy beat for a podcast intro
    2. a funky house with 80s hip hop vibes
    3. a chill song with influences from lofi, chillstep and downtempo
    \n\n'{response}'"""
    # 요약 요청을 모델에 전달
    summary_result = llm([HumanMessage(content=summary_prompt)])
    return summary_result

# 음악 생성 모델 초기화
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

def generate_music(summary, directory="./", duration_seconds=30):
    # Musicgen 모델을 사용하여 음악 생성
    inputs = processor(text=[summary], padding=True, return_tensors="pt")
    audio_values = model.generate(**inputs, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # 파일 저장
    output_filename = f"{directory}music.wav"
    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_values[0, 0].numpy())
    
    return output_filename

# Gradio 인터페이스용 통합 함수
def gradio_music_therapy(mood, genre, elements, tempo, duration_seconds):
    response = generate_response(mood, genre, elements, tempo)
    summary = music_features(response).content
    music_file = generate_music(summary, duration_seconds=duration_seconds)
    return response, music_file


# Gradio Blocks를 사용하여 인터페이스 설정
with gr.Blocks() as demo:
    gr.Markdown("# 성인 ADHD 진단 및 치료를 위한 상담 챗봇: ETA bot")
    gr.Markdown("#### ADHD를 위한 자가진단 및 치료 챗봇입니다. 자가진단을 통해 ADHD 결과를 분석하고, 병원 치료 혹은 음악 치료를 받을 수 있습니다.")
    with gr.Tabs():
        
        # ADHD 자가진단 챗봇 탭
        with gr.TabItem("ADHD 자가진단"):
            inputs = [gr.Radio(label=q, choices=["1", "2", "3", "4", "5"], type="index") for q in questions]
            diagnosis_output = gr.Textbox(label="진단 결과")
            gr.Button("진단 받기").click(
                gradio_adhd_assessment, 
                inputs=inputs, 
                outputs=diagnosis_output
            )
        
        # ADHD 병원추천 챗봇 탭
        with gr.TabItem("ADHD 병원 추천"):
            user_input = gr.Textbox(label="질문 입력", placeholder="병원 정보나 ADHD 관련 질문을 입력하세요.")
            hospital_response_output = gr.Textbox(label="병원 추천 결과")
            gr.Button("추천 받기").click(
                gradio_conversation, 
                inputs=user_input, 
                outputs=hospital_response_output
            )


        # ADHD 치료 음악 생성 탭
        with gr.TabItem("ADHD 치료 음악 생성"):
            with gr.Row():
                with gr.Column(scale=1):
                    mood_input = gr.Textbox(label="현재 기분 또는 상황", placeholder="예: 행복, 스트레스 받음 등")
                    genre_input = gr.Textbox(label="선호하는 음악 장르", placeholder="예: 클래식, 재즈 등")
                    elements_input = gr.Textbox(label="음악에 반영되었으면 하는 요소", placeholder="예: 자연 소리, 특정 악기 등")
                    tempo_input = gr.Textbox(label="원하는 음악의 분위기나 속도", placeholder="예: 차분하고 느린, 에너지 넘치는 등")
                    duration_input = gr.Slider(minimum=1, maximum=120, step=1, label="음악 길이 (초 단위)", value=30)
                    generate_button = gr.Button("음악 생성")
                with gr.Column(scale=1):
                    music_description_output = gr.Textbox(label="생성된 음악 설명")
                    music_file_output = gr.Audio(label="생성된 음악 파일")
            generate_button.click(gradio_music_therapy, 
                                  inputs=[mood_input, genre_input, elements_input, tempo_input, duration_input], 
                                  outputs=[music_description_output, music_file_output])
        

demo.launch(share=True)