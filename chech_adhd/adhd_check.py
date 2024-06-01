import os
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일 로드
load_dotenv()

# OpenAI API 키 환경 변수 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# PDF 파일 경로 설정
current_directory = os.path.dirname(os.path.abspath(__file__))
pdf_file_path = os.path.join(current_directory, 'data', 'adhd_checklist.pdf')

# PDF에서 텍스트 추출
def extract_questions_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    # 간단한 방법으로 질문들만 추출한다고 가정
    questions = [line for page in pages for line in page.split("\n") if line.strip().startswith("Q:")]
    return questions

# ADHD 체크리스트 질문 추출
questions = extract_questions_from_pdf(pdf_file_path)

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"
)

# 프롬프트 템플릿 설정
prompt_template = ChatPromptTemplate(
    input_variables=["user_input", "question"],
    template="""
You are a helpful assistant designed to help users determine if they might have ADHD based on a checklist. Below is the question you should ask the user:

{question}

The user has provided the following input describing their symptoms:
"{user_input}"

Based on the question and the user's input, provide the next appropriate question or a summary if all questions are answered.
"""
)

# 질문과 답변을 기록할 리스트
user_answers = []

# 인사와 첫 번째 질문
print("안녕하세요! ADHD 증상 체크를 도와드리겠습니다. 몇 가지 질문을 드릴게요.")
current_question_index = 0

# 대화 루프
while current_question_index < len(questions):
    current_question = questions[current_question_index]
    user_input = input(current_question + "\n")
    
    # 사용자 답변 기록
    user_answers.append((current_question, user_input))
    
    # 다음 질문으로 이동
    current_question_index += 1

# 최종 평가를 위한 프롬프트 생성
evaluation_prompt = "Based on the user's responses to the ADHD checklist, provide an assessment.\n\n"
for question, answer in user_answers:
    evaluation_prompt += f"Q: {question}\nA: {answer}\n"

# 프롬프트 실행 및 결과 출력
response = chat.run(evaluation_prompt)
print(response)