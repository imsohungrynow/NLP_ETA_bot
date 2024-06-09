import openai
import googlemaps
from datetime import datetime

# OpenAI GPT API 설정
openai.api_key = ''

# Google Maps API 설정
gmaps = googlemaps.Client(key='')

def generate_query(location):
    # GPT를 사용하여 사용자 위치를 기반으로 병원 검색 쿼리 생성
    prompt = f"What hospitals are near {location}?"
    response = openai.Completion.create(
        engine="davinci", 
        prompt=prompt, 
        temperature=0.5, 
        max_tokens=100
    )
    return response.choices[0].text.strip()

def find_hospital_nearby(location):
    # 병원 검색 쿼리 생성
    query = generate_query(location)

    # Google Maps API를 사용하여 병원 검색
    places = gmaps.places_nearby(location=query, type='hospital', radius=5000)

    if len(places['results']) > 0:
        # 병원 중 가장 가까운 병원 선택
        nearest_hospital = places['results'][0]

        # 가장 가까운 병원의 주소 및 좌표 가져오기
        hospital_address = nearest_hospital['vicinity']
        hospital_location = nearest_hospital['geometry']['location']

        # 현재 위치에서 가장 가까운 병원까지의 방향 찾기
        directions = gmaps.directions(location, hospital_address, mode="driving", departure_time=datetime.now())

        print("Nearest Hospital:")
        print("Name:", nearest_hospital['name'])
        print("Address:", hospital_address)
        print("Location:", hospital_location)
        print("Directions:", directions[0]['summary'])
    else:
        print("No hospitals found nearby.")

# 사용자의 현재 위치 입력
user_location = input("Enter your current location: ")

# 주변 병원 찾기 및 방향 탐색
find_hospital_nearby(user_location)