from fastapi import FastAPI
import os
import environ


import ml
from dto import IssueDto, ResponseDto

app = FastAPI()
BASE_URL = "http://backend:8080/"
env = environ.Env()
environ.Env.read_env()

def generate_file_url(issue: IssueDto):
    return f"{BASE_URL}api/v0/issues/{issue.id}/downloadFile"

def audio_file_url(issue: IssueDto):
    return f"{BASE_URL}api/v0/issues/{issue.id}/downloadAudioFile"

 
@app.post("/")
async def analyze_recommendation(
        issue: IssueDto
):
    token = env('HUGGINGFACE_TOKEN')
    file_url = generate_file_url(issue) #забрали картинку с описанием и категорией
    audio_file = audio_file_url(issue) # забрали аудио
    #иницилизация модели для распознавания картинок
    model = ml.init_model(token, issue.categoryId)
    #получение распознанных объектов 
    labels = ml.detection(model, file_url)
    if issue.isDescriptionByVoice:
        #получение коэффициента доверия c аудио 
        description = ml.get_description(audio_file)
        coef_of_trust = ml.get_response(token, description, labels, issue.categoryId)
    else:  
        #получение коэффициента доверия без аудио
        coef_of_trust = ml.get_response(token, issue.description, labels, issue.categoryId)
    return ResponseDto(prediction=coef_of_trust)