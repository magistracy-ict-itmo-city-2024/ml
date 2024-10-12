# from dto import IssueDto
from huggingface_hub import InferenceClient
import json
import random


def translate_text(text, token):
    translator = InferenceClient(model = "facebook/wmt19-ru-en", token = token)
    return translator.translation(text)


def get_response(token, inputs, labels, category_id):
        #обработка языка
        description = translate_text(inputs, token)

        #допустим до этого пришел какой-то коэфф доверия (его нет в MVP)
        coeff = round(random.uniform(0.9, 1), 2)

        #категории по распознанным объектам
        categories = {
               "trash" : ['paper', 'plastic', 'cardboard', 'glass', 'metal'],
               "accident": ['car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'stop sign', 'person'],
               "bad weather": ['umbrella', 'person'],
               "renovation work": ['truck', 'traffic light', 'stop sign', 'person', 'fire hydrant']
        }

        #названия категорий по id
        categories_names = {
               1: "situation with trash",
               2: "bad weather",
               3: "danger",
               4: "city problems"
        }

        #подготовка labels
        candidate_labels = []
        for i in categories:
                for j in labels:
                        if j in categories[i]:
                                candidate_labels.append(i)
        candidate_labels = list(set(candidate_labels))

        if not candidate_labels:
                candidate_labels.append(categories_names[category_id])

        #обработка результатов модели
        client_nlp = InferenceClient(token=token)
        params = {"candidate_labels": candidate_labels}
        response = client_nlp.post(json={"inputs": description, "parameters": params}, model="typeform/distilbert-base-uncased-mnli")
        response = response.decode()
        response = json.loads(response)
        
        #расчет коэффициента доверия
        pred_label, pred_score = response['labels'][0], response['scores'][0]

        if (pred_label == "trash" or pred_label == "situation with trash") and category_id == 1 and pred_score > 0.7:
                return 1.0  
        elif  (pred_label == "accident" or pred_label == "danger") and category_id == 3 and pred_score > 0.7:
                return 1.0 
        elif  pred_label == "bad weather" and category_id == 2 and pred_score > 0.7:
                return 1.0 
        elif  (pred_label == "renovation work" or pred_label == "city problems") and category_id == 4 and pred_score > 0.7:
                return 1.0 
        else:
                return coeff - 0.1
    

def init_model(token, category_id):  
        if category_id == 1:
                client = InferenceClient(model = "pvallej3/garbage_classifier", token=token)
        else:
                client = InferenceClient(model = "facebook/detr-resnet-50", token=token)
        
        return client

def detection(model, pic):
        response = model.object_detection(pic)
        labels = [] 
        for i in response:
              if i['score'] > 0.7:
                     labels.append(i["label"])
        return list(set(labels))