from fastapi import FastAPI

import ml
from dto import IssueDto, ResponseDto

app = FastAPI()
BASE_URL = "https://https://city24.civiltechgroup.ru/"


def generate_file_url(issue: IssueDto):
    return f"{BASE_URL}api/v0/issues/{issue.id}/downloadFile"


@app.post("/")
async def analyze_recommendation(
        issue: IssueDto
):
    file_url = generate_file_url(issue)
    ml.some_function(issue, file_url)
    return ResponseDto(prediction=0.5)
