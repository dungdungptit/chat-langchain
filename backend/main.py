"""Main entrypoint for the app."""
import os
import asyncio
from typing import *
from langchain_core.pydantic_v1 import BaseModel, Field
from uuid import UUID

import langsmith
from chain import (
    ChatRequest,
    chain_json,
    normalize_replace_abbreviation_text,
    get_source,
)
from chain_code import (
    get_answer,
    get_results,
    get_intent,
    get_results_intent,
    final_rag_chain,
    search,
)
from search_fqa import encode

from ingest import re_ingest, reload_server
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__5fc598259765473aab41796672a5776b"
os.environ["LANGCHAIN_PROJECT"] = "FTU"

client = Client()

origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

static_folder = "DataFTU"
app.mount("/static", StaticFiles(directory=static_folder), name="static")

add_routes(
    app,
    # answer_chain,
    chain_json,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/chain_code")
async def get_answer_code(body: ChatRequest):
    question = body.question
    chat_history = body.chat_history
    # return body
    print(body)
    res = get_answer(question, chat_history)
    print(res)
    return {"result": "successfully", "status": 200, "output": res}
    # try:

    # except:
    #     res = "Hmm, tôi không chắc."
    #     return {"result": "error", "status": 500, "output": res}


@app.post("/search_vector")
async def search_vector(question: str):
    try:
        res = search(question)
        print(res)
        return {"result": "successfully", "status": 200, "output": res}
    except:
        res = "Hmm, tôi không chắc."
        return {"result": "error", "status": 500, "output": res}


@app.get("/write")
async def write(jwt: str):
    with open("./jwt.txt", "w") as f:
        f.write(jwt)
        f.close()

    jwt = ""
    with open("./jwt.txt", "r") as f:
        jwt = f.read()
        f.close()

    return {"result": "preprocessed successfully", "code": 200, "res": jwt}


@app.post("/update_question_answer")
async def update_q_a(question: str):
    try:
        res = search(question)
        print(res)
        if res[0] != "No":
            return {"result": "can update", "status": 200, "output": res}
    except:
        res = "Hmm, tôi không chắc."
        return {"result": "error", "status": 500, "output": res}


@app.get("/encode")
async def get_encode(question: str):
    vector = str(encode(question)[0].tolist())
    return {"result": "encoded", "status": 200, "data": vector}


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


class GetTraceBody(BaseModel):
    run_id: UUID


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


@app.get("/preprocess")
async def preprocess(text: str):
    text = normalize_replace_abbreviation_text(text)
    return {"result": "preprocessed successfully", "code": 200, "text": text}


@app.get("/search_results")
async def search_results(question: str):
    res = get_results(question)
    return {"result": "preprocessed successfully", "code": 200, "res": res}


@app.get("/get_intent")
async def search_intent(question: str):
    res = get_intent(question)
    return {"result": "preprocessed successfully", "code": 200, "res": res}


@app.get("/get_results_intent")
async def search_results_intent(question: str):
    res = get_results_intent(question)
    return {"result": "preprocessed successfully", "code": 200, "res": res}


@app.get("/reload_server")
async def rei():
    reload_server()
    return {"result": "preprocessed successfully", "code": 200}


from typing import Annotated

from fastapi import FastAPI, File, UploadFile


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


class FeedBack(BaseModel):
    question: str
    chatbot_answer: str
    human_answer: str
    like: bool


import requests

ip_back_data = "https://ftu-qldl-be.ript.vn"


@app.post("/feedbacks")
async def feedbacks(data: FeedBack):
    vector = str(encode(data.question)[0].tolist())
    res = {}
    res["question"] = data.question
    res["chatbot_answer"] = data.chatbot_answer
    res["human_answer"] = data.human_answer
    res["like"] = data.like
    res["vector"] = vector

    url = f"{ip_back_data}/api/feedbacks"
    payload = {
        "data": {
            "question": res["question"],
            "chatbot_answer": res["chatbot_answer"],
            "human_answer": res["human_answer"],
            "vector": res["vector"],
            "like": res["like"],
        }
    }
    # break
    response = requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )

    # print(response.json())
    return {"status": response.status_code, "data": response.json()}


@app.get("/prep_output")
async def prep_output(text: str):
    text = normalize_replace_abbreviation_text(text)
    sources = get_source(text)
    """ 
    {"sources": [{
      "page_content": "Chỉ tiêu tuyển sinh đại học Ngoại thương (FTU) năm 2024\n\nTổng chỉ tiêu tuyển sinh 2024 của trường là 4130 gồm:\n\n1. Tổng chỉ tiêu tuyển sinh Trụ sở chính Hà Nội là 3080\n\n2. Tổng chỉ tiêu tuyển sinh CS Quảng Ninh là 100\n\n3. Tổng chỉ tiêu tuyển sinh Cơ sở II-TP Hồ Chí Minh là 950\n\nChi tiết chỉ tiêu tuyển sinh từng ngành/ nhóm ngành tại từng cơ sở theo từng phương thức xét tuyển:\n\n1. chỉ tiêu tuyển sinh Xét tuyển riêng PT3 và đặc thù tại Trụ sở chính Hà Nội (Mã ngành/ nhóm ngành xét tuyển: NTH409,",
      "metadata": {
        "source": "Đề án tuyển sinh 2024 (dự thảo 2)\\245.txt"
      },
      "type": "Document"
    }]}
       """
    list_sources = []
    for source in sources:
        # print(source)
        source.metadata = {"source": source.metadata["source"]}
        list_sources.append(source.metadata["source"])
    return {
        "result": "preprocessed successfully",
        "code": 200,
        "list_sources": list_sources,
        "text": text,
        "detail": sources,
    }


@app.get("/write")
async def write(jwt: str):
    with open("./jwt.txt", "w") as f:
        f.write(jwt)
        f.close()

    jwt = ""
    with open("./jwt.txt", "r") as f:
        jwt = f.read()
        f.close()

    return {"result": "preprocessed successfully", "code": 200, "res": jwt}


# ----------------------FUNCTION CALLING----------------------------------
from RAG_w_functionsCalling.function_calling import (
    function_calling,
    function_calling_re_ingest,
)
import re


def optimize_markdown(md_text):
    # Xóa các dòng trống liên tiếp (giữ lại 1 dòng trống duy nhất nếu có)
    md_text = re.sub(r"\n\s*\n+", "\n", md_text)

    # Chuẩn hóa dấu gạch đầu dòng của danh sách (loại bỏ khoảng trắng dư)
    md_text = re.sub(r"(\n\s*[-*])\s+", r"\1 ", md_text)

    # Chuẩn hóa danh sách số (loại bỏ khoảng trắng dư)
    md_text = re.sub(r"(\n\d+\.)\s+", r"\1 ", md_text)

    return md_text.strip()


@app.post("/function-calling")
async def function_calling_chatbot(data: ChatRequest):
    question = data.question
    response = function_calling(question=question)

    answer = response["response"] 
    recommendations = response["recommendations"]
    sources = response["sources"]
    return {
        "result": "successfully",
        "status": 200,
        "output": {
            "user_answer": False,
            "response": [{"type": "text", "content": optimize_markdown(answer), "recommendations": recommendations, "sources": sources}],
        },
    }


@app.get("/re_ingest")
async def reingest(jwt: str):
    res = re_ingest(jwt)
    res = function_calling_re_ingest()
    return {"result": "preprocessed successfully", "code": 200, "res": res}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
