# Import libraries
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Tuple

# import weaviate
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)

from ingest import get_retriever, get_intent_retriever

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import re
import string
import json

import requests

from search_fqa import searchSimilarity, encode
import pandas as pd
import torch

ip_back_data = "https://ftu-qldl-be.ript.vn"


def search(question):
    print(question)
    jwt = ""
    with open("./jwt.txt", "r") as f:
        jwt = f.read()
        f.close()
    data_qas = []
    for i in range(1, 100):
        list_topic_response = requests.get(
            f"{ip_back_data}/api/feedbacks?populate=*&pagination[pageSize]=100&pagination[page]={i}",
        )
        data_feedback = list_topic_response.json()["data"]
        if len(data_feedback) == 0:
            break
        for item in data_feedback:
            if item["attributes"]["used"] == True:
                if item["attributes"]["human_answer"] == "":
                    data_qas.append(
                        [
                            item["id"],
                            item["attributes"]["question"],
                            item["attributes"]["chatbot_answer"],
                            item["attributes"]["vector"],
                            "fqa",
                        ]
                    )
                else:
                    data_qas.append(
                        [
                            item["id"],
                            item["attributes"]["question"],
                            item["attributes"]["human_answer"],
                            item["attributes"]["vector"],
                            "fqa",
                        ]
                    )

    list_topic_response = requests.get(
        f"{ip_back_data}/api/intents?populate=*&pagination[pageSize]=100",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    data_res = [
        {"questions": i["attributes"]["questions"]["data"]}
        for i in list_topic_response.json()["data"]
    ]
    for i in data_res:
        for j in i["questions"]:
            data_qas.append(
                [
                    j["id"],
                    j["attributes"]["question"],
                    j["attributes"]["answer"],
                    j["attributes"]["vector"],
                    "intent",
                ]
            )

    data_qas

    df = pd.DataFrame(data_qas, columns=["id", "Question", "Answer", "Vector", "Model"])

    def getApproximateAnswer(q):
        max_score = 0
        answer = ""
        prediction = ""
        for idx, row in df.iterrows():
            score = searchSimilarity(q, torch.tensor(json.loads(row["Vector"])))
            if score >= 0.9:  # I'm sure, stop here
                return [row["id"], row["Answer"], score, row["Question"]]
            elif score > max_score:  # I'm unsure, continue
                max_score = score
                id = row["id"]
                answer = row["Answer"]
                prediction = row["Question"]
        if max_score > 0.7:
            return [id, answer, max_score, prediction]
        return [-1, "No", max_score, prediction]

    q_search = encode(question)

    res = getApproximateAnswer(q_search)
    return res


def get_intent(text):
    url = "http://localhost:5080/model/parse"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    return response.json()


dict_replace = {
    "2k": "200",
    "a": "anh",
    "ac": "anh chị",
    "add|ad|adm|addd": "admin",
    "ak|ah": "à",
    "attt": "An toàn thông tin",
    "ttdpt|ttđpt": "Truyền thông đa phương tiện",
    "b": "bạn",
    "baoh|bh": "bao giờ",
    "bhyt": "bảo hiểm y tế",
    "bn|bnh|baonh": "bao nhiêu",
    "bâyh": "bây giờ",
    "cccd": "căn cước công dân",
    "chòa": "chào",
    "cj": "chị",
    "ck": "chuyển khoản",
    "chỉ tiêu": "chỉ tiêu tuyển sinh",
    "chỉ tiêu tuyển sinh tuyển sinh": "chỉ tiêu tuyển sinh",
    "clc": "chất lượng cao",
    "cmnd": "chứng minh nhân dân",
    "cn dpt|cndpt": "Công nghệ đa phương tiện",
    "udu": "Công nghệ thông tin định hướng ứng dụng",
    "cntt udu": "Công nghệ thông tin định hướng ứng dụng",
    "cntt ứng dụng": "Công nghệ thông tin định hướng ứng dụng",
    "công nghệ thông tin ứng dụng": "Công nghệ thông tin định hướng ứng dụng",
    "cntt": "Công nghệ thông tin",
    "dgnl": "đánh giá năng lực",
    "dgtd": "đánh giá tư duy",
    "dhqg": "đại học quốc gia",
    "dkxt": "đăng ký xét tuyển",
    "dky|dki|đăng kí|dki": "đăng ký",
    "dpt|đpt": "đa phương tiện",
    "dtvt|dt vt|đtvt|đt vt": "điện tử viễn thông",
    "d|đ": "điểm",
    "e": "em",
    "gddt|gd dt|gdđt|gd đt": "giáo dục đào tạo",
    "gd": "giáo dục",
    "hc": "học",
    "hcm": "hồ chí minh",
    "hnay": "hôm nay",
    "hs": "học sinh",
    "hsg": "học sinh giỏi",
    "hssv": "hồ sơ sinh viên",
    "hqua": "hôm qua",
    "điểm tuyển sinh": "điểm trúng tuyển",
    "điểm đỗ": "điểm trúng tuyển",
    "điểm chuẩn|điểm": "điểm trúng tuyển",
    "điểm trúng tuyển đỗ": "điểm trúng tuyển",
    "điểm trúng tuyển điểm trúng tuyển": "điểm trúng tuyển",
    "điểm trúng tuyển trúng tuyển": "điểm trúng tuyển",
    "điểm trúng tuyển tuyển sinh": "điểm trúng tuyển",
    "điểm trúng tuyển nổi bật": "điểm nổi bật",
    "điểm trúng tuyển": "Điểm trúng tuyển",
    "hv": "học viện",
    "j": "gì",
    "ko|k|kh|khg|kg|hong|hok|khum": "không",
    "ktx": "ký túc xá",
    "kt": "Kế toán",
    "khmt": "Khoa học máy tính",
    "kv|khvuc|kvuc": "khu vực",
    "lm": "làm",
    "lpxt": "lệ phí xét tuyển",
    "m": "mình",
    "marketting|makettting": "marketing",
    "mk": "mình",
    "nnao": "như nào",
    "ntn": "như thế nào",
    "nv|nvong": "nguyện vọng",
    "oke|oki|okeee": "ok",
    "p": "phải",
    "pk": "phải không",
    "q1|q9": "quận",
    "r": "rồi",
    "sdt|sđt|dt|đt": "số điện thoại",
    "sv": "sinh viên",
    "t": "tôi",
    "thpt": "trung học phổ thông",
    "thptqg": "trung học phổ thông quốc gia",
    "tk": "tài khoản",
    "tmdt": "Thương mại điện tử",
    "tnthpt": "trắc nghiệm trung học phổ thông",
    "tp hcm": "thành phố hồ chí minh",
    "trg|tr|trườg": "trường",
    "ttnv": "thứ thự nguyện vọng",
    "tt": "thông tin",
    "uh|uk|um": "ừ",
    "vc": "việc",
    "vd": "ví dụ",
    "vs": "với",
    "vt": "Viễn thông",
    "xtkh": "xét tuyển kết hợp",
    "z|v": "vậy",
    "đc|dc": "được",
    "năm nay": "2025",
    "năm ngoái|năm trước": "2024",
    "A1|a1": "A01",
    "A0|a0": "A00",
    "D1|d1": "D01",
    "A01": "tổ hợp xét tuyển A01",
    "A00": "tổ hợp xét tuyển A00",
    "D01": "tổ hợp xét tuyển D01",
    "tổ hợp": "tổ hợp xét tuyển",
    "khối": "tổ hợp xét tuyển",
    "tổ hợp xét tuyển tổ hợp xét tuyển": "tổ hợp xét tuyển",
    "Tổ hợp xét tuyển": "tổ hợp xét tuyển",
}


def normalize_replace_abbreviation_text(text):
    # text = re.sub(
    #     r"[\.,\(\)]", " ", text
    # )  # thay thế các kí tự đặc biệt bằng khoảng trắng
    # text = re.sub("<.*?>", "", text).strip()
    # text = re.sub("(\s)+", r"\1", text)
    # chars = re.escape(string.punctuation)
    # text = re.sub(
    #     r"[" + chars + "]", " ", text
    # )  # thay thế các kí tự đặc biệt bằng khoảng trắng
    text = re.sub(r"\s+", " ", text)  # thay thế nhiều khoảng trắng bằng 1 khoảng trắng
    text = text.strip()  # xóa khoảng trắng ở đầu và cuối
    text = text.lower()  # chuyển về chữ thường
    """ 
    # "cntt" -> "công nghệ thông tin"
    text = re.sub(r'\bcntt\b', 'công nghệ thông tin', text)
    # "ntn" -> "như thế nào"
    text = re.sub(r'\bntn\b', 'như thế nào', text)
    # "ad, adm" -> "admin"
    text = re.sub(r'\b(ad|adm)\b', 'admin', text)
    text = re.sub(r'\b(gd dt|gddt)\b', 'giáo dục đào tạo', text) 
    # điểm chuẩn -> điểm trúng tuyển
    text = re.sub(r'\bđiểm chuẩn\b', 'điểm trúng tuyển', text)
    """

    for k, v in dict_replace.items():
        text = re.sub(r"\b" + "(" + k + ")" + r"\b", v, text)

    return text.lower()


from getpass import getpass
import os

# from langchain.llms import PromptLayerOpenAI

# PROMPTLAYER_API_KEY = getpass()
# os.environ["PROMPTLAYER_API_KEY"] = PROMPTLAYER_API_KEY

# load openai api key from file .env
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.prompts import ChatPromptTemplate

""" 
--
Nếu câu trả lời có link ảnh (type: image) hoặc link đường dẫn, hãy sử dụng dấu gạch chân (_) theo cú pháp [ảnh](url) để thể hiện link ảnh hoặc [đường dẫn](url) link đường dẫn.
--
 """

# Prompt
template = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của trường Đại học Ngoại thương (FTU).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản. 
Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu thông tin trong ngữ cảnh không liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc". Đừng cố bịa ra một câu trả lời và đừng cố khẳng định bất kì điều gì.

Question: {question}\n<context> {context} </context>\n---\n
    HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    
    Với mỗi câu trả lời có câu hỏi giống nhau, bạn cần trả lời giống nhau.
    ---
    Output:

"""

prompt = ChatPromptTemplate.from_template(template)

# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


embed_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "./DataFTU/DataNewLastest/VectorDB_FTU_27052024",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)
dict_vectorstore = {
    "ban_tuyen_sinh": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/ban_tuyen_sinh",
    "chi_tieu_tuyen_sinh": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/chi_tieu_tuyen_sinh",
    "co_hoi_viec_lam": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/co_hoi_viec_lam",
    "co_so_dao_tao": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/co_so_dao_tao",
    "de_an_2024": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/de_an_2024",
    "diem_trung_tuyen": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/diem_trung_tuyen",
    "doi_tuong_uu_tien_va_khu_vuc_uu_tien": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/doi_tuong_uu_tien_va_khu_vuc_uu_tien",
    "hoc_bong": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/hoc_bong",
    "hoc_phi": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/hoc_phi",
    "hop_tac_doi_ngoai": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/hop_tac_doi_ngoai",
    "ky_tuc_xa": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/ky_tuc_xa",
    "le_phi": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/le_phi",
    "nhap_hoc": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/nhap_hoc",
    "phuong_thuc_xet_tuyen": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/phuong_thuc_xet_tuyen",
    "thong_tin_truong": "./DataFTU/DataNewLastest/VectorDB/IntentOutline/thong_tin_truong",
}


for key, value in dict_vectorstore.items():
    vectorstore = FAISS.load_local(
        value,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    dict_vectorstore[key] = vectorstore.as_retriever(
        search_kwargs={"k": 3, "threshold": 0.5}
    )


retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "threshold": 0.5})
dict_vectorstore

if os.path.exists("data_137"):
    retriever = get_retriever()
    dict_vectorstore = get_intent_retriever()


template_vietnamese_fusion = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của trường Đại học Ngoại thương (FTU).
Bạn có thể tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất. \n
Tạo ra nhiều truy vấn tìm kiếm liên quan đến: {question} \n
Đầu ra (2 truy vấn):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template_vietnamese_fusion)
generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        # print(docs)
        for rank, doc in enumerate(docs):
            # print(rank, doc)
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k).
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# remove duplicate words


def remove_duplicate_words(text: str):
    text = text.split()
    text = " ".join(sorted(set(text), key=text.index))
    return text


def get_results_intent(question: str):
    response = get_intent(question)
    response["intent_ranking"]

    top_5_intent = response["intent_ranking"][:4]
    print(top_5_intent)
    list_docs = []
    for i in top_5_intent:
        if "nlu_fallback" in i["name"]:
            continue
        else:
            intent = i["name"]
            for key, value in dict_vectorstore.items():
                if intent == key:
                    print(intent, key)
                    vectorstore = dict_vectorstore[key]
                    docs1 = vectorstore.get_relevant_documents(question)
                    list_docs.append(docs1)
    return list_docs


def get_results(question: str):

    docs = retrieval_chain_rag_fusion.invoke({"question": question})

    docs1 = retriever.get_relevant_documents(question)
    docs.append(docs1)
    docs = reciprocal_rank_fusion(docs)

    docs_copy = docs.copy()
    docs_copy.sort(key=lambda x: len(x[0].page_content), reverse=True)
    combined_docs = []
    string_check = ""
    for doc in docs_copy:
        if doc[0].page_content not in string_check:
            string_check += doc[0].page_content
            combined_docs.append(doc)
    combined_docs

    return {"use_answer": False, "docs": combined_docs}


retrieval_chain_rag_fusion = generate_queries | retriever.map()


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def _format_chat_history(chat_history: List[Dict[str, str]]) -> List:
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": RunnableLambda(itemgetter("question")) | get_results,
    }
).with_types(input_type=ChatRequest)

final_rag_chain = _inputs | prompt | llm | StrOutputParser()
# remove duplicate words
# {
#         "question": itemgetter("question"),
#         "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#         "context": itemgetter("docs"),
#     }


def remove_duplicate_words(text: str):
    text = text.split()
    text = " ".join(sorted(set(text), key=text.index))
    return text


reliable_sources = """Website: https://www.ftu.edu.vn/
Fanpage Tuyển sinh Trường Đại học Ngoại thương: https://www.facebook.com/TuyensinhFTU
Fanpage Diễn đàn sinh viên trường Đại học Ngoại thương - FTU Forum: https://www.facebook.com/ForumFTU
Group tuyển sinh “K63 FTU (2006) - Light the Star!”: https://www.facebook.com/groups/k63ftulightthestar"""


def get_answer(question: str, chat_history: any):
    question = normalize_replace_abbreviation_text(question)
    fqa = search(question)
    # print(fqa)
    docs = get_results(question)
    if docs["use_answer"] == True:
        response = docs["docs"]
        return {"use_answer": True, "response": response}
    elif fqa[1] != "No" and fqa[1] != "":
        return {
            "use_answer": False,
            "response": [{"type": "text", "content": fqa[1]}],
        }
    else:
        docs = docs["docs"]
        response = final_rag_chain.invoke(
            {"question": question, "docs": docs, "chat_history": chat_history}
        )
        print(response)
        return {
            "use_answer": False,
            "response": [{"type": "text", "content": response}],
        }
