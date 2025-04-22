"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from parser import langchain_docs_extractor

from langchain_core.retrievers import BaseRetriever
import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ip_back_data = "https://ftu-qldl-be.ript.vn"
def get_embeddings_model() -> Embeddings:
    # return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
    return OpenAIEmbeddings()


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


import requests
import json
import re
import datetime
from dotenv import load_dotenv
import shutil

load_dotenv()


def remove_vietnamese_accent(s):
    s = s.lower()
    s = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
    s = re.sub(r"[èéẹẻẽêềếệểễ]", "e", s)
    s = re.sub(r"[ìíịỉĩ]", "i", s)
    s = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", s)
    s = re.sub(r"[ùúụủũưừứựửữ]", "u", s)
    s = re.sub(r"[ỳýỵỷỹ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    s = "_".join(s.split())
    return s


def get_retriever() -> BaseRetriever:
    vectorstore = FAISS.load_local(
        "./data_137/VectorDB_RAG",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "threshold": 0.5})
    return retriever


def get_intent_retriever():
    list_path = os.listdir("./data_137/VectorDB/IntentOutline")
    list_path = [i for i in list_path]
    list_path

    dict_vectorstore = {}
    for i in list_path:
        dict_vectorstore[i] = "./data_137/VectorDB/IntentOutline/" + i
    print(dict_vectorstore)
    for key, value in dict_vectorstore.items():
        vectorstore = FAISS.load_local(
            value,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        dict_vectorstore[key] = vectorstore.as_retriever(
            search_kwargs={"k": 3, "threshold": 0.5}
        )
    return dict_vectorstore


def re_ingest(jwt: str):
    list_intent_response = requests.get(
        f"{ip_back_data}/api/topics?populate=*&pagination[pageSize]=100",
        headers={"Authorization": f"Bearer {jwt}"},
    ).json()
    data_res = [
        {
            "folder_name": i["attributes"]["name"],
            "folder_path": remove_vietnamese_accent(i["attributes"]["name"]),
            "intents": [
                {
                    "intent_name": j["attributes"]["name"],
                    "intent_path": remove_vietnamese_accent(j["attributes"]["name"]),
                    "file": j["attributes"]["file"],
                }
                for j in i["attributes"]["intents"]["data"]
            ],
        }
        for i in list_intent_response["data"]
    ]
    data_res

    if not os.path.exists("data_137"):
        os.makedirs("data_137")

    for i in data_res:
        if not os.path.exists(f'data_137/{i["folder_path"]}'):
            os.makedirs(f'data_137/{i["folder_path"]}')
        for j in i["intents"]:
            with open(f'data_137/{i["folder_path"]}/{j["intent_path"]}.txt', "w") as f:
                f.write(j["file"])

    # remove dir and file not in data_res # os.rmdir(f'data_137/{i}') # shutil.rmtree(f'data_137/{i}/{j}')
    for i in os.listdir("data_137"):
        if i not in [j["folder_path"] for j in data_res]:
            if not i.startswith("VectorDB"):
                print(i)
                shutil.rmtree(f"data_137/{i}", ignore_errors=True)
        else:
            for j in os.listdir(f"data_137/{i}"):
                path_check = j.split(".")[0]
                if path_check not in [
                    k["intent_path"]
                    for k in [l["intents"] for l in data_res if l["folder_path"] == i][
                        0
                    ]
                ]:
                    os.remove(f"data_137/{i}/{j}")
                    print(f"data_137/{i}/{j}")

    # remove vector db not in data_res os.rmdir(f'data_137/VectorDB/IntentOutline/{i}')
    for i in os.listdir("data_137/VectorDB/IntentOutline"):
        if i not in [j["folder_path"] for j in data_res]:
            shutil.rmtree(f"data_137/VectorDB/IntentOutline/{i}", ignore_errors=True)

    # Khai bao bien
    vector_db_path = "./data_137/VectorDB/IntentOutline/"
    # Khai bao loader de quet toan bo thu muc dataa
    # loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)

    # Embeding
    embed_model = OpenAIEmbeddings()
    # db = FAISS.from_documents(chunks, embed_model, distance_strategy = DistanceStrategy.COSINE)

    for i in data_res:
        path_load = i["intents"]
        path_save = i["folder_path"]
        path_save = remove_vietnamese_accent(path_save)
        if len(path_load) > 0:
            print(path_load[0]["intent_path"], path_save)
            loader = TextLoader(
                "./data_137/" + path_save + "/" + path_load[0]["intent_path"] + ".txt"
            )
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            chunks = text_splitter.split_documents(documents)
            # print(chunks)

        if len(path_load) > 1:
            for p in path_load[1:]:
                print(p["intent_path"], path_save)
                loader = TextLoader(
                    "./data_137/" + path_save + "/" + p["intent_path"] + ".txt"
                )
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=150
                )
                chunks_item = text_splitter.split_documents(documents)
                chunks = chunks + chunks_item

        db = FAISS.from_documents(
            chunks, embed_model, distance_strategy=DistanceStrategy.COSINE
        )
        db.save_local(vector_db_path + path_save)

    """ Db rag """
    pdf_data_path = "./data_137/" + data_res[0]["folder_path"]
    #  + datetime.datetime.now().strftime("%Y%m%d") + "/"
    vector_db_path = "./data_137/VectorDB_RAG"
    loader = DirectoryLoader(pdf_data_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=256)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embed_model = OpenAIEmbeddings()
    if chunks != []:
        db = FAISS.from_documents(
            chunks, embed_model, distance_strategy=DistanceStrategy.COSINE
        )
    db.save_local(vector_db_path)
    for i in data_res[1:]:
        pdf_data_path = "./data_137/" + i["folder_path"]
        loader = DirectoryLoader(pdf_data_path, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)

        if chunks != []:
            db.add_documents(chunks)
        db.save_local(vector_db_path)

    reload_server()
    return list_intent_response


def reload_server():
    import datetime

    with open("./test.py", "w") as f:
        f.write(f"print({datetime.datetime.now()})")


if __name__ == "__main__":
    ingest_docs()
