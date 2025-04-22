import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

# import weaviate
from constants import WEAVIATE_DOCS_INDEX_NAME
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ingest import get_embeddings_model
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
    RunnablePassthrough,
    RunnableSequence,
    chain,
)

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


def load_dict_replace():
    with open("./dict_replace.json", "r", encoding="utf8") as json_file:
        dict_replace = json.load(json_file)
        print("load")
    return dict_replace


dict_replace = load_dict_replace()


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

    return text


# Define your desired data structure.
class Test(BaseModel):
    query: str = Field(description="Câu hỏi của người dùng")
    answer: str = Field(description="Câu trả lời của mô hình")
    question_1: str = Field(description="Câu hỏi thường gặp 1, được tạo từ văn bản")
    answer_1: str = Field(description="Câu trả lời của câu hỏi thường gặp 1")
    question_2: str = Field(description="Câu hỏi thường gặp 2, được tạo từ văn bản")
    answer_2: str = Field(description="Câu trả lời của câu hỏi thường gặp 2")
    # Set up a parser + inject instructions into the prompt template.


# And a query intented to prompt a language model to populate the data structure.
parser = JsonOutputParser(pydantic_object=Test)


RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 180 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""
RESPONSE_TEMPLATE = """\
Bạn là một lập trình viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Đại học Ngoọi Thương (FTU).

Tạo câu trả lời đầy đủ và đầy đủ thông tin từ 180 từ trở xuống cho \
câu hỏi đưa ra chỉ dựa trên kết quả tìm kiếm được cung cấp (Document và nội dung). Bạn phải \
chỉ sử dụng thông tin từ kết quả tìm kiếm được cung cấp. Sử dụng một cách khách quan và \
giọng điệu báo chí. Kết hợp các kết quả tìm kiếm lại với nhau thành một câu trả lời mạch lạc, thân thiện nhất với người dùng. Đừng \
lặp lại văn bản. Nếu như các kết quả khác nhau đề cập đến các thực thể khác nhau trong cùng một tên, hãy viết \
câu trả lời cho từng thực thể.

Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.

Mọi thứ giữa các khối `context` sau đây đều được lấy từ một kiến thức \
ngân hàng, không phải là một phần của cuộc trò chuyện với người dùng.

<context>
     {context}
<context/>

HÃY NHỚ: Nếu <context> chứa cả thông tin phía Bắc và phía Nam, bạn cần phải trích dẫn cả hai phần và ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
Nếu không có thông tin liên quan trong ngữ cảnh, bạn chỉ cần nói "Hmm, tôi \
không chắc chắn." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì. Bất cứ điều gì nằm giữa 'context' trước đó \
các Document và nội dung được lấy từ ngân hàng kiến thức, không phải là một phần của cuộc trò chuyện với \
người dùng.\
"""

COHERE_RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 180 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

COHERE_RESPONSE_TEMPLATE = """\
Bạn là một lập trình viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Đại học Ngoọi Thương (FTU).

Tạo câu trả lời đầy đủ và đầy đủ thông tin từ 180 từ trở xuống cho \
câu hỏi đưa ra chỉ dựa trên kết quả tìm kiếm được cung cấp (Document và nội dung). Bạn phải \
chỉ sử dụng thông tin từ kết quả tìm kiếm được cung cấp. Sử dụng một cách khách quan và \
giọng điệu báo chí. Kết hợp các kết quả tìm kiếm lại với nhau thành một câu trả lời mạch lạc, thân thiện nhất với người dùng. Đừng \
lặp lại văn bản. Nếu như các kết quả khác nhau đề cập đến các thực thể khác nhau trong cùng một tên, hãy viết \
câu trả lời cho từng thực thể.

Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc" Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.


HÃY NHỚ: Nếu <context> chứa cả thông tin phía Bắc và phía Nam, bạn cần phải trích dẫn cả hai phần và ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
Nếu không có thông tin liên quan trong ngữ cảnh, bạn chỉ cần nói "Hmm, tôi \
không chắc chắn." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì. Bất cứ điều gì nằm giữa 'context' trước đó \
các Document và nội dung được lấy từ ngân hàng kiến thức, không phải là một phần của cuộc trò chuyện với \
người dùng.\
"""
COHERE_RESPONSE_TEMPLATE = """Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi dựa trên văn bản và Tạo thêm 1 đến 2 Câu hỏi thường gặp nhất định (các câu hỏi thường gặp) từ văn bản. Tạo câu hỏi, câu trả lời và ngữ cảnh tương ứng.\
    Nếu không thể tạo câu hỏi từ văn bản, hãy nói "Hmm, tôi không chắc.". Đừng cố gắng tạo câu trả lời, câu hỏi liên quan từ văn bản không phù hợp hoặc không chính xác.\
        .\n{format_instructions}"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

REPHRASE_TEMPLATE = """\
Với cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu tiếp theo \
câu hỏi là một câu hỏi độc lập.

Lịch sử trò chuyện:
{chat_history}
Đầu vào tiếp theo: {question}
Câu hỏi độc lập:"""


# client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# WEAVIATE_URL = os.environ["WEAVIATE_URL"]
# WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


def get_retriever() -> BaseRetriever:
    # weaviate_client = weaviate.Client(
    #     url=WEAVIATE_URL,
    #     auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    # )
    # weaviate_client = Weaviate(
    #     client=weaviate_client,
    #     index_name=WEAVIATE_DOCS_INDEX_NAME,
    #     text_key="text",
    # embedding=get_embeddings_model(),
    #     by_text=False,
    #     attributes=["source", "title"],
    # )
    # return weaviate_client.as_retriever(search_kwargs=dict(k=6))
    embed_model = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "./DataFTU/DataNewLastest/VectorDB_FTU_27052024",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "threshold": 0.5})
    return retriever


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            # HumanMessagePromptTemplate.from_template("{question}"),
        ],
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    default_response_synthesizer = prompt | llm | parser

    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ],
    )

    cohere_prompt = cohere_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
            anthropic_claude_3_sonnet=default_response_synthesizer,
            fireworks_mixtral=default_response_synthesizer,
            google_gemini_pro=default_response_synthesizer,
            cohere_command=cohere_response_synthesizer,
        )
        | parser
    ).with_config(run_name="GenerateResponse")

    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


from getpass import getpass
import os

# load openai api key from file .env
from dotenv import load_dotenv

load_dotenv()

gpt_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
claude_3_sonnet = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=4096,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
)
fireworks_mixtral = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0,
    max_tokens=16384,
    fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", "not_provided"),
)
gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_tokens=16384,
    convert_system_message_to_human=True,
    google_api_key=os.environ.get("GOOGLE_API_KEY", "not_provided"),
)
cohere_command = ChatCohere(
    model="command",
    temperature=0,
    cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
)
llm = gpt_3_5.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="openai_gpt_3_5_turbo",
    anthropic_claude_3_sonnet=claude_3_sonnet,
    fireworks_mixtral=fireworks_mixtral,
    google_gemini_pro=gemini_pro,
    cohere_command=cohere_command,
).with_fallbacks(
    [gpt_3_5, claude_3_sonnet, fireworks_mixtral, gemini_pro, cohere_command]
)

retriever = get_retriever()
# answer_chain = create_chain(llm, retriever)


import os
from operator import itemgetter
from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

### Ingest code - you may need to run this the first time
# # Load
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# # Split
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = PineconeVectorStore.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
# )
# retriever = vectorstore.as_retriever()

# vectorstore = vectorstore
# retriever = vectorstore.as_retriever()

# Condense a chat history and follow-up question into a standalone question
# _template =
"""Đưa ra cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập, bằng ngôn ngữ gốc của nó.
Cuộc trò chuyện: {chat_history}
Câu hỏi tiếp theo: {question}
Câu hỏi độc lập:"""  # noqa: E501

# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt

# template =
"""Trả lời câu hỏi của người dùng dựa trên thông tin sau đây.:
<context>
{context}
</context>"""
"""ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

ANSWER_PROMPT = ANSWER_PROMPT.partial(
    format_instructions=parser.get_format_instructions()
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | parser,
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = (
    _inputs
    | ANSWER_PROMPT
    | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    | parser
)
 """

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)

RESPONSE_TEMPLATE = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Đại học Ngoọi Thương (FTU).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản.
Cùng với đó hãy Tạo thêm 1 đến 2 Câu hỏi thường gặp nhất định (các câu hỏi thường gặp) từ văn bản với độ dài.

Câu trả lời được trình bày theo cấu trúc Object JSON dựa theo mô tả bên dưới bao gồm 1 ví dụ mẫu và 1 Schema mô tả các key, value của Object JSON trả về.\n{format_instructions}\nQuestion: {question}\n<context> {context} </context>\n---
Bạn nên Định dạng lại câu trả lời cho dễ đọc hơn và có cấu trúc rõ ràng hơn. Tách đoạn trong câu trả lời nếu độ dài câu trả lời trong "answer" quá dài (lớn hơn 30 từ, sau các dấu ngắt câu), sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc hơn và dễ nhìn, rõ ràng hơn. 

---
Ví dụ tách đoạn: "answer": "Đại học Ngoọi Thương tuyển sinh các ngành như: Kinh tế, Tài chính - Ngân hàng, Quản trị kinh doanh, Kinh tế quốc tế, Kinh doanh quốc tế, Luật, Kế toán, Ngôn ngữ Anh, Ngôn ngữ Pháp, Ngôn ngữ Trung quốc, Ngôn ngữ Nhật, Quản trị khách sạn, Marketing, Kinh tế chính trị, Khoa học máy tính."

Sau khi tách đoạn sẽ là:
"answer": "Đại học Ngoọi Thương tuyển sinh các ngành như: 
1. Kinh tế
2. Tài chính - Ngân hàng
3. Quản trị kinh doanh
4. Kinh tế quốc tế
5. Kinh doanh quốc tế
6. Luật
7. Kế toán
8. Ngôn ngữ Anh
9. Ngôn ngữ Pháp
10. Ngôn ngữ Trung quốc
11. Ngôn ngữ Nhật
12. Quản trị khách sạn
13. Marketing
14. Kinh tế chính trị
15. Khoa học máy tính"
---
---
Ví dụ tách đoạn: "answer": "Điểm trúng tuyển Đại học Ngoại thương năm 2023 như sau:

1. Quản trị kinh doanh: mã tổ hợp A00: 27.00 điểm , mã tổ hợp A01: 26.50 điểm , mã tổ hợp D01: 26.50 điểm , mã tổ hợp D07: 26.50 điểm
2. Tài chính - Ngân hàng: mã tổ hợp A00: 27.50 điểm , mã tổ hợp A01: 27.00 điểm , mã tổ hợp D01: 27.00 điểm , mã tổ hợp D07: 27.00 điểm
3. Kế toán: mã tổ hợp A00: 27.45 điểm , mã tổ hợp A01: 26.95 điểm , mã tổ hợp D01: 26.95 điểm , mã tổ hợp D07: 26.95 điểm
4. Kinh doanh quốc tế: mã tổ hợp A00: 27.90 điểm , mã tổ hợp A01: 27.40 điểm , mã tổ hợp D01: 27.40 điểm , mã tổ hợp D06: 27.40 điểm , mã tổ hợp D07: 27.40 điểm
5. Marketing: mã tổ hợp A00: 27.80 điểm , mã tổ hợp A01: 27.30 điểm , mã tổ hợp D01: 27.70 điểm , mã tổ hợp D06: 27.30 điểm , mã tổ hợp D07: 27.30 điểm
6. Luật: mã tổ hợp A00: 26.90 điểm , mã tổ hợp A01: 26.40 điểm , mã tổ hợp D01: 26.40 điểm , mã tổ hợp D07: 26.40 điểm
7. Ngôn ngữ Anh: mã tổ hợp D01: 36.40 điểm
8. Ngôn ngữ Pháp: mã tổ hợp D03: 25.45 điểm , mã tổ hợp D01: 26.20 điểm
9. Ngôn ngữ Trung: mã tổ hợp D01: 28.50 điểm , mã tổ hợp D04: 27.75 điểm
10. Ngôn ngữ Nhật: mã tổ hợp D01: 26.80 điểm , mã tổ hợp D06: 26.05 điểm"

Sau khi tách đoạn sẽ là:
"answer": "Điểm trúng tuyển Đại học Ngoại thương năm 2023 như sau:

1. Quản trị kinh doanh: 
- Mã tổ hợp A00: 27.00 điểm 
- Mã tổ hợp A01: 26.50 điểm 
- Mã tổ hợp D01: 26.50 điểm 
- Mã tổ hợp D07: 26.50 điểm 
2. Tài chính - Ngân hàng: 
- Mã tổ hợp A00: 27.50 điểm 
- Mã tổ hợp A01: 27.00 điểm 
- Mã tổ hợp D01: 27.00 điểm 
- Mã tổ hợp D07: 27.00 điểm 
3. Kế toán: 
- Mã tổ hợp A00: 27.45 điểm 
- Mã tổ hợp A01: 26.95 điểm 
- Mã tổ hợp D01: 26.95 điểm 
- Mã tổ hợp D07: 26.95 điểm 
4. Kinh doanh quốc tế: 
- Mã tổ hợp A00: 27.90 điểm 
- Mã tổ hợp A01: 27.40 điểm 
- Mã tổ hợp D01: 27.40 điểm 
- Mã tổ hợp D06: 27.40 điểm 
- Mã tổ hợp D07: 27.40 điểm 
5. Marketing: 
- Mã tổ hợp A00: 27.80 điểm 
- Mã tổ hợp A01: 27.30 điểm 
- Mã tổ hợp D01: 27.70 điểm 
- Mã tổ hợp D06: 27.30 điểm 
- Mã tổ hợp D07: 27.30 điểm 
6. Luật: 
- Mã tổ hợp A00: 26.90 điểm 
- Mã tổ hợp A01: 26.40 điểm 
- Mã tổ hợp D01: 26.40 điểm 
- Mã tổ hợp D07: 26.40 điểm 
7. Ngôn ngữ Anh: 
- Mã tổ hợp D01: 36.40 điểm 
8. Ngôn ngữ Pháp: 
- Mã tổ hợp D03: 25.45 điểm 
- Mã tổ hợp D01: 26.20 điểm 
9. Ngôn ngữ Trung: 
- Mã tổ hợp D01: 28.50 điểm 
- Mã tổ hợp D04: 27.75 điểm 
10. Ngôn ngữ Nhật: 
- Mã tổ hợp D01: 26.80 điểm 
- Mã tổ hợp D06: 26.05 điểm
"
---
Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, tôi không chắc" và đặt "answer": "Hmm, tôi không chắc". Đừng cố bịa ra một câu trả lời không đúng. Đừng cố khẳng định bất kì điều gì.
---
HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến cả 3 cơ sở tại Hà Nội, tp Hồ Chí Minh và cơ sở Quảng Ninh, bạn cần phải trích dẫn thật chi tiết từng cơ sừ, ưu tiên trình bày thông tin theo thứ tự tại Hà Nội, tp Hồ Chí Minh và cơ sở Quảng Ninh.\
Nội dung trong "answer" phải trả lời bằng tiếng Việt, Định dạng lại câu trả lời cho dễ đọc hơn và có cấu trúc rõ ràng hơn. Nếu độ dài câu trả lời quá dài (lớn hơn 30 từ, sau các dấu ngắt câu), bạn cần phải tách thành các đoạn văn ngắn hơn để dễ đọc và dễ hiểu hơn.\
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
---
Output:"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        # HumanMessagePromptTemplate.from_template("{question}"),
    ],
)

prompt = prompt.partial(
    format_instructions=parser.get_format_instructions(),
)

import os
from operator import itemgetter
from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

### Ingest code - you may need to run this the first time
# # Load
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# # Split
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = PineconeVectorStore.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
# )
# retriever = vectorstore.as_retriever()

# vectorstore = vectorstore
# retriever = vectorstore.as_retriever()

# Condense a chat history and follow-up question into a standalone question
_template = """\
Với cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu tiếp theo \
câu hỏi là một câu hỏi độc lập. Nếu thông tin Lịch sử trò chuyện không liên quan đến Câu hỏi tiếp theo, hãy sử dụng câu hỏi tiếp theo làm Câu hỏi độc lập.

Lịch sử trò chuyện:
{chat_history}
Câu hỏi tiếp theo: {question}
Câu hỏi độc lập:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Đại học Ngoọi Thương (FTU).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản.
Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc" Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.

Question: {question}\n<context> {context} </context>\n---\n
    HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    ---
    Output:"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        # ("system", template),
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

from difflib import SequenceMatcher


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # docs.sort(key=lambda x: len(x.page_content), reverse=True)

    # for i in range(len(docs)):
    #     text = docs[i].page_content
    #     text = re.sub(r"\n", "|@#ript", text)
    #     text = re.sub(r"\s+", " ", text)
    #     text = re.sub(r"\|@#ript", "\n", text)
    #     text = re.sub(r" ,", "", text)

    #     docs[i].page_content = text

    # l = []
    # st = docs[0].page_content.strip()
    # for i in range(1, len(docs)):
    #     if docs[i].page_content.strip() not in st:
    #         l.append(docs[i])
    #         st += "\n"
    #         st += docs[i].page_content.strip()
    #         print(docs[i])
    # print(len(l))
    # doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # return document_separator.join(doc_strings)

    # remove one of the pair of similar documents > 90% similarity

    # for i in range(len(docs)-1):
    #     for j in range(i+1, len(docs)):
    #         print(i, j)
    #         print(SequenceMatcher(None, docs[i].page_content, docs[j].page_content).ratio())
    #         if SequenceMatcher(None, docs[i].page_content, docs[j].page_content).ratio() > 0.9:
    #             docs[j].page_content = ""
    # [print(i) for i in docs]
    # docs = [i for i in docs if i.page_content != ""]
    # print(len(docs))
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        # | parser,
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

# _inputs = RunnableParallel(
#     {
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#         "context": _search_query | retriever | _combine_documents,
#     }
# ).with_types(input_type=ChatHistory)

# chain = _inputs | ANSWER_PROMPT | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) | StrOutputParser()


def get_source(text: str):
    docs = retriever.get_relevant_documents(text)
    return docs


# FUSION
template_vietnamese_fusion = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).
Bạn có thể tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất. \n
Tạo ra nhiều truy vấn tìm kiếm liên quan đến: {question} \n
Đầu ra (3 truy vấn):"""

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
        for rank, doc in enumerate(docs):
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


# retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
retrieval_chain_rag_fusion = generate_queries | retriever.map()


def get_results(question: str):
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    docs1 = retriever.get_relevant_documents(question)
    docs.append(docs1)
    docs = reciprocal_rank_fusion(docs)
    # print(docs)
    docs_copy = docs.copy()
    docs_copy.sort(key=lambda x: len(x[0].page_content), reverse=True)
    combined_docs = []
    string_check = ""
    for doc in docs_copy:
        if doc[0].page_content not in string_check:
            string_check += doc[0].page_content
            combined_docs.append(doc)
    return combined_docs


_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": RunnableLambda(itemgetter("question")) | get_results,
    }
).with_types(input_type=ChatHistory)

chain_json = (
    _inputs
    | prompt
    | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)
