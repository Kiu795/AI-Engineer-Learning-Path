import agent_framework
import streamlit as st
from typing import List, Optional
import json
import httpx
from functools import wraps
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_opengauss import OpenGaussSettings
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader, ExcelLoader, PowerPointLoader
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeStore, KnowledgeDB
from mx_rag.reranker.service import TEIReranker
from mx_rag.retrievers import Retriever, FullTextRetriever
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam
from pymilvus import MilvusClient
from mx_rag.graphrag import GraphRAGPipeline
from mx_rag.llm import LLMParameterConfig, Text2TextLLM
from mx_rag.utils import Lang

CONFIG_CACHE = None

# 缓存配置文件避免频繁读取
CONFIG_FILE_PATH = "config/mindie_config.json"
def get_config():
    global CONFIG_CACHE
    if CONFIG_CACHE is None:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            CONFIG_CACHE = json.load(f)
    return CONFIG_CACHE
config = get_config()


# 定义异常处理装饰器
def catch_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"功能出错：{str(e)}")
            st.exception(e)  # 调试用，生产环境可注释

    return wrapper

#MindIE大模型客户端
class MindIE_LLM:
    def __init__(self, base_url, model_name, temperature=0.1, max_tokens=1024, top_p=None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
    def _build_payload(self, messages, stream:bool):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload
    
    def invoke(self,messages):
        payload = self._build_payload(messages, stream=False)
        resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    
    def stream(self,messages):
        payload = self._build_payload(messages, stream=True)
        with httpx.stream("POST", self.base_url, json=payload, timeout=None) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8') if isinstance(raw_line, (bytes, bytearray)) else raw_line
                if not line.startswith("data: "):
                    try:
                        obj = json.loads(line)
                        # 处理可能的 chunk（降级兼容）
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
                    continue
                content = line[len("data: "):].strip()
                if content == "[DONE]":
                    break
                try:
                    obj = json.loads(content)
                except Exception:
                    continue
                
                choices0 = obj.get("choices", [{}])[0]
                # 首先尝试 delta（流）
                delta = choices0.get("delta", {}).get("content", "")
                if delta:
                    yield delta
                    continue
                # 否则尝试完整 message.content
                message_content = choices0.get("message", {}).get("content", "")
                if message_content:
                    yield message_content
                    continue

# 打印聊天历史记录
def print_history_message():
    pass

# 捕获大模型生成的检索需求                
def deal_model_query(output:str) -> Optional[str]:
    pass

# 问题重写，用于生成更符合检索需求的查询
def query_rewrite(query, llm) -> str:
    pass

# Embedding服务对象获取
def get_embedding():
    return TEIEmbedding(url=config["embedding_url"], client_param=ClientParam(use_http=True))

# Reranker服务对象获取
def get_reranker():
    return TEIReranker(url=config["reranker_url"], client_param=ClientParam(use_http=True))

# 向量数据库对象获取
def get_vector_db(knowledge):
    pass

# 文档数据库对象获取
def get_doc_db(knowledge):
    pass

# 稠密查询
def dense_retrieval():
    pass

# 稀疏查询
def sparse_retrieval():
    pass

# 查询结果整理
def retrieval_result_process(docs: List[Document]) -> str:
    pass
    
# 获取提示词
def get_prompt():
    pass

# 获取用户查询
def deal_user_query():
    print_history_message()
    user_query = st.session_state["query"]
    
    # 配置大模型客户端对象


# 执行多轮循环查询
def transform():
    pass