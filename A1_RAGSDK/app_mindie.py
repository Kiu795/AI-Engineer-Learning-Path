import base64
import io
import json
import os
import re
import shutil
import httpx

from datasets import tqdm
from pathlib import Path
import subprocess

import sys
import httpx
from PIL import Image
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import OpenAI
from paddle.base import libpaddle
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

import streamlit as st

user_id = "7d1d04c1-dd5f-43f8-bad5-99795f24bce6"
# 工作目录
WORKSPACE_DIR = "/home/HwHiAiUser/workspace"
# 配置文件路径
CONFIG_FILE_PATH = WORKSPACE_DIR + "/" + "config.json"

if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

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

img_to_text_prompt = '''给定一张包含表格或图表的图片，请提供结构化且详细的中文描述，描述需包含两个粒度级别:
  概要描述:
  - 概括图片的整体内容
  - 简要说明图片呈现的数据或信息(例如比对、趋势、分布等信息)
  - 说明表格或图表传达的主要主题或信息

  精细描述:
  - 描述图片中呈现的具体细节
  - 对于表格：列出列标题和行标题、单位，以及任何值得注意的值、模式或异常情况
  - 对于图表（例如，绘图、图形）：解释坐标轴、数据系列、图例以及任何显著趋势、异常值或数据点
  - 请注意图像中包含的任何标签、标题或注释
  - 突出具体例子或值得注意的细节

  请用清晰、条理分明、易于阅读的方式进行描述，不要使用markdown的格式输出。'''

default_text_infer_prompt = '''
你是一位问答助手，你的任务是根据提供的问题和文本、图片描述片段信息生成图文交错的回复。以下是指示如何生成回复内容：
1. **文本或图片描述片段引用选择**:
   - 从文本和图片描述片段中，找出与回答问题真正相关的内容。重点关注其重要性和直接相关性
   - 每张图片片段都是对图片的概要和精细描述.

2. **答案生成**:
   - 请使用 Markdown格式在回复中嵌入文本和图像，避免使用明显的标题或分隔符；确保回复自然流畅、连贯一致.
   - 在答案最后使用简洁明了的句子直接回答问题

3. **引用格式**:
   - 引用图片描述片段时，请使用 `![{图片描述总结}](图片索引)` 格式；引用第一张图片，请使用 `![{图片描述总结}](图片1)`；{图片描述总结} 应为对图片内容的简洁描述，最好用一句话概括
   - 请确保图片引用必须严格遵循 `![{图片描述总结}](图片索引)` 格式，不要简单地写成“参见图片1”、“图片1显示”、“[图片1]”或“图片1”
   - 每张图片或文字只允许引用一次

- 不要引用无关的片段
- 请用条理清晰、结构严谨的语言，对这个问题作详细解答
- 请确保您的答案逻辑清晰、内容翔实，并与引文提供的证据直接相关
- 如果引用包含文本和图像，则答案必须同时包含文本和图像回复
- 如果引用内容仅包含文本，则答案只能包含文本内容，不能包含图片
'''

default_prompt = """<指令>以下是提供的背景知识，请简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答。<指令>\n背景知识：{context}\n用户问题：{question}"""


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


@catch_errors
def refresh_chat():
    print_history_message()


@catch_errors
def query_knowledge(knowledge_name):
    return get_document(knowledge_name)[1]

#思考内容过滤
def filter_think_content(text: str, show_think: False, state: dict) -> str:

    # 如果用户选择显示思考内容，直接返回
    if show_think:
        return text

    result = ""

    i = 0
    while i < len(text):
        # 检测进入 <think>
        if text.startswith("<think>", i):
            state["in_think"] = True
            i += len("<think>")
            continue

        # 检测退出 </think>
        if text.startswith("</think>", i):
            state["in_think"] = False
            i += len("</think>")
            continue

        # 不在 think 中才允许输出
        if not state["in_think"]:
            result += text[i]

        i += 1

    return result


# 删除知识库中的文件
@catch_errors
def delete_document_in_knowledge(knowledge_name: str):
    file_names = st.session_state.file_to_delete
    _, _, knowledge_db = get_knowledge_db(knowledge_name)
    upload_file_dir, ocr_store_dir = get_knowledge_dir(knowledge_name)
    try:
        for file_name in file_names.split(","):
            # 删除web上传时存放的文件
            name, extension = os.path.splitext(file_name)

            # 删除 name对应的所有文件，如删除test.md 将会删除test.md test.pdf test.docx等
            for file in Path(upload_file_dir).rglob(f'{name}.*'):
                os.remove(file)

            # 删除ocr解析目录下的所有文件及子目录
            shutil.rmtree(os.path.join(ocr_store_dir, name))
            # 删除知识库，向量库中的数据
            knowledge_db.delete_file(file_name)

    except Exception as e:
        logger.error(f"delete file [{file_names}] failed: {e}")


# clear 按钮
@catch_errors
def on_btn_click():
    del st.session_state['messages']


knowledge_store = KnowledgeStore(db_path=F"{WORKSPACE_DIR}/knowledge_store_sql.db")

KnowledgeDB_Map = {}


@catch_errors
def get_knowledge_dir(knowledge_name):
    upload_file_dir = WORKSPACE_DIR + "/" + knowledge_name + "_data"
    ocr_store_dir = WORKSPACE_DIR + "/" + knowledge_name + "_ocr"

    if not os.path.exists(upload_file_dir):
        os.makedirs(upload_file_dir)

    if not os.path.exists(ocr_store_dir):
        os.makedirs(ocr_store_dir)

    return upload_file_dir, ocr_store_dir


@catch_errors
def get_knowledge_name(knowledge_name: str):
    if knowledge_name is None or len(knowledge_name) == 0 or ' ' in knowledge_name:
        return 'test_wyj'
    else:
        return knowledge_name


@catch_errors
def get_embedding():
    # 初始化embedding客户端对象
    return TEIEmbedding(url=st.session_state["embedding_url"], client_param=ClientParam(use_http=True))


# 获取向量数据库对象
@catch_errors
def get_vector_store(knowledge_name: str):
    # 初始化向量数据库
    knowledge_name = get_knowledge_name(knowledge_name)
    milvus_client = MilvusClient(st.session_state["milvus_url"])
    return MilvusDB.create(client=milvus_client, x_dim=int(st.session_state["embedding_dim"]),
                           collection_name=f"{knowledge_name}_vector")


# 获取文本数据库对象
@catch_errors
def get_chunk_store(knowledge_name: str):
    knowledge_name = get_knowledge_name(knowledge_name)
    milvus_client = MilvusClient(st.session_state["milvus_url"])
    return MilvusDocstore(milvus_client, collection_name=f"{knowledge_name}_chunk")


# 创建新的知识库
@catch_errors
def get_knowledge_db(knowledge_name: str):
    knowledge_name = get_knowledge_name(knowledge_name)
    if knowledge_name in KnowledgeDB_Map.keys():
        return KnowledgeDB_Map["knowledge_name"][2]
    logger.info(f"get knowledge_name:{knowledge_name}")

    # 初始化向量数据库
    vector_store = get_vector_store(knowledge_name)
    # 初始化文档chunk关系数据库
    chunk_store = get_chunk_store(knowledge_name)

    knowledge_store.add_knowledge(knowledge_name, user_id=user_id)
    # 初始化知识库管理
    knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                               chunk_store=chunk_store,
                               vector_store=vector_store,
                               knowledge_name=knowledge_name,
                               white_paths=["/tmp"],
                               user_id=user_id)
    KnowledgeDB_Map["knowledge_name"] = (vector_store, chunk_store, knowledge_db)

    return vector_store, chunk_store, knowledge_db


# 获取知识库中文档列表
@catch_errors
def get_document(knowledge_name: str):
    knowledge_db = get_knowledge_db(knowledge_name)[2]
    doc_names = [doc_model.document_name for doc_model in knowledge_db.get_all_documents()]
    return knowledge_name, doc_names, len(doc_names)


# 清空知识库中文档列表
@catch_errors
def clear_knowledge(knowledge_name: str):
    logger.info(f"start to delete all files")
    knowledge_name = get_knowledge_name(knowledge_name)
    vector_store, chunk_store, knowledge_db = get_knowledge_db(knowledge_name)
    knowledge_db.delete_all()

    upload_file_dir, ocr_store_dir = get_knowledge_dir(knowledge_name)
    # 删除从文件解析出来的图片
    try:
        shutil.rmtree(upload_file_dir)
        if st.session_state.parse_image == 'True':
            shutil.rmtree(ocr_store_dir)
    except Exception as e:
        logger.info(f"-------- delete {upload_file_dir} failed: {e}")


# 将DOCX或PPTX文件转换为PDF
@catch_errors
def convert_to_pdf(input_file, output_dir=None):
    """
    将DOCX或PPTX文件转换为PDF
    :param input_file: 输入文件的完整路径
    :param output_dir: 输出目录，默认为输入文件所在目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    # 构建转换命令
    # --headless: 无界面模式
    # --convert-to pdf: 指定转换为PDF格式
    # --outdir: 指定输出目录
    command = [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        input_file
    ]

    try:
        # 执行命令
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"convert to pdf success, output dir：{output_dir}")
        return input_file.with_suffix(".pdf")
    except subprocess.CalledProcessError as e:
        logger.error(f"convert to pdf failed：{e}")


@catch_errors
def upload_file(knowledge_name: str, file):
    logger.info(f"start to upload file: {file.name}")

    upload_file_dir, ocr_store_dir = get_knowledge_dir(knowledge_name)

    if not os.path.exists(upload_file_dir):
        os.makedirs(upload_file_dir)

    # Construct the full path for the saved file
    file_path = os.path.join(upload_file_dir, file.name.split("/")[-1])

    # Write the file content to disk in binary write mode
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    file_obj = Path(file_path)

    if st.session_state["parse_image"] == 'True':
        if file_obj.suffix in [".docx", ".pptx"]:
            file_obj = convert_to_pdf(file_obj)
        if file_obj.suffix == ".pdf":
            parse_pdf_file(file_obj, ocr_store_dir, server_url=st.session_state["ocr_url"])
            # ocr 处理后，将原始的pdf格式转换为md文件
            file_obj = Path(os.path.join(ocr_store_dir, file_obj.stem, "vlm", f"{file_obj.stem}.md"))

            # 第二步 对image_out_dir下的所有图片进行vlm 多粒度理解，结果存放在image_out_dir/image_info.json中
            cur_file_image_path = os.path.join(ocr_store_dir, file_obj.stem, "vlm/images")
            # vlm描述图片
            extract_images_info_by_vlm(cur_file_image_path, file_obj)

    # 根据文类型，获取loader类和splitter类信息
    loader_info, splitter_info = get_document_loader_splitter(file_obj.suffix)

    # 获取embedding对象
    emb = get_embedding()
    # 获取知识库管理对象
    knowledge_db = get_knowledge_db(knowledge_name)[2]

    # 检查当前文件是否已经入过库
    if knowledge_db.check_document_exist(file_obj.name):
        logger.warning(f"file {file_obj.name} exists in knowledge db")
        return

    # 创建文件解析器和切分器
    loader = loader_info.loader_class(file_path=file_obj.as_posix(), **loader_info.loader_params)
    splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
    # 解析文件并切分
    docs = loader.load_and_split(splitter)
    # 获取文档片段chunk内容和元数据信息
    texts = [doc.page_content for doc in docs if doc.page_content]
    meta_data = [{**doc.metadata, "type": "text"} for doc in docs if doc.page_content]

    if st.session_state.parse_image == 'True':
        # 解析的图片存放目录
        cur_file_image_path = os.path.join(ocr_store_dir, file_obj.stem, "vlm/images")

        # 读取图片多粒度解析信息
        try:
            with open(os.path.join(cur_file_image_path, "image_info.json"), "r", encoding='utf-8') as f:
                images_description = json.load(f)
        except Exception as e:
            logger.warning(f"read image info failed: {e}")
            images_description = {}

        for description in images_description:
            texts.append(description["image_description"])
            meta_data.append({"type": "image",
                              "source": file_path,
                              "image_path": description.get("image_path")
                              })
    # 存储到文本、向量数据库中
    knowledge_db.add_file(file_obj, texts, {"dense": emb.embed_documents}, meta_data)

    logger.info(f"upload file {file.name} to knowledge successfully")


@catch_errors
def file_upload(knowledge_name: str):
    if st.session_state.uploaded_files is None:
        return

    for uploaded_file in st.session_state.uploaded_files:
        upload_file(knowledge_name, uploaded_file)

    print_history_message()


@catch_errors
def new_kgfile_upload():
    if st.session_state.uploaded_files is None:
        return

    for uploaded_file in st.session_state.uploaded_files:
        graphrag_build(uploaded_file)
    print_history_message()


@catch_errors
def graphrag_build(file):
    pipeline, _, _ = get_pipeline()

    logger.info(f"start to upload file: {file}")
    _, upload_kgfile_dir = get_graph_dir()
    # Construct the full path for the saved file
    file_path = os.path.join(upload_kgfile_dir, file.name)
    # Write the file content to disk in binary write mode
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # 获取文档处理管理器
    loader_mng = get_loader_mng()
    pipeline.upload_files([file_path], loader_mng)
    pipeline.build_graph(lang=Lang.CH)


@catch_errors
def get_graph_dir():
    graph_name = st.session_state.graph_name
    graph_type = st.session_state.graph_type
    work_dir = f"{WORKSPACE_DIR}/knowledge_graph/{graph_type}/{graph_name}"
    upload_kgfile_dir = f"{work_dir}_data"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(upload_kgfile_dir):
        os.makedirs(upload_kgfile_dir)
    return work_dir, upload_kgfile_dir


@catch_errors
def create_new_db():
    get_knowledge_db(st.session_state.knowledge_name)

    print_history_message()


# 创建llm_chain 客户端
@catch_errors
def create_llm_chain(base_url, model_name):
    temperature = float(st.session_state.get("temperature", 0.3))
    max_tokens = int(st.session_state.get("max_length", 1024))
    
    top_p = st.session_state.get("top_p", None)

    # 返回我们的原生 HTTP 客户端实例
    return MindIE_LLM(base_url=base_url, model_name=model_name,
                      temperature=temperature, max_tokens=max_tokens, top_p=top_p)


@catch_errors
def get_kg_files() -> list[str]:
    _, folder_path = get_graph_dir()
    # 获取文件夹中所有条目
    entries = os.listdir(folder_path)
    # 筛选出文件（排除文件夹）
    file_names = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]

    return file_names


@catch_errors
def get_pipeline():
    work_dir, _ = get_graph_dir()
    llm = Text2TextLLM(
        base_url=st.session_state["llm_url"] + "/chat/completions",
        model_name=st.session_state["llm_name"],
        llm_config=LLMParameterConfig(max_tokens=st.session_state.max_length,
                                      temperature=st.session_state.temperature,
                                      top_p=st.session_state.top_p),
        client_param=ClientParam(timeout=180, use_http=True),
    )

    # 获取embedding对象
    embedding_model = get_embedding()

    graph_name = st.session_state.graph_name
    graph_type = st.session_state.graph_type
    if graph_type == "opengauss":
        graph_conf = OpenGaussSettings(user=st.session_state.oguser, password=st.session_state.ogpassword,
                                       host=st.session_state.oghost, port=st.session_state.ogport,
                                       database=st.session_state.ogdatabase)
    else:
        graph_conf = None
    pipeline = GraphRAGPipeline(work_dir, llm, embedding_model, st.session_state.embedding_dim,
                                graph_name=graph_name, graph_type=graph_type, graph_conf=graph_conf)
    return pipeline, graph_name, graph_type


@catch_errors
def compose_text_messages(question, text_docs, img_docs):
    # 1. Add text quotes
    user_message = ""
    for i, doc in enumerate(text_docs):
        if i == 0:
            user_message += "以下是文本片段信息:"

        user_message += f"\n文本片段[{i + 1}] {doc.page_content}"

    # 1. Add image quotes vlm-text or ocr-text
    for i, doc in enumerate(img_docs):
        if i == 0:
            user_message += "\n以下是图片描述片段信息:"

        user_message += f"\n图片[{i + 1}]描述信息内容: {doc.page_content}"
    user_message += "\n\n"

    # 3. add user question
    user_message += f"以下是用户的问题: {question}"

    return user_message


@catch_errors
def do_parse(
        output_dir,  # Output directory for storing parsing results
        pdf_file_names: list[str],  # List of PDF file names to be parsed
        pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
        backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
        server_url=None,  # Server URL for vlm-http-client backend
        f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    if backend.startswith("vlm-"):
        backend = backend[4:]

    parse_method = "vlm"
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
        middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend,
                                                    server_url=server_url)

        pdf_info = middle_json["pdf_info"]

        # 处理输出文件
        image_dir = str(os.path.basename(local_image_dir))

        md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

        logger.info(f"local output dir is {local_md_dir}")


@catch_errors
def parse_pdf_file(
        path: Path,
        output_dir,
        backend="vlm-http-client",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-vllm-engine: Faster(engine).
            vlm-http-client: Faster(client).
            without method specified, pipeline will be used by default.
        server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """

    target_file = os.path.join(output_dir, path.stem, "vlm", path.stem + ".md")

    if os.path.exists(target_file):
        logger.warning(f"{target_file} have been extracted, no need to ocr extraction")
        return

    try:
        file_name_list = []
        pdf_bytes_list = []

        file_name = str(Path(path).stem)
        pdf_bytes = read_fn(path)

        file_name_list.append(file_name)
        pdf_bytes_list.append(pdf_bytes)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            backend=backend,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )

    except Exception as e:
        logger.exception(e)


@catch_errors
def get_loader_mng():
    # 初始化文档加载切分管理器
    loader_mng = LoaderMng()
    # 注册文档加载器，可以使用mxrag提供的，也可以使用langchain提供的，同时也可实现langchain_community.document_loaders.base.BaseLoader
    # 接口类自定义实现文档解析功能
    loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt", ".md"])
    loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
    loader_mng.register_loader(loader_class=ExcelLoader, file_types=[".xlsx"])
    loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
    loader_mng.register_loader(loader_class=PowerPointLoader, file_types=[".pptx"])

    # 注册文档切分器，可自定义实现langchain_text_splitters.base.TextSplitter基类对文档进行切分
    loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                 file_types=[".docx", ".txt", ".md", ".pdf", ".xlsx", ".pptx"],
                                 splitter_params={"chunk_size": 750,
                                                  "chunk_overlap": 150,
                                                  "keep_separator": False
                                                  }
                                 )
    return loader_mng


# 获取文档加载器，和切分器
@catch_errors
def get_document_loader_splitter(file_suffix):
    loader_mng = get_loader_mng()

    # 根据文件后缀获取对应的文件解析器信息，包含解析类，及参数
    loader_info = loader_mng.get_loader(file_suffix)
    # 根据文件后缀获取对应的文件切分器信息，包含切分类，及参数
    splitter_info = loader_mng.get_splitter(file_suffix)

    return loader_info, splitter_info


# 根据问题从数据库中检索相似片段
@catch_errors
def retrieve_similarity_docs(knowledge_name: str, query, top_k, score_threshold):
    knowledge_name = get_knowledge_name(knowledge_name)
    # 获取embedding对象
    emb = get_embedding()
    # 获取文本和向量数据库对象
    chunk_store = get_chunk_store(knowledge_name)
    vector_store = get_vector_store(knowledge_name)

    # 配置向量检索器，
    dense_retriever = Retriever(vector_store=vector_store,
                                document_store=chunk_store,
                                embed_func=emb.embed_documents,
                                k=top_k,
                                score_threshold=score_threshold
                                )

    # 调用检索器从向量数据库中查找出和query最相近的tok个文档chunk
    dense_res = dense_retriever.invoke(query)

    # 配置全文检索器，其实现原理为BM25检索
    full_text_retriever = FullTextRetriever(document_store=chunk_store, k=top_k)

    full_text_res = full_text_retriever.invoke(query)

    # 合并检索结果
    docs = dense_res + full_text_res

    # 两路检索，可能检索到重复的片段，去重处理
    contents = set()
    new_docs = []
    for doc in docs:
        if doc.page_content not in contents:
            new_docs.append(doc)
            contents.add(doc.page_content)

    logger.info(f"retrieve similarity chunks from knowledge successfully")
    return new_docs


# 调用vlm对图片进行多粒度理解
@catch_errors
def extract_image_info_by_vlm(image_path):
    # 将图像转换为 base64 编码的字符串
    with Image.open(image_path) as img:
        width, height = img.size
        # 如果图片小于256*256，直接返回
        if width < 256 and height < 256:
            logger.warning(f"----------- image:{image_path} size: ({width},{height}) too little, will be discarded")
            return ""

        buffer = io.BytesIO()
        if Path(image_path).suffix == ".png":
            img = img.convert("RGB")

        if width > 1024 or height > 1024:
            img = img.resize(size=(width // 2, height // 2))

        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构造请求消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": img_to_text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{img_str}"}}
            ]
        }
    ]

    vlm = create_llm_chain(base_url=st.session_state["vlm_url"], model_name=st.session_state["vlm_name"])

    try:
        return vlm.invoke(messages).content
    except Exception as e:
        logger.error(f"call vlm invoke failed:{e}")
        return ""


# 获取目录下的所有图片文件路径
@catch_errors
def find_images_files(directory, recursive=False):
    base_path = Path(directory)
    exts = ('.jpg', '.jpeg', ".png")
    files = []

    for ext in exts:
        pattern = f'**/*{ext}' if recursive else f'*{ext}'
        file_list = list(base_path.glob(pattern))
        files.extend(str(p) for p in file_list)

    return files


# 将图片目录下的所有有图片调用vlm进行多粒度理解述并保存到json文件中
@catch_errors
def extract_images_info_by_vlm(image_dir, md_file_obj=None):
    # 避免同一个文件中的所有图片重复调用vlm 提取信息，如果image_info.json如果存在，表示已经提取，不用再次提取，降低vlm算力
    if os.path.exists(os.path.join(image_dir, "image_info.json")):
        logger.warning(f"all images in {image_dir} have been extracted, no need to repeat extraction")
        return

    logger.info(f"start to extract images info by vlm ...")

    image_files = find_images_files(image_dir)
    info = []
    for image_file in tqdm(image_files, desc='parse images by vlm'):
        logger.info(f"start to deal {[image_file]} by vlm")
        res = extract_image_info_by_vlm(image_file)
        if res:
            info.append({"image_path": image_file, "image_description": res})

    if len(info) > 0:
        with open(os.path.join(image_dir, "image_info.json"), "w", encoding='utf-8') as f:
            f.write(json.dumps(info, indent=4, ensure_ascii=False))

    logger.info(f"extract images info successfully")


# 检查会话状态中是否存在 messages 列表，如果不存在，则初始化为空列表
# 用于存储交流过程中的消息
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.title("Chat with Mind RAG SDK")  # 显示聊天界面的标题


@catch_errors
def print_history_message():
    st.title("Chat with Mind RAG SDK")  # 显示聊天界面的标题
    # 遍历保存在会话状态中的消息，并根据消息类型（人类或AI）分别显示
    for message in st.session_state["messages"]:
        if message["type"] == "human":  # 这里不应该用 system message
            with st.chat_message("user"):  # 不用 container; user
                st.markdown(message["content"])
        elif message["type"] == "ai":
            with st.chat_message("assistant"):  # 不用 container；assistant
                st.markdown(message["content"])
            with st.expander("背景知识"):
                for i, context in enumerate(message.get("contexts", [])):
                    st.markdown(f"## -----------------context {i + 1}----------------------")
                    st.markdown(context)
                    if context.metadata.get("image_path", "") != "":
                        st.image(context.metadata.get("image_path"))


@catch_errors
def generate_question(query, llm):
    prompt = """现在你有一个上下文依赖问题补全任务，任务要求:请根据对话历史和用户当前的问句，重写问句。\n
    历史问题依次是:\n
    {}\n
    注意如果当前问题不依赖历史问题直接返回none即可\n
    请根据上述对话历史重写用户当前的问句,仅输出重写后的问句,不需要附加任何分析。\n
    重写问句: \n
    """
    if len(st.session_state["messages"]) <= 2:
        return query

    history_n = st.session_state["history_n"]
    messages = st.session_state["messages"]
    history_list = [f"第{idx + 1}轮：{message['content']}"
                    for idx, message in enumerate(messages[len(messages) - history_n:]) if message['type'] == "human"]

    history_str = "\n\n".join(history_list)
    re_query = prompt.format(history_str)

    invoke_message = [
        {"role": "system", "content": re_query},
        {"role": "user", "content": query}
    ]

    logger.info(f"======================改写前问题：{query}")

    try:
        new_query = ""
        for chunk in llm.stream(invoke_message):
            new_query += chunk

        if new_query == "":
            query = new_query
        # else:
        #     pos = new_query.rfind("</think>")
        #     pos += len("</think>")
        #     query = new_query[pos:].strip()
    except Exception as e:
        logger.error(f"call llm invoke failed:{e}")

    logger.info(f"======================改写后问题：{query}")
    return query


@catch_errors
def generate_interleaved_answer(query, q_docs, llm_chain):
    # 拆分知识片段分为原始文本和图片多粒度信息文本
    text_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "text"]
    img_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "image"]

    text = compose_text_messages(query, text_docs, img_docs)

    # 构造请求消息
    messages = [
        {"role": "system", "content": st.session_state["interleaved_prompt"]},
        {"role": "user", "content": text}
    ]

    return llm_chain.stream(messages)


@catch_errors
def generate_text_answer(query, q_docs, llm_chain):
    text_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "text"]

    context = "\n".join([text.page_content for text in text_docs])
    llm_prompt = st.session_state["text_prompt"].format(context=context, question=query)
    # 构造请求消息
    messages = [
        {"role": "system", "content": "你是一个专业的知识问答助手，请直接回答用户问题,不要输出思考过程,不要使用 <think> 标签。"},
        {"role": "user", "content": llm_prompt}
    ]

    return llm_chain.stream(messages)


@catch_errors
def replace_image_paths(text, image_docs):
    """
    将response中的imageN替换为image_docs中对应的image_path
    """
    if text == "":
        return text

    # 创建imageN到image_path的映射
    image_mapping = {}
    for i, doc in enumerate(image_docs):
        image_key = f"image{i + 1}"
        image_path = doc.metadata.get("image_path", "")
        image_mapping[image_key] = image_path

    # 使用正则表达式找到所有的![...](imageN)模式并替换
    def replace_match(match):
        full_match = match.group(0)  # 完整匹配的字符串
        alt_text = match.group(1)  # 图片描述文本
        image_des = match.group(2)  # image或图片字符串
        image_id = match.group(3)  # image id
        # 如果找到对应的image_path，则替换
        if f"image{image_id}" in image_mapping:
            return f"![{alt_text}]({image_mapping[f'image{image_id}']})"
        else:
            # 如果没有找到对应的路径，保持原样
            return full_match

    # 匹配 ![...](imageN) 模式
    pattern = r'!\[([^\]]*)\]\((image(\d+))\)'
    updated_response = re.sub(pattern, replace_match, text)
    pattern = r'!\[([^\]]*)\]\((图片(\d+))\)'
    updated_response = re.sub(pattern, replace_match, updated_response)

    logger.info(f"Image paths replacement completed")
    return updated_response


@catch_errors
def render_markdown_with_images(markdown_text):
    # 匹配 Markdown 图片语法 ![alt text](image_url)
    pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    # 记录上一个位置
    last_pos = 0

    # 查找所有匹配项
    with st.chat_message("ai"):
        # pos = markdown_text.rfind("</think>")
        # pos += len("</think>")
        # # 显示思考部分
        # st.markdown(markdown_text[:pos])
        markdown_text = markdown_text[pos:]
        # 非思考部分图片内容进行替换显示
        for match in pattern.finditer(markdown_text):
            # 显示上一个位置到匹配位置之间的文本
            st.markdown(markdown_text[last_pos:match.start()], unsafe_allow_html=True)

            # 显示图片
            img_url = match.group(1)
            # 更新上一个位置
            last_pos = match.end()

            if os.path.exists(img_url):
                st.image(img_url)
            else:
                st.markdown(markdown_text[match.start():last_pos], unsafe_allow_html=True)

        # 显示剩余的文本
        st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)


@catch_errors
def answer_without_knowledge(llm_chain, query):
    # 构造请求消息
    messages = [
        {"role": "system", "content": "你是一个专业的知识问答助手。请直接回答用户问题,不要输出思考过程,不要使用 <think> 标签。"},
        {"role": "user", "content": query}
    ]

    with st.chat_message("user"):  # 不用 container; user
        st.markdown(query)

    placeholder = st.empty()
    full_answer = ""
    
    think_state = {"in_think": False}
    
    for chunk in llm_chain.stream(messages):
        content = chunk.strip()
        filtered = filter_think_content(content, False, think_state)
        full_answer += filtered
        display_text = filter_think_content(full_answer, False, think_state)
        placeholder.markdown(display_text)
        
    placeholder.empty() 
    with st.chat_message("ai"):  # 不用 container; user
        st.markdown(full_answer)

    st.session_state["messages"].append({'content': full_answer, 'type': "ai"})  # 保存ai msg


@catch_errors
def answer_with_knowledge(llm_chain, query):
    if st.session_state.graph_pipeline == "True":
        pipeline, graph_name, graph_type = get_pipeline()
        contexts = pipeline.retrieve_graph(graph_name, query, batch_size=st.session_state.batch_size,
                                           similarity_tail_threshold=st.session_state.similarity_tail_threshold,
                                           retrieval_top_k=st.session_state.retrieval_top_k,
                                           # reranker_top_k=st.session_state.reranker_top_k,
                                           subgraph_depth=st.session_state.subgraph_depth)
        q_docs = [Document(page_content=context, metadata={"source": "graph", "type": "text"}) for context in contexts]
        logger.debug(f"检索到的相关的文本： {q_docs}")
    else:
        q_docs = retrieve_similarity_docs(st.session_state.knowledge_name, query, st.session_state.top_k,
                                          st.session_state.similarity_threshold)

        text_reranker = TEIReranker(url=st.session_state["reranker_url"], k=st.session_state.top_k,
                                    client_param=ClientParam(use_http=True))

        if text_reranker is not None and len(q_docs) > 0:
            score = text_reranker.rerank(query, [doc.page_content for doc in q_docs])
            q_docs = text_reranker.rerank_top_k(q_docs, score)

    img_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "image"]
    # for doc in q_docs:
    #     logger.info(f"00000000000000000000 {doc}")
    full_answer = ""
    with st.chat_message("user"):  # 不用 container; user
        st.markdown(query)

    placeholder = st.empty()

    think_state = {"in_think": False}
    
    if st.session_state.graph_pipeline == "True":
        interleaved_answer = None
    else:
        interleaved_answer = st.session_state.interleaved_answer

    if interleaved_answer == "True":
        answer = generate_interleaved_answer(query, q_docs, llm_chain)
        # 流式显示
        for chunk in answer:
            content = chunk.strip()
            filtered = filter_think_content(content, False, think_state)
            full_answer += filtered
            display_text = filter_think_content(full_answer, False, think_state)
            placeholder.markdown(display_text)

        # 删除临时流式结果
        placeholder.empty()
        logger.info(f"------------------------{full_answer}")
        full_answer = replace_image_paths(full_answer, img_docs)
        render_markdown_with_images(full_answer)
    else:
        answer = generate_text_answer(query, q_docs, llm_chain)
        
        # 流式显示
        for chunk in answer:
            content = chunk.strip()
            full_answer += content
            display_text = filter_think_content(full_answer, False, think_state)
            placeholder.markdown(display_text)

        # 删除临时流式结果
        placeholder.empty()
        with st.chat_message("ai"):
            st.markdown(full_answer)

    # 存储到历史消息中
    contexts = q_docs
    st.session_state["messages"].append({'content': full_answer, 'type': "ai",
                                         "contexts": contexts})  # 保存ai msg
    with st.expander("背景知识"):
        for i, context in enumerate(contexts):
            st.markdown(f"## -----------------context {i + 1}----------------------")
            st.markdown(context)
            if context.metadata.get("image_path", "") != "":
                st.image(context.metadata.get("image_path"))


@catch_errors
def deal_user_query():
    print_history_message()
    user_query = st.session_state["query"]

    # 配置大模型客户端对象
    llm_chain = create_llm_chain(base_url=st.session_state["llm_url"], model_name=st.session_state["llm_name"])

    if st.session_state.modify_query == "True":
        user_query = generate_question(user_query, llm_chain)

    st.session_state["messages"].append({'content': user_query, 'type': "human"})  # 保存human

    if st.session_state.use_knowledge == "False":
        answer_without_knowledge(llm_chain, user_query)
    else:
        answer_with_knowledge(llm_chain, user_query)


@catch_errors
def init_config():
    # """初始化配置：新增回复默认值项，首次运行生成默认配置"""

    default_config = {
        # 服务参数
        "llm_url": "http://127.0.0.1:1025",
        "llm_name": "Qwen2.5-0.6B-Instruct",
        "ocr_url": "http://127.0.0.1:30003",
        "ocr_name": "Qwen3-32B",
        "vlm_url": "http://127.0.0.1:9097/v1",
        "vlm_name": "Qwen2.5-VL-7B-Instruct",
        "embedding_url": "http://127.0.0.1:7999/embed",
        "embedding_dim": 1024,
        "reranker_url": "http://127.0.0.1:7998/rerank",
        "rerank_top_k": 3,
        "milvus_url": "http://127.0.0.1:19530",
        "use_knowledge": "True",
        "graph_pipeline": "False",
        "graph_name": "graph",
        "graph_type": "networkx",
        "oghost": "127.0.0.1",
        "ogport": "xxxx",
        "ogdatabase": "xxx",
        "oguser": "xxx",
        "ogpassword": 'xxx',
        "retrieval_top_k": 3,
        "similarity_tail_threshold": 0.5,
        "subgraph_depth": 3,
        "batch_size": 4,
        "knowledge_name": "test_1",
        "parse_image": 'False',
        "interleaved_answer": 'False',
        "interleaved_prompt": default_text_infer_prompt,
        "text_prompt": default_prompt,
        "temperature": 0.95,
        "top_p": 0.95,
        "max_length": 1024,
        "top_k": 3,
        "similarity_threshold": 0.5,
        "modify_query": "False",
        "history_n": 3,
    }

    if not os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        return default_config
    else:
        # 读取现有配置，若缺少新增项则补充默认值（兼容旧配置文件）
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
        for key, value in default_config.items():
            if key not in existing_config:
                existing_config[key] = value
        # 保存补充后的配置
        save_config(existing_config)
        return existing_config


@catch_errors
def save_config(config_data):
    """通用保存配置函数：被自动保存事件调用"""
    with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


@catch_errors
def auto_save_config():
    """自动保存逻辑：收集所有参数，覆盖配置文件"""
    # 定义所有可能用到的参数键（作为“白名单”，避免保存无关键）
    possible_keys = {
        "llm_url", "llm_name", "ocr_url",
        "ocr_name", "vlm_url", "vlm_name", "embedding_url", "embedding_dim", "reranker_url",
        "rerank_top_k", "milvus_url", "use_knowledge", "graph_pipeline", "graph_name", "graph_type",
        "oghost", "ogport", "ogdatabase", "oguser", "ogpassword", "retrieval_top_k", #"reranker_top_k",
        "similarity_tail_threshold", "subgraph_depth", "batch_size", "knowledge_name", "parse_image",
        "interleaved_answer", "interleaved_prompt", "text_prompt", "temperature", "top_p", "max_length",
        "top_k", "similarity_threshold", "modify_query", "history_n"
    }

    # 动态收集：只保留 session_state 中已存在的键
    current_config = {}
    for key in possible_keys:
        if key in st.session_state:  # 仅保存已初始化的键
            current_config[key] = st.session_state[key]

    save_config(current_config)
    # 可选：显示短暂提示（避免频繁弹窗干扰）
    st.toast("配置已自动保存", icon="✅")


@catch_errors
def set_service_para():
    with st.expander("服务参数配置"):
        llm_columns = st.columns([3, 2])
        vlm_columns = st.columns([3, 2])
        ocr_columns = st.columns([3, 2])
        emb_columns = st.columns([3, 2])
        reranker_columns = st.columns([3, 2])
        with llm_columns[0]:
            st.text_input("llm_base_url", value=st.session_state["llm_url"], on_change=auto_save_config,
                          key="llm_url", help="llm服务基地址, 格式为http://ip:port/openai/v1")
        with llm_columns[1]:
            st.text_input("llm模型名", value=st.session_state["llm_name"], on_change=auto_save_config,
                          key="llm_name", help="llm模型名")

        with ocr_columns[0]:
            st.text_input("ocr_base_url", value=st.session_state["ocr_url"], on_change=auto_save_config,
                          key="ocr_url", help="ocr服务基地址,格式为 http://ip:port")
        with ocr_columns[1]:
            st.text_input("ocr模型名", value=st.session_state["ocr_name"], on_change=auto_save_config,
                          key="ocr_name", help="ocr模型名")

        with vlm_columns[0]:
            st.text_input("vlm_base_url", value=st.session_state["vlm_url"], on_change=auto_save_config,
                          key="vlm_url", help="vlm服务基地址, 格式为http://ip:port/v1")
        with vlm_columns[1]:
            st.text_input("vlm模型名", value=st.session_state["vlm_name"], on_change=auto_save_config,
                          key="vlm_name", help="vlm模型名")

        with emb_columns[0]:
            st.text_input("embedding url", value=st.session_state["embedding_url"], on_change=auto_save_config,
                          key="embedding_url", help="emb 服务地址,格式为http://ip:port/embed")
        with emb_columns[1]:
            st.number_input("embedding dim", value=st.session_state["embedding_dim"], on_change=auto_save_config,
                            key="embedding_dim", help="emb 向量维度")

        with reranker_columns[0]:
            st.text_input("reranker url", value=st.session_state["reranker_url"], on_change=auto_save_config,
                          key="reranker_url",
                          help="reranker服务地址,格式为http://ip:port/rerank")
        with reranker_columns[1]:
            st.number_input("rerank_top_k", value=st.session_state["rerank_top_k"], on_change=auto_save_config,
                            key="rerank_top_k", help="rerank_top_k值")

        st.text_input("milvus_url", value=st.session_state["milvus_url"], on_change=auto_save_config,
                      key="milvus_url", help="milvus服务基地址,格式为http://ip:port")


@catch_errors
def set_web():
    with st.sidebar:
        # 1. 初始化配置并同步到session_state
        config = init_config()
        for key, value in config.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
        set_service_para()
        st.session_state.setdefault("show_think", False)

        # st.radio(
        #     "是否显示思考内容（<think> 部分）：",
        #     ["True", "False"],
        #     index=0 if st.session_state.get("show_think", "False") == "True" else 1,
        #     help="关闭后，模型输出中的 <think> 思考内容将被隐藏，仅显示最终回答",
        #     on_change=lambda: [refresh_chat(), auto_save_config()],
        #     key="show_think"
        # )
        # st.session_state.show_think = st.checkbox(
        #     "显示思考内容（<think> 部分）",
        #     value=st.session_state.show_think
        # )
        st.radio("是否使用外部知识库问答：", ["True", "False"],
                 index=0 if st.session_state["use_knowledge"] == "True" else 1,
                 help="启用后，系统会结合外部知识库内容进行问答",
                 on_change=lambda: [refresh_chat(), auto_save_config()], key="use_knowledge")
        if st.session_state.use_knowledge == "True":
            st.radio("是否启用知识图谱：", ["True", "False"],
                     index=0 if st.session_state["graph_pipeline"] == "True" else 1,
                     help="启用后，知识库内容将以图谱形式存储、检索",
                     on_change=lambda: [refresh_chat(), auto_save_config()], key="graph_pipeline")
        if st.session_state.use_knowledge == "True":
            if st.session_state.get("graph_pipeline", "False") == "True":
                st.text_input("设置知识图谱名称", value=st.session_state["graph_name"], key="graph_name",
                              on_change=lambda: [get_pipeline(), auto_save_config()])
                st.selectbox("选择知识图谱类型", ["networkx", "opengauss"],
                             index=0 if st.session_state["graph_type"] == "networkx" else 1,
                             on_change=lambda: [get_pipeline(), auto_save_config()], key="graph_type",
                             help="networkx则使用mindfaiss作为向量数据库；opengauss则使用opengauss作为向量数据库")
                if st.session_state.graph_type == "opengauss":
                    with st.expander("opengauss服务参数配置"):
                        opengauss_columns = st.columns([1.2, 0.8, 1, 1, 1])
                        with opengauss_columns[0]:
                            st.text_input("host", value=st.session_state["oghost"], key="oghost", on_change=auto_save_config,
                                          help="opengauss服务ip")
                        with opengauss_columns[1]:
                            st.text_input("port", value=st.session_state["ogport"], key="ogport", on_change=auto_save_config,
                                          help="opengauss服务端口")
                        with opengauss_columns[2]:
                            st.text_input("数据库名", value=st.session_state["ogdatabase"], key="ogdatabase", on_change=auto_save_config,
                                          help="opengauss数据库名")
                        with opengauss_columns[3]:
                            st.text_input("用户名", value=st.session_state["oguser"], key="oguser", on_change=auto_save_config,
                                          help="opengauss数据库用户名")
                        with opengauss_columns[4]:
                            st.text_input("密码", value=st.session_state["ogpassword"], key="ogpassword", on_change=auto_save_config,
                                          help="opengauss数据库密码")

                st.file_uploader("上传知识文档", key="uploaded_files", accept_multiple_files=True,
                                 type=["docx", "txt", "md", "pdf", "xlsx", "pptx"],
                                 on_change=new_kgfile_upload)
                st.text_area("知识图谱文件详情", value=get_kg_files())

                with st.expander("设置知识图谱检索参数"):

                    st.slider('retrieval_top_k', 1, 1000, st.session_state["retrieval_top_k"], step=1,
                              key="retrieval_top_k",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="最相似的k个知识片段")
                    st.slider('similarity_threshold', 0.1, 1.0, st.session_state["similarity_tail_threshold"],
                              step=0.01,
                              key="similarity_tail_threshold",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="值越大，越相似")
                    st.slider('subgraph_depth', 1, 5, st.session_state["subgraph_depth"], step=1, key="subgraph_depth",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="图检索最大探索的深度")
                    st.slider('batch_size', 1, 1024, st.session_state["batch_size"], step=2, key="batch_size",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="对节点向量化时的批次大小")

                st.text_area("设置提示词", st.session_state["text_prompt"],
                             help="设置的提示词需包含{context}和{question}",
                             on_change=auto_save_config, key="text_prompt")

        if st.session_state.use_knowledge == "True":
            if st.session_state.get("graph_pipeline", "False") == "False":
                st.text_input("设置知识库名", st.session_state["knowledge_name"], key="knowledge_name",
                              on_change=lambda: [create_new_db(), auto_save_config()])

                cur_knowledge_name = st.session_state.get("knowledge_name", "test_1")
                st.file_uploader("上传知识文档", key="uploaded_files", accept_multiple_files=True,
                                 type=["docx", "txt", "md", "pdf", "xlsx", "pptx"],
                                 on_change=lambda: [file_upload(cur_knowledge_name), auto_save_config()])
                st.radio("文档入库时是否提取多模态信息", ['True', 'False'],
                         index=0 if st.session_state["parse_image"] == "True" else 1,
                         help="开启后，调用ocr模型提取文档中的图片、表格信息，当前支持docx、pptx和pdf格式，会自动转成md文档入库",
                         on_change=lambda: [refresh_chat(), auto_save_config()], key="parse_image")
                st.text_input("待删除知识文档名", key="file_to_delete", help="如果一次需要删除多个文件，使用逗号分隔")
                st.button("删除知识库中指定的文档", key="delete_document", help="删除知识库中指定的文档",
                          type="primary",
                          on_click=delete_document_in_knowledge, args=(cur_knowledge_name,))

                st.button("清空知识库", key="clear_knowledge", help="删除知识库中的所有文档", type="primary",
                          on_click=clear_knowledge, args=(cur_knowledge_name,))
                st.text_area("知识库文件详情", value=query_knowledge(cur_knowledge_name))
                interleaved_answer = st.radio("是否图文嵌入答复", ['True', 'False'],
                                              index=0 if st.session_state["interleaved_answer"] == "True" else 1,
                                              help="根据检索到的文本片段和图片描述片段，生成图文嵌入内容",
                                              on_change=lambda: [refresh_chat(), auto_save_config()],
                                              key="interleaved_answer")

                if interleaved_answer == "True":
                    st.text_area("设置提示词", st.session_state["interleaved_prompt"],
                                 on_change=auto_save_config, key="interleaved_prompt")
                else:
                    st.text_area("设置提示词", st.session_state["text_prompt"], help="设置的提示词需包含{context}和{question}",
                                 on_change=auto_save_config, key="text_prompt")

                with st.expander("设置检索参数"):
                    st.slider('top_k', 1, 100, st.session_state["top_k"], step=1, key="top_k",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="最相似的k个知识片段")
                    st.slider('similarity_threshold', 0.1, 1.0, st.session_state["similarity_threshold"], step=0.1,
                              key="similarity_threshold",
                              on_change=lambda: [refresh_chat(), auto_save_config()],
                              help="值越大，越相似")

        with st.expander("设置大模型对话参数"):
            st.slider("temperature", 0.1, 1.0, st.session_state["temperature"], step=0.1,
                      on_change=lambda: [refresh_chat(), auto_save_config()],
                      key="temperature")
            st.slider("top_p", 0.1, 1.0, st.session_state["top_p"], step=0.1,
                      on_change=lambda: [refresh_chat(), auto_save_config()],
                      key="top_p")
            st.slider("max_length", min_value=64, max_value=2048, step=128, value=st.session_state["max_length"],
                      key="max_length", on_change=lambda: [refresh_chat(), auto_save_config()],
                      help="大模型输出的最大token数")

        st.radio("是否开启问题改写：", ["True", "False"], index=0 if st.session_state["modify_query"] == "True" else 1,
                 help="开启问题改写，会根据历史问题进行改写当前问题，更准确理解当前问题语义",
                 on_change=lambda: [refresh_chat(), auto_save_config()],
                 key="modify_query")
        st.slider('历史对话轮数', 1, 20, st.session_state["history_n"], step=1, key="history_n",
                  on_change=auto_save_config,
                  help="改写问题时采纳的历史对话轮数")

        st.button("clear chat history", on_click=on_btn_click, type="primary")

    st.chat_input("请输入内容...", key="query", on_submit=deal_user_query)


if __name__ == "__main__":
    set_web()
