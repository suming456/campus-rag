# rag_engine.py

import logging
import os
from typing import List, Tuple, Optional

import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

import config_data as config

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 关键：全局设置 dashscope API Key（确保所有调用都使用正确的 key）
dashscope.api_key = config.QWEN_API_KEY


def get_embeddings():
    """获取嵌入模型 Qwen（使用 DashScope）"""
    return DashScopeEmbeddings(
        model=config.QWEN_EMBEDDING_MODEL,
        dashscope_api_key=config.QWEN_API_KEY  # 正确参数名
    )


def process_pdf_to_chroma(pdf_path: str, chroma_dir: str) -> Chroma:
    """处理 PDF 文件：加载、分块、生成 embeddings、存入 Chroma"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info(f"成功加载 {len(pages)} 页")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,
    )

    chunks = []
    for page_number, page in enumerate(pages, 1):
        page_chunks = splitter.split_text(page.page_content)
        for chunk_text in page_chunks:
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={"source": os.path.basename(pdf_path), "page": page_number}
                )
            )

    logger.info(f"文本已被分为 {len(chunks)} 个块，每块约 {config.CHUNK_SIZE} 字符")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,           # 注意参数名是 documents
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    # 新版本 Chroma 会自动持久化，但调用 persist() 也无妨
    vector_store.persist()
    logger.info(f"成功将 {len(chunks)} 个块存入向量库 {chroma_dir}")
    return vector_store


def load_vector_store(chroma_dir: str) -> Optional[Chroma]:
    """从指定目录加载已存在的向量库"""
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        return None
    embeddings = get_embeddings()
    try:
        vector_store = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )
        _ = vector_store._collection.count()
        logger.info(f"已成功加载向量库 {chroma_dir}")
        return vector_store
    except Exception as e:
        logger.error(f"加载向量库失败: {e}")
        return None


def retrieve_documents(vector_store: Chroma, query: str, k: int = 4) -> List[Document]:
    """检索最相关的 k 个文档"""
    docs = vector_store.similarity_search(query=query, k=k)
    return docs



def format_history(history: List[Tuple[str, str]]) -> str:
    """格式化历史记录"""
    if not history:
        return ""
    #只保留最近 MAX_HISTORY_LENGTH 条记录
    history = history[-config.MAX_HISTORY_LENGTH:]
    formatted = "\n [对话历史] \n"
    for user_msg,assistant_msg in history:
        formatted +=f"[用户] {user_msg}\n[助手] {assistant_msg}\n\n"
        formatted +=" [当前对话]\n"
    return formatted



def build_prompt_with_history(context: str, question: str,history: List[Tuple[str,str]] = None) -> str:
    """构建提示词"""
    history_text = format_history(history) if history else ""
    template = PromptTemplate(
        input_variables=["context", "question"],
        template="""你是一位校园助手，只能根据《学生手册》中的内容回答问题。
如果无法从提供的上下文中找到答案，请直接说“手册中没有相关信息”，不要编造答案。
{history_text}
参考资料：
{context}

用户问题：{question}

请基于上述参考资料给出准确、简洁的答案。"""
    )
    return template.format(history_text = history_text ,context=context, question=question)


def generate_answer(llm, prompt: str) -> str:
    """调用 LLM 生成答案（支持流式输出）"""
    try:
        full_answer = ""
        print("\n🤖 AI 回答：")
        for chunk in llm.stream(prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_answer += chunk.content
        print()
        return full_answer
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return "生成答案失败，请检查 API Key 或网络连接。"


def get_llm():
    """获取 ChatTongyi 实例（使用 Qwen 模型）"""
    return ChatTongyi(
        model=config.QWEN_CHAT_MODEL,
        dashscope_api_key=config.QWEN_API_KEY,   # 修正参数名
        temperature=0.1,
    )


def format_sources(docs: List[Document]) -> str:
    """返回 (来源文本, 页码列表) 用于前端显示"""
    scources = []
    pages = set()
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "未知")
        source_file = doc.metadata.get("source", "学生手册")
        scources.append(f"[来源{i}] {source_file} | 第 {page} 页")
        if page != "未知":
            pages.add(int(page))
    return "\n".join(scources), sorted(pages)

def answer_question(vector_store: Chroma, question: str,history: List[Tuple[str,str]] = None, k: int = 4) -> Tuple[str, str]:
    """
    完整问答流程：检索 -> 生成答案 -> 返回 (答案, 引用文本)
    """
    docs = retrieve_documents(vector_store, question, k)   
    if not docs:
        return "没找到相关的手册内容", "",[]

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt_with_history(context, question,history)

    llm = get_llm()
    answer = generate_answer(llm, prompt)
    scources,pages = format_sources(docs)
    return answer, scources,pages