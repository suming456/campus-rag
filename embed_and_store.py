#embed_and_store.py

"""
将校园手册文本分块、生成 embeddings 并存入 Chroma 向量数据库。
用法: python embed_and_store.py
"""
from pydoc import text
import sys
import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

import config_data as config

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

def load_text_from_pdf(file_path: str) -> str:
    """读取完整的手册文本"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            logger.info(f"成功从 {file_path} 读取文本,共 {len(text)} 字符")
            return text

    except FileNotFoundError:        
        logger.error(f"文件 {file_path} 不存在,请先运行loader.py生成文件")
        sys.exit(1)#退出程序
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        sys.exit(1)


def split_text(text: str,chunk_size: int,chunk_overlap: int)->List[str]:
    """将文本分块"""
    # 创建文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,        #分割后的文本最大长度
        chunk_overlap = chunk_overlap,  #分割后的文本之间的重叠长度
        separators=config.SEPARATORS,   
        length_function = len           
    )
    chunks = splitter.split_text(text) 
    logger.info(f"文本已被分为{len(chunks)}个块,每块约{chunk_size}字符")
    return chunks

def get_embedding():
    """获取 embedding 模型"""
    from langchain_community.embeddings import DashScopeEmbeddings
    logger.info(f"使用 Qwen embedding 模型:{config.QWEN_EMBEDDING_MODEL}")
    embeddings = DashScopeEmbeddings(
        model = config.QWEN_EMBEDDING_MODEL,
        
    )
    return embeddings


def store_to_chroma(chunks: List[str],embeddings, persist_dir: str):
    """将分块的文本和 embeddings 存储到 Chroma 向量数据库中"""
    #为每个chunk 生成一个metadata
    metadatas = [{"source":"student_handbook","chunk_id":1} for i in range(len(chunks))]

    try:
        # 创建 Chroma 向量数据库
        vectorstore = Chroma.from_texts(
            texts = chunks,
            embedding = embeddings,
            persist_directory = persist_dir,
            metadatas = metadatas,
        )
        #持久化
        vectorstore.persist()   #持久化
        logger.info(f"成功存入{len(chunks)}个向量块到 {persist_dir}")
        return vectorstore
    except Exception as e:
        logger.error(f"向量数据库存储失败: {e}")
        sys.exit(1)

def main():
    
    logger.info("开始执行 embed_and_store.py")

    #1. 加载文本
    full_text = load_text_from_pdf(config.STUDENT_HARDBOOK)

    #2.切分文本
    chunks = split_text(full_text,config.CHUNK_SIZE,config.CHUNK_OVERLAP)

    #3. 获取 embedding 模型
    embeddings = get_embedding()

    #4. 存储到 Chroma 向量数据库中
    vector_store = store_to_chroma(chunks,embeddings,config.CHROMA_PERSIST_DIR)

    logger.info("向量数据库存储成功")


if __name__ == "__main__":
    main()