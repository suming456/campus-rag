# rag_chain.py

"""
RAG-LangChain问答链:
        根据用户问题检索相关手册,使用Qwen 生成答案
用法:
        python rag_chain.py "你的问题"
"""

import sys
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
import config_data as config

#日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
)
logger = logging.getLogger(__name__)

def load_vector_store() :
    """加载已存储的 Chroma向量数据库"""
    #使用与存储时完全相同的嵌入模型
    embeddings = DashScopeEmbeddings(
        model = config.QWEN_EMBEDDING_MODEL,
        dashscope_api_key=config.QWEN_API_KEY
    )
    try:
        vector_store = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,#数据库文件路径
            embedding_function=embeddings,#嵌入模型
        )
        #打印数据库信息
        logger.info(f"成功加载向量数据库,包括{vector_store._collection.count()}个向量")
        return vector_store
    
    except Exception as e:
        logger.error(f"加载向量数据库失败:{e}")
        sys.exit(1)


def retrieve_documents(vector_store,query: str,k: int=4):
    """检索与问题最相关的k个文档"""
    docs = vector_store.similarity_search(query,k= k)
    logger.info(f"检索到{len(docs)}个相关文档")
    return docs


def build_prompt(context:str,question:str)->str:
    """构建发送给 LLM  的提示词"""
    prompt_template = PromptTemplate(
        input_variables=["context","question"],
        template = """你是一位校园助手，只能根据《学生手册》中的内容回答问题。
如果无法从提供的上下文中找到答案，请直接说“手册中没有相关信息”，不要编造答案。

参考资料：
{context}

用户问题：{question}

请基于上述参考资料给出准确、简洁的答案."""
    ) 
    return prompt_template.format(context=context,question=question)


def generate_answer(llm,prompt: str)->str:
    """使用 LLM 生成答案"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM 调用失败:{e}")
        return "生成答案失败"



def format_sources(docs)->str:
    """格式化引用来源（由于缺乏页码，暂时显示块索引）"""
    sources = []
    for i ,doc in enumerate(docs,1):
        #获取元数据中的chunk_id
        chunk_id = doc.metadata.get("chunk_id","未知")#块索引
        sources.append(f"[片段 {i} ](chunk_id: {chunk_id})")    
    return "\n".join(sources)


def main():
    # 获取问题：优先使用命令行参数，否则交互输入
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:])
    else:
        print("\n欢迎使用校园问答助手（基于学生手册）")        
        while True:
            try:
                question = input("请输入你的问题: ").strip()
                if question:
                    break
                print("问题不能为空")
            # 捕获 Ctrl+C 中断，优雅退出
            except KeyboardInterrupt:
                print("\n\n已退出程序，欢迎下次使用！")
                sys.exit(0)
        
    logger.info(f"用户问题: {question}")
    #1.加载向量数据库
    vector_store = load_vector_store()

    #2.检索最相关的文档
    docs = retrieve_documents(vector_store,question,k=4)
    if not docs:
        print("没有找到相关的文档")
        return
    
    #3.合并上下文(保留原文顺序)
    context = "\n\n".join([doc.page_content for doc in docs])

    #4.构建提示词
    prompt = build_prompt(context,question)
    logger.debug(f"提示词: {prompt[:200]}...")

    #5.初始化LLM (使用Qwen)
    llm = ChatTongyi(
        model = config.QWEN_CHAT_MODEL,
        dashscope_api_key = config.QWEN_API_KEY,
        temperature = 0.1,#降低随机性
    )

    #6.生成答案
    answer = generate_answer(llm,prompt)

    #7.输出结果
    print("\n" + "="*50)
    print(f"问题: {question}\n")
    print(f"答案: {answer}\n")
    print("引用来源:")
    print(format_sources(docs))
    print("="*50)

    #将结果记录到日志中
    logger.info(f"问题: {question}\n答案: {answer}\n引用来源:{format_sources(docs)}")


if __name__ == "__main__":
    main()
