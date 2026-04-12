# app.py
#        streamlit run "app.py"
import os
import shutil   
import tempfile
import streamlit as st
from rag_engine import process_pdf_to_chroma, load_vector_store, answer_question
import config_data as config

# 页面配置
st.set_page_config(page_title="校园手册问答助手", layout="wide")
st.title("📚 校园手册智能问答助手")
st.markdown("基于 **Qwen LLM + RAG** 的学生手册问答系统")

# ------------------ 初始化聊天历史 -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 元素: {"role": "user"/"assistant", "content": str, "sources": str, "pages": list}
if "vectorstore" not in st.session_state:   # 统一使用 vectorstore (无下划线)
    st.session_state.vectorstore = None
if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = None

# ------------------ 侧边栏: 文件上传与向量库管理 -------------------------
with st.sidebar:
    st.header("📁 手册管理")

    use_default = st.radio(
        "选择手册来源",
        ["使用默认学生手册（student_hardbook.pdf）", "上传新的 PDF 手册"]
    )

    if use_default == "使用默认学生手册（student_hardbook.pdf）":
        chroma_dir = config.CHROMA_PERSIST_DIR
        if st.button("加载默认手册向量库..."):
            vector_store = load_vector_store(chroma_dir)
            if vector_store:
                st.session_state.vectorstore = vector_store   # 统一键名
                st.session_state.chroma_dir = chroma_dir
                st.success("向量库加载成功！")
                st.session_state.messages = []
            else:
                if os.path.exists(config.DEFAULT_PDF_PATH):
                    st.info("向量库不存在，正在构建...")
                    with st.spinner("构建向量库..."):
                        vector_store = process_pdf_to_chroma(
                            config.DEFAULT_PDF_PATH, 
                            chroma_dir,
                        )
                        st.session_state.vectorstore = vector_store
                        st.session_state.chroma_dir = chroma_dir
                        st.success("默认手册向量库构建成功！")
                        st.session_state.messages = []
                else:
                    st.error(f"默认手册不存在: {config.DEFAULT_PDF_PATH}，请上传新的手册！")
    else:
        uploaded_file = st.file_uploader("上传新的手册", type=["pdf"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
                safe_name = uploaded_file.name.replace(".", "_").replace(" ", "_")
                chroma_dir = os.path.join(config.CHROMA_PERSIST_DIR, safe_name)
                if st.button("构建手册向量库"):
                    with st.spinner(f"正在处理 {uploaded_file.name}，请稍候..."):
                        if os.path.exists(chroma_dir):
                            shutil.rmtree(chroma_dir)
                        vector_store = process_pdf_to_chroma(temp_file_path, chroma_dir)
                        st.session_state.vectorstore = vector_store
                        st.session_state.chroma_dir = chroma_dir
                        st.success(f"手册向量库构建成功！保存在 {chroma_dir}")
                        st.session_state.messages = []
        else:
            st.info("请上传 PDF 文件，然后点击下方按钮构建向量库。")

    if st.session_state.vectorstore:
        st.info(f"✅ 当前激活手册：{st.session_state.chroma_dir}")
    else:
        st.warning("⚠️ 尚未加载任何手册，请先在左侧加载或上传手册。")

# ----------------------------- 主区域: 问答交互 --------------------------------------------
st.header("💬 向手册提问")

# 显示历史对话
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if "pages" in msg and msg["pages"]:
                pages_str = "、".join(str(p) for p in msg["pages"])
                st.caption(f"📖 答案来自《学生手册》第 {pages_str} 页")
            elif "sources" in msg and msg["sources"]:
                st.caption(msg["sources"])

# 输入框
if question := st.chat_input("请输入你的问题，例如：'奖学金申请条件是什么？'"):
    if not st.session_state.vectorstore:   # 修正：使用 vectorstore
        st.error("未加载手册，请先在左侧加载或上传手册。")
        st.stop()
    
    # 将用户问题添加到界面
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 构建历史记录 (从 messages 中提取 user 和 assistant 配对)
    history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            user_msg = st.session_state.messages[i]["content"]
            assistant_msg = st.session_state.messages[i + 1]["content"]
            history.append((user_msg, assistant_msg))
    
    with st.chat_message("assistant"):
        with st.spinner("正在检索手册并生成答案..."):
            try:
                # rag_engine.answer_question 返回三个值: answer, sources, pages
                answer, sources, pages = answer_question(
                    st.session_state.vectorstore,
                    question,
                    history,
                    k=4,
                )
                st.write(answer)
                if pages:
                    pages_str = "、".join(str(p) for p in pages)
                    st.caption(f"📖 答案来自《学生手册》第 {pages_str} 页")
                elif sources:
                    st.caption(sources)
                
                # 存储助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "pages": pages,
                })
            except Exception as e:
                st.error(f"发生错误: {e}")