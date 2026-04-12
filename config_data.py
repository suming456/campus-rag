# config_data.py
import os

BASE_DIR = os.path.dirname(__file__)


# PDF 路径
PDF_PATH = os.path.join(BASE_DIR, "student_hardbook.pdf")
# 文本提取保存路径
STUDENT_HARDBOOK = os.path.join(BASE_DIR, "handbook_full_text.txt")

# 向量数据库存储目录
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# ========= 临时上传文件配置 ==========
# 默认 PDF 路径
DEFAULT_PDF_PATH = PDF_PATH
# 临时上传文件的向量库目录（每次上传会新建）
TEMP_CHROMA_DIR = os.path.join(BASE_DIR, "temp_chroma")

# ========== Qwen API 配置 ==========
# 使用 Qwen API（需申请 API Key）
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen3-max")
QWEN_EMBEDDING_MODEL = "text-embedding-v4"

# ========== 文本分块参数 ==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

#========= 对话历史保留次数 ==========
MAX_HISTORY_LENGTH = 20