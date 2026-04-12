# loader.py  - 使用PyPDFLoader 加载校园手册 PDF 并输出文本

from langchain_community.document_loaders import PyPDFLoader
import config_data as config

#指定 PDF 文件路径
pdf_path = config.PDF_PATH
def load_pdf(pdf_path):
    print("正在加载 PDF 文件 {padf_path} ...")
    loader = PyPDFLoader(pdf_path)#创建 PyPDFLoader 对象
    documents = loader.load()      #加载 PDF 文件

    print(f"共加载 {len(documents)} 个页\n")

    #输出每一页的前 500 个字符
    for i , doc in enumerate(documents,start=1):
        pag_content = doc.page_content  #获取每一页的文本
        print(f"======第 {i} 页 (共 {len(documents)} 字符)======")
        print(pag_content[:500])
        if len(pag_content) > 500:
            print("...(片段省略)...")
            print("\n"+"="*50+"\n")

    #将全部文本保存到文件
    with open(config.STUDENT_HARDBOOK, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.page_content+"\n\n")
    print(f"全部文本已保存到 {config.STUDENT_HARDBOOK} 文件")


if __name__ == "__main__":
    load_pdf(pdf_path)