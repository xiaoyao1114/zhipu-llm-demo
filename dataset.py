import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 文本向量化
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
def get_files():
    """
        读取数据集
    """
    file_paths = []
    folder_path = './data_base/knowledge_db'

    if not os.path.exists(folder_path):
        print(f"指定的路径不存在: {folder_path}")
    else:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths
# 文档读取
def get_docs():  
    docs = []
    for file_path in get_files():
        file_type = file_path.split(".")[-1]
        if file_type == 'pdf':
            current_pdf_pages = PyMuPDFLoader(file_path).load()
            docs.extend(current_pdf_pages)
        elif file_type == 'md':
            current_md_pages = UnstructuredMarkdownLoader(file_path).load()
            docs.extend(current_md_pages)
    return docs

def get_vectordb():
    docs = get_docs()
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents(docs)

    # 定义 Embeddings
    # embedding = OpenAIEmbeddings() 
    embedding = ZhipuAIEmbeddings()
    # embedding = QianfanEmbeddingsEndpoint()

    # 定义持久化路径
    persist_directory = '../data_base/vector_db/chroma'



    vectordb = Chroma.from_documents(
        documents=split_docs, # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
   

    print(f"向量库中存储的数量：{vectordb._collection.count()}")

   
    return vectordb


def test():
    vectordb = docs_embedding()
    """
        测试向量数据库
    """
    question="什么是大语言模型"
    # 相似度检索
    sim_docs = vectordb.similarity_search(question,k=3)
    print(f"相似度检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")



    # MMR 检索 最大边际相关性检索
    mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
 
def get_model():
    from zhipuai_llm import ZhipuAILLM
    api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

    llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=api_key)
    return llm


# from langchain.prompts import PromptTemplate

# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
# {context}
# 问题: {question}
# """

# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
#                                  template=template)

# from langchain.chains import RetrievalQA
# vectordb = get_vectordb()
# qa_chain = RetrievalQA.from_chain_type(get_model(),
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})


# question_1 = "什么是南瓜书？"
# question_2 = "王阳明是谁？"
# result = qa_chain.invoke({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])
# result = qa_chain.invoke({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])

# llm = get_model()

# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)

# ### 基于大模型的问答
# ll1 = llm.invoke(prompt_template)

# print("大模型zijide 回答 question_1 的结果：")
# print(ll1)



# prompt_template2 = """请回答下列问题:
#                             {}""".format(question_2)

# ### 基于大模型的问答
# ll2 = llm.invoke(prompt_template2)

# print("大模型zijide 回答 question_2 的结果：")
# print(ll2)

if __name__ == '__main__':
    get_vectordb()







