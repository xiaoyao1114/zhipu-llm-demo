import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file

import streamlit as st
from zhipuai_llm import ZhipuAILLM
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

def get_vectordb():
    persist_directory = './data_base/vector_db/chroma'
    embedding = ZhipuAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function= embedding # 允许我们将persist_directory目录保存到磁盘上
    )
    return vectordb

def get_llm(openai_api_key):
    llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=openai_api_key)
    return llm

def generate_response(input_text,openai_api_key):
    # Initialize the ZhipuAILLM model
    llm = get_llm(openai_api_key=openai_api_key)
    # Generate a response using the model
    output = llm.invoke(input_text)
    out_put_parser = StrOutputParser()
    output = out_put_parser.invoke(output)
    return output
def get_retrieve_result(question):
    db = get_vectordb()
    raw_docs = db.similarity_search_with_relevance_scores(query=question, k=3)  
    my_docs = [doc for doc, score in raw_docs]  
    context = "\n\n".join([doc.page_content for doc in my_docs])     
    return context, my_docs                                        
def get_qa_chain(question:str,openai_api_key:str):
    vectordb = get_vectordb()
    llm = get_llm(openai_api_key=openai_api_key)
    sys_prompt = SystemMessagePromptTemplate.from_template(template="""
     你是一个大模型开发助手，请根据用户从私有知识库检索出来的上下文来回答用户的问题！
        请注意：
            1，如果你不知道答案，就说你不知道，不要试图编造答案
            2，最多使用五句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！""")
    user_prompt = HumanMessagePromptTemplate.from_template(template="""
        检索出的上下文为：
            {context}
            用户的问题为：
            {question}
            答案为：
        """)
    prompt = ChatPromptTemplate.from_messages(messages=[sys_prompt, user_prompt])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    context,docs = get_retrieve_result(question)
    results = chain.invoke(input={"context": context, "question": question})
    return results
def main():
    """
    主函数，用于构建Streamlit应用界面，处理用户输入，并显示对话历史。
    """
    # 设置页面标题
    st.title('🦜🔗 Nanxi 动手学大模型应用开发')
    
    # 通过radio按钮选择对话模式
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", ],
        captions = ["不使用检索问答的普通模式", "检索问答模式"])

    # 在侧边栏输入OpenAI API Key
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password',
            value=os.environ['ZHIPUAI_API_KEY']) 
    
    # 初始化会话状态中的消息列表
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 创建一个容器用于显示消息
    msgBox = st.container(height=300)
    
    # 获取用户输入的提示信息
    prompt = st.chat_input("Say something")
    
    # 如果有提示信息，则添加到会话状态的消息列表中
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 根据选择的模式调用不同的函数获取回答
        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, openai_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,openai_api_key)
      
        # 如果有回答，则添加到会话状态的消息列表中
        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # 遍历消息列表，根据角色在容器中显示消息
        for message in st.session_state.messages:       
            if message["role"] == "user":
                msgBox.chat_message("user").write(message["content"])
            else:
                msgBox.chat_message("assistant").write(message["content"])


if __name__ == '__main__':
    main()