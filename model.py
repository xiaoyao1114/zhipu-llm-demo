from zhipuai_llm import ZhipuAILLM


from dotenv import find_dotenv, load_dotenv
import os


_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

zhipuai_model = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=api_key)

res = zhipuai_model.invoke("你好，请你自我介绍一下！")
print(res)