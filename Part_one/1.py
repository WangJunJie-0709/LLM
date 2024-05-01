import os
import openai
from dotenv import load_dotenv, find_dotenv
import os
from dotenv import load_dotenv, find_dotenv
# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key:", openai_api_key)



def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']


if __name__ == '__main__':
    openai.api_key = get_openai_key()