from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *




B_INST,E_INST = "[INST]","[/INST]"
B_SYS,E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

instruction = "Give proper summery of he text: \n\n {text}"

SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

prompt = PromptTemplate(template=template,input_variables=["text"])


llm = CTransformers(model='C:\GenerativeAI-Basics\Project-Generative-AI\Llama-2-Project-on-CPU\model\llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':128,
                            'temperature':0.01}
                            )

LLMChain = LLMChain(prompt=prompt,llm=llm)

#print(LLMChain.run("How are you?"))

#print(LLMChain.run("I am Curious to learn Artificial intelligence"))

print(LLMChain.run("Harry Porter"))


