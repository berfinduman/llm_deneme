

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd 
import datetime
# Make sure the model path is correct for your system!
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
            model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin",
            n_gpu_layers=60,
            n_batch=64,
            temperature=0.1,
            max_tokens=40,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True,
        )
#prompt = """
#Question: A rap battle between Stephen Colbert and John Oliver
#"""
#print(llm(prompt))

#template = """
#              Write a concise and descriptive topics that defines the main objectives of text,which actually contains a customer comment, delimited by triple backquotes.
#              The topics should be a single word or a short phrase.
#              Return your response in bullet points that cover the main topics of the text.
#             ```{text}```
#              BULLET POINT TOPICS:
#           """
#template=r"""I am giving you the customer coment in parentheses ,  give me the Broader category like poor quality product, dissatisfied customer, favorite product  or the related high level topics in one word in the format[Topic: your primary topic] for the 
#text '{text}' """

#template="""I am the owner of an automotive company that has a comment system written by our customers. I don't want all the comments so I need your help:
#Regardless of the complaint, suggestion, or request, you should convey to me the summary of important topics in the customer comments, delimited by max triple backquotes.
#The topics should be a single word or a short phrase. I put customer reviews in three italic quotes.
#Return ONLY your response in bullet points that cover the main topics of the text, no extra comment!
#             ```{text}```
#              BULLET POINT TOPICS:"""

template="""Given the following user review ```REVIEW``` extract the key complaints the user has,
summarized into either 2 or 3 words for each key complaint. 
Return ONLY your response in bullet points that cover the main topics of the text.
             ```{text}```
              BULLET POINT TOPICS:"""
#template="""\n\n{text}The above customer comment contains the following topic(s):""" 
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

data=pd.read_csv("filtered_data_kia.csv",engine='python',index_col=False)
reviews=data["Review"]




answers=[]
for text in reviews[:]:

    
      print("TEXT:",text)
      answers.append(llm_chain.run(text))
      print("<enddddd>")

data2=pd.DataFrame()
data2=data[["Review_Title","Vehicle_Title","Review","Rating"]][:]
data2["answer_llma"] = answers
print("that is end")

name = f'carBMW_{datetime.datetime.now().strftime("%H%M_%m%d%Y")}.csv'
data2.to_csv(name)


"""
all_rs=pd.read_csv("amazon_done.csv")
import random
randomlist = []
for i in range(0,20):
  randomlist.append(random.randint(1,all_rs.shape[0]))

  texts=all_rs.iloc[randomlist]["review_content"]
answers=[]
for text in texts:
  print(text)
  answer=llm_chain.run(text)
  print(answer)
  answers.append(answer)
  print("###################")
data2=all_rs.iloc[randomlist]['Title', 'Review Text']
data2["answer_llama2"]=answers
name = f'amazon_en_v3.csv'
data2.to_csv(name)




data=pd.read_csv("women_ecc.csv")

texts=data["Review Text"].to_list()
answers=[]
for text in texts[:20]:
    print("TEXT:",text)
    answers.append(llm_chain.run(text))

data2=pd.DataFrame()
data2=data.head(20)[["Title","Review Text"]]
data2["answer_llma"] = answers

name = f'wcc_en_{datetime.datetime.now().strftime("%H%M_%m%d%Y")}.csv'
data2.to_csv(name)
"""