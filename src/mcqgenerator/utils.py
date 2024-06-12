from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_together import TogetherEmbeddings
from langchain_community.llms import Together

vectorstore = FAISS.from_texts(
    ["Together AI provides the fastest cloud platform for building and running generative AI.",
     "Together AI recently released their StripedHyena models, which is the first alternative model competitive with the best open-source Transformers in short and long-context evaluations.",
     "Together AI is also the creator of the RedPajama Datasets. RedPajama-Data-v2 is an open dataset with 30 trillion tokens from 84 CommonCrawl dumps.",
     "Together AI recently raised $102.5M in a Series A financing to build the future of AI."],
     TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
)

retriever = vectorstore.as_retriever()

model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128,
    top_k=50,
    # together_api_key="..."
)

# Provide a template following the LLM's original chat template.
template = """<s>[INST] Answer the question in a simple sentence based only on the following context:
{context}

Question: {question} [/INST] 
"""
prompt = ChatPromptTemplate.from_template(template) 

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

input_query = "What are some recent highlights of Together AI?"
output = chain.invoke(input_query)

print(output)
