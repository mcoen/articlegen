import os
import streamlit as st

from apikey import apikey
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain

os.environ["OPENAI_API_KEY"] = apikey

st.title('Generic Article Generator')
topic = st.text_input('Please enter the topic of interest for the article')

# Create the queries to be used by the LLM
titleTemplate = PromptTemplate(
    input_variables = ['topic'],
    template = 'Give me an article title on {topic}'
)

articleTemplate = PromptTemplate(
    input_variables = ['title'],
    template = 'Give me an article for title: {title}'
)

# Set up the model by chaining together two distinct queries of LLM
GPT_MODEL = 'gpt-3.5-turbo'

llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.95)
titleChain = LLMChain(llm=llm, prompt=titleTemplate, output_key='title', verbose=True)
articleChain = LLMChain(llm=llm, prompt=articleTemplate, output_key='article', verbose=True)
overallChain = SequentialChain(chains=[titleChain, articleChain],
                               input_variables=['topic'],
                               output_variables=['title', 'article'],
                               verbose=True)

# Output the results
if topic:
    results  = overallChain(topic)
    print(type[results])
    print(results.keys)
    st.write(results['title'])
    st.write()
    st.write(results['article'])