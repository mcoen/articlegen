import os
from apikey import apikey

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = apikey

st.title('Article Generator')
topic = st.text_input('Please enter the topic of interest for the article')
#language = st.text_input('Please enter the language of the article')

titleTemplate = PromptTemplate(
    input_variables = ['topic'],
#    input_variables = ['topic', 'language'],
#    template = 'Give me a Medium article title on {topic} in {language} language'
    template = 'Give me an article title on {topic}'
)

articleTemplate = PromptTemplate(
    input_variables = ['title'],
#    template = 'Give me a Medium article title on {topic} in {language} language'
    template = 'Give me an article for title: {title}'
)

GPT_MODEL = 'gpt-3.5-turbo'

llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.95)
#llm = OpenAI(temperature=0.9)
#titleLLM = ChatOpenAI(temperature=0.9)
#titleChain = LLMChain(llm=titleLLM, prompt=titleTemplate, verbose=True)
titleChain = LLMChain(llm=llm, prompt=titleTemplate, verbose=True)

#articleLLM = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
#articleChain = LLMChain(llm=articleLLM, prompt=articleTemplate, verbose=True)
articleChain = LLMChain(llm=llm, prompt=articleTemplate, verbose=True)

overallChain = SimpleSequentialChain(chains=[titleChain, articleChain], verbose=True)

if topic:
#    response = llm(titleTemplate.format(topic=topic, language='english'))
#    response = titleChain.run({'topic':topic, 'language':language})
#    response = titleChain.run(topic)
    response = overallChain.run(topic)
    st.write(response)
