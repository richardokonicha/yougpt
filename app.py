import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

st.title("LangChain Essay")

prompt = st.text_input('Plug in your prompt here')

st.write("OPENAI_API_KEY: ", st.secrets['OPENAI_API_KEY'])
st.write("DEBUG:", st.secrets["dev"]["DEBUG"])

title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write a title for your essay about ${topic}.'
)

body_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write a body for your essay about ${title}. while leveraging wiki {wikipedia_research}'
)

title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')
body_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')


llm = OpenAI(temperature=0.9)
title_chain = LLMChain(
    llm=llm, prompt=title_template,
    output_key='title', memory=title_memory)
body_chain = LLMChain(llm=llm, prompt=body_template,
                      output_key='body', memory=body_memory)

# sequential_chain = SequentialChain(
#     chains=[title_chain, body_chain], input_variables=['topic'], output_variables=['title', 'body'], verbose=True)


wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    body = body_chain.run(title=title, wikipedia_research=wiki_research)
    # response = sequential_chain({'topic': prompt})
    st.write(title)
    st.write(body)

    with st.expander('Title Message History'):
        st.info(title_memory.buffer)

    with st.expander('Body Message History'):
        st.info(body_memory.buffer)

    with st.expander('Wikipedia Message History'):
        st.info(wiki_research)


# if prompt:
#     response = body_chain.run(topic=prompt)
#     st.write(response)
