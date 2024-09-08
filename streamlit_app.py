from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Gen AI: Analysis Tool", page_icon="🤖")
st.title("🤖 Gen AI: Analysis Tool")

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type="CSV",
    help = "Only supports CSV files",
    on_change=clear_submit
)

df = pd.DataFrame([])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)
else:
    st.write("Please upload a CSV file")

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:  
    st.chat_message(msg["role"]).write(msg["content"])
    
if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
        
    llm = ChatOpenAI(
        model = "gpt-4o-mini", 
        temperature = 0, 
        openai_api_key = openai_api_key,
        streaming = True,
        max_tokens = 3000
        )
        
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="openai-functions",
        handle_parsing_errors=True,
    )
        
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.invoke(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.write(response["output"])

