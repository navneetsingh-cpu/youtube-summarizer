import streamlit as st  # Import Streamlit library for creating web apps
import os  # Import os module for environment variables
from openai import OpenAI  # OpenAI library for accessing GPT-3 API
from dotenv import (
    find_dotenv,
    load_dotenv,
)  # Import dotenv for loading environment variables
import re

# Import necessary classes and functions from LangChain library
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"


youtube_url_pattern = re.compile(r"^(https?\:\/\/)?(www\.youtube\.com|youtu\.be)\/.+$")


def is_valid_youtube_url(url):
    """Check if the provided URL is a valid YouTube URL."""
    return youtube_url_pattern.match(url) is not None


# Set the title of the Streamlit app
st.title("🔎 YouTube Video Summarizer")

# Initialize messages in the session state if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can summarize youtube videos. How can I help you?",
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


with st.form("my_form"):
    st.write("## Enter YouTube URL")
    url = st.text_input("URL", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        if url:
            # Check if URL is valid
            if not is_valid_youtube_url(url):
                st.write("Please enter a valid YouTube URL.")

            else:
                st.write("## Summary")
                # Add a loading indicator while waiting for the backend
                with st.spinner("Summarizing Video..."):
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    result = loader.load()
                    st.write(
                        f"Found Video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long"
                    )
                    llm = OpenAI(temperature=0.6)
                    chain = load_summarize_chain(
                        llm=llm, chain_type="stuff", verbose=True
                    )
                    st.write(chain.run(result))

        else:
            st.write("Please enter a YouTube URL.")

# footer
st.write("---")
st.write("Made by [Navneet](https://medium.com/@navneetskahlon)")
