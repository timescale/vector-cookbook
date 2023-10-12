# Copyright (c) Timescale, Inc. (2023)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

from llama_index.vector_stores import TimescaleVectorStore
from llama_index import ServiceContext, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index import set_global_service_context

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from timescale_vector import client

from typing import List, Tuple

from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
import psycopg2

def get_repos():
    with psycopg2.connect(dsn=st.secrets["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            try:
                select_data_sql = "SELECT * FROM time_machine_catalog;"
                cursor.execute(select_data_sql)
            except psycopg2.errors.UndefinedTable as e:
                return {}

            catalog_entries = cursor.fetchall()

            catalog_dict = {}
            for entry in catalog_entries:
                repo_url, table_name = entry
                catalog_dict[repo_url] = table_name

            return catalog_dict

def get_auto_retriever(index, retriever_args):
    from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
    vector_store_info = VectorStoreInfo(
        content_info="Description of the commits to PostgreSQL. Describes changes made to Postgres",
        metadata_info=[
            MetadataInfo(
                name="commit_hash",
                type="str",
                description="Commit Hash",
            ),
            MetadataInfo(
                name="author",
                type="str",
                description="Author of the commit",
            ),
            MetadataInfo(
                name="__start_date",
                type="datetime in iso format",
                description="All results will be after this datetime",
    
            ),
            MetadataInfo(
                name="__end_date",
                type="datetime in iso format",
                description="All results will be before this datetime",
    
            )
        ],
    )
    from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
    retriever = VectorIndexAutoRetriever(index, 
                                         vector_store_info=vector_store_info, 
                                         service_context=index.service_context,
                                         **retriever_args)
    
    # build query engine
    from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, service_context=index.service_context
    )

    from llama_index.tools.query_engine import QueryEngineTool
    # convert query engine to tool
    query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

    from llama_index.agent import OpenAIAgent
    chat_engine = OpenAIAgent.from_tools(
        tools=[query_engine_tool],
        llm=index.service_context.llm,
        verbose=True
        #service_context=index.service_context
    )
    return chat_engine

def tm_demo():
    repos = get_repos()

    months = st.sidebar.slider('How many months back to search (0=no limit)?', 0, 130, 0)

    if "config_months" not in st.session_state.keys() or months != st.session_state.config_months:
        st.session_state.clear()

    topk = st.sidebar.slider('How many commits to retrieve', 1, 150, 20)
    if "config_topk" not in st.session_state.keys() or topk != st.session_state.config_topk:
        st.session_state.clear()
        
    if len(repos) > 0:
        repo = st.sidebar.selectbox("Choose a repo", repos.keys())
    else:
        st.error("No repositiories found, please [load some data first](/LoadData)")
        return
    
    if "config_repo" not in st.session_state.keys() or repo != st.session_state.config_repo:
        st.session_state.clear()
    
    st.session_state.config_months = months
    st.session_state.config_topk = topk
    st.session_state.config_repo = repo


    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Please choose a repo and time filter on the sidebar and then ask me a question about the git history"}
        ]

    vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name=repos[repo],
        time_partition_interval=timedelta(days=7),
    );

    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0.1))
    set_global_service_context(service_context)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
    
        
    #chat engine goes into the session to retain history
    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        retriever_args = {"similarity_top_k" : int(topk)}
        if months > 0:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(weeks=4*months)
            retriever_args["vector_store_kwargs"] = ({"start_date": start_dt, "end_date":end_dt})
        st.session_state.chat_engine = get_auto_retriever(index, retriever_args)
        #st.session_state.chat_engine = index.as_chat_engine(chat_mode="best", similarity_top_k=20, verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt, function_call="query_engine_tool")
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

st.set_page_config(page_title="Time machine demo", page_icon="🧑‍💼")
st.markdown("# Time Machine")
st.sidebar.header("Welcome to the Time Machine")

debug_llamaindex = False
if debug_llamaindex:
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

tm_demo()

#show_code(tm_demo)
