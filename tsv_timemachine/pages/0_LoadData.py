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

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import subprocess
import shutil
import psycopg2

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

from llama_index.vector_stores import TimescaleVectorStore
from llama_index import StorageContext
from llama_index.indices.vector_store import VectorStoreIndex

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from timescale_vector import client

from typing import List, Tuple

from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from git import Repo

from llama_index.text_splitter import SentenceSplitter

def create_uuid(date_string: str):
    datetime_obj = datetime.fromisoformat(date_string)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)

# Create a Node object from a single row of data
def create_nodes(row):
    text_splitter = SentenceSplitter(chunk_size=1024)

    record = row.to_dict()
    record_content = (
        "Date: "+ str(record["Date"])
        + " "
        + "Author: "+ record['Author']
        + " "
        + str(record["Subject"])
        + " "
        + str(record["Body"])
    )

    text_chunks = text_splitter.split_text(record_content)
    nodes = [TextNode(
        id_=create_uuid(record["Date"]),
        text=chunk,
        metadata={
            "commit_hash": record["Commit Hash"],
            "author": record['Author'],
            "date": record["Date"],
        },
    ) for chunk in text_chunks]

    return nodes

def github_url_to_table_name(github_url):
    repository_path = github_url.replace("https://github.com/", "")
    table_name = "li_"+repository_path.replace("/", "_")
    return table_name

def record_catalog_info(repo):
    with psycopg2.connect(dsn=st.secrets["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            # Define the Git catalog table creation SQL command
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS time_machine_catalog (
                repo_url TEXT PRIMARY KEY,
                table_name TEXT
            );
            """
            cursor.execute(create_table_sql)

            delete_sql = "DELETE FROM time_machine_catalog WHERE repo_url = %s"
            cursor.execute(delete_sql, (repo,))

            insert_data_sql = """
            INSERT INTO time_machine_catalog (repo_url, table_name)
            VALUES (%s, %s);
            """
            
            table_name = github_url_to_table_name(repo)
            cursor.execute(insert_data_sql, (repo, table_name))
            return table_name


def load_into_db(table_name, df_combined):
    embedding_model = OpenAIEmbedding()
    embedding_model.api_key = st.secrets["OPENAI_API_KEY"]

    ts_vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name=table_name,
        time_partition_interval=timedelta(days=365),
    )

    ts_vector_store._sync_client.drop_table()
    ts_vector_store._sync_client.create_tables()

    cpus = cpu_count()
    min_splits = len(df_combined.index) / 1000 #no more than 1000 rows/split
    num_splits = int(max(cpus, min_splits))


    st.spinner("Processing...")
    progress = st.progress(0, f"Processing, with {num_splits} splits")
    start = time.time()

    nodes_combined = [item for sublist in [create_nodes(row) for _, row in df_combined.iterrows()] for item in sublist]
    node_tasks = np.array_split(nodes_combined, num_splits)
    
    def worker(nodes): 
        start = time.time()
        texts = [n.get_content(metadata_mode="all") for n in nodes] 
        embeddings = embedding_model.get_text_embedding_batch(texts)
        for i, node in enumerate(nodes):
            node.embedding = embeddings[i]
        duration_embedding = time.time()-start
        start = time.time()
        ts_vector_store.add(nodes)
        duration_db = time.time()-start
        return (duration_embedding, duration_db)

    embedding_durations = []
    db_durations = []
    with ThreadPoolExecutor() as executor:
        times = executor.map(worker, node_tasks)

        for index, worker_times in enumerate(times):
            duration_embedding, duration_db = worker_times
            embedding_durations.append(duration_embedding)
            db_durations.append(duration_db)
            progress.progress((index+1)/num_splits, f"Processing, with {num_splits} splits")


    progress.progress(100, f"Processing embeddings took {sum(embedding_durations)}s. Db took {sum(db_durations)}s. Using {num_splits} splits")
    
    st.spinner("Creating the index...")
    progress = st.progress(0, "Creating the index")
    start = time.time()
    ts_vector_store.create_index()
    duration = time.time()-start
    progress.progress(100, f"Creating the index took {duration} seconds")
    st.success("Done")

def get_history(repo, branch, limit): 
    st.spinner("Fetching git history...")
    start = time.time()
    progress = st.progress(0, "Fetching git history")
    # Clean up any existing "tmprepo" directory
    shutil.rmtree("tmprepo", ignore_errors=True)

    # Clone the Git repository with the specified branch
    res = subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            "--branch=" + branch,
            repo + ".git",
            "tmprepo",
        ],
        capture_output=True,
        text=True,
        cwd=".",  # Set the working directory here
    )

    if res.returncode != 0:
        st.error("Error running Git \n\n"+str(res.stderr))
        raise ValueError(f"Git failed: {res.returncode}")

    repo = Repo('tmprepo')

    # Create lists to store data
    commit_hashes = []
    authors = []
    dates = []
    subjects = []
    bodies = []

    # Iterate through commits and collect data
    for commit in repo.iter_commits():
        commit_hash = commit.hexsha
        author = commit.author.name
        date = commit.committed_datetime.isoformat()
        message_lines = commit.message.splitlines()
        subject = message_lines[0]
        body = "\n".join(message_lines[1:]) if len(message_lines) > 1 else ""

        commit_hashes.append(commit_hash)
        authors.append(author)
        dates.append(date)
        subjects.append(subject)
        bodies.append(body)

    # Create a DataFrame from the collected data
    data = {
        "Commit Hash": commit_hashes,
        "Author": authors,
        "Date": dates,
        "Subject": subjects,
        "Body": bodies
    }

    df = pd.DataFrame(data)

    # Light data cleaning on DataFrame
    df = df.astype(str)
    if limit > 0:
        df = df[:limit]

    duration = time.time()-start
    progress.progress(100, f"Fetching git history took {duration} seconds")
    return df


def load_git_history():
    repo = st.text_input("Repo", "https://github.com/postgres/postgres")
    branch = st.text_input("Branch", "master")
    limit = int(st.text_input("Limit number commits (0 for no limit)", "1000"))
    if st.button("Load data into the database"):
        df = get_history(repo, branch, limit)
        table_name = record_catalog_info(repo)
        load_into_db(table_name, df)

st.set_page_config(page_title="Load git history", page_icon="💿")
st.markdown("# Load git history for analysis")
st.sidebar.header("Load git history")
st.write(
    """Load Git history!"""
)
if  st.secrets.get("ENABLE_LOAD") == 1:
    load_git_history()
else:
    st.warning("Loading is disabled on the demo site. Please follow the instructions in the [README](https://github.com/cevian-streamlit/tsv-timemachine/tree/main) to enable loading.")
#show_code(tm_demo)
