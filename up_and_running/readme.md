
# pgvector and Timescale Vector Up and Running

These examples will provide a barebones introduction to pgvector and Timescale Vector.

We will be starting with a CSV file containing git commits that were made to the timescaledb git repository. Each record in the CSV contains an ID, author, date, commit (hash), summary, and details.

Sample:

```text
┌───────┬──────────────────────┬─────────────────────┬──────────────────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  id   │        author        │        date         │        commit        │       summary        │                                                     details                                                     │
│ int64 │       varchar        │      timestamp      │       varchar        │       varchar        │                                                     varchar                                                     │
├───────┼──────────────────────┼─────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│   145 │ Zoltan Haindrich     │ 2023-05-12 06:30:41 │ ab2cccb6e2b10008c2…  │ Post-release 2.11.0  │ Adjust the upgrade/downgrade scripts and add the tests.  (cherry picked from commit d5fea0a842cbd38d2d72db16e…  │
│   239 │ Fabrízio de Royes …  │ 2023-04-04 20:31:33 │ 6440bb3477eef18345…  │ Remove unused func…  │ Remove unused function `invalidation_threshold_htid_found`.                                                     │
│   305 │ Maheedhar PV         │ 2023-02-21 17:33:46 │ c8c50dad7eca4f7425…  │ Post-release fixes…  │ Bumping the previous version and adding tests for 2.10.0                                                        │
│   308 │ Sven Klemm           │ 2023-02-20 12:31:19 │ 09766343997aa903f9…  │ Set name for COPY …  │ Having the hash table named makes debugging easier as the name is used for the MemoryContext used by the hash…  │
│   379 │ Mats Kindahl         │ 2023-01-16 08:24:32 │ 8f4fa8e4cca73f11d3…  │ Add build matrix t…  │ Build matrix is missing from the ignore workflows for the Windows and Linux builds, so this commit adds them.   │
└───────┴──────────────────────┴─────────────────────┴──────────────────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Using this CSV file, we will:

1. generate embeddings of the git commits using OpenAI
2. store the git commits and their embeddings in a Timescale database
3. perform semantic similarity search on the embeddings in the database
4. perform semantic similarity search while filtering on time
5. perform semantic similarity search while filtering on time AND filtering on metadata
6. doing retrieval augmented generation using similarity search 

This will demonstrate the functionality unlocked by using postgres, pgvector, timescale_vector, and timescaledb together for such workloads.

The data will be stored in this postgres table:

```text
                      Table "public.commit_history"
┌───────────┬──────────────────────────┬───────────┬──────────┬─────────┐
│  Column   │           Type           │ Collation │ Nullable │ Default │
├───────────┼──────────────────────────┼───────────┼──────────┼─────────┤
│ id        │ integer                  │           │          │         │
│ date      │ timestamp with time zone │           │ not null │         │
│ metadata  │ jsonb                    │           │          │         │
│ content   │ text                     │           │          │         │
│ embedding │ vector(1536)             │           │          │         │
└───────────┴──────────────────────────┴───────────┴──────────┴─────────┘
Indexes:
    "commit_history_date_idx" btree (date DESC)
    "commit_history_embedding_idx" tsv (embedding)
Triggers:
    ts_insert_blocker BEFORE INSERT ON commit_history FOR EACH ROW EXECUTE FUNCTION _timescaledb_functions.insert_blocker()
Number of child tables: 94 (Use \d+ to list them.)
```

## Setup

Create and activate a python virtual environment. Install the dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Head over to the Timescale console and create a service using the "Time Series an Analytics" service time. Choose a region near you.

https://console.cloud.timescale.com/login

Create a `.env` file in this project directory. Add the URL for your Timescale service and your OpenAI API key as below.

```bash
TIMESCALE_SERVICE_URL="postgres://<user>:<password>@<hostname>:<port>/tsdb?sslmode=require"
OPENAI_API_KEY="<your key here>"
```

## Examples

### 0_embed.py

In the [0_embed.py](./0_embed.py) script, we will generate embeddings for the git commits using OpenAI. Then, we will create a table, load it with our data, and build a vector index on the embeddings.

```bash
./0_embed.py
```

### 1_similarity_search.py

In the [1_similarity_search.py](./1_similarity_search.py) script, we will use the table and index created in [0_embed.py](./0_embed.py) to search for git commits that are semantically relevant to a user's question.

```bash
./1_similarity_search.py
```

### 2_similarity_search_with_time.py

The [2_similarity_search_with_time.py](./2_similarity_search_with_time.py) script builds on the previous script. We will still search for git commits that are semantically relevant to a user's question, but we will additionally constrain our results to commits that are more recent than a user-provided date. This time filtering utilizes the power of hypertables -- a foundational feature of the timescaledb extension.

```bash
./2_similarity_search_with_time.py
```

### 3_similarity_search_with_time_author.py

The [3_similarity_search_with_time_author.py](./3_similarity_search_with_time_and_author.py) script will extend the prior script to additionally filter by metadata -- in this case filtering by the commit's author. In a single SQL query, we can do semantic search, time filtering, and metadata filtering.

```bash
./3_similarity_search_with_time_author.py
```

### 4_rag.py

Finally, the [4_rag.py](./4_rag.py) script uses the prior work to demonstrate retrieval-augmented generation. We use similarity search with time filtering to retrieve commits relevant to a user's query. Then, we construct a prompt that is augmented with additional context -- the relevant commits' information. This prompt is presented to an LLM which generates a text response that we display to the user.

```bash
./4_rag.py
```

