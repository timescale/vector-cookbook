#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import openai
import psycopg2
import click


# In the this script, we will generate embeddings for the git commits using 
# OpenAI. Then, we will create a table, load it with our data, and build a vector
# index on the embeddings.
#
# We will be starting with a CSV file containing git commits that were made to 
# the timescaledb git repository. Each record in the CSV contains an ID, author,
# date, commit (hash), summary, and details.
#
# Sample of the CSV file:
# ┌───────┬──────────────────────┬─────────────────────┬──────────────────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
# │  id   │        author        │        date         │        commit        │       summary        │                                                     details                                                     │
# │ int64 │       varchar        │      timestamp      │       varchar        │       varchar        │                                                     varchar                                                     │
# ├───────┼──────────────────────┼─────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
# │   145 │ Zoltan Haindrich     │ 2023-05-12 06:30:41 │ ab2cccb6e2b10008c2…  │ Post-release 2.11.0  │ Adjust the upgrade/downgrade scripts and add the tests.  (cherry picked from commit d5fea0a842cbd38d2d72db16e…  │
# │   239 │ Fabrízio de Royes …  │ 2023-04-04 20:31:33 │ 6440bb3477eef18345…  │ Remove unused func…  │ Remove unused function `invalidation_threshold_htid_found`.                                                     │
# │   305 │ Maheedhar PV         │ 2023-02-21 17:33:46 │ c8c50dad7eca4f7425…  │ Post-release fixes…  │ Bumping the previous version and adding tests for 2.10.0                                                        │
# │   308 │ Sven Klemm           │ 2023-02-20 12:31:19 │ 09766343997aa903f9…  │ Set name for COPY …  │ Having the hash table named makes debugging easier as the name is used for the MemoryContext used by the hash…  │
# │   379 │ Mats Kindahl         │ 2023-01-16 08:24:32 │ 8f4fa8e4cca73f11d3…  │ Add build matrix t…  │ Build matrix is missing from the ignore workflows for the Windows and Linux builds, so this commit adds them.   │
# └───────┴──────────────────────┴─────────────────────┴──────────────────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#
# We will use OpenAI to generate embeddings for each CSV record.
#
# Then, we will load this data into a postgres table, convert it to a hypertable,
# and build a vector index on the embeddings.
#
# The table will look like this:
#
#                       Table "public.commit_history"
# ┌───────────┬──────────────────────────┬───────────┬──────────┬─────────┐
# │  Column   │           Type           │ Collation │ Nullable │ Default │
# ├───────────┼──────────────────────────┼───────────┼──────────┼─────────┤
# │ id        │ integer                  │           │          │         │
# │ date      │ timestamp with time zone │           │ not null │         │
# │ metadata  │ jsonb                    │           │          │         │
# │ content   │ text                     │           │          │         │
# │ embedding │ vector(1536)             │           │          │         │
# └───────────┴──────────────────────────┴───────────┴──────────┴─────────┘
# Indexes:
#     "commit_history_date_idx" btree (date DESC)
#     "commit_history_embedding_idx" tsv (embedding)
# Triggers:
#     ts_insert_blocker BEFORE INSERT ON commit_history FOR EACH ROW EXECUTE FUNCTION _timescaledb_functions.insert_blocker()
# Number of child tables: 94 (Use \d+ to list them.)


_ = load_dotenv(find_dotenv())
TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
openai.api_key  = os.environ['OPENAI_API_KEY']

client = openai.OpenAI()


def read_csv(path="commit_history.csv"):
    records: list[dict] = []
    with open(path) as f:
        r = csv.reader(f)
        r = csv.DictReader(f, fieldnames=["id", "author", "date", "commit", "summary", "details"])
        for row in r:
            # we'll have a jsonb column in the database containing objects with these keys
            metadata = json.dumps({
                "author": row["author"],
                "date": row["date"],
                "commit": row["commit"],
                "summary": row["summary"],
                "details": row["details"],
            })
            # concatenate multiple fields to be our content
            content = " ".join([row["author"], row["date"], row["commit"], row["summary"], row["details"]])
            records.append({
                "id": row["id"],
                "date": row["date"],
                "metadata": metadata,
                "content": content,
            })
    return records


def embed(records: list[dict]):
    with click.progressbar(records, label="embedding...", show_eta=False, show_pos=True) as bar:
        for record in bar:
            content = record["content"]
            content = content.replace("\n", " ")
            # ask openai for a vector representation of the content
            embedding = client.embeddings.create(input = [content], model="text-embedding-3-small").data[0].embedding
            record["embedding"] = embedding


def write_embedded_csv(records: list[dict], path="commit_history_embedded.csv"):
    with open(path, mode="w") as f:
        w = csv.DictWriter(f,fieldnames=["id", "date", "metadata", "content", "embedding"])
        w.writerows(records)


def read_embedded_csv(path="commit_history_embedded.csv") -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        r = csv.DictReader(f, fieldnames=["id", "date", "metadata", "content", "embedding"])
        for row in r:
            records.append(row)
    return records


def load_db(records: list[dict]) -> None:
    with psycopg2.connect(TIMESCALE_SERVICE_URL) as con:
        with con.cursor() as cur:
            # create the extensions
            cur.execute("create extension if not exists vector") # pgvector
            cur.execute("create extension if not exists timescale_vector")
            cur.execute("create extension if not exists timescaledb")
            # create the hypertable
            print("creating hypertable...")
            cur.execute("drop table if exists commit_history")
            cur.execute("""
                create table commit_history
                ( id int
                , "date" timestamptz
                , metadata jsonb
                , content text             -- the content that was embedded
                , embedding vector(1536)   -- vector type from pgvector extension stores the embedding
                )
                """)
            # transform the plain table into a hypertable. this functionality is from the timescaledb extension
            cur.execute("select create_hypertable('commit_history', by_range('date', interval '1 month'))")
            # insert the records into the hypertable
            with click.progressbar(records, label="inserting...", show_eta=False, show_pos=True) as bar:
                for record in bar:
                    cur.execute("""
                        insert into commit_history (id, date, metadata, content, embedding)
                        values (%(id)s, %(date)s, %(metadata)s, %(content)s, %(embedding)s)
                        """, record)
            con.commit()
            print("creating vector index...")
            # create a tsv index on our vector data. this index type is from the timescale_vector extension
            cur.execute("create index on commit_history using tsv (embedding)")
            con.commit()


if __name__ == "__main__":
    gen_embeddings = True
    if Path("commit_history_embedded.csv").exists():
        gen_embeddings = click.confirm("regenerate embeddings?")
    if gen_embeddings:
        print("reading commit_history.csv")
        records = read_csv()
        embed(records)
        write_embedded_csv(records)
    else:
        records = read_embedded_csv()
    print("loading database...")
    load_db(records)
    print("done")

