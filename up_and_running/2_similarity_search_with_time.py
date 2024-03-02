#!/usr/bin/env python3
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import openai
import psycopg2
from psycopg2.extras import DictCursor
import click
from rich.console import Console
from rich.table import Table


# This script builds on 1_similarity_search.py. We will still search for git 
# commits that are semantically relevant to a user's question, but we will 
# additionally constrain our results to commits that are more recent than a 
# user-provided date. This time filtering utilizes the power of hypertables 
# -- a foundational feature of the timescaledb extension.
#
# Adding a time filter is as simple as adding a WHERE clause that constrains
# results based on the "date" column. This is easy to use but powerful. Behind 
# the scenes, the timescaledb extension "looks" at your filter criteria and is 
# able to exclude any chunks that would not match. In this way, a large amount 
# of the dataset can be ignored and no compute resources wasted on processing 
# those rows.
#
# If you need a good question to ask, try this:
# > describe how changes to decompression have improved performance


_ = load_dotenv(find_dotenv())

TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

openai.api_key  = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()


def similarity_search(question: str, since: datetime, k=5) -> list[dict]:
    # turn the question into an embedding/vector using the openai client
    embedding = client.embeddings.create(input = [question], model="text-embedding-3-small").data[0].embedding
    matches = []
    # connect to the database and search for relevant commits while filtering on time
    with psycopg2.connect(TIMESCALE_SERVICE_URL) as con:
        with con.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(f"""
                select 
                  "date"
                , metadata->>'author' as author
                , metadata->>'commit' as "commit"
                , metadata->>'summary' as summary
                , metadata->>'details' as details
                , content
                from commit_history 
                where "date" >= %s::timestamptz     -- time based filtering
                order by embedding <=> %s::vector   -- order by semantic similarity
                limit %s                            -- only return the k most similar
                """, (since , embedding, k))
            for row in cur.fetchall():
                matches.append({k:v for k, v in row.items()})
    return matches


def print_results(matches: list[dict]) -> None:
    table = Table(title="Matches")
    table.add_column("Date")
    table.add_column("Author")
    table.add_column("Commit")
    table.add_column("Summary")
    table.add_column("Details")
    for match in matches:
        table.add_row(
            datetime.strftime(match["date"], "%Y-%m-%d"),
            match["author"],
            match["commit"],
            match["summary"],
            match["details"])
    console = Console()
    console.print(table)


if __name__ == "__main__":
    while True:
        question = click.prompt("Enter your question", type=str)
        since = click.prompt("Only find results more recent than (YYYY-MM-DD)", type=str)
        since = datetime.strptime(since, "%Y-%m-%d")
        click.echo("Searching...")
        matches = similarity_search(question, since)
        click.echo(f"Here are the {len(matches)} most similar rows since {since}:")
        print_results(matches)
        click.echo("\n\n")
        if not click.confirm('Do you want to continue?'):
            break

