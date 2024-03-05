#!/usr/bin/env python3
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import openai
import psycopg2
from psycopg2.extras import DictCursor
import click
from rich.console import Console
from rich.table import Table


# In this script, we will use the table and index created in 0_embed.py to search
# for git commits that are semantically relevant to a user's question.
#
# The `similarity_search` function takes a user-provided question, embeds it 
# using OpenAI, and then uses a single SQL query to find the 5 git commits most
# semantically relevant to the user's question.
#
# If you need a good question to ask, try this:
# > describe how changes to decompression have improved performance


_ = load_dotenv(find_dotenv())

TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

openai.api_key  = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()


def similarity_search(question: str, k=5) -> list[dict]:
    # turn the question into an embedding/vector using the openai client
    embedding = client.embeddings.create(input = [question], model="text-embedding-3-small").data[0].embedding
    matches = []
    # connect to the database and search for relevant commits
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
                order by embedding <=> %s::vector   -- order by semantic similarity
                limit %s                            -- only return the k most similar
                """, (embedding, k))
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
        click.echo("Searching...")
        matches = similarity_search(question)
        click.echo(f"Here are the {len(matches)} most similar rows:")
        print_results(matches)
        click.echo("\n\n")
        if not click.confirm('Do you want to continue?'):
            break

