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


# This script will extend the prior script to additionally filter by metadata 
# -- in this case filtering by the commit's author. In a single SQL query, we can
# do semantic search, time filtering, and metadata filtering.
#
# If you need a good question to ask, try this:
# > describe how changes to decompression have improved performance


_ = load_dotenv(find_dotenv())

TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

openai.api_key  = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()


def similarity_search(question: str, since: datetime, author: str, k=5) -> list[dict]:
    # turn the question into an embedding/vector using the openai client
    embedding = client.embeddings.create(input = [question], model="text-embedding-3-small").data[0].embedding
    matches = []
    # connect to the database and search for relevant commits while filtering on time and author
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
                where "date" >= %s::timestamptz                   -- time based filtering
                and metadata @> jsonb_build_object('author', %s)  -- metadata filtering
                order by embedding <=> %s::vector                 -- order by semantic similarity
                limit %s                                          -- only return the k most similar
                """, (since, author, embedding, k))
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
        author = click.prompt("Enter the author to limit results to", type=str)
        click.echo("Searching...")
        matches = similarity_search(question, since, author)
        click.echo(f"Here are the {len(matches)} most similar rows since {since} by {author}:")
        print_results(matches)
        click.echo("\n\n")
        if not click.confirm('Do you want to continue?'):
            break

