#!/usr/bin/env python3
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import openai
import psycopg2
import click


# This script uses the prior work to demonstrate retrieval-augmented generation. 
# We use similarity search with time filtering to retrieve commits relevant to a 
# user's query. Then, we construct a prompt that is augmented with additional 
# context -- the relevant commits' information. This prompt is presented to an 
# LLM which generates a text response that we display to the user.
#
# If you need a good question to ask, try this:
# > describe how changes to decompression have improved performance


_ = load_dotenv(find_dotenv())

TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

openai.api_key  = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()


def similarity_search(question: str, k=5) -> list[str]:
    # turn the question into an embedding/vector using the openai client
    embedding = client.embeddings.create(input = [question], model="text-embedding-3-small").data[0].embedding
    matches = []
    # connect to the database and search for relevant commits while filtering on time and author
    with psycopg2.connect(TIMESCALE_SERVICE_URL) as con:
        with con.cursor() as cur:
            cur.execute("""
                select content
                from commit_history
                order by embedding <=> %s::vector   -- order by semantic similarity
                limit %s                            -- only return the k most similar
                """, (embedding, k))
            for row in cur.fetchall():
                matches.append(row[0])
    return matches


def generate_response(question: str, matches: list[str]) -> str:
    # construct a prompt
    matches = "\n* ".join(matches)
    prompt = f"""
    Use the git commit records from the timescaledb git repository to answer the subseqent question.
    Do not describe the commits individually. Provide an overall summary to address the question.

    Git Commit Records:
    {matches}

    Question: {question}
    """
    # ask the GPT to respond to the prompt
    response = client.chat.completions.create(
        messages=[
            {
                'role': 'system', 
                'content': 'You answer questions about the git commit history for the timescaledb repository.'
            },
            {'role': 'user', 'content': prompt},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    # return the GPT's response
    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        # 1. get the user's question
        question = click.prompt("Enter your question", type=str)
        # 2. do a similarity search for git commits relevant to the question
        click.echo("Searching...")
        matches = similarity_search(question)
        click.echo(f"Found {len(matches)} matches.")
        # 3. provide the relevant commits in a prompt and ask the GPT for a response
        click.echo("Generating response...")
        response = generate_response(question, matches)
        # 4. display the response to the user
        click.echo("\n\n")
        click.echo(response)
        click.echo("\n\n")
        # 5. profit!
        if not click.confirm('Do you want to continue?'):
            break
