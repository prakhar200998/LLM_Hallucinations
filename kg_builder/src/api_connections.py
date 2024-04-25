
from langchain_community.graphs import Neo4jGraph
import os

# Neo4j connection setup
url = "neo4j+s://2f409740.databases.neo4j.io"
username = "neo4j"
password = "oe7A9ugxhxcuEtwci8khPIt2TTdz_am9AYDx1r9e9Tw"
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

# OpenAI API key setup
os.environ["OPENAI_API_KEY"] = "sk-proj-hceIL56CC2zfjAvAlMjbT3BlbkFJyHKX2wbiQxsG9yy8dGJN"
