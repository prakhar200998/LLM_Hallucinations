
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env into os.environ

# Now use os.getenv to access your variables
url = os.getenv("NEO4J_URL")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)


