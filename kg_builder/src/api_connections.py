
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models import KnowledgeGraph
from typing import Optional, List


load_dotenv()  # This loads the variables from .env into os.environ

def get_graph_connection(category):
    if category == "Chemotherapy":
        url = os.getenv("CHEMO_NEO4J_URL")
        username = os.getenv("CHEMO_NEO4J_USERNAME")
        password = os.getenv("CHEMO_NEO4J_PASSWORD")
    elif category == "Traffic Law":
        url = os.getenv("TRAFFIC_NEO4J_URL")
        username = os.getenv("TRAFFIC_NEO4J_USERNAME")
        password = os.getenv("TRAFFIC_NEO4J_PASSWORD")
    else:
        raise ValueError(f"Unknown category: {category}")

    return Neo4jGraph(url=url, username=username, password=password)

openai_api_key = os.getenv("OPENAI_API_KEY")

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables.")
    return ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

def get_extraction_chain(
    category,
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    if category == "Chemotherapy":
        # Chemotherapy-specific prompt
        prompt_text = ""
    elif category == "Traffic Law":
        # Traffic Law-specific prompt
        prompt_text = "[Traffic Law-specific instructions]"
    else:
        raise ValueError("Unknown category")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [(
                    "system",prompt_text),
                    ("human", "Use the given format to extract information from the following input: {input}"),
                    ("human", "Tip: Precision in the node and relationship creation is vital for the integrity of the knowledge graph."),
        ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt)
