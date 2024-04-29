
from api_connections import get_graph_connection

from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
from models import Node, Relationship, KnowledgeGraph
from utils import map_to_base_node, map_to_base_relationship
from api_connections import get_extraction_chain

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_runnable,
    create_structured_output_chain,
)


def extract_and_store_graph(
    document: Document,
    category: str,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    
    graph = get_graph_connection(category)
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(category, nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])

