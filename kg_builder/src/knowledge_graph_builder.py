import logging

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

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_and_store_graph(
    document: Document,
    data_source_name: str,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    """
    Extract data from text document and add nodes and relationships to knowledge graph
    :param document: Text document
    :param data_source_Name: Data source name, e.g. "Traffic Law"
    :param nodes: TODO
    :param rels: TODO
    """

    logger.info("Extract graph data using OpenAI functions ...")
    extract_chain = get_extraction_chain(data_source_name, nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )

    # Store information into a graph
    graph = get_graph_connection(data_source_name)
    graph.add_graph_documents([graph_document])

