
# Add to knowledge_graph_builder.py
from api_connections import graph

from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(..., description="List of relationships in the knowledge graph")

def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    properties["name"] = node.id.title()  # Assuming nodes have an 'id' attribute for this operation
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env into os.environ

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_runnable,
    create_structured_output_chain,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Setting the OpenAI API key for usage in LLM calls
os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
        [(
          "system",
        f"""# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a sophisticated algorithm tailored for parsing Wikipedia pages to construct a knowledge graph about chemotherapy and related cancer treatments.
- **Nodes** symbolize entities such as medical conditions, drugs, symptoms, treatments, and associated medical concepts.
- The goal is to create a precise and comprehensible knowledge graph, serving as a reliable resource for medical practitioners and scholarly research.

## 2. Labeling Nodes
- **Consistency**: Utilize uniform labels for node types to maintain clarity.
  - For instance, consistently label drugs as **"Drug"**, symptoms as **"Symptom"**, and treatments as **"Treatment"**.
- **Node IDs**: Apply descriptive, legible identifiers for node IDs, sourced directly from the text.

{'- **Allowed Node Labels:**' + ", ".join(['Drug', 'Symptom', 'Treatment', 'MedicalCondition', 'ResearchStudy']) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(['Treats', 'Causes', 'Researches', 'Recommends']) if allowed_rels else ""}

## 3. Handling Numerical Data and Dates
- Integrate numerical data and dates as attributes of the corresponding nodes.
- **No Isolated Nodes for Dates/Numbers**: Directly associate dates and numerical figures as attributes with pertinent nodes.
- **Property Format**: Follow a straightforward key-value pattern for properties, with keys in camelCase, for example, `approvedYear`, `dosageAmount`.

## 4. Coreference Resolution
- **Entity Consistency**: Guarantee uniform identification of each entity across the graph.
  - For example, if "Methotrexate" and "MTX" reference the same medication, uniformly apply "Methotrexate" as the node ID.

## 5. Relationship Naming Conventions
- **Clarity and Standardization**: Utilize clear and standardized relationship names, preferring uppercase with underscores for readability.
  - For instance, use "HAS_SIDE_EFFECT" instead of "HASSIDEEFFECT", use "CAN_RESULT_FROM" instead of "CANRESULTFROM" etc. You keep making the same mistakes of storing the relationships without the "_" in between the words. Any further similar errors will lead to termination.
- **Relevance and Specificity**: Choose relationship names that accurately reflect the connection between nodes, such as "INHIBITS" or "ACTIVATES" for interactions between substances.

## 6. Strict Compliance
Rigorous adherence to these instructions is essential. Failure to comply with the specified formatting and labeling norms will necessitate output revision or discard.
        """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Precision in the node and relationship creation is vital for the integrity of the knowledge graph."),
        ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt)

def extract_and_store_graph(
    document: Document,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])

