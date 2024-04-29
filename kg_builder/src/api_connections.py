
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
        prompt_text = f"""# Knowledge Graph Instructions for GPT-4
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
    """

    elif category == "Traffic Law":
        # Traffic Law-specific prompt
        prompt_text = f"""# Knowledge Graph Instructions for GPT-4
    ## 1. Overview
    You are a sophisticated algorithm tailored for parsing Wikipedia pages to construct a knowledge graph about traffic laws and regulations in the United States.
    - **Nodes** symbolize entities such as types of traffic violations, penalties, driving regulations, and relevant legal statutes.
    - The goal is to create a precise and comprehensible knowledge graph, serving as a reliable resource for legal professionals, law enforcement agencies, and the general public.

    ## 2. Labeling Nodes
    - **Consistency**: Utilize uniform labels for node types to maintain clarity.
    - For instance, consistently label violations as **"Violation"**, penalties as **"Penalty"**, and statutes as **"Statute"**.
    - **Node IDs**: Apply descriptive, legible identifiers for node IDs, sourced directly from the text.
    {'- **Allowed Node Labels:**' + ", ".join(['Violation', 'Penalty', 'Statute', 'VehicleType', 'LegalDocument']) if allowed_nodes else ""}
    {'- **Allowed Relationship Types**:' + ", ".join(['Violates', 'Penalizes', 'Governs', 'Cites']) if allowed_rels else ""}

    ## 3. Handling Numerical Data and Dates
    - Integrate numerical data and dates as attributes of the corresponding nodes.
    - **No Isolated Nodes for Dates/Numbers**: Directly associate dates and numerical figures as attributes with pertinent nodes.
    - **Property Format**: Follow a straightforward key-value pattern for properties, with keys in camelCase, for example, `fineAmount`, `lawEffectiveDate`.

    ## 4. Coreference Resolution
    - **Entity Consistency**: Guarantee uniform identification of each entity across the graph.
    - For example, if "Vehicle Code 22350" and "Speed Law" reference the same statute, uniformly apply "Vehicle Code 22350" as the node ID.

    ## 5. Relationship Naming Conventions
    - **Clarity and Standardization**: Utilize clear and standardized relationship names, preferring uppercase with underscores for readability.
    - For instance, use "IS_PENALIZED_BY" instead of "ISPENALIZEDBY", use "IS_GOVERNED_BY" instead of "ISGOVERNEDBY" etc. You keep making the same mistakes of storing the relationships without the "_" in between the words. Any further similar errors will lead to termination.
    - **Relevance and Specificity**: Choose relationship names that accurately reflect the connection between nodes, such as "REQUIRES" or "PROHIBITS" for legal requirements or prohibitions.

    ## 6. Strict Compliance
    Rigorous adherence to these instructions is essential. Failure to comply with the specified formatting and labeling norms will necessitate output revision or discard.
    """

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
