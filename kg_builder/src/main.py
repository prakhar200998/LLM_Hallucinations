import os
from openai import OpenAI
from api_connections import get_graph_connection
from knowledge_graph_builder import extract_and_store_graph
from query_graph import query_knowledge_graph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables.")
    return OpenAI(api_key=api_key)

def classify_query(query):
    llm = get_llm()
    response = llm.Completion.create(
        model="text-davinci-003",  # Consider updating to the latest model as necessary
        prompt=f"Classify the following query into 'Chemotherapy' or 'Traffic Law': {query}",
        max_tokens=60
    )
    return response.choices[0].text.strip()

def main():
    print("Starting the script...")

    # Get user query
    query = input("Please enter your query: ")
    
    # Classify the query
    category = classify_query(query)
    print(f"Query classified into category: {category}")
    
    # Get the correct graph connection
    graph = get_graph_connection(category)
    
    # Query the correct graph
    result = query_knowledge_graph(graph, query)
    print(f"Query result: {result}")

if __name__ == "__main__":
    main()
