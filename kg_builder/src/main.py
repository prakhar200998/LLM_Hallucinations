import os
from openai import OpenAI
from api_connections import get_graph_connection
from query_graph import query_knowledge_graph

import openai

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables.")
    openai.api_key = api_key
    return openai

def classify_query(query):
    llm = get_llm()
    try:
        response = llm.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": f"Classify the following query into 'Chemotherapy' or 'Traffic Law': {query}"}],
            max_tokens=60
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error during classification: {e}")
        return None


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
