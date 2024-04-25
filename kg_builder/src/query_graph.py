from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from api_connections import graph  # Importing 'graph' from 'api_connections.py'

def query_knowledge_graph(query):
    print("Refreshing the graph schema...")
    # Refresh the graph schema before querying
    graph.refresh_schema()

    print("Setting up the Cypher QA Chain...")
    # Setup the Cypher QA Chain with specific LLM configurations
    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        #verbose=True
    )

    print(f"Executing the query: {query}")
    # Execute the query and return results
    result = cypher_chain.invoke({"query": query})
    print("Query executed. Processing results...")
    return result
