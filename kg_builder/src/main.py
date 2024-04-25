from knowledge_graph_builder import extract_and_store_graph
from query_graph import query_knowledge_graph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm

def main():
    print("Starting the script...")
    # Take Wikipedia article name as input
    article_name = input("Enter the Wikipedia article name: ")  # Corrected to proper input usage

    print(f"Loading documents for: {article_name}")
    # Load and process the Wikipedia article
    raw_documents = WikipediaLoader(query=article_name).load()
    text_splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=96)
    documents = text_splitter.split_documents(raw_documents[:5])  # Only process the first 5 documents

    print("Building the knowledge graph...")
    # Build the knowledge graph from the documents
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(d)

    print("Graph construction complete. Please enter your query.")
    # Take a query related to the graph
    user_query = input("Enter your query related to the graph: ")

    print(f"Querying the graph with: {user_query}")
    # Query the graph and print the answer
    answer = query_knowledge_graph(user_query)
    print("Answer to your query:", answer)

if __name__ == "__main__":
    main()
