from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from knowledge_graph_builder import extract_and_store_graph
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# IMPORTANT: Make sure data source names match with values inside api_connections.py
# Define articles / topics to load
#articles = {
#    "Chemotherapy": "Chemotherapy",
#    "Traffic Law": "Traffic laws in the United States"
#}
# Switzerland: https://www.fedlex.admin.ch/eli/cc/1962/1364_1409_1420/de
# Connecticut: https://en.wikipedia.org/wiki/Transportation_in_Connecticut#Rules_of_the_road
articles = {
    "Traffic Law": "Traffic laws in the United States"
}

def build_graph_for_article(query, data_source_name):
    """
    Build knowledge graph from loaded articles / documents of a particular topic
    :param query: The query string to search on Wikipedia, e.g. "Traffic laws in the United States"
    :param data_source_name: Data source name, e.g. "Traffic Law"
    :return:
    """
    load_max_documents = 5
    #chunk_size=4096
    #chunk_overlap=96
    chunk_size=400
    chunk_overlap=10

    print(f"Loading document(s) from Wikipedia using query '{query}' ...")
    raw_documents = WikipediaLoader(query=query, load_max_docs=load_max_documents).load()
    if not raw_documents:
        print(f"Failed to load content for query: {query}")
        return

    print(f"{str(len(raw_documents))} document(s) loaded from Wikipedia.")
    for doc in raw_documents:
        print(f"Document: {doc.metadata['source']}")
        #print(f"Document: {doc.page_content}")

    print(f"Split document(s) into chunk(s) (Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}) ...")
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunkDocs = text_splitter.split_documents(raw_documents[:load_max_documents])  # Only process the first 5 documents
    print(f"{str(len(raw_documents))} document(s) split into {str(len(chunkDocs))} chunk(s)")

    print(f"Building the knowledge graph for document(s) found by query '{query}' ...")
    for i, chunkDoc in tqdm(enumerate(chunkDocs), total=len(chunkDocs)):
        print(f"Extract data from chunk {str(i)} ...")
        #print(f"Extract data from chunk {str(i)}: {chunkDoc.page_content}")
        extract_and_store_graph(chunkDoc, data_source_name)

def main():
    for data_source_name, query in articles.items():
        build_graph_for_article(query, data_source_name)

if __name__ == "__main__":
    main()

# import os
# from openai import OpenAI
# from api_connections import get_graph_connection
# from knowledge_graph_builder import extract_and_store_graph
# from query_graph import query_knowledge_graph
# from langchain_community.document_loaders import WikipediaLoader
# from langchain.text_splitter import TokenTextSplitter
# from tqdm import tqdm

# def get_llm():
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("No OpenAI API key found in environment variables.")
#     return OpenAI(api_key=api_key)

# def classify_query(query):
#     llm = get_llm()
#     response = llm.Completion.create(
#         model="text-davinci-003",  # Consider updating to the latest model as necessary
#         prompt=f"Classify the following query into 'Chemotherapy' or 'Traffic Law': {query}",
#         max_tokens=60
#     )
#     return response.choices[0].text.strip()

# def main():
#     print("Starting the script...")
#     # Take Wikipedia article name as input
#     article_name = input("Enter the Wikipedia article name: ")  

#     print(f"Loading documents for: {article_name}")
#     # Load and process the Wikipedia article
#     raw_documents = WikipediaLoader(query=article_name).load()
#     text_splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=96)
#     documents = text_splitter.split_documents(raw_documents[:5])  # Only process the first 5 documents

#     print("Building the knowledge graph...")
#     # Build the knowledge graph from the documents
#     for i, d in tqdm(enumerate(documents), total=len(documents)):
#         extract_and_store_graph(d)

#     print("Graph construction complete. Please enter your query.")
#     # Take a query related to the graph
#     user_query = input("Enter your query related to the graph: ")

#     print(f"Querying the graph with: {user_query}")
#     # Query the graph and print the answer
#     answer = query_knowledge_graph(user_query)
#     print("Answer to your query:", answer)

# if __name__ == "__main__":
#     main()
