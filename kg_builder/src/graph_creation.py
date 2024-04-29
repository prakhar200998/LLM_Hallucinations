from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from knowledge_graph_builder import extract_and_store_graph
from langchain.schema import Document
from dotenv import load_dotenv
from tqdm import tqdm
import os

# Load environment variables
load_dotenv()

# Define articles to load
articles = {
    "Chemotherapy": "Chemotherapy",
    "Traffic Law": "Traffic laws in the United States"
}

def build_graph_for_article(article_name, category):
    print(f"Loading documents for: {article_name}")
    # Load and process the Wikipedia article
    raw_documents = WikipediaLoader(query=article_name).load()
    if not raw_documents:
        print(f"Failed to load content for {article_name}")
        return
    
    text_splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=96)
    documents = text_splitter.split_documents(raw_documents[:5])  # Only process the first 5 documents

    print("Building the knowledge graph...")
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(document, category)

def main():
    for category, title in articles.items():
        build_graph_for_article(title, category)

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
