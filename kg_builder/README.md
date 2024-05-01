# Knowledge Graph Builder

## Description
This project builds and queries knowledge graphs from Wikipedia articles using the LangChain library and OpenAI's language models, storing data in a Neo4j database.

## Features
- **Knowledge Graph Construction**: Build graphs from Wikipedia articles.
- **Graph-Based Querying**: Utilize graphs to answer queries with a Graph Cypher QA Chain.
- **Environment Flexibility**: Manages dependencies and environment variables through `.env` files.

## Prerequisites
- Python 3.8+
- pip and virtualenv (optional)
- Access to a Neo4j database
- OpenAI API key
- Extra change

## Installation
1. **Clone the repository**:
    ```bash
    git clone git@hf.co:Master-Thesis-Prakhar/GraphRAG
    cd GraphRAG
    ```
2. **Set up a Python virtual environment (optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your environment variables:**
   - Copy the `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to include your specific configurations such as `OPENAI_API_KEY`, `NEO4J_URL`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.

## Usage

1. **Run the main script:**
    ```bash
    python main.py
    ```

## Contributing

Contributions are welcome! To contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
