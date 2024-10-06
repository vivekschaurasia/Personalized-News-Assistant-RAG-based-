# Personalized News Assistant (RAG-based)


## Overview
The Personalized News Assistant is an AI-driven application that leverages Retrieval-Augmented Generation (RAG) and NLP technologies to retrieve and summarize real-time news articles. It integrates multiple APIs to ensure that users receive personalized, up-to-date news based on their queries. The system is also equipped with a question-answering feature to clarify specific details from the articles, making it a versatile tool for staying informed.

## Key Features
1. Real-Time News Retrieval: Retrieves and processes 100+ real-time news articles per query using NewsAPI and newspaper3k.
2. NLP Summarization: Applies state-of-the-art NLP models to summarize articles with 85% accuracy, ensuring the user gets concise and relevant information.
3. Question-Answering System: Allows users to ask follow-up questions based on the summarized news articles to clarify any doubts.
4. Dynamic Query Adaptation: The system adapts dynamically to user queries, ensuring that the most relevant and up-to-date content is provided.
5. API Security: Environment variables are used to manage API keys and secure the system.


## Technology Stack
* Language: Python
* APIs: NewsAPI, newspaper3k
* NLP: Transformers, LangChain, Summarization Models
* RAG: Retrieval-Augmented Generation framework
* Environment Management: dotenv for API key management
* Version Control: Git


