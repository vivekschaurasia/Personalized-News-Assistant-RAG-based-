"""  Getting the NEWS  """

from dotenv import load_dotenv
import os
import requests
from newspaper import Article

# Load environment variables from .env
load_dotenv()

content = []
query = input(str("What  news do you wanna know? "))

def get_latest_news(query):
    api_key = "API key"  # Replace with your API key
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize=100&sortBy=publishedAt&apiKey={api_key}"
    #url = f"https://www.nbcnews.com/search/?q=israel
    response = requests.get(url)
    data = response.json()

    if data["status"] == "ok":
        print(f"Top 100 latest news articles about {query}:")
        with open("latest_news.txt", "w", encoding="utf-8") as file:
            for i, article in enumerate(data["articles"]):
                title = article['title']
                url = article['url']

                # Fetch full article content using newspaper3k
                try:
                    news_article = Article(url)
                    news_article.download()
                    news_article.parse()
                    article_content = news_article.text
                except Exception as e:
                    article_content = "Unable to retrieve full content."

                # Store the content in the list
                content.append(article_content)

                # Write the article details into the text file
                file.write(f"{i+1}. {title}\n")
                file.write(f"Content: {article_content}\n\n")

        print("News content saved to 'latest_news.txt'.")

    else:
        print(f"Failed to retrieve news: {data.get('message', 'Unknown error')}")

# Run the function
get_latest_news(query)

# Check the length of content (optional)
print(f"Total articles with content: {len(content)}")



# Specify the folder where you want to save the files
folder_path = "news_data"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Loop through the content list and save each item to a separate .txt file in the specified folder
for i, news in enumerate(content):
    # Define the filename with the folder path
    filename = os.path.join(folder_path, f"news_{i+1}.txt")
    
    # Write the content of each news item to a .txt file with utf-8 encoding
    with open(filename, "w", encoding="utf-8") as file:
        file.write(news)

print("All news items have been saved to separate text files in the specified folder.")



import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Define the directory containing the text file and the persistent directory
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "latest_news.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()


    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key = "API key")  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
    
    



# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small" , api_key = "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")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)




# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 70, "score_threshold": 0.1},
)


# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", api_key = "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")


# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)



# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use 7-8 sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


from langchain_core.messages import HumanMessage, SystemMessage


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        # Display only the AI's final response
        print(f"AI: {result['answer']}")
        
        # Update the chat history with the user and AI messages
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
    
    
    
