""" Load and Preprocess the Text Files"""

import os

# Path to the folder containing your news text files
folder_path = "C:/Users/vivek/OneDrive/Desktop/langchain/news_data"

# Read all text files and store them in a list
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())

news_documents = []
for i in documents:
    if i != "Failed to retrieve the webpage.":
        news_documents.append(i)
        




# Split the document into chunks

from langchain.text_splitter import CharacterTextSplitter

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",       # Split on newlines or any other delimiter
    chunk_size=1000,       # Maximum size of each chunk in characters
    chunk_overlap=200      # Overlap between chunks to maintain context
)

all_chunks = []
for i in news_documents:
    chunks = text_splitter.split_text(i)
    all_chunks.extend(chunks)



"""

# Check if vector store contains entries; avoid re-adding on re-runs
if len(all_chunks) > 0:
    ids = [f"doc_{i}" for i in range(len(all_chunks))]
    db.add_texts(all_chunks, ids=ids)
    print(f"Added {len(all_chunks)} new documents.")
"""




""" Generate Embeddings for News Documents """

# Define the embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small" , api_key = "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")

# Load the existing vector store with the embedding function
from langchain_chroma import Chroma

db = Chroma(persist_directory="C:/Users/vivek/OneDrive/Desktop/langchain/db/chroma_db",
            embedding_function=embeddings)


ids = [f"doc_{i}" for i in range(len(all_chunks))]  # Generate unique IDs
db.add_texts(all_chunks, ids=ids)



#Retreival
# Retrieve relevant documents based on the query

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)


# Here’s how you can use the retriever to search based on a query:
#query = "Latest advancements in AI"
#results = retriever.get_relevant_documents(query)

# Print the retrieved documents
#for i, result in enumerate(results, 1):
#    print(f"\nResult {i}:\n{result.page_content}")



# Create a ChatOpenAI model
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", api_key = "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")


query = "what is happening between israel"

relevant_docs = retriever.invoke(query)


# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question and answer in not more than 4 lines: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)



# Define the messages for the model
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]




# Invoke the model with the combined input
result = llm.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)

