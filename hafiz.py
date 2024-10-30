import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_chroma import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="persist", embedding_function=embedding_model)

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/", glob="**/*.json", loader_cls=JSONLoader,
    loader_kwargs={"jq_schema": ".", "text_content": False}, show_progress=True)
  if PERSIST:
    index = VectorstoreIndexCreator(
      vectorstore=vectorstore,
      vectorstore_kwargs={"persist_directory":"persist"},
      embedding=embedding_model).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

llm = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'exit']:
    sys.exit()
  llm_result = llm.invoke({"question": query, "chat_history": chat_history})
  print(llm_result['answer'])

  print("llm_result.usage_metadata")
  print(llm_result.usage_metadata)

  chat_history.append((query, llm_result['answer']))
  query = None