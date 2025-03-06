from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os  

os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdef1234567890abcdef"  # Replace with your OpenAI API key

loader = TextLoader("belajar_javascript.txt")
docs = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

llm = OpenAI()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

question = "Bagaimana cara belajar Javascript untuk pemula?"
response = qa_chain.run(question)
print(response)

