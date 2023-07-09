import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

pinecone.init(api_key="016f225a-0ae8-4915-8aac-ed2b6143eb72", environment="us-west4-gcp-free")
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    loader = TextLoader("/Users/monisha.rajan/Projects/langchain/vector-db-intro/mediumblogs/blog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-embeddings-index")

    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query":query})
    print(result)