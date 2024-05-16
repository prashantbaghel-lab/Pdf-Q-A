from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import openai
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader

from  set_up import AstraCS_DB_Application_Token, Astra_DB_ID, open_api_key




pdfreader=PdfReader("JD - AI Engineer.pdf")

from typing_extensions import Concatenate

raw_text=""
for i , page in enumerate(pdfreader.pages):
    content=page.extract_text()
    if content:
        raw_text+=content
print(raw_text)


cassio.init(token=AstraCS_DB_Application_Token, database_id=Astra_DB_ID)

llm=OpenAI(api_key=open_api_key)
embedding=OpenAIEmbeddings(api_key=open_api_key)

astra_vector_store=Cassandra(
    embedding=embedding,
    table_name="qa",
    session=None,
    keyspace=None
)

from langchain.text_splitter import CharacterTextSplitter

text_splitter= CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function= len
)

texts= text_splitter.split_text(raw_text)


astra_vector_store.add_texts(texts[:50])
print("Instered %i  headline." % len(texts[:50]))
astra_vector_index =VectorStoreIndexWrapper(vectorstore=astra_vector_store)


first_question=True
while True:
    if first_question:
        query_text=input("\nEnter your question (or type 'quit' to exit):").strip()
    else:
        query_text=input("\n What's your next question (or type 'quit' to exit):").strip()
    if query_text.lower()=="quit":
        break
    first_question=False
    print("\nQuestion: \" %s\"" % query_text)
    answer=astra_vector_index.query(query_text, llm=llm)
    print("Answer:\"%s\"\n" % answer)
    print("First Documents By Relevance:")
    for doc , score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("  [%0.4f]\" %s ...\"" %(score, doc.page_content[:84]))
