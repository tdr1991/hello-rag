#coding:utf-8

from pathlib import Path
from llama_index.readers.file.pymu_pdf import PyMuPDFReader

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.llama_cpp import LlamaCPP

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader

from llama_index.core.schema import TextNode


import chromadb

from vector import VectorDBRetriever

gemma_2b = Ollama(model="gemma:2b", request_timeout=600)

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

text_parser = SentenceSplitter(chunk_size=1024)

text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_chunks)
    doc_idxs.extend([doc_idx] * len(cur_chunks))
    
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)



emb_model = HuggingFaceEmbedding(model_name = "/mnt/d/work/models/bge-small-en-v1.5")

for node in nodes:
    node_embedding = emb_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

client = chromadb.EphemeralClient()
conn = client.create_collection(name="rag")

vector_store = ChromaVectorStore.from_collection(collection=conn)


vector_store.add(nodes)


# storage_context = StorageContext.from_defaults(vector_store=vec_store)

# vec_index = VectorStoreIndex(documents, embed_model=emb_model, storage_context=storage_context)

# vec_retriever = VectorIndexRetriever(index=vec_index)

retriever = VectorDBRetriever(
    vector_store, emb_model, query_mode="default", similarity_top_k=2
)

q_engine = RetrieverQueryEngine.from_args(retriever, llm=gemma_2b)


# q_engine = vec_index.as_query_engine(llm=gemma_2b, similarity_top_k=4)

query_str = "How does Llama 2 perform compared to other open-source models?"

res = q_engine.query(query_str)

print(res)