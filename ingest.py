from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.document_stores import WeaviateDocumentStore
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack import Pipeline

print("Import Successfully")

path_doc = ["Data/2024-conocophillips-proxy-statement.pdf"]

# Initialize Weaviate Document Store with correct settings
document_store = WeaviateDocumentStore(
    host='http://localhost',
    port=8080,
    embedding_dim=768
)

print("Document Store: ", document_store)
print("#####################")

converter = PyPDFToDocument()
print("Converter: ", converter)
print("#####################")

# Convert PDF into documents
output = converter.run(paths=path_doc)
docs = output["documents"]
print("Docs: ", docs)
print("#####################")

# Prepare documents for processing
final_doc = []
for doc in docs:
    print(doc.text)
    new_doc = {
        'content': doc.text,
        'meta': doc.metadata
    }
    final_doc.append(new_doc)
    print("#####################")

# Pre-process the documents
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)

print("Preprocessor: ", preprocessor)
print("#####################")

preprocessed_docs = preprocessor.process(final_doc)
print("Preprocessed Docs: ", preprocessed_docs)
print("#####################")

# Write processed documents to Weaviate Document Store
document_store.write_documents(preprocessed_docs)

# Initialize the Embedding Retriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Retriever: ", retriever)

# Update embeddings in Weaviate Document Store
document_store.update_embeddings(retriever)

print("Embeddings Done.")
