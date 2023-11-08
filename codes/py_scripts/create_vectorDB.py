import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import time

DATA_PATH = "../../data/foodON.csv"
CHUNK_SIZE=650
CHUNK_OVERLAP=200
BATCH_SIZE=200
SENTENCE_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_NAME="../../data/vectorDB/foodON"


def main():
    start_time = time.time()
    data_df = pd.read_csv(DATA_PATH)
    data = list(data_df.food_name)
    metadata_list = list(data_df.food_id)
    metadata_list_of_dict = []
    for i in metadata_list:
        metadata_list_of_dict.append({"foodON_ID": i})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.create_documents(data, metadatas=metadata_list_of_dict)
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    vectorstore = Chroma(embedding_function=SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL), 
                         persist_directory=VECTOR_DB_NAME)
    for batch in batches:
        vectorstore.add_documents(documents=batch)
    end_time = round((time.time() - start_time)/(60*60), 2)
    print("VectorDB is created in {} hrs".format(end_time))


if __name__ == "__main__":
    main()