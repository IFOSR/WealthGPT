from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

import os, glob

from config import openai_key


def vectorlization(directory_path, dbname, collection_name):
    """
    build vector and persist in disk
    :param directory_path of files to be vectorised
    :param dbname: persist db name
    :param collection_name: persist collection name
    :return: None
    """
    loader = DirectoryLoader(directory_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=dbname, collection_name=collection_name)
    vectorstore.persist()


def does_vectorstore_exist(persist_directory):
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def build(companyId):
    dbname = f"company/{companyId}_db"
    collection_name = f"{companyId}_col"

    if does_vectorstore_exist(dbname):
        # skip
        print(f"{dbname} already exists")
        pass
    else:
        vectorlization(f"company/{companyId}", dbname, collection_name)
        print(f"Ingestion complete! You can now chat with your document")



if __name__ == '__main__':

    dirlist = os.listdir('./company')
    for d in dirlist:
        print(d)
        if d[-2:] == 'db':
            print('not target dir, pass')
            continue
        build(d)



    # directory_path = 'carbon_projects'
    # dbname = 'carbon_projects_db'
    # collection_name = 'redd_projects_db'
    # directory_path = 'INPO'
    # dbname = 'INPO_db'
    # collection_name = 'inpo_db_col'
