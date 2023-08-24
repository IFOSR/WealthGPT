# coding: utf-8
# main.py
import os
import time

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import json

from prompt_engineering.financial_prompt import QA_PROMPT
from config import openai_key
from financial_queries import company_queries
from financial_queries import macro_queries
from financial_queries import sector_queries


def chat(knowledge_db, collection_name, chain_type='stuff', verbose=True):

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, openai_api_key=openai_key, max_tokens=512)

    retriever = Chroma(persist_directory=knowledge_db,
                       embedding_function=embeddings, collection_name=collection_name).as_retriever()

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=compression_retriever,
        chain_type_kwargs={"verbose": verbose, "prompt": QA_PROMPT},
    )

    while True:
        query = input("input question ('q' quit, 'clear' clear chat history): ")
        if query.lower() == 'q':
            break
        if query.lower().strip() == '':
            continue

        result = chain({"query": query}, return_only_outputs=True)
        # result = chain({"question": query, "context": chat_history}, return_only_outputs=True)
        # result = chain({"question": query, "chat_history": chat_history}, return_only_outputs=True)
        # chat_history.append((query, result['answer']))

        print("")
        print(result['result'])
        print("")


def query_from_cofig(querys, fd, knowledge_db, collection_name, chain_type='stuff', verbose=False):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, openai_api_key=openai_key)

    retriever = Chroma(persist_directory=knowledge_db,
                       embedding_function=embeddings, collection_name=collection_name).as_retriever()

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=compression_retriever,
        chain_type_kwargs={"verbose": verbose, "prompt": QA_PROMPT},
        return_source_documents=True
    )

    for q in querys:
        print(q)
        result = chain({"query": q})
        print(result['result'])
        format_q_c_a(fd, q, result['result'], result['source_documents'], collection_name)
        time.sleep(5)


def format_q_c_a(fd, query, answer, context, collection_name):
    res = {}
    res['query'] = query
    res['answer'] = answer
    res['context'] = []
    res['collection_name'] = collection_name[:-4]
    for c in context:
        res['context'].append(c.page_content)
    str = json.dumps(res)
    fd.write(f'{str}\n')
    fd.flush()


if __name__ == "__main__":
    dbpath = 'company/db/'
    chain_type = 'stuff'
    result_file = 'result.txt'
    fd = open(result_file, 'w')
    dlist = os.listdir(dbpath)
    for d in dlist:
        print(f'start company {d}')
        knowledge_db = f'{dbpath}{d}'
        collection_name = f'{d[:-3]}_col'
        print('process company queries')
        query_from_cofig(company_queries, fd, knowledge_db, collection_name, chain_type)
        print('process macro queries')
        query_from_cofig(macro_queries, fd, knowledge_db, collection_name, chain_type)
        print('process sector queries')
        query_from_cofig(sector_queries, fd, knowledge_db, collection_name, chain_type)
    print('process over')
    fd.close()
    # chat(f'{dbpath}000001_db', '000001_col', verbose=False)



