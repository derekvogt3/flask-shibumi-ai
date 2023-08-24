
from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS
from flask import Blueprint, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
import os
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os

app = Flask(__name__)
load_dotenv()

embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

pinecone.init(api_key=os.getenv("PINECONE_API"),environment=os.getenv("PINECONE_ENV"))

index = pinecone.Index(index_name='shibumi-retrieval-agent')
vectorstore = Pinecone(index, embeddings_model.embed_query,"text")

@app.route('/', methods=['POST'])
def index():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing from request body"}), 400
    vectorstore.similarity_search(query=query, k=3)
    llm = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'),model_name='gpt-3.5-turbo', temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    return jsonify({"message": qa.run(data)}), 200

if __name__ == '__main__':
  app.run(port=5000)
