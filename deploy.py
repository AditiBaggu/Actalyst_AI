import streamlit as st
import pandas as pd
import numpy as np
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch the OpenAI API key
def fetch_api_key():
    url = "http://52.66.239.27:8504/get_keys"
    email = {"email": "aditi.baggu_2025@woxsen.edu.in"}
    response = requests.post(url, json=email)
    if response.status_code == 200:
        return response.json().get('key')
    else:
        raise Exception("Failed to fetch API key")

# Set the OpenAI API key
openai.api_key = fetch_api_key()

# Load the embeddings
news_df = pd.read_pickle('aluminium_news_embeddings.pkl')

# Streamlit app
st.title("💬 Aluminium Industry News Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with aluminium industry news today?"}]

# Function to get embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Function to get the most relevant article
def get_most_relevant_article(query):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity(
        [query_embedding],
        news_df['embedding'].tolist()
    )
    most_relevant_idx = np.argmax(similarities)
    return news_df.iloc[most_relevant_idx]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box for user prompt
if prompt := st.chat_input("Ask me about the latest news in the aluminum industry:"):
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    try:
        # Get the most relevant article
        article = get_most_relevant_article(prompt)
        response_message = f"**Title:** {article['title']}\n\n**Summary:** {article['summary']}\n\n**Date:** {article['date']}"
    except Exception as e:
        response_message = f"Error: {str(e)}"

    # Add assistant message to session state
    st.session_state["messages"].append({"role": "assistant", "content": response_message})
    st.chat_message("assistant").write(response_message)
