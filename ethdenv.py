import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from typing import List
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="ETHDenver Event Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("ETHDenver Event Finder 🦬")
st.write("Find the perfect events to attend at ETHDenver based on your interests and schedule.")

# Check for API key in secrets
if 'openai_api_key' not in st.secrets:
    st.error("OpenAI API key not found. Please set it in the Streamlit secrets.")
    st.stop()

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

# Initialize models
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.1)


def load_and_process_data():
    """Load and process the ETHDenver events data"""
    df = pd.read_csv("ethdenver_events.csv")

    # Create documents for vectorstore
    documents = []
    for _, row in df.iterrows():
        content = f"""
        Event: {row['Event']}
        Date: {row['Date']}
        Time: {row['Time']}
        Organizer: {row['Organizer']}
        Registration: {row['Registration']}
        """
        documents.append({"content": content, "metadata": dict(row)})

    return df, documents


def create_vectorstore(documents):
    """Create FAISS vectorstore from documents"""
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def analyze_query(query: str, vectorstore) -> dict:
    """Analyze query and find matching events"""
    template = """You are an expert event coordinator for ETHDenver. Help find and recommend the most relevant events based on the following query:

    Query: {query}

    Based on the provided context about ETHDenver events, identify the top 3 most relevant events and explain why they would be good matches.
    Pay special attention to:
    1. The relevance to the query topic/interest
    2. Time and date convenience
    3. The event organizer's reputation

    Context about available events:
    {context}

    Please provide your response in the following JSON format:
    {{
        "matches": [
            {{
                "event": "Name of event",
                "datetime": "Date and time of event",
                "organizer": "Name of organizer",
                "registration": "Registration link",
                "rationale": "2-3 sentence explanation of why this event matches the query"
            }}
        ],
        "summary": "2-3 sentence overall summary of the recommendations"
    }}
    """

    prompt = PromptTemplate.from_template(template)
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query, "context": context})

    return json.loads(result)


# Load data
with st.spinner("Loading event database..."):
    df, documents = load_and_process_data()
    vectorstore = create_vectorstore(documents)

# Query input
st.subheader("What are you looking for?")
query = st.text_area(
    "Enter your query about ETHDenver events",
    height=100,
    placeholder="Example: 'What blockchain gaming events are happening?' or 'What events should I attend if I'm interested in DeFi?'"
)

# Date filter
st.subheader("Date Filter (Optional)")
selected_date = st.selectbox(
    "Filter events by date",
    options=["All Dates"] + sorted(df["Date"].unique().tolist())
)

if st.button("Find Events"):
    if query:
        with st.spinner("Searching for relevant events..."):
            results = analyze_query(query, vectorstore)

            # Display results
            st.subheader("🎯 Recommended Events")

            # Summary
            st.info(results["summary"])

            # Matches
            for i, match in enumerate(results["matches"], 1):
                # Skip if date filter is active and doesn't match
                if selected_date != "All Dates" and selected_date not in match["datetime"]:
                    continue

                with st.expander(f"#{i} - {match['event']}", expanded=True):
                    st.write("**Why this event?**")
                    st.write(match["rationale"])

                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Event Details:**")
                        st.write(f"**Date & Time:** {match['datetime']}")
                        st.write(f"**Organizer:** {match['organizer']}")
                    with cols[1]:
                        st.write("**Registration:**")
                        st.markdown(f"[Register Here]({match['registration']})")

            # Export options
            st.download_button(
                "Export Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name="event_recommendations.json",
                mime="application/json"
            )
    else:
        st.warning("Please enter a query first.")

# Sidebar
with st.sidebar:
    st.subheader("About")
    st.write("""
    This tool helps you discover and plan your ETHDenver experience by finding the most relevant events based on your interests.

    It uses:
    - AI-powered event matching
    - Natural language understanding
    - Smart scheduling recommendations

    To get the best results:
    1. Be specific about your interests
    2. Mention any time preferences
    3. Include any specific topics you're interested in
    4. Specify if you're looking for networking opportunities
    """)

    st.markdown("---")
    st.caption("ETHDenver 2025 Event Finder")