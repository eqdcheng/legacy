import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import time
import json

load_dotenv()

@st.cache_resource
def load_model():
    # NOTE: all-MiniLM-L6-v2 chosen for optimal speed/quality tradeoff (5x faster than BERT with 90% quality)
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data_and_embeddings():
    df = pd.read_csv('processed_conversations.csv')
    embeddings = np.load('embeddings.npy')
    
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # TODO: add more data preprocessing and simple validation
    # TODO: implement data versioning with timestamp in metadata
    
    return df, embeddings, metadata

@st.cache_resource
def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv('ANTHROPIC_API_KEY'))
    if not api_key:
        st.error("Please set ANTHROPIC_API_KEY in Streamlit secrets or .env file")
        st.stop()
    return Anthropic(api_key=api_key)

def find_similar_conversations(query, df, embeddings, model, top_k=3, min_threshold=0.2):
    if not query or len(query.strip()) < 3:
        return []
    
    # NOTE: truncate at 2000 chars (~400 words) to prevent embedding quality degradation
    if len(query) > 2000:
        query = query[:2000]
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # TODO: implement diversity sampling to avoid returning very similar examples
    # TODO: can use semantic clustering to group similar responses
    
    results = []
    for idx in top_indices:
        if similarities[idx] >= min_threshold:
            results.append({
                'context': df.iloc[idx]['Context'],
                'response': df.iloc[idx]['Response'],
                'similarity': similarities[idx],
                'category': df.iloc[idx]['category'],
                'quality_score': df.iloc[idx]['response_quality_score']
            })
    
    # NOTE: always return at least one result for user feedback - even low similarity 
    # cases can provide useful therapeutic principles
    if not results and len(top_indices) > 0:
        idx = top_indices[0]
        results.append({
            'context': df.iloc[idx]['Context'],
            'response': df.iloc[idx]['Response'],
            'similarity': similarities[idx],
            'category': df.iloc[idx]['category'],
            'quality_score': df.iloc[idx]['response_quality_score'],
            'low_confidence': True
        })
    
    return results

def generate_counselor_guidance(user_input, similar_examples, client, temperature=0.3):
    # NOTE: sort by quality first, then similarity - quality score indicates therapeutic best practices
    sorted_examples = sorted(similar_examples, 
                           key=lambda x: (x['quality_score'], x['similarity']), 
                           reverse=True)
    
    examples_text = ""
    for i, example in enumerate(sorted_examples, 1):
        examples_text += f"""
Example {i} (Similarity: {example['similarity']:.1%}, Category: {example['category']}):
Patient: {example['context']}
Counselor: {example['response']}
---"""
    
    # TODO: add prompt variation based on detected crisis keywords (suicide, self-harm, abuse)
    
    prompt = f"""You are an experienced clinical supervisor helping a mental health counselor.

Based on these similar counseling conversations from experienced practitioners:
{examples_text}

The counselor needs guidance for this situation:
"{user_input}"

Provide specific, actionable guidance that:
1. Identifies key therapeutic principles demonstrated in the examples
2. Suggests concrete response strategies tailored to this situation
3. Highlights important clinical considerations and potential risks
4. Recommends specific therapeutic techniques or interventions

Keep advice evidence-based and practical. Focus on what the counselor should actually say or do."""

    # NOTE: temperature 0.3 default keeps responses focused while allowing some variation
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

def main():
    st.set_page_config(
        page_title="Counselor Guidance Tool",
        page_icon="🧠",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .stTextArea textarea {font-size: 16px;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Mental Health Counselor Guidance Tool")
    st.markdown("Get evidence-based guidance by finding similar cases from experienced counselors.")
    
    # load resources
    with st.spinner("Loading resources..."):
        model = load_model()
        df, embeddings, metadata = load_data_and_embeddings()
        client = get_anthropic_client()
    
    # show dataset info
    with st.expander("About this tool"):
        st.markdown(f"""
        - **Database**: {metadata['total_conversations']} counseling conversations
        - **Categories**: {', '.join(metadata['categories'].keys())}
        - **Approach**: Finds similar cases using semantic search, then generates personalized guidance
        """)

    def set_user_input(text):
        st.session_state.user_input = text

    # main input
    user_input = st.text_area(
        "Describe the patient's situation or challenge:",
        height=150,
        placeholder="Example: My teenage client is struggling with social anxiety and avoiding school. They haven't attended in 2 weeks and parents are frustrated...",
        key='user_input' 
    )
    
    # advanced options
    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.slider("Similar cases to retrieve:", 1, 5, 3)
        # NOTE: 0.2 threshold from EDA - balances relevance with always having results
        min_confidence = st.slider("Minimum similarity:", 0.1, 0.5, 0.2)
        temperature = st.slider("Response creativity:", 0.0, 0.5, 0.3, 0.1,
                                help="Lower = more focused, Higher = more creative")
    
    if st.button("Get Guidance", type="primary", disabled=not user_input):
        start_time = time.time()
        
        with st.spinner("Finding similar cases..."):
            similar_cases = find_similar_conversations(
                user_input, df, embeddings, model, 
                top_k=top_k, min_threshold=min_confidence
            )
        
        if not similar_cases:
            st.warning("No similar cases found. Try rephrasing your query.")
            return
        
        # display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recommended Guidance")
            
            if any(case.get('low_confidence') for case in similar_cases):
                st.info("Note: Similarity scores are lower than usual. The guidance may be less specific to your situation.")
            
            with st.spinner("Generating personalized guidance..."):
                try:
                    guidance = generate_counselor_guidance(user_input, similar_cases, client, temperature)
                    st.markdown(guidance)
                    
                    # TODO: add "Copy to Clipboard" button for easy integration into session notes
                    # TODO: implement suggested follow-up questions based on the guidance
                    
                except Exception as e:
                    st.error(f"Error generating guidance: {str(e)}")
                    st.info("Showing similar cases as reference:")
                    
                    # NOTE: fallback ensures counselors always have actionable information
                    best_case = max(similar_cases, key=lambda x: x['similarity'])
                    st.markdown(f"**Most similar case response:**\n\n{best_case['response']}")
        
        with col2:
            st.subheader("Similar Cases")
            for i, case in enumerate(similar_cases, 1):
                # NOTE: star indicates comprehensive responses (2+ therapeutic elements)
                quality_emoji = "⭐" if case.get('quality_score', 0) >= 2 else ""
                
                with st.expander(
                    f"Case {i} - {case['category']} "
                    f"({case['similarity']:.0%} match) {quality_emoji}"
                ):
                    st.markdown("**Patient Situation:**")
                    st.text(case['context'][:500] + "..." if len(case['context']) > 500 else case['context'])
                    
                    st.markdown("**Counselor Response:**")
                    st.text(case['response'][:500] + "..." if len(case['response']) > 500 else case['response'])
                    
                    st.caption(f"Response quality score: {case.get('quality_score', 'N/A')}/3")
        
        # metrics and feedback
        st.divider()
        processing_time = time.time() - start_time
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{processing_time:.1f}s")
        with col2:
            st.metric("Similar Cases Found", len(similar_cases))
        with col3:
            avg_similarity = np.mean([c['similarity'] for c in similar_cases]) if similar_cases else 0
            st.metric("Avg Similarity", f"{avg_similarity:.0%}")
        
        # feedback section
        st.markdown("Was this guidance helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes, helpful"):
                st.success("Thank you for your feedback!")
                # TODO: save query and positive feedback to improve future recommendations
        with col2:
            if st.button("👎 Could be better"):
                st.info("Thank you. Try adding more details to your query.")
                # TODO: log failed queries for manual review and dataset expansion
    
    # sidebar with example queries
    with st.sidebar:
        st.header("Example Queries")
        # NOTE: examples cover diverse age groups, settings, and clinical presentations
        example_queries = [
            "My client is dealing with severe anxiety about returning to work after medical leave",
            "Teenage patient with depression refuses to engage in therapy sessions",
            "Couple experiencing communication breakdown after infidelity",
            "Child showing signs of ADHD but parents refuse evaluation",
            "Client with trauma history having panic attacks during sessions"
        ]
        
        st.markdown("Click any example to use it:")

        for eq in example_queries:
            st.button(
                eq, 
                key=eq, 
                use_container_width=True,
                on_click=set_user_input,
                args=(eq,)
            )
        
        # TODO: add category filter to search within specific mental health domains

if __name__ == "__main__":
    main()