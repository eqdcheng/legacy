# Mental Health Counselor Guidance Tool

A POC web application that helps mental health counselors during sessions.

## Overview

This tool addresses a critical need in mental health counseling: providing evidence-based guidance when counselors face challenging clinical situations. This project leverages a few ML engineering topics:

- **Semantic Similarity**: Finds the most relevant historical counseling cases using cosine similarity
- **RAG**: Provides actionable recommendations based on similar cases
- **Quality Scoring**: Prioritizes counselor responses that demonstrate best practices. Sort of like a data filtering model

### Design Decisions
- **Embedding Model**: MiniLM provides 5x speed improvement over BERT with 90% quality retention
- **Similarity Threshold (0.2)**: Based on EDA showing this captures ~9% of results while filtering noise
- **Temperature Default (0.3)**: Balances consistency with appropriate variation for clinical advice
- **Always return results**: Even low-confidence matches can provide valuable therapeutic principles
- **Single File Architecture**: All functionality in `app.py` for easier review and deployment
- **Dataset**: Chose a dataset that would be quick to work with, but there are definetly better ones

## Technical Highlights

### Embedding Strategy
* Pre-compute embeddings for all conversations (potentially a database)
* Cache embeddings (NumPy) for fast loading
* Use cosine similarity for semantic matching

### Quality Scoring System
Responses are scored 0-3 based on therapeutic best practices:
* Contains question (+1): Encourages patient engagement
* Contains validation (+1): Shows empathy and understanding
* Contains advice (+1): Provides actionable guidance

### Performance Optimizations
* Streamlit caching for model and data loading
* Batch embedding computation in preprocessing
* Truncation of long queries to maintain embedding quality

## Future Enhancements

### Next iteration immediate ideas
* Copy to Clipboard: Easy integration with session notes
* Category Filtering: Search within specific domains (anxiety, depression, etc.)
* Follow-up Questions: Generate relevant prompts based on guidance
* Export Functionality: Save guidance as PDF for documentation

### Medium-term Improvements
* Diversity Sampling: find unique cases and also reduce similar cases
* Crisis Detection: Flag high-risk situations (self-harm, suicide) with special handling
* Feedback Loop: Store successful queries to improve recommendations
* Multi-dataset Support: Integrate additional counseling conversation sources - could use a huge database!

### Long-term ideas
* Session Progress Tracking: Monitor patient journey across sessions
* Collaborative Filtering: Recommend based on similar counselor preferences
