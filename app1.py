import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv("arxiv_sample.csv")
    return df

@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

@st.cache_resource
def load_embeddings(df, model):
    # Suppose df["abstract"] has text
    abstracts = df["abstract"].tolist()
    embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

@st.cache_resource
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def semantic_search(query, model, index, df, embeddings, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k=top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        paper = df.iloc[idx]
        results.append({
            "title": paper["title"],
            "abstract": paper["abstract"],
            "categories": paper.get("categories", ""),
            "year": paper.get("year", ""),
            "score": float(dist)
        })
    return results

def main():
    st.title("Semantic Research Paper Finder")

    # Load data
    df = load_data("arxiv_sample.csv")
    model = load_embedding_model()
    embeddings = load_embeddings(df, model)
    index = build_faiss_index(embeddings)

    # Sidebar controls
    top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)
    year_filter = st.sidebar.slider("Year range", int(df["year"].min()), int(df["year"].max()), (2000, 2025))
    category_filter = st.sidebar.multiselect("Categories", sorted(set(df["categories"].apply(lambda s: s.split()).sum())))  # need adjust

    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        with st.spinner("Searching..."):
            results = semantic_search(query, model, index, df, embeddings, top_k=top_k)
            # Filter results
            filtered = []
            for r in results:
                yr = int(r["year"])
                if yr < year_filter[0] or yr > year_filter[1]:
                    continue
                if category_filter:
                    # check if any category in r["categories"] matches
                    paper_cats = r["categories"].split()
                    if not any(cat in paper_cats for cat in category_filter):
                        continue
                filtered.append(r)
            
            for res in filtered:
                st.markdown(f"**Title:** {res['title']}  \n" +
                            f"**Year:** {res['year']} — **Categories:** {res['categories']}  \n" +
                            f"**Score:** {res['score']:.4f}")
                # show first 300 chars of abstract
                st.write(res["abstract"][:300] + "..." if len(res["abstract"]) > 300 else res["abstract"])
                st.write("---")

if __name__ == "__main__":
    main()
