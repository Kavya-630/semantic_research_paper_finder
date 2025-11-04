#!/usr/bin/env python3
"""
Load arXiv dataset from Google Drive using gdown and run your research paper finder.
"""

import os
import gdown
from sentence_transformers import SentenceTransformer
from your_module import load_arxiv_data, embed_texts, extract_keywords, search_papers  # update this import

# Google Drive file ID (replace this with your own)
FILE_ID = "1aBcD123EFgHiJKLmn0PQrsTuVWxyzAB"
OUTPUT_PATH = "arxiv-metadata-oai-snapshot.json"

if __name__ == "__main__":
    # Download dataset from Google Drive if not already present
    if not os.path.exists(OUTPUT_PATH):
        print("Downloading dataset from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_PATH, quiet=False)
    else:
        print("Dataset already exists locally.")

    print("Loading dataset...")
    papers = load_arxiv_data(OUTPUT_PATH, max_entries=500000)

    print("Generating embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_texts(papers, embed_model)

    print("Extracting keywords...")
    papers = extract_keywords(papers)

    # Interactive search
    while True:
        query = input("\nEnter your research query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = search_papers(query, papers, embeddings, embed_model, top_k=5)
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Title: {res['title']}")
            print(f"Authors: {res['authors']}")
            print(f"Categories: {res['categories']}")
            print(f"Similarity Score: {res['similarity']:.4f}")
            print(f"Abstract: {res['abstract']}")
            if 'summary' in res:
                print(f"Summary: {res['summary']}")
