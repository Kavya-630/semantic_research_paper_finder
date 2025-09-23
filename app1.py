import streamlit as st
import json
import pandas as pd

# Load dataset (replace with your dataset file)
@st.cache_data(hash_funcs={str: lambda _: None})
def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data(r"C:\Users\Navya sree\Downloads\arxiv-metadata-oai-snapshot.json")



# Simple search function
def search_papers(df, query):
    results = df[df['title'].str.contains(query, case=False, na=False) |
                 df['abstract'].str.contains(query, case=False, na=False)]
    return results

# Streamlit UI
def main():
    st.set_page_config(page_title="Semantic Research Paper Finder", layout="wide")
    st.title("📚 Semantic Research Paper Finder")

    df = load_data()

    query = st.text_input("🔎 Enter your search query")
    if query:
        results = search_papers(df, query)
        st.write(f"Found {len(results)} matching papers:")
        st.dataframe(results[['title', 'authors', 'year']])

        for _, row in results.iterrows():
            with st.expander(row['title']):
                st.write(f"**Authors:** {row['authors']}")
                st.write(f"**Year:** {row['year']}")
                st.write(f"**Abstract:** {row['abstract']}")
                if 'url' in row:
                    st.markdown(f"[Read Paper]({row['url']})")

if __name__ == "__main__":
    main()
