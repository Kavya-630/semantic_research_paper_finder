import streamlit as st
import json
import pandas as pd

# Load dataset (replace with your dataset file)
import pandas as pd

@st.cache_data
def load_data():
    # Load the smaller sample CSV that is inside your GitHub repo
    df = pd.read_csv("arxiv_sample.csv")
    return df

df = load_data()




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
    st.write("Columns in dataset:", df.columns.tolist())  # 👈 debug

    query = st.text_input("🔎 Enter your search query")

    if query:
        results = search_papers(df, query)
        st.write(f"Found {len(results)} matching papers:")

        # 👇 ADD THIS CONTROL (slider to choose number of papers)
        max_results = st.slider("How many papers to display?", 
                                min_value=1, 
                                max_value=min(len(results), 50), 
                                value=min(10, len(results)))

        # Show only the top N results
        results = results.head(max_results)

        # Show only existing columns
        cols_to_show = [c for c in ['title', 'authors', 'year'] if c in results.columns]
        st.dataframe(results[cols_to_show])

        for _, row in results.iterrows():
            with st.expander(row.get('title', 'No Title')):
                if 'authors' in row:
                    st.write(f"**Authors:** {row['authors']}")
                if 'year' in row:
                    st.write(f"**Year:** {row['year']}")
                if 'abstract' in row:
                    st.write(f"**Abstract:** {row['abstract']}")
                if 'url' in row:
                    st.markdown(f"[Read Paper]({row['url']})")


if __name__ == "__main__":
    main()
