from whoosh.qparser import MultifieldParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_with_ranking(ix, query, top_n=5):
    """
    Melakukan pencarian pada index Whoosh dan perangkingan dengan cosine similarity.
    """
    with ix.searcher() as searcher:
        parser = MultifieldParser(["judul", "clean_text"], ix.schema)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=None)

        if not results:
            print("Tidak ditemukan hasil untuk query tersebut.")
            return []

        docs = [r['clean_text'] for r in results]
        titles = [r['judul'] for r in results]
        sources = [r['sumber'] for r in results]  # ambil sumber

        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])

        cosine_scores = cosine_similarity(query_vec, doc_vectors).flatten()
        ranked_indices = cosine_scores.argsort()[::-1][:top_n]

        # kembalikan judul, sumber, dan skor
        ranked_results = [(titles[i], sources[i], cosine_scores[i]) for i in ranked_indices]
        return ranked_results
