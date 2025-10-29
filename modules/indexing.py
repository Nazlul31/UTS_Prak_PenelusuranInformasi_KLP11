import os
import pandas as pd
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer

def create_search_index(data_dir="data", index_dir="indexdir"):
    """
    Membuat index Whoosh dari semua file *_clean.csv di folder data/
    """
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    schema = Schema(
        id=ID(stored=True),
        judul=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        clean_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        sumber=TEXT(stored=True)  # tambahan field sumber
    )

    ix = create_in(index_dir, schema)
    writer = ix.writer()

    csv_files = [f for f in os.listdir(data_dir) if f.endswith("_clean.csv")]
    for file in csv_files:
        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"Gagal baca {file}: {e}")
            continue

        for _, row in df.iterrows():
            writer.add_document(
                id=str(row.get("row_id", "")),  # menggunakan row_id unik
                judul=str(row.get("judul", "")),
                clean_text=str(row.get("clean_text", "")),
                sumber=str(row.get("sumber", row.get("file", file)))  # ambil sumber dari kolom sumber/file
            )

    writer.commit()
    print(f"Index berhasil dibuat di folder: {index_dir}")