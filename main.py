import os
import pandas as pd
from modules.preprocessing import preprocess_text
from modules.indexing import create_search_index
from modules.search import search_with_ranking
from whoosh import index
from tqdm import tqdm

DATA_DIR = "datasets"
CLEAN_DIR = "datasets_clean"
INDEX_DIR = "indexdir"

os.makedirs(CLEAN_DIR, exist_ok=True)

CANDIDATE_CONTENT_COLS = [
    "text", "content", "abstract", "abstrak", "isi", "body", "article",
    "judul_dan_isi", "clean_text", "dokumen", "teks"
]
CANDIDATE_TITLE_COLS = ["title", "judul", "headline"]

def pick_first_existing(cands, df):
    for c in cands:
        if c in df.columns:
            return c
    return None

def combine_title_content(row, title_col, content_col):
    parts = []
    if title_col and pd.notna(row.get(title_col, None)):
        parts.append(str(row[title_col]))
    if content_col and pd.notna(row.get(content_col, None)):
        parts.append(str(row[content_col]))
    return " - ".join(parts) if parts else ""

def load_documents_from_csvs():
    docs, meta, titles = [], [], []

    if not os.path.isdir(DATA_DIR):
        print(f'Folder "{DATA_DIR}" tidak ditemukan.')
        return docs, meta, titles

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"Tidak ada file .csv di folder {DATA_DIR}/")
        return docs, meta, titles

    for csv_name in sorted(csv_files):
        dataset_name = os.path.splitext(csv_name)[0]
        clean_path = os.path.join(CLEAN_DIR, f"{dataset_name}_clean.csv")

        # Jika sudah ada, beritahu dan load langsung
        if os.path.exists(clean_path):
            df_clean = pd.read_csv(clean_path, sep=None, engine="python", on_bad_lines='skip')
            docs.extend(df_clean["clean_text"].tolist())
            titles.extend(df_clean["judul"].tolist())
            meta.extend(df_clean.to_dict("records"))
            print(f"Memuat hasil preprocessing: {clean_path}")
            continue

        # Kalau belum ada, lakukan preprocessing dan simpan
        fpath = os.path.join(DATA_DIR, csv_name)
        try:
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="latin-1")

            df.columns = [c.strip().lower() for c in df.columns]
            title_col = pick_first_existing(CANDIDATE_TITLE_COLS, df)
            content_col = pick_first_existing(CANDIDATE_CONTENT_COLS, df)

            if not content_col:
                print(f"Tidak ditemukan kolom teks di {csv_name}")
                continue

            clean_records = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Memproses {csv_name}"):
                raw = combine_title_content(row, title_col, content_col).strip()
                if not raw:
                    continue
                cleaned = preprocess_text(raw)
                title = str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else f"{dataset_name}_row{i}"
                sumber = os.path.splitext(csv_name)[0]
                docs.append(cleaned)
                titles.append(title)
                clean_records.append({
                    "judul": title,
                    "clean_text": cleaned,
                    "dataset": dataset_name,
                    "file": csv_name,
                    "row_id": i,
                    "sumber": sumber
                })

            df_clean = pd.DataFrame(clean_records)
            df_clean.to_csv(clean_path, index=False, encoding="utf-8")
            print(f"Hasil preprocessing disimpan ke: {clean_path}")
            meta.extend(clean_records)

        except Exception as e:
            print(f"Gagal memproses {csv_name}: {e}")

    return docs, meta, titles

def search_from_index(query):
    if not os.path.exists(INDEX_DIR):
        print("Index belum dibuat. Jalankan menu [1] dulu untuk membuat index.")
        return

    ix = index.open_dir(INDEX_DIR)
    results = search_with_ranking(ix, query)

    print("\n=== Hasil Pencarian ===")
    if not results:
        print("Tidak ada hasil ditemukan.")
        print("=========================================================\n")
        return

    for rank, (judul, sumber, skor) in enumerate(results, start=1):
        print(f"{rank}. {judul} (score: {skor:.4f})")
        print(f"   Sumber: {sumber}\n")
    print("=========================================================\n")

def main():
    print("=== INFORMATION RETRIEVAL SYSTEM ===")
    print("[1] Load & Index Dataset (CSV)")
    print("[2] Search Query (Whoosh + Cosine)")
    print("[3] Exit")
    print("====================================")

    while True:
        choice = input("Pilih menu: ").strip()

        if choice == "1":
            print("Memproses dataset...")
            load_documents_from_csvs()  # akan buat dataset bersih jika belum ada
            create_search_index(data_dir=CLEAN_DIR, index_dir=INDEX_DIR)
            print("\nDataset selesai diproses dan diindeks.\n")

        elif choice == "2":
            query = input("Masukkan query: ").strip()
            if not query:
                print("Query tidak boleh kosong.")
                continue
            search_from_index(query)

        elif choice == "3":
            print("Keluar dari program.")
            break

        else:
            print("Pilihan tidak valid. Coba lagi.")

if __name__ == "__main__":
    main()
