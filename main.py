import os
import pandas as pd
from modules.preprocessing import preprocess_text
from modules.vectorization import vectorize_documents
from modules.search import search_query
from tqdm import tqdm

DATA_DIR = "datasets"   # folder berisi *.csv dari asisten lab
CLEAN_DIR = "datasets_clean"  # folder untuk hasil preprocessing
os.makedirs(CLEAN_DIR, exist_ok=True)

# Kolom-kolom umum untuk konten & judul (opsional, judul akan ditampilkan)
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
    """
    Baca semua CSV di DATA_DIR, lakukan preprocessing jika belum pernah dilakukan.
    Simpan hasil preprocessing ke datasets_clean/ agar tidak perlu diulang.
    """
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

        # 🔹 Jika sudah pernah dipreprocess, langsung load
        if os.path.exists(clean_path):
            df_clean = pd.read_csv(clean_path)
            docs.extend(df_clean["clean_text"].tolist())
            titles.extend(df_clean["title"].tolist())
            meta.extend(df_clean.to_dict("records"))
            print(f"✅ Memuat hasil preprocessing lama: {clean_path}")
            continue

        # 🔹 Kalau belum ada, lakukan preprocessing
        fpath = os.path.join(DATA_DIR, csv_name)
        try:
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="latin-1")

            df.columns = [c.strip().lower() for c in df.columns]
            title_col = pick_first_existing(CANDIDATE_TITLE_COLS, df)
            content_col = pick_first_existing(CANDIDATE_CONTENT_COLS, df)

            clean_records = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Memproses {csv_name}"):
                raw = combine_title_content(row, title_col, content_col).strip()
                if not raw:
                    continue
                cleaned = preprocess_text(raw)
                title = str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else f"{dataset_name}_row{i}"
                docs.append(cleaned)
                titles.append(title)
                clean_records.append({
                    "title": title,
                    "clean_text": cleaned,
                    "dataset": dataset_name,
                    "file": csv_name,
                    "row_id": i
                })

            # 🔹 Simpan hasil preprocessing
            df_clean = pd.DataFrame(clean_records)
            df_clean.to_csv(clean_path, index=False, encoding="utf-8")
            print(f"✅ Hasil preprocessing disimpan ke: {clean_path}")

            meta.extend(clean_records)

        except Exception as e:
            print(f"Gagal memproses {csv_name}: {e}")

    return docs, meta, titles


def main():
    print("=== INFORMATION RETRIEVAL SYSTEM ===")
    print("[1] Load & Index Dataset (CSV)")
    print("[2] Search Query")
    print("[3] Exit")
    print("====================================")

    docs, meta, titles = [], [], []
    vectorizer = None

    while True:
        choice = input("Pilih menu: ").strip()

        if choice == "1":
            docs, meta, titles = load_documents_from_csvs()
            if not docs:
                print("Dokumen tidak ditemukan / kosong.")
                continue
            _, vectorizer = vectorize_documents(docs)
            print(f"{len(docs)} dokumen berhasil dimuat dan diproses dari CSV.\n")

        elif choice == "2":
            if not docs or vectorizer is None:
                print("Dataset belum dimuat. Pilih [1] dulu.")
                continue
            query = input("Masukkan query: ").strip()
            qproc = preprocess_text(query)

            results = search_query(qproc, docs, vectorizer)

            print("\n=== Hasil Pencarian (Top-5) ===")
            if not results:
                print("Tidak ada hasil.")
            for rank, (idx, score) in enumerate(results, start=1):
                title = titles[idx] if idx < len(titles) else "(Tanpa Judul)"
                ds = meta[idx]["dataset"]
                rid = meta[idx]["row_id"]
                print(f"{rank}. {title} [{ds} - row#{rid}] | Skor: {score:.4f}")
            print("================================\n")

        elif choice == "3":
            print("Udah Keluar Nichh...")
            break

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()