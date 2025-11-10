# Dokumentasi Proses Ingestion

Proses ini menyiapkan knowledge base dalam PostgreSQL (pgvector) dari CSV.

Langkah-langkah:

- Siapkan file `dataset/dataset_assignment.csv`.
- Create database customer_kb.
- Jalankan `python -m src.ingest` untuk melakukan:
  - Preprocessing dan pembersihan teks.
  - Kategorisasi heuristik (jika kolom `category` tidak tersedia).
  - Tagging sederhana berdasarkan kata kunci.
  - Pembuatan embedding batch (dinormalisasi) menggunakan SentenceTransformers.
  - Insert ke tabel `kb_entries` beserta indeks lexical dan vector.

Skema Tabel:

- `id SERIAL PRIMARY KEY`
- `category TEXT`
- `tags TEXT[]`
- `content TEXT NOT NULL`
- `content_tsv TSVECTOR` (generated)
- `embedding VECTOR(dimensi model)`

Catatan:

- Pastikan PostgreSQL dengan ekstensi `pgvector` aktif pada `localhost:5430`.
- `EMBEDDING_MODEL_NAME` default `all-MiniLM-L6-v2`.