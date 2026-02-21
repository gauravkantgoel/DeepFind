# 🔍 DeepFind: Semantic File Search for Pharma & Beyond

A local, privacy-first semantic file search engine. Find any document by **meaning** and not just filename keywords — with zero cloud dependency. All AI processing happens entirely on your machine.

Built with pharmacovigilance professionals in mind, but useful for anyone managing large collections of documents.

---

## Why DeepFind?

Windows Search is keyword-only and frequently misses files with non-obvious names. DeepFind uses AI embeddings to understand the *meaning* of your search:

| You search for | It finds |
|---|---|
| `Novartis safety agreement` | `PV_NVS_partnership_v3_FINAL.pdf` |
| `quarterly financial report` | `Q3_2024_summary.xlsx` |
| `signal detection procedure` | `SOP_ADR_SignalMgmt_v2.docx` |
| `PSUR` | All periodic safety update reports, even if named differently |

No file ever leaves your laptop.

---

## Features

- 🧠 **Biomedical AI search** — uses `PubMedBERT` to understand pharma/PV terminology, MedDRA terms, drug names, and regulatory language
- 💊 **Pharma synonym expansion** — automatically expands queries with known PV synonyms (PSUR ↔ PBRER ↔ periodic safety update report)
- 🔤 **4-tier hybrid ranking** — exact filename matches always appear first, then partial, then content, then semantic
- 📑 **Two search tabs** — Keyword & Content tab and Semantic AI tab, clearly separated
- 📄 **Content extraction** — reads inside PDF, Word, Excel, and text files during indexing
- 👁️ **Auto file watching** — new/modified/deleted files update the index automatically within seconds
- ⏹️ **Stoppable indexing** — stop and resume indexing at any time without losing progress
- ⚙️ **Visual safeguard controls** — configure file size limits, throttle, similarity threshold, and model from the UI
- 🔒 **100% local & offline** — no internet required after model download, no data leaves your machine
- 🪟 **Browser UI** — clean, modern interface that opens automatically in your browser

---

## Quick Start

### Requirements
- Windows 10 or 11
- [Anaconda](https://www.anaconda.com/download) or Python 3.9+

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/file-finder.git
cd file-finder
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```
Your browser opens automatically at `http://localhost:5000/ui`

**4. Configure folders**
- Go to ⚙️ Settings
- Add folders to watch under **Watched Folders**
- Click **💾 Save All Settings**

**5. Index your files**
- Click **▶ Start Indexing**
- Wait for status to show **Idle** (may take several minutes for large folders)
- The AI model downloads automatically on first run (~440 MB, one time only)

**6. Search**
- Type naturally — `PSUR 2024` or `Novartis PV agreement` or `signal detection SOP`
- Use the **Keyword & Content** tab for exact matches
- Use the **Semantic AI** tab for meaning-based results

---

## Search Modes

### Keyword & Content Tab
Results ranked in strict order:
1. ✅ **Exact match** — all search words found in filename
2. 🔤 **Filename** — some search words in filename  
3. 📄 **Content** — words found inside document text

### Semantic AI Tab
- 🧠 **Semantic match** — files found by meaning, even with no keyword overlap
- Score shown as percentage (e.g. `🧠 78% match`)

---

## Pharma Domain Features

DeepFind has built-in knowledge of pharmacovigilance terminology:

- **Abbreviation expansion** — searching `PSUR` also matches files containing *periodic safety update report*, *PBRER*, *periodic report*
- **Regulatory synonyms** — `RMP` ↔ *risk management plan* ↔ *safety specification*
- **Agency awareness** — `EMA` ↔ *European Medicines Agency* ↔ *CHMP* ↔ *European regulatory*
- **Case report terms** — `ICSR` ↔ *individual case safety report* ↔ *case report* ↔ *AER*
- **Inspection terms** — `GVP inspection` ↔ *regulatory audit* ↔ *mock inspection* ↔ *CAPA*

---

## Settings & Safeguards

All configurable from the UI — no code changes needed:

| Setting | Description | Default |
|---|---|---|
| Max file size | Skip files larger than this | 10 MB |
| Content length cap | Max characters read per file | 2,000 |
| Throttle delay | Pause between files to protect CPU | 100 ms |
| Similarity threshold | How strict the AI match must be | 40% |
| Keyword fallback | Use keyword search if AI finds nothing | On |
| Embedding model | Choose between Biomedical or General model | PubMedBERT |

---

## Supported File Types

**Content extraction** (text read from inside):
PDF, Word (.docx, .doc), Excel (.xlsx, .xls), Text (.txt), CSV

**Filename indexing** (all other types):
PowerPoint, Images, ZIP, Email (.eml, .msg), and all other extensions

---

## How It Works

1. **Indexing** — walks your watched folders, extracts text from documents, generates AI embeddings using `neuml/pubmedbert-base-embeddings`
2. **Cache** — all embeddings loaded into a NumPy matrix in memory for millisecond search
3. **Searching** — query embedded and compared via vectorised cosine similarity; results tiered by keyword then semantic rank
4. **Watching** — background file system watcher updates the index in real time as files change

---

## Privacy

- All indexing, embedding, and search runs locally
- No files, filenames, or queries are ever sent to any server
- The AI model runs fully offline after the one-time download
- Index stored as a local SQLite database (`file_index.db`)

---

## Roadmap

- [ ] Windows standalone executable (.exe)
- [ ] Document type auto-detection (PSUR, SOP, ICSR, RMP badges)
- [ ] Inspection readiness query mode
- [ ] Version grouping (detect v1/v2/FINAL variants)
- [ ] AI-powered document summarisation
- [ ] Date range filter
- [ ] Pinned / bookmarked files
- [ ] macOS support

---

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change or add.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built by [Gaurav Goel](https://github.com/gauravkantgoel)  
Pharmacovigilance Analytics Professional | [LinkedIn](https://www.linkedin.com/in/gauravkantgoel/)
