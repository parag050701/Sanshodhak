# ✅ SANSHODHAK - COMPLETE INTEGRATION TEST

## 🚀 Server Status

```bash
✅ Running: http://localhost:5000
✅ Health Check: PASSED
✅ WebSocket: Connected
✅ Agent: Initialized
```

---

## 🧪 HOW TO TEST - STEP BY STEP

### Step 1: Open Browser
```
http://localhost:5000
```

### Step 2: Enter Research Query
In the search box, type:
```
Deep Learning in Healthcare
```

### Step 3: Select Mode
Click: **🔄 Full Pipeline**

This will execute ALL 6 stages:
1. 🔍 **SEARCH** - 10X search (300 papers from CrossRef → Unpaywall → OpenAlex → CORE → S2)
2. 📥 **DOWNLOAD** - PDF retrieval with fallbacks (Unpaywall → Sci-Hub → LibGen)
3. 📄 **PARSE** - Text extraction (PyMuPDF → pdfplumber → PyPDF2)
4. 🧬 **EMBED** - Build NEW embeddings (bge-m3 + FAISS index)
5. 💡 **RETRIEVE** - Query NEW index (DeepSeek-R1:7b)
6. 🎯 **RECOMMEND** - Find resources (HuggingFace + GitHub + Web)

### Step 4: Click Execute
Button: **🚀 Execute**

### Step 5: Watch Progress Panel
You should see:
```
🔄 Pipeline Progress

🔍 SEARCH: running
   10X SEARCH: User wants 30 papers → Searching 300 papers for best quality

🔍 SEARCH: running
   [1/3] 📚 Searching CrossRef + Unpaywall for 300 papers...

🔍 SEARCH: running
   [1/3] ✅ CrossRef: Found 245 papers (127 with OA PDFs)

🔍 SEARCH: running
   [2/3] 🌐 Searching OpenAlex + CORE + Semantic Scholar + arXiv...

... and so on for all 6 stages
```

---

## 📊 WHAT TO EXPECT

### Stage 1: SEARCH (2-5 minutes)
- **10X Search Strategy** - Request 30 papers → Search 300 papers
- **Sources:**
  - CrossRef: 300 papers (metadata + DOIs)
  - Unpaywall: Enrich ALL DOIs with OA links
  - OpenAlex: 300 papers (open access)
  - CORE: 300 papers (repositories)
  - Semantic Scholar: 300 papers (AI-powered)
  - arXiv: Preprints
- **Merge & Deduplicate** - Unique papers only
- **Rank** - Citations × Recency × OA availability
- **Filter** - Select TOP 30 highest quality

**Expected Output:**
```
✅ 10X Search complete: Selected TOP 30 papers from 312 candidates
   Avg citations: 156.3
   OA percentage: 73.3%
   Avg year: 2022.1
```

### Stage 2: DOWNLOAD (3-10 minutes)
- **Primary:** Unpaywall OA PDFs
- **Fallback 1:** Sci-Hub (if enabled)
- **Fallback 2:** LibGen (if enabled)
- **Parallel:** 5 concurrent downloads

**Expected Output:**
```
Downloaded 27/30 PDFs
Success rate: 90%
```

### Stage 3: PARSE (1-3 minutes)
- **Method 1:** PyMuPDF (fast, reliable)
- **Method 2:** pdfplumber (good for tables)
- **Method 3:** PyPDF2 (lightweight)
- **Fallback chain** - Tries all until success

**Expected Output:**
```
Parsed 27 PDFs
Total chars: 1,234,567
```

### Stage 4: EMBED (5-15 minutes)
- **Model:** bge-m3 (1024-dim embeddings)
- **Chunking:** 512 tokens, 128 overlap
- **Index:** FAISS (cosine similarity)
- **Session:** NEW index per research topic

**Expected Output:**
```
NEW embeddings created
Chunk count: 3,847
Index saved: /sessions/{hash}_{timestamp}/index/
```

### Stage 5: RETRIEVE (10-30 seconds)
- **Model:** DeepSeek-R1:7b
- **Strategy:** Retrieves from NEW index
- **Top-K:** 5 most relevant chunks
- **Generation:** Answers with citations

**Expected Output:**
```
Answer generated with 5 citations
Response time: 12.3s
```

### Stage 6: RECOMMEND (5-10 seconds)
- **HuggingFace:** Models & datasets
- **GitHub:** Top repositories
- **Web:** Tutorials & docs

**Expected Output:**
```
Found 47 resources
- HuggingFace: 15 models
- GitHub: 20 repos
- Web: 12 tutorials
```

---

## 🔍 MONITORING PROGRESS

### Terminal Logs
```bash
tail -f /tmp/sanshodhak.log
```

You should see:
```
[SEARCH] 10X SEARCH: User wants 30 papers → Searching 300 papers
[SEARCH] [1/3] 📚 Searching CrossRef + Unpaywall for 300 papers...
[SEARCH] CrossRef: 600 papers found
[SEARCH] Enriching with Unpaywall OA info...
[SEARCH] [1/3] ✅ CrossRef: Found 245 papers (127 with OA PDFs)
[SEARCH] [2/3] 🌐 Searching OpenAlex + CORE + Semantic Scholar + arXiv...
[SEARCH] SemanticScholar: 30 papers (2427ms)
[SEARCH] OpenAlex: 89 papers
[SEARCH] CORE: 45 papers
[SEARCH] [2/3] ✅ Open-source: Found 164 papers
[SEARCH] [3/3] 🔗 Merging 409 papers → deduplicating → ranking...
[SEARCH] After dedup: 312 unique papers
[SEARCH] [3/3] 🏆 Ranking by citations × recency × OA...
[SEARCH] ✅ 10X Search complete: Selected TOP 30 papers
[DOWNLOAD] Downloading 30 PDFs with fallbacks...
[DOWNLOAD] Downloaded 27/30 PDFs
[PARSE] Parsing 27 PDFs...
[PARSE] Parsed [1/27]: Paper title...
... continues ...
```

### Browser Console
Press `F12` → Console tab

You should see:
```javascript
Progress update: {stage: "search", status: "running", message: "10X SEARCH: ...", ...}
Progress update: {stage: "search", status: "running", message: "[1/3] 📚 Searching...", ...}
Progress update: {stage: "download", status: "running", message: "Downloading 30 PDFs...", ...}
... etc ...
```

---

## 🐛 TROUBLESHOOTING

### Issue: "No progress showing on frontend"

**Solution 1:** Check WebSocket connection
```javascript
// In browser console (F12)
socket.connected  // Should return true
```

**Solution 2:** Check server logs
```bash
tail -50 /tmp/sanshodhak.log | grep -i "progress\|emit\|socket"
```

**Solution 3:** Hard refresh browser
```
Ctrl + Shift + R  (Linux/Windows)
Cmd + Shift + R   (Mac)
```

### Issue: "Search taking too long"

**Normal!** 10X search (300 papers) takes 2-5 minutes:
- CrossRef: ~1-2 minutes
- Unpaywall enrichment: ~1-2 minutes  
- OpenAlex/CORE/S2: ~1-2 minutes in parallel

**Monitor:** Watch `/tmp/sanshodhak.log` for progress

### Issue: "Download fails"

**Check:** Are you behind a firewall/proxy?

**Solution:** Enable shadow libraries in `.env`:
```bash
ENABLE_SCIHUB=true
ENABLE_LIBGEN=true
```

Then restart:
```bash
pkill -9 -f "python.*sanshodhak_app"
cd /home/admin-/Desktop/Sanshodhak/paper-intel
python3 sanshodhak_app.py
```

### Issue: "Embedding takes forever"

**Normal!** Embedding 30 papers (27 PDFs × ~15 pages = ~400 pages) takes 5-15 minutes:
- Text extraction: ~1-3 minutes
- Chunking: ~30 seconds
- Embedding generation: ~5-10 minutes (depends on CPU)
- FAISS index creation: ~10 seconds

**Monitor:** Watch for "Embedding chunk X/Y" messages

---

## 📁 OUTPUT STRUCTURE

After successful execution, check:

```bash
ls -la /home/admin-/Desktop/Sanshodhak/paper-intel/sessions/
```

You should see:
```
sessions/
  3fbdc3b4_20251206_080437/  # Session ID: {query_hash}_{timestamp}
    papers/                   # Downloaded PDFs (30 files)
      paper1.pdf
      paper2.pdf
      ...
    text/                     # Extracted text (27 files)
      paper1.txt
      paper2.txt
      ...
    index/                    # NEW FAISS index
      index.faiss             # Vector store
      metadata.json           # Chunk metadata
      config.json             # RAG configuration
```

---

## ✨ KEY FEATURES IMPLEMENTED

### ✅ 10X Search Strategy
- User requests 30 papers → System searches **300 papers**
- Ensures highest quality by selecting TOP 30 from large pool
- Multi-source: CrossRef, Unpaywall, OpenAlex, CORE, S2, arXiv

### ✅ Smart Ranking
- **Citations** (40% weight): More cited = higher quality
- **Recency** (30% weight): Recent papers preferred
- **Open Access** (30% weight): +10 bonus for OA availability

### ✅ NEW Embeddings Per Topic
- Creates **NEW** FAISS index for EACH research topic
- Session-based directories: No cross-contamination
- Old indexes NOT reused

### ✅ Complete PDF Pipeline
- **Download:** Unpaywall → Sci-Hub → LibGen fallbacks
- **Parse:** PyMuPDF → pdfplumber → PyPDF2 fallbacks
- **Extract:** Full text with metadata

### ✅ Real-Time Progress
- WebSocket updates for all 6 stages
- Live status changes: pending → running → completed
- Detailed progress messages with emoji indicators

### ✅ Resource Recommendations
- **HuggingFace:** Pre-trained models & datasets
- **GitHub:** Top repositories (by stars)
- **Web:** Tutorials, docs, courses

---

## 🎯 SUCCESS CRITERIA

After clicking "🚀 Execute", you should see:

1. **Progress Panel Activates** - Shows all 6 stages
2. **Search Stage Runs** - 2-5 minutes, 300 papers searched
3. **Download Stage Runs** - 3-10 minutes, PDFs downloaded
4. **Parse Stage Runs** - 1-3 minutes, text extracted
5. **Embed Stage Runs** - 5-15 minutes, NEW index created
6. **Retrieve Stage Runs** - 10-30 seconds, answer generated
7. **Recommend Stage Runs** - 5-10 seconds, resources found
8. **Results Display** - Answer + Citations + Resources shown

**Total Time:** 15-35 minutes (varies by paper count and system)

---

## 🚀 NEXT STEPS

### Test Different Topics
```
1. "Graph Neural Networks"
2. "Transfer Learning in NLP"
3. "Quantum Machine Learning"
4. "Federated Learning Privacy"
```

### Verify Session Isolation
Each query creates a NEW session:
```bash
ls -l /home/admin-/Desktop/Sanshodhak/paper-intel/sessions/
# Should show multiple directories, one per query
```

### Check Index Quality
```bash
# Count chunks in latest session
find /home/admin-/Desktop/Sanshodhak/paper-intel/sessions/ -name "index.faiss" -exec ls -lh {} \;
```

---

## 📝 ARCHITECTURE SUMMARY

```
USER QUERY
    ↓
┌─────────────────────────────────────────┐
│ 1. SEARCH (10X Strategy)                │
│    - CrossRef: 300 papers               │
│    - Unpaywall: Enrich ALL DOIs         │
│    - OpenAlex/CORE/S2/arXiv: 300 papers │
│    - Merge → Deduplicate → Rank         │
│    - Select TOP 30                      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. DOWNLOAD (Multi-fallback)            │
│    - Unpaywall OA PDFs                  │
│    - Sci-Hub fallback                   │
│    - LibGen fallback                    │
│    - Parallel (5 workers)               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. PARSE (Multi-method)                 │
│    - PyMuPDF (fast)                     │
│    - pdfplumber (accurate)              │
│    - PyPDF2 (lightweight)               │
│    - Fallback chain                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. EMBED (NEW index per topic)          │
│    - bge-m3 embeddings (1024-dim)       │
│    - Chunk: 512 tokens, 128 overlap     │
│    - FAISS IndexFlatIP                  │
│    - Session: {hash}_{timestamp}        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. RETRIEVE (RAG query)                 │
│    - DeepSeek-R1:7b LLM                 │
│    - Query NEW index                    │
│    - Top-5 chunks                       │
│    - Generate answer + citations        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. RECOMMEND (Resources)                │
│    - HuggingFace models/datasets        │
│    - GitHub repositories                │
│    - Web tutorials/docs                 │
└─────────────────────────────────────────┘
    ↓
COMPLETE RESULTS
```

---

**Status:** 🟢 FULLY INTEGRATED AND READY TO TEST
**Server:** http://localhost:5000
**Logs:** /tmp/sanshodhak.log
