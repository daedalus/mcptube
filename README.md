# 🎬 mcptube-vision

**YouTube video knowledge engine — transcripts, vision, and persistent wiki.**

[![PyPI](https://img.shields.io/pypi/v/mcptube-vision)](https://pypi.org/project/mcptube-vision/)
[![Python](https://img.shields.io/pypi/pyversions/mcptube-vision)](https://pypi.org/project/mcptube-vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

mcptube-vision transforms YouTube videos into a persistent, structured knowledge base using both transcripts and visual frame analysis. Built on the [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern: knowledge compounds with every video you add.

> **Evolved from [mcptube](https://pypi.org/project/mcptube/) v0.1** — mcptube-vision replaces semantic chunk search with a persistent wiki that gets smarter with every video ingested.

---

## 🧠 How It Works

Traditional video tools re-discover knowledge from scratch on every query. mcptube-vision is different:

```
           mcptube v0.1                    mcptube-vision
    ┌─────────────────────┐         ┌─────────────────────────┐
    │ Query → vector search│         │ Video ingested → LLM     │
    │ → raw chunks → LLM  │         │ extracts knowledge →     │
    │ → answer (from scratch│        │ wiki pages created →     │
    │   every time)        │         │ cross-references built   │
    └─────────────────────┘         │                         │
                                    │ Query → FTS5 + agent    │
                                    │ → reasons over compiled  │
                                    │   knowledge → answer     │
                                    └─────────────────────────┘
```

| | v0.1 (Video Search Engine) | vision (Video Knowledge Engine) |
|---|---|---|
| **On ingest** | Chunk transcript, embed in vector DB | LLM watches + reads, writes wiki pages |
| **On query** | Find similar chunks | Agent reasons over compiled knowledge |
| **Frames** | Timestamp or keyword extraction | Scene-change detection + vision model |
| **Cross-video** | Re-search all chunks each time | Connections already in the wiki |
| **Over time** | Library of isolated videos | Compounding knowledge base |

---

## ✨ Features

| Feature | CLI | MCP Server |
|---------|:---:|:----------:|
| Add/remove YouTube videos | ✅ | ✅ |
| Wiki knowledge base (auto-built) | ✅ | ✅ |
| Scene-change frame extraction + vision analysis | ✅ | ✅ |
| Full-text wiki search (FTS5) | ✅ | ✅ |
| Agentic Q&A over wiki | ✅ | ✅ |
| Browse wiki pages (entities, topics, concepts) | ✅ | ✅ |
| Wiki version history | ✅ | ✅ |
| Wiki export (markdown, HTML) | ✅ | — |
| Illustrated reports (single & cross-video) | ✅ (BYOK) | ✅ (passthrough) |
| YouTube discovery + clustering | ✅ (BYOK) | ✅ |
| Cross-video synthesis | ✅ (BYOK) | ✅ (passthrough) |
| Text-only processing mode | ✅ | ✅ |

**BYOK** = Bring Your Own Key (Anthropic, OpenAI, or Google)
**Passthrough** = The MCP client's own LLM does the analysis

---

## 📦 Installation

### Prerequisites

- **Python 3.12 or 3.13**
- **ffmpeg** — required for frame extraction ([install guide](https://ffmpeg.org/download.html))

### Recommended: pipx

```bash
pipx install mcptube-vision --python python3.12
```

### Alternative: pip

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install mcptube-vision
```

### Verify installation

```bash
mcptube --help
```

---

## 🚀 Quick Start

```bash
# 1. Add a video (builds wiki automatically)
mcptube add "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 2. Add with text-only processing (cheaper, faster)
mcptube add "https://www.youtube.com/watch?v=abc123" --text-only

# 3. Browse the wiki
mcptube wiki list
mcptube wiki show "video-dQw4w9WgXcQ"

# 4. Search the knowledge base
mcptube search "main topic"

# 5. Ask a question (agentic retrieval over wiki)
mcptube ask "What are the key ideas discussed?"

# 6. View the table of contents
mcptube wiki toc
```

> 💡 **Always wrap multi-word arguments in double quotes.**

---

## 📖 CLI Reference

### Library Management

| Command | Description | Example |
|---------|-------------|---------|
| `mcptube add "<url>"` | Ingest video + build wiki (full analysis) | `mcptube add "https://youtu.be/dQw4w9WgXcQ"` |
| `mcptube add "<url>" --text-only` | Ingest without vision processing | `mcptube add "https://youtu.be/abc" --text-only` |
| `mcptube list` | List all videos with tags | `mcptube list` |
| `mcptube info <query>` | Show full video details | `mcptube info 1` |
| `mcptube remove <query>` | Remove video + clean wiki references | `mcptube remove 1` |

### Wiki Knowledge Base

| Command | Description | Example |
|---------|-------------|---------|
| `mcptube wiki list` | Browse all wiki pages | `mcptube wiki list` |
| `mcptube wiki list --type entity` | Filter by type | `mcptube wiki list --type concept` |
| `mcptube wiki show <slug>` | Read a specific wiki page | `mcptube wiki show "entity-openai"` |
| `mcptube wiki search "<query>"` | Full-text search | `mcptube wiki search "attention"` |
| `mcptube wiki toc` | Table of contents | `mcptube wiki toc` |
| `mcptube wiki history <slug>` | Version history | `mcptube wiki history "topic-ml"` |
| `mcptube wiki export` | Export as markdown or HTML | `mcptube wiki export --format html -o wiki.html` |

### Search & Ask

| Command | Description | Example |
|---------|-------------|---------|
| `mcptube search "<query>"` | Search wiki via FTS5 | `mcptube search "transformers"` |
| `mcptube ask "<question>"` | Agentic Q&A over wiki | `mcptube ask "What is self-attention?"` |

### Frames

| Command | Description | Example |
|---------|-------------|---------|
| `mcptube frame <video> <timestamp>` | Extract frame at timestamp | `mcptube frame 1 30` |
| `mcptube frame-query <video> "<query>"` | Extract frame by transcript match | `mcptube frame-query 1 "key moment"` |

### Analysis & Reports (BYOK)

| Command | Description | Example |
|---------|-------------|---------|
| `mcptube classify <video>` | LLM classification/tagging | `mcptube classify 1` |
| `mcptube report <video> [--format html] [-o file]` | Single-video report | `mcptube report 1 -o report.html` |
| `mcptube report-query "<topic>" [--format html] [-o file]` | Cross-video report | `mcptube report-query "AI" -o report.html` |
| `mcptube discover "<topic>"` | YouTube search + clustering | `mcptube discover "prompt engineering"` |
| `mcptube synthesize-cmd "<topic>" -v <id1> -v <id2>` | Cross-video synthesis | `mcptube synthesize-cmd "AI" -v id1 -v id2` |

### Server

| Command | Description |
|---------|-------------|
| `mcptube serve` | Start MCP server (HTTP) |
| `mcptube serve --stdio` | Start MCP server (stdio) |

---

## 🧩 Wiki Page Types

When you ingest a video, mcptube-vision builds four types of wiki pages:

| Page Type | Created From | Update Policy |
|-----------|-------------|---------------|
| **Video** | Each ingested video | Write-once (immutable) |
| **Entity** | People, companies, tools mentioned | Append-only (new references added) |
| **Topic** | Broad themes (e.g., "Machine Learning") | Synthesis rewritten, per-video contributions immutable |
| **Concept** | Specific ideas (e.g., "Scaling Laws") | Synthesis rewritten, per-video contributions immutable |

**Principle:** Raw source content (what was said/shown in each video) is never modified. Only synthesis summaries evolve as new videos are added. Version history is maintained for all changes.

---

## 🔍 How Search Works (Hybrid Retrieval)

mcptube-vision uses a two-step hybrid approach:

1. **SQLite FTS5** — keyword search narrows thousands of wiki pages to a handful of candidates (milliseconds, zero LLM cost)
2. **LLM Agent** — reads candidates + wiki table of contents, reasons about relevance, synthesizes an answer

This gives you the speed of keyword search with the intelligence of an LLM agent.

---

## 👁️ Vision Pipeline

When you ingest a video without `--text-only`, mcptube-vision:

1. Extracts key frames using **ffmpeg scene-change detection** (`select='gt(scene,0.4)'`)
2. Sends frames to a **vision-capable LLM** (GPT-4o, Claude, Gemini) for description
3. Combines frame descriptions with transcript in the knowledge extraction pass

This captures visual content (slides, code, diagrams, demos) that transcripts alone miss.

---

## 🔌 MCP Server

### Configuration (Claude Desktop)

```json
{
    "mcpServers": {
        "mcptube": {
            "command": "mcptube",
            "args": ["serve", "--stdio"]
        }
    }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `add_video` | Ingest video + build wiki |
| `list_videos` | List library |
| `remove_video` | Remove video + clean wiki |
| `wiki_list` | Browse wiki pages |
| `wiki_show` | Read a wiki page |
| `wiki_search` | Full-text search |
| `wiki_toc` | Table of contents |
| `wiki_ask` | Agentic Q&A |
| `wiki_history` | Version history |
| `get_frame` | Extract frame (inline image) |
| `get_frame_by_query` | Frame by transcript match |
| `classify_video` | Get metadata for classification |
| `generate_report` | Get data for single-video report |
| `generate_report_from_query` | Get data for cross-video report |
| `synthesize` | Get data for theme synthesis |
| `discover_videos` | Search YouTube |
| `ask_video` | Single-video Q&A data |
| `ask_videos` | Multi-video Q&A data |

---

## ⚙️ Configuration

All settings can be overridden via environment variables prefixed with `MCPTUBE_`:

| Setting | Default | Env Var |
|---------|---------|---------|
| Data directory | `~/.mcptube` | `MCPTUBE_DATA_DIR` |
| Server host | `127.0.0.1` | `MCPTUBE_HOST` |
| Server port | `9093` | `MCPTUBE_PORT` |
| Default LLM model | `gpt-4o` | `MCPTUBE_DEFAULT_MODEL` |

### BYOK API Keys

Set one or more to enable LLM features:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AI..."
```

Auto-detection priority: Anthropic → OpenAI → Google.

---

## 📁 Data Layout

```
~/.mcptube/
├── mcptube.db          # Video metadata (SQLite)
├── wiki.db             # FTS5 search index (SQLite)
├── wiki/
│   ├── video/          # Video pages (JSON)
│   ├── entity/         # Entity pages (JSON)
│   ├── topic/          # Topic pages (JSON)
│   ├── concept/        # Concept pages (JSON)
│   └── _history/       # Version history
└── frames/
    ├── <id>_<ts>.jpg   # Single extracted frames
    └── <id>_scenes/    # Scene-change frames + metadata
```

---

## 🧪 Development

```bash
git clone https://github.com/0xchamin/mcptube.git
cd mcptube
git checkout vision
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## 🗺️ Roadmap

- [x] Wiki knowledge engine (entities, topics, concepts)
- [x] Scene-change frame extraction + vision analysis
- [x] Hybrid retrieval (FTS5 + agentic)
- [x] CLI + MCP server
- [ ] Playlist/series support
- [ ] Web app with early access sign-up
- [ ] Token-based payment integration

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
