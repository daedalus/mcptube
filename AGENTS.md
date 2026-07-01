# AGENTS.md — mcptube

## Overview

mcptube is a YouTube video knowledge engine that transforms videos into a persistent wiki knowledge base. It provides both a CLI and MCP server for ingestion, search, Q&A, and report generation over compiled video knowledge.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format` | Format code |
| `mdformat` | Format markdown |
| `prospector --with-tool ruff --with-tool mypy --with-tool pylint src/` | Lint + type check (with blending) |
| `opengrep --config=auto --severity=ERROR src/` | Security and pattern scanning |
| `vulture --min-confidence 90 src/` | Dead/unused code detection |
| `lizard src/ --min-cyclomatic-complexity 10` | Code complexity analysis |
| `impactguard-check-staged` | API impact analysis for staged changes |

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format
ruff format src/ tests/

# Format markdown
mdformat .

# Lint + type check (prospector runs ruff check + mypy + pylint together)
prospector --with-tool ruff --with-tool mypy --with-tool pylint src/
opengrep --config=auto --severity=ERROR src/

# Find unused code
vulture --min-confidence 90 src/

# Analyze code complexity
lizard src/ --min-cyclomatic-complexity 10

# Track API impact
impactguard-check-staged
```

## Testing

Tests use pytest with pytest-asyncio for async operations. All external dependencies (YouTube extraction, LLM calls, ffmpeg) are mocked in conftest.py fixtures. The `service` fixture provides a fully wired McpTubeService with all mocked dependencies.

## Code Style

- Format: ruff format
- Lint + Type check: prospector (runs ruff check + mypy + pylint with blending)
- Docstrings: Google style

## Release

Use `tools/release.sh` to automate version bumps, builds, and GitHub releases:

```bash
./tools/release.sh          # bump patch (default)
./tools/release.sh minor
./tools/release.sh major
```

The script:

- Checks working tree is clean; warns if not on `master`/`main`
- Runs `bumpversion <part> --tag --verbose`
- Pushes commit + tags
- Builds the package
- Creates a GitHub release with auto-generated notes

## MCP Server

```bash
# Install and use
pip install mcptube
```

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "mcptube": {
      "command": "mcptube"
    }
  }
}
```
