# Retrieval Server

Local retrieval server for search-augmented RL training.

---

## Overview

The retrieval server provides document retrieval capabilities for search-based RL training. It uses dense retrieval to find relevant documents for search queries.

---

## Quick Start

### 1. Start the Retrieval Server

```bash
bash examples/search/retriever/retrieval_launch.sh
```

This starts the retrieval server on the default port.

### 2. Run Search Training

With the server running, start search training:

```bash
bash examples/search/run_search.sh
```

---

## Components

### retrieval_server.py

The main retrieval server implementation providing:

- Dense document retrieval
- HTTP API for search queries
- Batch retrieval support

### retrieval_launch.sh

Launch script for the retrieval server with default configuration.

---

## Configuration

The retrieval server can be configured via environment variables or command-line arguments. See `retrieval_server.py` for available options.

---

## Integration with Training

The search environment uses the retrieval server to:
1. Process search queries from the model
2. Return relevant documents
3. Score based on retrieval relevance

---

## Related Documentation

- [Search Example](../README.md)
- [Custom Environments](../../../docs/CUSTOM_ENVIRONMENTS.md)
