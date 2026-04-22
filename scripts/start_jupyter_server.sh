#!/bin/bash
# Start Jupyter Server for MCP integration
# This runs JupyterLab with collaboration enabled for jupyter-mcp-server

# Resolve project root (one level up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/src/.venv"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

# Start JupyterLab with collaboration features enabled
# Token must match the one in .mcp.json
"$VENV_DIR/bin/python" -m jupyter lab --port 8888 \
    --IdentityProvider.token=my_secure_token_123 \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=true \
    --ip=0.0.0.0 \
    --no-browser \
    --LabApp.collaborative=true
