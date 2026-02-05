#!/usr/bin/env bash
# Run the complete CI pipeline locally using act
# This builds base images, tags them for local use, and runs the full PR preview workflow
#
# Usage: ./scripts/dev/run_full_ci_locally.sh [--skip-build] [--skip-smoke]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SKIP_BUILD=false
SKIP_SMOKE=false
TAG="local-ci"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --skip-smoke) SKIP_SMOKE=true; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--skip-build] [--skip-smoke] [--tag TAG]"
      echo "  --skip-build  Skip building base images (use existing)"
      echo "  --skip-smoke  Skip smoke deployment test"
      echo "  --tag TAG     Tag for base images (default: local-ci)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}       archi Full CI - Local Runner${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required${NC}"; exit 1; }
command -v act >/dev/null 2>&1 || { echo -e "${RED}act is required. Install with: brew install act${NC}"; exit 1; }

# Find Python - prefer venv
if [[ -f ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  echo -e "${RED}Python not found${NC}"; exit 1
fi
echo -e "${GREEN}✓${NC} Prerequisites OK (using $PYTHON)"

# Step 1: MkDocs build (quick check first)
echo ""
echo -e "${YELLOW}Step 1: MkDocs Build${NC}"
if $PYTHON -m mkdocs build --config-file docs/mkdocs.yml --strict 2>&1; then
  echo -e "${GREEN}✓${NC} MkDocs build passed"
  rm -rf site
else
  echo -e "${RED}✗${NC} MkDocs build failed"
  exit 1
fi

# Step 2: Python syntax check
echo ""
echo -e "${YELLOW}Step 2: Python Syntax Check${NC}"
SYNTAX_OK=true
while IFS= read -r -d '' file; do
  if ! $PYTHON -m py_compile "$file" 2>/dev/null; then
    echo -e "${RED}✗${NC} Syntax error in: $file"
    SYNTAX_OK=false
  fi
done < <(find src -name "*.py" -print0 2>/dev/null)
if $SYNTAX_OK; then
  echo -e "${GREEN}✓${NC} All Python files have valid syntax"
else
  exit 1
fi

# Step 3: Build base images (if not skipped)
if ! $SKIP_BUILD; then
  echo ""
  echo -e "${YELLOW}Step 3: Building Base Images (this may take 10-15 minutes)...${NC}"
  
  # Build python base (use --no-cache if FORCE_REBUILD is set)
  echo "  Building archi-python-base..."
  CACHE_FLAG=""
  if [[ "${FORCE_REBUILD:-}" == "1" ]]; then
    CACHE_FLAG="--no-cache"
    echo "  (--no-cache enabled via FORCE_REBUILD=1)"
  fi
  docker build $CACHE_FLAG \
    -t "localhost/archi/archi-python-base:${TAG}" \
    -t "localhost/archi/archi-python-base:latest" \
    -t "archi/archi-python-base:${TAG}" \
    -t "archi/archi-python-base:latest" \
    -f src/cli/templates/dockerfiles/base-python-image/Dockerfile \
    src/cli/templates/dockerfiles/base-python-image/ 2>&1 | tail -5
  
  echo -e "${GREEN}✓${NC} archi-python-base built"
  
  # Check if pytorch base is needed (skip for faster CI)
  echo "  Skipping pytorch base (not needed for smoke test)"
else
  echo ""
  echo -e "${YELLOW}Step 3: Skipping base image build (--skip-build)${NC}"
fi

# Step 4: Clean up any previous test deployment
echo ""
echo -e "${YELLOW}Step 4: Cleaning up previous test deployments...${NC}"
docker rm -f chatbot-manual-1 data-manager-manual-1 postgres-manual-1 2>/dev/null || true
docker volume rm archi-data-manual-1 archi-manual-1 archi-pg-manual-1 2>/dev/null || true
rm -rf archi-local/archi-manual-1 2>/dev/null || true
echo -e "${GREEN}✓${NC} Cleanup complete"

# Step 5: Run smoke test (if not skipped)
if ! $SKIP_SMOKE; then
  echo ""
  echo -e "${YELLOW}Step 5: Running Smoke Test via act...${NC}"
  echo "  This runs the full PR preview workflow locally"
  echo ""
  
  # Run act with bind mount for proper path mapping
  act pull_request -j preview \
    --container-architecture linux/amd64 \
    -P submit76-runner=catthehacker/ubuntu:act-latest \
    --bind \
    --env SKIP_BASE_IMAGE_BUILD=true \
    2>&1 | tee /tmp/act-local-ci.log | grep -E "(✅|❌|⭐|🏁|Error)" || true
  
  # Check result - must find "Job succeeded" for the preview job
  if grep -q "preview.*Job succeeded" /tmp/act-local-ci.log || (grep -q "Job succeeded" /tmp/act-local-ci.log && ! grep -q "Job failed" /tmp/act-local-ci.log); then
    echo ""
    echo -e "${GREEN}✓${NC} Smoke test passed!"
  else
    echo ""
    echo -e "${RED}✗${NC} Smoke test failed. Check /tmp/act-local-ci.log for details"
    echo "  Showing last 20 lines of logs:"
    tail -20 /tmp/act-local-ci.log
    echo ""
    echo "  Common issues:"
    echo "  - Base images outdated (run with FORCE_REBUILD=1)"
    echo "  - Code dependencies missing (frontend needs backend code)"
    exit 1
  fi
else
  echo ""
  echo -e "${YELLOW}Step 5: Skipping smoke test (--skip-smoke)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All CI checks passed!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
