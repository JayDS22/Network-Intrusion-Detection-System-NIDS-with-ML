#!/usr/bin/env bash
#
# NIDS-ML demo launcher.
#
# Usage:
#   ./run_demo.sh             # install deps if missing, train if needed, run dashboard
#   ./run_demo.sh --train     # retrain models even if cached
#   ./run_demo.sh --no-install
#
set -euo pipefail

cd "$(dirname "$0")"

PY=${PYTHON:-python3}
DO_INSTALL=1
DO_RETRAIN=0
for arg in "$@"; do
  case "$arg" in
    --no-install) DO_INSTALL=0 ;;
    --train)      DO_RETRAIN=1 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# //'
      exit 0 ;;
  esac
done

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
echo -e "${CYAN}=============================================="
echo -e "  NIDS-ML  Network Intrusion Detection (demo)"
echo -e "==============================================${NC}"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  echo -e "${YELLOW}>> Installing requirements...${NC}"
  $PY -m pip install -q -r requirements.txt
fi

if [[ "$DO_RETRAIN" -eq 1 ]] || [[ ! -f models/saved/ensemble_meta.joblib ]]; then
  echo -e "${YELLOW}>> Training ensemble on synthetic traffic...${NC}"
  $PY scripts/train_model.py --packets 25000 --epochs 3 --no-tf
fi

echo -e "${GREEN}>> Launching dashboard at http://localhost:8501${NC}"
exec $PY -m streamlit run dashboard/streamlit_app.py
