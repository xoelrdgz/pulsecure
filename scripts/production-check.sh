set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Rust compile checks =="
cargo check
cargo check --benches

echo "== Model contract =="
python3 scripts/validate-model-contract.py models/calibrated_model.json

echo "== Contract and storage tests =="
cargo test test_bundled_model_contract_loads_calibrated_threshold
cargo test test_bundled_v2_model_matches_python_reference_cases
cargo test test_exported_luts_and_threshold_are_applied
cargo test test_diagnosis_clinical_result_is_encrypted_at_rest
cargo test test_phi_safe_audit_event_created_for_screening
cargo test patient_reference_rejects_control_characters
cargo test patient_reference_rejects_excessive_length
cargo test security_headers_are_applied

echo "== Web build =="
(cd web && npm run build)

echo "== English visible-content QA =="
if rg -n -i "\b(paciente|diagn[oó]stico|cribado|contrase[nñ]a|historial|riesgo|umbral|resultado|cifrado)\b" README.md docs web/src; then
  echo "Non-English visible content candidate found; review the matches above." >&2
  exit 1
fi

echo "PoC smoke checks passed."
