import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  FileClock,
  LockKeyhole,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

type HealthResponse = {
  status: string;
  intended_use: string;
};

type ModelResponse = {
  intended_use: string;
  schema_version: number;
  feature_count: number;
  features: string[];
  precision_bits: number;
  scale_factor: number;
  clinical_threshold: number | null;
  recall_at_threshold: number | null;
  precision_at_threshold: number | null;
  auc_calibrated: number | null;
  auc_quantized: number | null;
  brier_score: number | null;
  dataset_sha256: string | null;
  dataset_n_samples: number | null;
  trained_at: string | null;
  metadata_complete: boolean;
  signed_models_required_in_production: boolean;
};

type ScreeningResponse = {
  id: string;
  patient_id: string | null;
  screening_positive: boolean;
  calibrated_probability: number;
  threshold_used: number;
  confidence: number;
  risk_level: string;
  encrypted_computation: boolean;
  created_at: string;
  limitations: string;
  recommended_follow_up: string;
};

type HistoryResponse = {
  items: ScreeningResponse[];
};

type AuditEvent = {
  id: string;
  event_type: string;
  subject_id: string | null;
  details_json: string;
  created_at: string;
};

type AuditResponse = {
  items: AuditEvent[];
};

type FeatureForm = {
  patient_id: string;
  age: string;
  sex: string;
  race: string;
  bmi: string;
  hypertension: string;
  sys_bp: string;
  smoking: string;
  total_chol: string;
  hdl_chol: string;
  creatinine: string;
  serum_glucose: string;
  urine_albumin: string;
  waist_circ: string;
  diabetes: string;
  hba1c: string;
};

const emptyForm: FeatureForm = {
  patient_id: "",
  age: "",
  sex: "0",
  race: "3",
  bmi: "",
  hypertension: "0",
  sys_bp: "",
  smoking: "0",
  total_chol: "",
  hdl_chol: "",
  creatinine: "",
  serum_glucose: "",
  urine_albumin: "",
  waist_circ: "",
  diabetes: "0",
  hba1c: "",
};

const sampleForm: FeatureForm = {
  patient_id: "local-pilot-001",
  age: "55",
  sex: "1",
  race: "3",
  bmi: "30.1",
  hypertension: "1",
  sys_bp: "142",
  smoking: "1",
  total_chol: "205",
  hdl_chol: "45",
  creatinine: "1.1",
  serum_glucose: "104",
  urine_albumin: "12",
  waist_circ: "102",
  diabetes: "0",
  hba1c: "5.9",
};

const featureFields = [
  ["age", "Age", "years"],
  ["bmi", "BMI", "kg/m2"],
  ["sys_bp", "Systolic BP", "mmHg"],
  ["total_chol", "Total cholesterol", "mg/dL"],
  ["hdl_chol", "HDL cholesterol", "mg/dL"],
  ["creatinine", "Creatinine", "mg/dL"],
  ["serum_glucose", "Serum glucose", "mg/dL"],
  ["urine_albumin", "Urine albumin", "mg/L"],
  ["waist_circ", "Waist circumference", "cm"],
  ["hba1c", "HbA1c", "%"],
] as const;

const binaryFields = [
  ["sex", "Sex", "Female", "Male"],
  ["hypertension", "Hypertension"],
  ["smoking", "Smoking history"],
  ["diabetes", "Diabetes"],
] as const;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new Error(
      "Cannot reach the local Pulsecure API. Start the Rust web server and open the same address, for example http://127.0.0.1:8080.",
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(body.error ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function optionalPercent(value: number | null | undefined): string {
  return typeof value === "number" ? percent(value) : "Not available";
}

function computeEgfr(ageValue: string, sexValue: string, creatinineValue: string): number {
  const age = Number(ageValue);
  const creatinine = Number(creatinineValue);
  const male = Number(sexValue) === 1;
  if (!Number.isFinite(age) || !Number.isFinite(creatinine) || creatinine <= 0) {
    return 0;
  }
  const kappa = male ? 0.9 : 0.7;
  const alpha = male ? -0.302 : -0.241;
  const ratio = creatinine / kappa;
  const sexFactor = male ? 1 : 1.012;
  return 142 * Math.min(ratio, 1) ** alpha * Math.max(ratio, 1) ** -1.2 * 0.9938 ** age * sexFactor;
}

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [history, setHistory] = useState<ScreeningResponse[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [form, setForm] = useState<FeatureForm>(emptyForm);
  const [result, setResult] = useState<ScreeningResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    const [healthData, modelData, historyData, auditData] = await Promise.all([
      api<HealthResponse>("/api/health"),
      api<ModelResponse>("/api/model"),
      api<HistoryResponse>("/api/screenings"),
      api<AuditResponse>("/api/audit"),
    ]);
    setHealth(healthData);
    setModel(modelData);
    setHistory(historyData.items);
    setAudit(auditData.items);
  };

  useEffect(() => {
    loadStatus().catch((err) => setError(err.message));
  }, []);

  const summary = useMemo(() => {
    const positives = history.filter((item) => item.screening_positive).length;
    return {
      total: history.length,
      positives,
      encrypted: history.filter((item) => item.encrypted_computation).length,
      audit: audit.length,
    };
  }, [audit.length, history]);

  const egfr = useMemo(
    () => computeEgfr(form.age, form.sex, form.creatinine),
    [form.age, form.creatinine, form.sex],
  );

  const updateField = (field: keyof FeatureForm, value: string) => {
    setForm((current) => ({ ...current, [field]: value }));
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const response = await api<ScreeningResponse>("/api/screenings", {
        method: "POST",
        body: JSON.stringify({
          patient_id: form.patient_id.trim() || null,
          features: {
            age: Number(form.age),
            sex: Number(form.sex),
            race: Number(form.race),
            bmi: Number(form.bmi),
            hypertension: Number(form.hypertension),
            sys_bp: Number(form.sys_bp),
            smoking: Number(form.smoking),
            total_chol: Number(form.total_chol),
            hdl_chol: Number(form.hdl_chol),
            creatinine: Number(form.creatinine),
            serum_glucose: Number(form.serum_glucose),
            egfr,
            urine_albumin: Number(form.urine_albumin),
            waist_circ: Number(form.waist_circ),
            diabetes: Number(form.diabetes),
            hba1c: Number(form.hba1c),
          },
        }),
      });
      setResult(response);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Screening failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="workstation">
      <header className="topbar">
        <div>
          <h1>Pulsecure</h1>
        </div>
      </header>

      <section className="notice" aria-label="Intended use">
        <ShieldCheck size={20} aria-hidden="true" />
        <span>{health?.intended_use ?? "Proof-of-concept encrypted cardiovascular screening; not for medical decisions."}</span>
      </section>

      {error && (
        <section className="error-panel" role="alert">
          <AlertTriangle size={20} aria-hidden="true" />
          <span>{error}</span>
        </section>
      )}

      <div className="grid">
        <section className="panel panel-primary" aria-labelledby="screening-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Input</p>
              <h2 id="screening-title">Feature capture</h2>
            </div>
            <button className="icon-button" type="button" onClick={() => setForm(sampleForm)} title="Load sample" aria-label="Load sample">
              <RotateCcw size={18} />
            </button>
          </div>

          <form onSubmit={submit} className="screening-form">
            <label className="field wide">
              <span>Local reference <small>optional</small></span>
              <input
                value={form.patient_id}
                onChange={(event) => updateField("patient_id", event.target.value)}
                placeholder="Pseudonymized before storage"
              />
            </label>

            <label className="field wide">
              <span>Race/ethnicity <small>NHANES RIDRETH1 category</small></span>
              <select value={form.race} onChange={(event) => updateField("race", event.target.value)}>
                <option value="1">Mexican American</option>
                <option value="2">Other Hispanic</option>
                <option value="3">Non-Hispanic White</option>
                <option value="4">Non-Hispanic Black</option>
                <option value="5">Other / multiracial</option>
              </select>
            </label>

            <div className="field-grid">
              {featureFields.map(([key, label, unit]) => (
                <label className="field" key={key}>
                  <span>
                    {label} <small>{unit}</small>
                  </span>
                  <input
                    inputMode="decimal"
                    value={form[key]}
                    onChange={(event) => updateField(key, event.target.value)}
                    required
                  />
                </label>
              ))}
            </div>

            <div className="segmented-row" aria-label="Binary clinical features">
              {binaryFields.map(([key, label, noLabel = "No", yesLabel = "Yes"]) => (
                <fieldset className="binary-field" key={key}>
                  <legend>{label}</legend>
                  <div className="segmented-control">
                    <button
                      type="button"
                      aria-pressed={form[key] === "0"}
                      onClick={() => updateField(key, "0")}
                    >
                      {noLabel}
                    </button>
                    <button
                      type="button"
                      aria-pressed={form[key] === "1"}
                      onClick={() => updateField(key, "1")}
                    >
                      {yesLabel}
                    </button>
                  </div>
                </fieldset>
              ))}
            </div>

            <div className="derived-strip" aria-live="polite">
              <span>Derived eGFR</span>
              <strong>{egfr > 0 ? egfr.toFixed(1) : "Waiting for age, sex, creatinine"}</strong>
            </div>

            <button className="primary-action" type="submit" disabled={busy}>
              <Activity size={18} />
              {busy ? "Running encrypted screening" : "Run screening"}
            </button>
          </form>
        </section>

        <aside className="panel" aria-labelledby="ops-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">System</p>
              <h2 id="ops-title">PoC runtime</h2>
            </div>
            <LockKeyhole size={20} aria-hidden="true" />
          </div>
          <dl className="metric-list">
            <Metric label="Stored screenings" value={summary.total.toString()} />
            <Metric label="Screening positive" value={summary.positives.toString()} />
            <Metric label="Encrypted runs" value={summary.encrypted.toString()} />
            <Metric label="Audit events" value={summary.audit.toString()} />
          </dl>
          <div className="model-block">
            <p className="eyebrow">Model contract</p>
            <p>{model?.features.join(" · ") ?? "Loading model metadata"}</p>
            {model && <ModelReadiness model={model} />}
          </div>
        </aside>

        <section className="panel result-panel" aria-live="polite" aria-labelledby="result-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Output</p>
              <h2 id="result-title">Screening result</h2>
            </div>
            <CheckCircle2 size={20} aria-hidden="true" />
          </div>
          {result ? <ResultView result={result} /> : <EmptyResult busy={busy} />}
        </section>

        <section className="panel history-panel" aria-labelledby="history-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Audit</p>
              <h2 id="history-title">Recent local history</h2>
            </div>
            <FileClock size={20} aria-hidden="true" />
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <p className="muted">No screenings recorded.</p>
            ) : (
              history.slice(0, 8).map((item) => <HistoryItem key={item.id} item={item} />)
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return <span className={ok ? "pill pill-ok" : "pill pill-warn"}>{label}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function ModelReadiness({ model }: { model: ModelResponse }) {
  const dataset = model.dataset_sha256 ? `${model.dataset_sha256.slice(0, 10)}...` : "Not available";
  return (
    <div className="readiness-grid" aria-label="Model readiness">
      <Metric label="Threshold" value={optionalPercent(model.clinical_threshold)} />
      <Metric label="Recall" value={optionalPercent(model.recall_at_threshold)} />
      <Metric label="Precision" value={optionalPercent(model.precision_at_threshold)} />
      <Metric label="AUC quantized" value={typeof model.auc_quantized === "number" ? model.auc_quantized.toFixed(3) : "Not available"} />
      <Metric label="Brier score" value={typeof model.brier_score === "number" ? model.brier_score.toFixed(3) : "Not available"} />
      <Metric label="Dataset hash" value={dataset} />
      {!model.metadata_complete && (
        <p className="metadata-warning">
          Missing model lineage metadata. Keep this in demo territory.
        </p>
      )}
    </div>
  );
}

function EmptyResult({ busy }: { busy: boolean }) {
  return (
    <div className="empty-result">
      <p>{busy ? "Encrypted inference is running in the local FHE queue." : "Submit patient features to generate a calibrated screening result."}</p>
    </div>
  );
}

function ResultView({ result }: { result: ScreeningResponse }) {
  return (
    <div className="result-view">
      <div className={result.screening_positive ? "result-banner positive" : "result-banner negative"}>
        <strong>{result.screening_positive ? "Screening positive" : "Screening negative"}</strong>
        <span>{result.risk_level}</span>
      </div>
      <dl className="result-metrics">
        <Metric label="Calibrated probability" value={percent(result.calibrated_probability)} />
        <Metric label="Threshold used" value={percent(result.threshold_used)} />
        <Metric label="Screening confidence" value={percent(result.confidence)} />
      </dl>
      <p className="follow-up">{result.recommended_follow_up}</p>
      <p className="muted">{result.limitations}</p>
      {result.screening_positive && (
        <p className="muted">
          High-sensitivity threshold: expect false positives. This is useful for evaluating the pipeline, not for patient care.
        </p>
      )}
    </div>
  );
}

function HistoryItem({ item }: { item: ScreeningResponse }) {
  return (
    <article className="history-item">
      <div>
        <strong>{item.screening_positive ? "Positive" : "Negative"}</strong>
        <span>{new Date(item.created_at).toLocaleString()}</span>
      </div>
      <span>{percent(item.calibrated_probability)}</span>
    </article>
  );
}
