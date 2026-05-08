from pathlib import Path

import pandas as pd

REPORT_PATH = Path("docs/SUBGROUP_EVALUATION_V2.md")
CSV_PATH = Path("docs/subgroup_evaluation_v2.csv")


def write_report(rows: list[dict[str, object]], report_path: Path = REPORT_PATH, csv_path: Path = CSV_PATH) -> None:
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    lines = ["# Subgroup Evaluation V2", ""]
    if df.empty:
        lines.append("No subgroup rows were produced.")
    else:
        lines.append(df.to_markdown(index=False))
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    write_report([])


if __name__ == "__main__":
    main()
