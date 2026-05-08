from pathlib import Path

import pandas as pd

from ..data import FEATURE_SCHEMA


def generate_report(output_path: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(FEATURE_SCHEMA)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    print(generate_report().to_string(index=False))
