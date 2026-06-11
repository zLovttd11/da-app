"""Load data from CSV, Excel, and Tableau .hyper files with encoding auto-detection."""

import io
import pandas as pd
import chardet


def detect_encoding(file_bytes: bytes) -> str:
    """Detect the character encoding of raw bytes. Falls back to utf-8."""
    result = chardet.detect(file_bytes)
    enc = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0)
    if enc is None or confidence < 0.5:
        return "utf-8"
    enc_lower = enc.lower()
    if enc_lower in ("gb2312", "gbk", "gb18030"):
        return "gbk"
    return enc


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns as numeric, categorical, datetime, or text."""
    numeric: list[str] = []
    categorical: list[str] = []
    datetime_cols: list[str] = []
    text_cols: list[str] = []

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype):
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio < 0.5:
                categorical.append(col)
            else:
                text_cols.append(col)
        else:
            categorical.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "text": text_cols,
    }


def _load_hyper(file_bytes: bytes) -> pd.DataFrame:
    """Load a .hyper file into a DataFrame using tableauhyperapi."""
    try:
        from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode
    except ImportError:
        raise ImportError("tableauhyperapi is not installed. Install it with: pip install tableauhyperapi")
    import tempfile
    import os

    tmp_path = os.path.join(tempfile.gettempdir(), "temp_upload.hyper")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=tmp_path, create_mode=CreateMode.NONE) as conn:
            schema_names = [s.name for s in conn.catalog.get_schema_names()]
            if not schema_names:
                raise ValueError("No schema found in .hyper file")
            schema = schema_names[0]
            table_names = [t.name for t in conn.catalog.get_table_names(schema=schema)]
            if not table_names:
                raise ValueError("No tables found in .hyper file")
            table_name = table_names[0]
            table_def = conn.catalog.get_table_definition(name=table_name)
            columns = [c.name for c in table_def.columns]
            rows = list(conn.execute_query(f'SELECT * FROM "{schema}"."{table_name}"'))
            df = pd.DataFrame(rows, columns=columns)
    return df


def load_data(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Load an uploaded file (CSV, Excel, .hyper) into a DataFrame.

    Returns (df, source_name).
    """
    file_bytes = uploaded_file.read()
    fname = uploaded_file.name.lower()

    if fname.endswith(".csv"):
        encoding = detect_encoding(file_bytes)
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", errors="replace")

    elif fname.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(file_bytes))

    elif fname.endswith(".hyper"):
        df = _load_hyper(file_bytes)

    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df, uploaded_file.name
