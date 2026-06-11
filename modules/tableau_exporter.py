"""Export processed data as Tableau .hyper extract files."""

import pandas as pd
import os
import tempfile


def export_to_hyper(df: pd.DataFrame, output_path: str | None = None) -> str:
    """Export a DataFrame to a Tableau .hyper file."""
    try:
        from tableauhyperapi import (
            HyperProcess, Telemetry, Connection, CreateMode,
            TableDefinition, SqlType, TableName, Inserter,
        )
    except ImportError:
        raise ImportError(
            "tableauhyperapi is not installed. Install it with: pip install tableauhyperapi"
        )

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "da_export.hyper")

    def _dtype_to_sql(dtype) -> SqlType:
        if pd.api.types.is_integer_dtype(dtype):
            return SqlType.big_int()
        elif pd.api.types.is_float_dtype(dtype):
            return SqlType.double()
        elif pd.api.types.is_bool_dtype(dtype):
            return SqlType.bool()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return SqlType.timestamp()
        else:
            return SqlType.text()

    columns = [
        TableDefinition.Column(str(col), _dtype_to_sql(df[col].dtype))
        for col in df.columns
    ]
    table_def = TableDefinition(TableName("Extract", "Extract"), columns)

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(
            endpoint=hyper.endpoint,
            database=output_path,
            create_mode=CreateMode.CREATE_AND_REPLACE,
        ) as conn:
            conn.catalog.create_schema("Extract")
            conn.catalog.create_table(table_def)
            with Inserter(conn, table_def) as inserter:
                inserter.add_rows(df.values.tolist())
                inserter.execute()
    return output_path
