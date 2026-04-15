from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

from data_loader import load_and_prep_obd_data


class PALExecutionError(RuntimeError):
    pass


@dataclass
class PALResult:
    answer: str
    code: str
    result_preview: str


def _extract_python_code(text: str) -> str:
    marker = "```"
    if marker not in text:
        return text.strip()

    chunks = text.split(marker)
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped.startswith("python"):
            return stripped.removeprefix("python").strip()
    return chunks[1].strip() if len(chunks) > 1 else text.strip()


def _is_code_safe(code: str) -> tuple[bool, str]:
    blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Lambda,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Delete,
        ast.Raise,
    )

    blocked_names = {
        "__import__",
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }

    blocked_attributes = {
        "to_pickle",
        "to_sql",
        "to_parquet",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_csv",
    }

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"Generated code has invalid syntax: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            return False, f"Blocked Python construct: {type(node).__name__}"

        if isinstance(node, ast.Name) and node.id in blocked_names:
            return False, f"Blocked symbol used: {node.id}"

        if isinstance(node, ast.Attribute) and node.attr in blocked_attributes:
            return False, f"Blocked DataFrame operation: .{node.attr}"

    return True, ""


def _normalize_generated_code(code: str) -> str:
    """Normalize common LLM script-style output to PAL-style dataframe code.

    Keeps useful computation while removing redundant setup lines such as:
    - import pandas as pd
    - df = pd.read_csv(...)
    """

    normalized_lines: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if lower in {"import pandas as pd", "import pandas"}:
            continue

        if lower.startswith("df = pd.read_csv(") or lower.startswith("df=pd.read_csv("):
            continue

        normalized_lines.append(raw_line)

    return "\n".join(normalized_lines).strip()


class OllamaPAL:
    def __init__(
        self,
        model: str = "granite-code:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        req = request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise ConnectionError(
                "Could not reach Ollama. Ensure Ollama is running and granite-code:8b is installed."
            ) from exc

        data = json.loads(raw)
        return data["message"]["content"]

    def _build_schema_context(self, df: pd.DataFrame) -> str:
        dtypes = ", ".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())
        sample = self._render_table(df.head(5))
        return (
            f"Rows: {len(df)}\n"
            f"Columns ({len(df.columns)}): {list(df.columns)}\n"
            f"Dtypes: {dtypes}\n\n"
            f"Sample:\n{sample}"
        )

    @staticmethod
    def _render_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return df.to_string(index=False)

    def _generate_code(
        self,
        df: pd.DataFrame,
        question: str,
        conversation_context: str = "",
        execution_error: str = "",
    ) -> str:
        schema_context = self._build_schema_context(df)

        system_prompt = (
            "You are a strict pandas code generator. "
            "Write Python code only, no explanations. "
            "Assume a pandas DataFrame named df already exists. "
            "Do not import anything. "
            "Store the final answer object in variable result. "
            "Only reference columns that exist in the provided dataset context."
        )

        conversation_block = ""
        if conversation_context.strip():
            conversation_block = f"Prior conversation context (for context only, not data):\n{conversation_context}\n\n"

        error_block = ""
        if execution_error.strip():
            error_block = (
                "Previous generated code failed. Fix it and regenerate valid code.\n"
                f"Execution error:\n{execution_error}\n\n"
            )

        user_prompt = (
            "Dataset context:\n"
            f"{schema_context}\n\n"
            f"{conversation_block}"
            f"{error_block}"
            "Question:\n"
            f"{question}\n\n"
            "Rules:\n"
            "1) Use only pandas operations on df.\n"
            "2) Return concise object in result (scalar, Series, or small DataFrame).\n"
            "3) If question is ambiguous, choose the most reasonable interpretation.\n"
            "4) Never select columns that are not present in df.columns.\n"
            "5) For follow-up conversion questions (e.g., mph to km/h), compute from relevant existing columns only."
        )

        text = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _extract_python_code(text)

    def _execute_code(self, df: pd.DataFrame, code: str) -> Any:
        code = _normalize_generated_code(code)

        is_safe, reason = _is_code_safe(code)
        if not is_safe:
            raise PALExecutionError(f"Rejected generated code. {reason}\n\nCode:\n{code}")

        safe_builtins = {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "abs": abs,
        }

        local_vars: dict[str, Any] = {"df": df.copy(deep=False), "pd": pd}
        global_vars: dict[str, Any] = {"__builtins__": safe_builtins}

        try:
            exec(compile(code, "<ollama_pal>", "exec"), global_vars, local_vars)
        except Exception as exc:
            raise PALExecutionError(f"Generated code failed: {exc}\n\nCode:\n{code}") from exc

        if "result" not in local_vars:
            raise PALExecutionError("Generated code did not set 'result'.")

        return local_vars["result"]

    def _summarize_result(self, question: str, result: Any, conversation_context: str = "") -> tuple[str, str]:
        if isinstance(result, pd.DataFrame):
            preview = self._render_table(result.head(10))
        elif isinstance(result, pd.Series):
            preview = result.head(10).to_string()
        else:
            preview = str(result)

        conversation_block = ""
        if conversation_context.strip():
            conversation_block = f"Prior conversation context:\n{conversation_context}\n\n"

        prompt = (
            "Answer the question using the computed result. "
            "Be concise and data-focused."
            f"\n\n{conversation_block}"
            f"Current question:\n{question}"
            f"\n\nComputed result:\n{preview}"
        )

        answer = self._chat(
            [
                {"role": "system", "content": "You are a precise data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return answer.strip(), preview

    def ask(self, df: pd.DataFrame, question: str, conversation_context: str = "") -> PALResult:
        code = self._generate_code(df=df, question=question, conversation_context=conversation_context)

        try:
            result = self._execute_code(df=df, code=code)
        except PALExecutionError as first_exc:
            retry_code = self._generate_code(
                df=df,
                question=question,
                conversation_context=conversation_context,
                execution_error=str(first_exc),
            )
            result = self._execute_code(df=df, code=retry_code)
            code = retry_code

        answer, preview = self._summarize_result(question=question, result=result, conversation_context=conversation_context)
        return PALResult(answer=answer, code=code, result_preview=preview)


def ask_question_on_csv(
    csv_path: str | Path,
    question: str,
    model: str = "granite-code:8b",
    conversation_context: str = "",
) -> PALResult:
    resolved_path = _resolve_csv_path(csv_path)
    df = load_and_prep_obd_data(
        resolved_path,
        coerce_numeric=True,
        parse_timestamp=True,
        drop_rows_missing_required=True,
        drop_duplicates=True,
        preserve_empty_cells=True,
    )
    pal = OllamaPAL(model=model)
    return pal.ask(df=df, question=question, conversation_context=conversation_context)


def _resolve_csv_path(csv_path: str | Path) -> Path:
    original = Path(csv_path)
    if original.exists() and original.is_file():
        return original

    cwd = Path.cwd()
    candidates = [
        cwd / original,
        cwd / "data" / "obdiidata" / original.name,
        cwd / "data" / original.name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    hint = (
        f"CSV file not found: '{csv_path}'. "
        "Try an explicit path like 'data/obdiidata/drive1.csv' or pass a full path."
    )
    raise FileNotFoundError(hint)
