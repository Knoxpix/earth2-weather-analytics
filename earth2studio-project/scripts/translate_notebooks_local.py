#!/usr/bin/env python3
"""Translate Jupyter notebooks into Thai without an API key.

This keeps original notebooks untouched and writes sibling `*.th.ipynb` files.
It translates:
- Markdown cells
- Human-language comments in code cells

Executable code is preserved.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


PROTECTED_RE = re.compile(
    r"(:py:(?:class|meth|func|mod|obj|attr|data|exc|ref):`[^`]+`|`[^`]+`|\.\.\s+\w+::[^\n]*|https?://\S+|[A-Za-z0-9_./+-]+\([^)]+\)|[A-Za-z0-9_./+-]+\[[^]]+\])"
)

EXACT_LINE_MAP = {
    "## Set Up": "## การเตรียมองค์ประกอบ",
    "## Execute the Workflow": "## การรัน Workflow",
    "## Post Processing": "## การทำ Post-Processing",
    "In this example you will learn:": "ในตัวอย่างนี้คุณจะได้เรียนรู้:",
    "- How to instantiate a built in prognostic model": "- วิธีสร้างอินสแตนซ์ของโมเดลพยากรณ์ที่มีมาให้ในระบบ",
    "- Creating a data source and IO object": "- วิธีสร้างแหล่งข้อมูลและออบเจ็กต์ IO",
    "- Running a simple built in workflow": "- วิธีรัน workflow พื้นฐานที่มีมาให้ในระบบ",
    "- Post-processing results": "- วิธีทำ post-processing กับผลลัพธ์",
    "Thus, we need the following:": "ดังนั้น เราจึงต้องมีองค์ประกอบต่อไปนี้:",
    "We need the following:": "เราจำเป็นต้องมีองค์ประกอบต่อไปนี้:",
    "We need the following components:": "เราจำเป็นต้องมีคอมโพเนนต์ต่อไปนี้:",
    "This example needs the following:": "ตัวอย่างนี้ต้องใช้องค์ประกอบต่อไปนี้:",
    "This example requires the following components:": "ตัวอย่างนี้ต้องใช้คอมโพเนนต์ต่อไปนี้:",
    "### Input/Output Coordinates": "### พิกัดอินพุต/เอาต์พุต",
    "### :py:func:`__call__` API": "### API ของ :py:func:`__call__`",
    "Let's instantiate the components needed.": "มาสร้างอินสแตนซ์ของคอมโพเนนต์ที่จำเป็นกัน",
    "For more information on cBottle see:": "ดูข้อมูลเพิ่มเติมเกี่ยวกับ cBottle ได้ที่:",
    "see": "ดู",
    "Studio.": "Studio",
}

EXACT_CELL_MAP = {
    "## Set Up\nAll workflows inside Earth2Studio require constructed components to be\nhanded to them. In this example, let's take a look at the most basic:\n:py:meth:`earth2studio.run.deterministic`.\n\n": "## การเตรียมองค์ประกอบ\nworkflow ทุกตัวภายใน Earth2Studio จำเป็นต้องได้รับคอมโพเนนต์ที่สร้างไว้ล่วงหน้าแล้วส่งเข้าไปให้ใช้งาน ในตัวอย่างนี้เราจะดูเวิร์กโฟลว์พื้นฐานที่สุดคือ\n:py:meth:`earth2studio.run.deterministic`.\n\n",
    "## Post Processing\nThe last step is to post process our results. Cartopy is a great library for plotting\nfields on projections of a sphere.\n\nNotice that the Zarr IO function has additional APIs to interact with the stored data.\n\n": "## การทำ Post-Processing\nขั้นตอนสุดท้ายคือการนำผลลัพธ์มาทำ post-process ต่อ Cartopy เป็นไลบรารีที่เหมาะมากสำหรับการพล็อตฟิลด์ข้อมูลบน projection ของทรงกลม\n\nสังเกตว่า Zarr IO function มี API เพิ่มเติมสำหรับใช้เข้าถึงและจัดการข้อมูลที่จัดเก็บไว้\n\n",
}

PROTECTED_TERMS = [
    "Earth2Studio",
    "Earth-2 Inference Studio",
    "Cartopy",
    "GFS",
    "WB2ERA5",
    "HRRR",
    "CBottle3D",
    "CBottle",
    "StormCast",
    "StormScope",
    "DLWP",
    "FourCastNet",
    "FCN",
    "Zarr",
    "ZarrBackend",
    "IO",
    "IOBackend",
    "API",
    "workflow",
    "inference",
    "deterministic",
    "ensemble",
    "forecast",
    "post-processing",
    "post-process",
    "lead time",
    "checkpoint",
    "batch",
    "projection",
    "Datasource",
    "Prognostic Model",
    "Diagnostic Model",
    "IO Backend",
]

GLOSSARY = {
    "Prognostic Model": "Prognostic Model",
    "Diagnostic Model": "Diagnostic Model",
    "Datasource": "Datasource",
    "IO Backend": "IO Backend",
    "built in": "built-in",
    "Earth 2": "Earth-2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", help="Notebook files or directories")
    parser.add_argument("--suffix", default=".th", help="Suffix before .ipynb")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.15,
        help="Pause in seconds between translation requests",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per chunk",
    )
    return parser.parse_args()


def iter_notebooks(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_file() and path.suffix == ".ipynb":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.glob("**/*.ipynb")))
        else:
            raise FileNotFoundError(target)
    return sorted(dict.fromkeys(paths))


def translated_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def split_protected(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = f"QXZKEEP{len(replacements)}ZXQ"
        replacements[key] = match.group(0)
        return key

    protected = PROTECTED_RE.sub(repl, text)
    for term in sorted(PROTECTED_TERMS, key=len, reverse=True):
        if term in protected:
            key = f"QXZTERM{len(replacements)}ZXQ"
            replacements[key] = term
            protected = protected.replace(term, key)
    return protected, replacements


def restore_protected(text: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def apply_glossary(text: str) -> str:
    out = text
    for src, dst in GLOSSARY.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def normalize_thai(text: str) -> str:
    text = text.replace("Earth 2", "Earth-2")
    text = text.replace("Earth2 Studio", "Earth2Studio")
    text = text.replace("สตูดิโอ Earth-2 Inference", "Earth-2 Inference Studio")
    text = text.replace("อินพุต/เอาท์พุต", "อินพุต/เอาต์พุต")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def should_skip_markdown(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(".. literalinclude::")


def split_markdown_chunks(text: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) > 900 and current:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def translate_known_line(line: str) -> str | None:
    stripped = line.strip()
    if stripped in EXACT_LINE_MAP:
        prefix = line[: len(line) - len(line.lstrip())]
        return prefix + EXACT_LINE_MAP[stripped]

    patterns = [
        (
            r"^- Prognostic Model: Use the built-?in (.+?) Model (.+)$",
            r"- Prognostic Model: ใช้โมเดล \1 ที่มีมาให้ในระบบ \2",
        ),
        (
            r"^- Diagnostic Model: Use the built-?in (.+?) Model (.+)$",
            r"- Diagnostic Model: ใช้โมเดล \1 ที่มีมาให้ในระบบ \2",
        ),
        (
            r"^- Datasource: Pull data from the (.+?) data api (.+)$",
            r"- Datasource: ดึงข้อมูลจาก \1 data API ผ่าน \2",
        ),
        (
            r"^- Datasource: Generate data from the (.+?) data api (.+)$",
            r"- Datasource: สร้างข้อมูลจาก \1 data API ผ่าน \2",
        ),
        (
            r"^- IO Backend: (?:Let's )?save the outputs into a Zarr store (.+)$",
            r"- IO Backend: บันทึกผลลัพธ์ลงใน Zarr store ผ่าน \1",
        ),
        (
            r"^- time: Input list of datetimes / strings to run inference for$",
            r"- time: รายการ datetime / string ที่ต้องการใช้รัน inference",
        ),
        (
            r"^- data: Initialized data source to fetch initial conditions from$",
            r"- data: data source ที่เตรียมไว้แล้วสำหรับดึง initial conditions",
        ),
        (
            r"^- nsteps: Number of forecast steps to predict$",
            r"- nsteps: จำนวน forecast steps ที่ต้องการพยากรณ์",
        ),
        (
            r"^- prognostic: Our initialized prognostic model$",
            r"- prognostic: โมเดลพยากรณ์ที่เราเตรียมอินสแตนซ์ไว้แล้ว",
        ),
        (
            r"^- io: IOBackend$",
            r"- io: IOBackend",
        ),
    ]
    for pattern, repl in patterns:
        if re.match(pattern, stripped):
            prefix = line[: len(line) - len(line.lstrip())]
            return prefix + re.sub(pattern, repl, stripped)
    return None


def build_translator():
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="en", target="th")


def translate_text(translator, text: str, pause: float, retries: int) -> str:
    protected, replacements = split_protected(apply_glossary(text))
    chunks = split_markdown_chunks(protected)
    translated_parts: list[str] = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        last_error = None
        result = chunk
        for attempt in range(retries):
            try:
                result = translator.translate(chunk)
                if not result:
                    result = chunk
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                time.sleep((attempt + 1) * 0.8)
        else:
            raise RuntimeError(f"Translation failed after {retries} retries: {last_error}")
        translated_parts.append(result)
        time.sleep(pause)
    return normalize_thai(restore_protected("".join(translated_parts), replacements))


def is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#")


def translate_comment_line(translator, line: str, pause: float, retries: int) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line.lstrip()
    body = stripped[1:]
    if not re.search(r"[A-Za-z]", body):
        return line
    if (
        stripped.startswith("# ///")
        or stripped.startswith("# dependencies")
        or "git+" in stripped
        or '"' in stripped
        or "'" in stripped
        or stripped.startswith("# ]")
    ):
        return line
    translated = translate_text(translator, body.strip(), pause, retries)
    return f"{indent}# {translated}"


def translate_inline_comment(translator, line: str, pause: float, retries: int) -> str:
    if "#" not in line or re.match(r"^\s*#", line):
        return line
    head, tail = line.split("#", 1)
    if not re.search(r"[A-Za-z]", tail):
        return line
    translated = translate_text(translator, tail.strip(), pause, retries)
    return f"{head}# {translated}"


def translate_code(translator, text: str, pause: float, retries: int) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if is_comment_line(line):
            out_lines.append(translate_comment_line(translator, line, pause, retries))
        else:
            out_lines.append(translate_inline_comment(translator, line, pause, retries))
    if text.endswith("\n"):
        return "\n".join(out_lines) + "\n"
    return "\n".join(out_lines)


def translate_markdown(translator, text: str, pause: float, retries: int) -> str:
    if should_skip_markdown(text):
        return text
    if text in EXACT_CELL_MAP:
        return EXACT_CELL_MAP[text]

    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        known = translate_known_line(line)
        if known is not None:
            out_lines.append(known)
            continue
        out_lines.append(translate_text(translator, line, pause, retries))
    if text.endswith("\n"):
        return "\n".join(out_lines) + "\n"
    return "\n".join(out_lines)


def process_notebook(path: Path, suffix: str, translator, pause: float, retries: int) -> Path:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            cell["source"] = [translate_markdown(translator, source, pause, retries)]
        elif cell.get("cell_type") == "code":
            cell["source"] = [translate_code(translator, source, pause, retries)]
    out = translated_path(path, suffix)
    out.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    args = parse_args()
    translator = build_translator()
    notebooks = iter_notebooks(args.targets)
    for path in notebooks:
        if path.name.endswith(f"{args.suffix}.ipynb"):
            continue
        out = process_notebook(path, args.suffix, translator, args.pause, args.retries)
        print(f"{path} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
