#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject


ABSTRACT_LINES = [
    "Large language models can produce",
    "plausible forecasting rationales, but it is",
    "unclear whether adding structure improves",
    "forecast quality. We evaluate 1,580",
    "resolved binary questions from Metaculus",
    "with three models (Qwen2.5-7B-Instruct,",
    "Qwen3-32B, and GPT-OSS-120B), nine",
    "prompt variants, and six temperatures per",
    "model. We score 162 runs with strict",
    "accuracy, Brier score, and Expected",
    "Calibration Error. Across the full sweep,",
    "the neutral baseline is the strongest overall",
    "setting on average, showing that extra",
    "rationale constraints often hurt both",
    "accuracy and calibration. Among structured",
    "prompts, temporal anchors and credibility",
    "are the most robust: temporal anchors",
    "produce the best single run, reaching",
    "83.0% accuracy with GPT-OSS-120B at",
    "T=0.125, while credibility prompting gives",
    "Qwen3-32B its best calibration. By",
    "contrast, predicted-event restatement, key",
    "attributes, key conditions, step-by-step",
    "reasoning, and uncertainty language",
    "usually underperform the baseline.",
    "Temperature effects are model-specific",
    "rather than monotonic, so rationale design",
    "should be tuned jointly with model family",
    "and decoding settings.",
]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_overlay_stream() -> bytes:
    lines = [
        "q",
        "1 1 1 rg",
        "0 195 300 430 re",
        "f",
        "BT",
        "/FABS 12 Tf",
        "0 g",
        "70 610 Td",
        f"({pdf_escape('Abstract')}) Tj",
        "ET",
        "BT",
        "/FREG 10 Tf",
        "0 g",
        "70 590 Td",
        "11.7 TL",
    ]
    for index, text in enumerate(ABSTRACT_LINES):
        if index == 0:
            lines.append(f"({pdf_escape(text)}) Tj")
        else:
            lines.append("T*")
            lines.append(f"({pdf_escape(text)}) Tj")
    lines.extend(["ET", "Q"])
    return "\n".join(lines).encode("utf-8")


def ensure_fonts(writer: PdfWriter, page) -> None:
    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
    else:
        resources = resources.get_object()
    fonts = resources.get("/Font")
    if fonts is None:
        fonts = DictionaryObject()
    else:
        fonts = fonts.get_object()

    regular_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    bold_font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
        }
    )

    fonts[NameObject("/FREG")] = writer._add_object(regular_font)
    fonts[NameObject("/FABS")] = writer._add_object(bold_font)
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources


def append_stream(writer: PdfWriter, page, stream_data: bytes) -> None:
    overlay_stream = DecodedStreamObject()
    overlay_stream.set_data(stream_data)
    overlay_ref = writer._add_object(overlay_stream)

    contents = page.get("/Contents")
    if contents is None:
        page[NameObject("/Contents")] = overlay_ref
        return
    if isinstance(contents, ArrayObject):
        contents.append(overlay_ref)
        return
    page[NameObject("/Contents")] = ArrayObject([contents, overlay_ref])


def update_pdf(input_path: Path, output_path: Path, backup_path: Path | None) -> None:
    if backup_path is not None and not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    reader = PdfReader(str(input_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    first_page = writer.pages[0]
    ensure_fonts(writer, first_page)
    append_stream(writer, first_page, build_overlay_stream())

    with output_path.open("wb") as handle:
        writer.write(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay an updated abstract onto the ACL PDF.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.pdf"),
        help="Input PDF path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=Path("ACL_Evaluating Rationale of LLMs.original.pdf"),
        help="Optional backup path. Pass an empty string to disable backups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_path = None if str(args.backup) == "" else args.backup
    update_pdf(args.input, args.output, backup_path)


if __name__ == "__main__":
    main()
