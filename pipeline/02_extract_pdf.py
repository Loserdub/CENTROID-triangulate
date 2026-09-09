#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 1: PDF & Document Text Extraction
Specification: AGENT.md Section 4 Stage 1

Description:
  - Scans `inputs/pdfs/` for PDF files and extracts text using PyMuPDF (fitz).
  - Scans `inputs/docs/` for DOCX and TXT files (using python-docx for DOCX).
  - Appends extracted text to `outputs/corpus.txt` with a clear source-boundary marker per file.
  - Logs filename and character count extracted per file to the console.
  - Skips and logs (without crashing) any corrupted or password-protected files.
  - If `inputs/pdfs/` and `inputs/docs/` are both empty, logs an informative message and exits cleanly.

Usage:
  python pipeline/02_extract_pdf.py
  python pipeline/02_extract_pdf.py --pdf-dir inputs/pdfs --docs-dir inputs/docs --output outputs/corpus.txt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Try importing PyMuPDF
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # Fallback for older PyMuPDF import style
    except ImportError:
        fitz = None

# Try importing python-docx
try:
    import docx
except ImportError:
    docx = None


# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PDFS_DIR = BASE_DIR / "inputs" / "pdfs"
DEFAULT_DOCS_DIR = BASE_DIR / "inputs" / "docs"
DEFAULT_RAW_DIR = BASE_DIR / "inputs" / "raw"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("02_extract_pdf")


def extract_text_from_pdf(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts plain text from a PDF file using PyMuPDF (fitz).

    Returns:
        (extracted_text, None) on success.
        (None, error_description) on failure or password-protected/corrupted files.
    """
    if fitz is None:
        return None, "PyMuPDF (fitz) library is not installed"

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return None, f"Failed to open PDF (possibly corrupted or invalid header): {e}"

    try:
        # Check if the document is encrypted / requires a password
        if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
            doc.close()
            return None, "File is encrypted or password-protected"

        page_texts: List[str] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                page_texts.append(text.strip())

        doc.close()
        full_text = "\n\n".join(page_texts).strip()
        return full_text, None
    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass
        return None, f"Error extracting text from PDF: {e}"


def extract_text_from_docx(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts plain text from a .docx file using python-docx.

    Returns:
        (extracted_text, None) on success.
        (None, error_description) on failure or corrupted files.
    """
    if docx is None:
        return None, "python-docx library is not installed"

    try:
        doc = docx.Document(file_path)
        content_lines: List[str] = []

        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                content_lines.append(text)

        # Extract text from tables if present
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    content_lines.append(" | ".join(row_cells))

        full_text = "\n\n".join(content_lines).strip()
        return full_text, None
    except Exception as e:
        return None, f"Failed to parse DOCX file (possibly corrupted or invalid format): {e}"


def extract_text_from_txt(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from a plain text file (.txt).
    Tries utf-8 first, falling back to latin-1 and cp1252 if decoding errors occur.

    Returns:
        (extracted_text, None) on success.
        (None, error_description) on failure.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, mode="r", encoding=enc) as f:
                content = f.read()
            return content.strip(), None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, f"Failed to read TXT file: {e}"

    # Final fallback with replace errors
    try:
        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content.strip(), None
    except Exception as e:
        return None, f"Failed to read TXT file with fallback encoding: {e}"


def append_to_corpus(corpus_path: Path, source_type: str, file_path: Path, text: str) -> None:
    """
    Appends extracted text to corpus.txt with a clear source boundary marker.
    Creates parent directories and corpus.txt if they do not already exist.
    """
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    filename = file_path.name
    
    # Check if corpus already exists and is non-empty
    is_non_empty = corpus_path.exists() and corpus_path.stat().st_size > 0
    prefix = "\n\n" if is_non_empty else ""
    marker = f"{prefix}--- SOURCE: {source_type} | file: {filename} ---\n"

    with open(corpus_path, mode="a", encoding="utf-8") as f:
        f.write(marker)
        f.write(text.strip())
        f.write("\n")


def scan_directory_files(directory: Path, allowed_extensions: Set[str]) -> List[Path]:
    """
    Scans a directory for files matching allowed extensions (case-insensitive).
    Returns a sorted list of Path objects.
    """
    if not directory.exists() or not directory.is_dir():
        return []

    matched_files: List[Path] = []
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in allowed_extensions:
            matched_files.append(item)
    return matched_files


def run_extraction(
    pdf_dir: Optional[Path] = None,
    docs_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_corpus: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Main orchestration function for Stage 1 document extraction:
      1. Scans pdf_dir for .pdf files.
      2. Scans docs_dir for .docx and .txt files.
      3. Scans raw_dir for raw pasted .txt, .text, and .md files.
      4. If all directories contain no matching files, logs message and exits cleanly.
      5. Extracts text per file with appropriate parser, skipping corrupted/encrypted files.
      6. Logs filename + character count extracted per file.
      7. Appends text to output_corpus with source-boundary markers.

    Returns:
        Dictionary with extraction metrics.
    """
    target_pdf_dir = pdf_dir or DEFAULT_PDFS_DIR
    target_docs_dir = docs_dir or DEFAULT_DOCS_DIR
    target_raw_dir = raw_dir or DEFAULT_RAW_DIR
    corpus_file = output_corpus or DEFAULT_CORPUS_FILE

    # Ensure input directories exist
    target_pdf_dir.mkdir(parents=True, exist_ok=True)
    target_docs_dir.mkdir(parents=True, exist_ok=True)
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = scan_directory_files(target_pdf_dir, {".pdf"})
    doc_files = scan_directory_files(target_docs_dir, {".docx", ".txt"})
    raw_files = scan_directory_files(target_raw_dir, {".txt", ".text", ".md"})
    total_files = len(pdf_files) + len(doc_files) + len(raw_files)

    if total_files == 0:
        logger.info(
            f"No PDF, DOCX, or TXT/RAW files found in '{target_pdf_dir}', '{target_docs_dir}', or '{target_raw_dir}'. Exiting cleanly."
        )
        return {
            "files_found": 0,
            "files_extracted": 0,
            "files_skipped": 0,
            "characters_extracted": 0,
        }

    logger.info(
        f"Found {len(pdf_files)} PDF file(s) in '{target_pdf_dir}', {len(doc_files)} document(s) in '{target_docs_dir}', "
        f"and {len(raw_files)} raw text file(s) in '{target_raw_dir}'."
    )

    extracted_count = 0
    skipped_count = 0
    total_characters = 0

    # 1. Process PDF files
    for pdf_path in pdf_files:
        text, error_msg = extract_text_from_pdf(pdf_path)
        if error_msg is not None:
            logger.warning(f"Skipped PDF '{pdf_path.name}': {error_msg}")
            skipped_count += 1
            continue

        char_count = len(text)
        if char_count == 0:
            logger.warning(f"Skipped PDF '{pdf_path.name}': Document contains no extractable text.")
            skipped_count += 1
            continue

        append_to_corpus(corpus_file, source_type="pdf", file_path=pdf_path, text=text)
        logger.info(f"Extracted {char_count:,} characters from '{pdf_path.name}' (PDF)")
        extracted_count += 1
        total_characters += char_count

    # 2. Process DOCX and TXT files
    for doc_path in doc_files:
        suffix = doc_path.suffix.lower()
        if suffix == ".docx":
            text, error_msg = extract_text_from_docx(doc_path)
            source_type = "docx"
        elif suffix == ".txt":
            text, error_msg = extract_text_from_txt(doc_path)
            source_type = "txt"
        else:
            continue

        if error_msg is not None:
            logger.warning(f"Skipped '{doc_path.name}': {error_msg}")
            skipped_count += 1
            continue

        char_count = len(text)
        if char_count == 0:
            logger.warning(f"Skipped '{doc_path.name}': File is empty.")
            skipped_count += 1
            continue

        append_to_corpus(corpus_file, source_type=source_type, file_path=doc_path, text=text)
        logger.info(f"Extracted {char_count:,} characters from '{doc_path.name}' ({source_type.upper()})")
        extracted_count += 1
        total_characters += char_count

    # 3. Process Raw pasted text files
    for raw_path in raw_files:
        text, error_msg = extract_text_from_txt(raw_path)
        if error_msg is not None:
            logger.warning(f"Skipped raw file '{raw_path.name}': {error_msg}")
            skipped_count += 1
            continue

        char_count = len(text)
        if char_count == 0:
            logger.warning(f"Skipped raw file '{raw_path.name}': File is empty.")
            skipped_count += 1
            continue

        append_to_corpus(corpus_file, source_type="raw", file_path=raw_path, text=text)
        logger.info(f"Extracted {char_count:,} characters from '{raw_path.name}' (RAW)")
        extracted_count += 1
        total_characters += char_count

    logger.info(
        f"Extraction complete: {extracted_count} file(s) extracted ({total_characters:,} total characters), "
        f"{skipped_count} skipped. Output appended to '{corpus_file}'."
    )

    return {
        "files_found": total_files,
        "files_extracted": extracted_count,
        "files_skipped": skipped_count,
        "characters_extracted": total_characters,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 1: PDF & Document Text Extraction"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDFS_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_PDFS_DIR})",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory containing .txt and .docx files (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing raw pasted .txt files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=f"Destination corpus file path (default: {DEFAULT_CORPUS_FILE})",
    )

    args = parser.parse_args()
    try:
        run_extraction(
            pdf_dir=args.pdf_dir,
            docs_dir=args.docs_dir,
            raw_dir=args.raw_dir,
            output_corpus=args.output,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
