import argparse
import sys
from pathlib import Path

sys.path.append(".")
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_pages, get_tokenizer

DEFAULT_PDF_PATH = "data/samples/test.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test the RAG ingestion pipeline against a PDF file.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=DEFAULT_PDF_PATH,
        help=f"Path to the PDF file to ingest (default: {DEFAULT_PDF_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.is_file():
        print(f"Error: PDF file not found at '{pdf_path}'")
        print("Please provide a valid path, e.g.:")
        print("    python scripts/manual_test.py path/to/your_file.pdf")
        print(f"Or place a test PDF at the default location: {DEFAULT_PDF_PATH}")
        sys.exit(1)

    pages = load_pdf(str(pdf_path))
    print(f"Loaded {len(pages)} pages")

    tokenizer = get_tokenizer()
    chunks = chunk_pages(pages, tokenizer)
    print(f"Created {len(chunks)} chunks")
    print("\n--- First chunk ---")
    print(chunks[0])


if __name__ == "__main__":
    main()
