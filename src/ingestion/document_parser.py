from typing import List
from docx import Document as DocxDocument
import pypdf
import logging

logger = logging.getLogger(__name__)


class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: str) -> List[str]:
        try:
            texts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        texts.append(
                            f"[Page {page_num + 1}]\n{text}"
                        )

            logger.info(f"Loaded PDF: {len(texts)} pages")

            return texts
        except Exception as e:
            logger.exception(f"PDF loading failed: {e}")
            return []

    @staticmethod
    def parse_txt(file_path: str) -> List[str]:
        try:
            with open(
                    file_path,
                    "r",
                    encoding="utf-8"
            ) as f:
                return [f.read()]

        except Exception as e:
            logger.exception(f"TXT loading failed: {e}")
            return []

    @staticmethod
    def parse_docx(file_path: str) -> List[str]:
        try:
            doc = DocxDocument(file_path)
            text = "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text
            )

            return [text]
        except Exception as e:
            logger.error(f"DOCX loading failed: {e}")
            return []

    def parse(self, file_path: str, file_type: str) -> List[str]:
        if file_type == 'pdf':
            return self.parse_pdf(file_path)
        elif file_type == 'txt':
            return self.parse_txt(file_path)
        elif file_type == 'docx':
            return self.parse_docx(file_path)
        else:
            logger.warning(f"Unsupported file type for parsing: {file_type}")
            return []
