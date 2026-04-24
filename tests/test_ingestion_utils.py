import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ingestion_utils


class NamedBytesIO(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


class IngestionUtilsTests(unittest.TestCase):
    def setUp(self):
        ingestion_utils._GENAI_CONFIGURED = False

    def test_clean_text_returns_empty_string_for_none(self):
        self.assertEqual(ingestion_utils.clean_text(None), "")

    def test_chunk_text_rejects_overlap_that_can_loop_forever(self):
        with self.assertRaises(ValueError):
            ingestion_utils.chunk_text("one two three", chunk_size=3, overlap=3)

    def test_read_txt_rewinds_before_reading(self):
        uploaded_file = NamedBytesIO(b"hello world", "note.txt")
        uploaded_file.read()

        self.assertEqual(ingestion_utils.read_txt(uploaded_file), "hello world")

    def test_extract_text_reads_text_files(self):
        uploaded_file = NamedBytesIO(b"line one\nline two", "note.txt")

        self.assertEqual(
            ingestion_utils.extract_text(uploaded_file),
            "line one\nline two",
        )

    def test_get_embedding_raises_when_key_is_missing(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            with self.assertRaises(ValueError):
                ingestion_utils.get_embedding("test")

    def test_get_embedding_configures_genai_once(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch.object(ingestion_utils.genai, "configure") as configure_mock:
                with patch.object(
                    ingestion_utils.genai,
                    "embed_content",
                    return_value={"embedding": [0.1, 0.2]},
                ) as embed_mock:
                    first = ingestion_utils.get_embedding("first")
                    second = ingestion_utils.get_embedding("second")

        self.assertEqual(first, [0.1, 0.2])
        self.assertEqual(second, [0.1, 0.2])
        configure_mock.assert_called_once_with(api_key="test-key")
        self.assertEqual(embed_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
