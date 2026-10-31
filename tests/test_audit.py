import json
import lzma
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_data import audit_source_annotations
from audit_predictions import _duplicate_audit


class DuplicateAuditTests(unittest.TestCase):
    def test_strips_source_titles_and_reports_cross_split_exact_duplicates(self) -> None:
        context = "This is a repeated contract text with enough words."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.json"
            processed_path = root / "processed.jsonl.xz"
            raw_path.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "title": "contract-a ",
                                "paragraphs": [{"context": context}],
                            },
                            {
                                "title": "contract-b",
                                "paragraphs": [{"context": context}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with lzma.open(processed_path, "wt", encoding="utf-8") as output:
                output.write(json.dumps({"contract_id": "contract-a", "split": "test"}) + "\n")
                output.write(json.dumps({"contract_id": "contract-b", "split": "train"}) + "\n")

            report, pairs = _duplicate_audit(raw_path, processed_path, 4)

        self.assertEqual(report["exact_duplicate_contract_pairs"], 1)
        self.assertEqual(report["cross_split_near_duplicate_pairs"], 1)
        self.assertEqual(pairs[0]["contract_a"], "contract-a")
        self.assertTrue(pairs[0]["exact_text_match"])

    def test_source_audit_counts_empty_and_invalid_answer_metadata(self) -> None:
        payload = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "qas": [
                                {"id": "c__Legal", "answers": []},
                                {
                                    "id": "c__Legal",
                                    "answers": [{"text": "", "answer_start": 0}],
                                },
                                {"id": "c__Legal", "answers": [{"text": "clause"}]},
                                {
                                    "id": "c__Legal",
                                    "answers": [{"text": "clause", "answer_start": 0}],
                                },
                                {"id": "c__Legal"},
                            ]
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "raw.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            report = audit_source_annotations(source)

        self.assertEqual(report["questions"], 5)
        self.assertEqual(report["empty_answer_questions"], 2)
        self.assertEqual(report["missing_answer_lists"], 1)
        self.assertEqual(report["empty_answer_entries"], 1)
        self.assertEqual(report["nonempty_answers_missing_offsets"], 1)
        self.assertEqual(report["valid_nonempty_spans"], 1)


if __name__ == "__main__":
    unittest.main()
