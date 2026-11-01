import csv
import json
import lzma
import random
import sys
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_experiments import (
    _import_full_transformer_results,
    _per_seed_results,
    _saved_paired_comparisons,
    _spearman,
    _write_loss_figure,
)

from cuad_tagging.baselines import _write_summary
from cuad_tagging.config import BIO_LABELS
from cuad_tagging.metrics import (
    decode_bio,
    paired_bootstrap_difference,
    span_metrics,
    token_metrics,
)
from cuad_tagging.pipeline import (
    Annotation,
    make_windows,
    normalize_text,
    preprocess,
    stratified_group_split,
    validate_annotations,
)
from cuad_tagging.transformer import (
    Window,
    _collate,
    _repair_bio_sequence,
    _subsample_o_only_windows,
    _subword_windows,
    _tokenizer_options,
    _windows,
)


class TextAndOffsetTests(unittest.TestCase):
    def test_whitespace_cleaning_keeps_original_offset_map(self) -> None:
        normalized, offsets, changes = normalize_text(
            " \r\nA\t\u00a0B\fC \x00\x01\u00ad "
        )
        self.assertEqual(normalized, "A B C")
        self.assertEqual([offsets[index] for index in (0, 2, 4)], [3, 6, 8])
        self.assertEqual(changes["control_characters_removed"], 1)
        self.assertEqual(changes["format_characters_removed"], 1)

    def test_exact_span_is_kept_and_nearby_unique_span_is_repaired(self) -> None:
        text = "one clause ends. local law applies."
        spans = [
            Annotation("License Grant", 4, 10, "clause"),
            Annotation("Governing Law", 19, 22, "law"),
        ]
        kept, changes = validate_annotations(text, spans, "contract")
        self.assertEqual([(span.start, span.end) for span in kept], [(4, 10), (23, 26)])
        self.assertEqual([change["action"] for change in changes], ["corrected"])

    def test_ambiguous_nearby_span_is_reported_as_dropped(self) -> None:
        text = "x clause clause"
        spans, changes = validate_annotations(
            text, [Annotation("License Grant", 0, 6, "clause")], "contract"
        )
        self.assertEqual(spans, [])
        self.assertEqual(changes[0]["action"], "dropped")


class LabelAndSplitTests(unittest.TestCase):
    def test_roberta_tokenizer_options_support_pretokenized_words(self) -> None:
        self.assertEqual(
            _tokenizer_options("roberta-base"),
            {"use_fast": True, "add_prefix_space": True},
        )
        self.assertEqual(
            _tokenizer_options("bert-base-cased"),
            {"use_fast": True},
        )

    def test_bio_decode_closes_malformed_inside_tag(self) -> None:
        self.assertEqual(
            decode_bio(["B-License Grant", "I-License Grant", "I-Governing Law", "O"]),
            [
                ("License Grant", 0, 2),
                ("Governing Law", 2, 3),
            ],
        )

    def test_windows_cover_long_spans_without_looping(self) -> None:
        tags = ["O"] * 4 + ["B-License Grant"] + ["I-License Grant"] * 5 + ["O"] * 8
        windows = make_windows(tags, max_tokens=8, stride=3)
        self.assertEqual(windows[0], {"start": 0, "end": 4})
        self.assertEqual(windows[1]["start"], 4)
        self.assertEqual(windows[-1]["end"], len(tags))
        for previous, current in pairwise(windows):
            self.assertLessEqual(current["start"], previous["end"])

    def test_subword_windows_respect_piece_budget_and_keep_fitting_span_intact(
        self,
    ) -> None:
        tags = ["O", "B-License Grant", "I-License Grant", "I-License Grant", "O", "O"]
        counts = [1, 1, 1, 1, 2, 1]
        windows = _subword_windows(
            tags,
            counts,
            max_words=6,
            overlap_words=1,
            max_content_pieces=3,
            contract_id="contract",
        )
        self.assertEqual(windows[0], {"start": 0, "end": 1})
        self.assertEqual(windows[1], {"start": 1, "end": 4})
        self.assertEqual(windows[-1]["end"], len(tags))
        for window in windows:
            self.assertLessEqual(sum(counts[window["start"] : window["end"]]), 3)
        for previous, current in pairwise(windows):
            self.assertLessEqual(current["start"], previous["end"])

    def test_o_only_subsampling_preserves_every_labeled_window(self) -> None:
        windows = [
            Window("contract", 0, 2),
            Window("contract", 2, 4),
            Window("contract", 4, 6),
        ]
        records = {"contract": {"labels": [0, 1, 0, 0, 0, 0]}}
        selected, labeled_count, o_only_count = _subsample_o_only_windows(
            windows, records, 0.0, random.Random(42)
        )
        self.assertEqual(selected, windows[:1])
        self.assertEqual(labeled_count, 1)
        self.assertEqual(o_only_count, 0)

    def test_transformer_batches_keep_all_bio_labels_and_ignore_padding(self) -> None:
        import torch

        self.assertEqual(len(BIO_LABELS), 83)
        batch = _collate(
            [
                {
                    "inputs": {"input_ids": [101, 102], "attention_mask": [1, 1]},
                    "labels": [0, 82],
                },
                {
                    "inputs": {"input_ids": [103], "attention_mask": [1]},
                    "labels": [1],
                },
            ],
            pad_token_id=0,
        )
        self.assertEqual(batch["labels"].tolist(), [[0, 82], [1, -100]])
        self.assertEqual(batch["input_ids"].tolist(), [[101, 102], [103, 0]])
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_transformer_repairs_malformed_bio_without_changing_decoded_spans(
        self,
    ) -> None:
        raw = ["O", "I-License Grant", "I-License Grant", "O", "I-Parties"]
        repaired = _repair_bio_sequence(raw)
        self.assertEqual(
            repaired,
            ["O", "B-License Grant", "I-License Grant", "O", "B-Parties"],
        )
        self.assertEqual(decode_bio(raw), decode_bio(repaired))

    def test_validation_window_boundaries_do_not_use_gold_bio_tags(self) -> None:
        record = {
            "contract_id": "contract",
            "tokens": ["a", "b", "c", "d", "e", "f"],
            "labels": [
                0,
                BIO_LABELS.index("B-License Grant"),
                BIO_LABELS.index("I-License Grant"),
                BIO_LABELS.index("I-License Grant"),
                BIO_LABELS.index("I-License Grant"),
                0,
            ],
        }
        gold_boundaries = _windows([record], max_words=3, stride_words=0)
        blind_boundaries = _windows(
            [record], max_words=3, stride_words=0, use_gold_boundaries=False
        )
        self.assertNotEqual(gold_boundaries, blind_boundaries)
        self.assertEqual(
            blind_boundaries,
            [Window("contract", 0, 3), Window("contract", 3, 6)],
        )

    def test_contract_split_is_deterministic_and_grouped(self) -> None:
        labels = {
            f"contract-{index}": {"License Grant", "Parties"}
            if index % 2
            else {"Governing Law"}
            for index in range(20)
        }
        first = stratified_group_split(labels, seed=7)
        self.assertEqual(first, stratified_group_split(labels, seed=7))
        self.assertEqual(set(first), set(labels))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})


class EvaluationTests(unittest.TestCase):
    def test_spearman_uses_average_ranks_for_ties(self) -> None:
        self.assertAlmostEqual(_spearman([1, 2, 2, 4], [10, 20, 20, 40]), 1.0)
        self.assertIsNone(_spearman([1, 1, 1], [1, 2, 3]))

    def test_token_macro_f1_and_span_boundary_metrics(self) -> None:
        gold = [["B-License Grant", "I-License Grant", "O"]]
        predicted = [["B-License Grant", "O", "O"]]
        token = token_metrics(gold, predicted)
        spans = span_metrics(gold, predicted)
        self.assertGreater(token["macro_f1"], 0)
        self.assertLess(spans["strict"]["micro_f1"], spans["partial"]["micro_f1"])

    def test_o_only_predictions_do_not_get_span_true_positives(self) -> None:
        gold = [["B-License Grant", "I-License Grant", "O"]]
        predicted = [["O", "O", "O"]]
        spans = span_metrics(gold, predicted)
        self.assertEqual(spans["strict"]["micro_f1"], 0.0)
        self.assertEqual(spans["partial"]["micro_f1"], 0.0)

    def test_paired_bootstrap_p_value_uses_finite_sample_correction(self) -> None:
        gold = [
            ["B-License Grant", "I-License Grant"],
            ["B-License Grant", "I-License Grant"],
        ]
        first = [sequence[:] for sequence in gold]
        second = [["O", "O"], ["O", "O"]]
        result = paired_bootstrap_difference(gold, first, second, iterations=10, seed=7)
        self.assertGreater(result["two_sided_p"], 0.0)
        self.assertLessEqual(result["two_sided_p"], 1.0)


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_small_cuad_file_writes_contract_level_outputs(self) -> None:
        categories = ["Document Name"]
        documents = []
        for index in range(10):
            title = f"contract-{index}"
            text = f"Document {index} grants a license."
            documents.append(
                {
                    "title": title,
                    "paragraphs": [
                        {
                            "context": text,
                            "qas": [
                                {
                                    "id": f"{title}__{categories[0]}",
                                    "question": "Document Name",
                                    "answers": [
                                        {
                                            "text": "Document",
                                            "answer_start": 0,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps({"data": documents}), encoding="utf-8")
            output = root / "processed"
            stats = preprocess(source, output, seed=17, max_tokens=8, stride=2)
            with lzma.open(
                output / "contracts.jsonl.xz", "rt", encoding="utf-8"
            ) as source:
                rows = [json.loads(line) for line in source]
            self.assertEqual(stats["contracts"], 10)
            self.assertEqual(stats["raw_spans"], 10)
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                {row["split"] for row in rows}, {"train", "validation", "test"}
            )
            self.assertEqual(
                len(
                    (output / "span_corrections.csv")
                    .read_text(encoding="utf-8-sig")
                    .splitlines()
                ),
                1,
            )


class ResultExportTests(unittest.TestCase):
    def test_summary_import_is_optional_without_ignored_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                _import_full_transformer_results(
                    root / "results", artifacts_dir=root / "missing-artifacts"
                ),
                [],
            )

    def test_summary_retains_each_seed_score_separately(self) -> None:
        runs = [
            {
                "seed": 43,
                "token_macro_f1": 0.53,
                "span_strict_micro_f1": 0.20,
                "span_partial_micro_f1": 0.40,
                "train_seconds": 3600,
                "bootstrap_95_lower": 0.45,
                "bootstrap_95_upper": 0.55,
            },
            {
                "seed": 42,
                "token_macro_f1": 0.51,
                "span_strict_micro_f1": 0.19,
                "span_partial_micro_f1": 0.39,
                "train_seconds": 3000,
                "bootstrap_95_lower": 0.44,
                "bootstrap_95_upper": 0.54,
            },
        ]

        self.assertEqual(
            _per_seed_results(runs),
            [
                {
                    "seed": 42,
                    "token_macro_f1": 0.51,
                    "strict_span_micro_f1": 0.19,
                    "partial_span_micro_f1": 0.39,
                    "training_minutes": 50,
                    "bootstrap_95": [0.44, 0.54],
                },
                {
                    "seed": 43,
                    "token_macro_f1": 0.53,
                    "strict_span_micro_f1": 0.2,
                    "partial_span_micro_f1": 0.4,
                    "training_minutes": 60,
                    "bootstrap_95": [0.45, 0.55],
                },
            ],
        )

    def test_summary_writes_scope_and_ignores_model_specific_details(self) -> None:
        result = {
            "run_id": "transformer-seed42",
            "model": "bert",
            "architecture": "transformer",
            "checkpoint": "bert-base-cased",
            "seed": 42,
            "token_macro_f1": 0.1,
            "span_strict_micro_f1": 0.2,
            "span_strict_macro_f1": 0.3,
            "span_partial_micro_f1": 0.4,
            "span_partial_macro_f1": 0.5,
            "bootstrap_95_lower": 0.01,
            "bootstrap_95_upper": 0.2,
            "train_seconds": 1.0,
            "inference_ms_per_contract": 2.0,
            "bootstrap_iterations": 50,
            "training_contracts": 1,
            "test_contracts": 1,
            "training_scope": "smoke",
            "evaluation_scope": "smoke",
            "best_validation_macro_f1": 0.6,
            "checkpoint_path": "artifacts/checkpoint",
            "loss_curve": "loss/transformer-seed42.csv",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            _write_summary(output_dir, [result])
            with (output_dir / "metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as source:
                row = next(csv.DictReader(source))

        self.assertEqual(row["evaluation_scope"], "smoke")
        self.assertEqual(row["best_validation_macro_f1"], "0.6")
        self.assertEqual(row["loss_curve"], "loss/transformer-seed42.csv")

    def test_imports_completed_full_transformer_test_and_validation_results(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts_dir = root / "artifacts"
            run_dir = artifacts_dir / "bert-full-seed42"
            loss_dir = run_dir / "loss"
            results_dir = root / "results"
            loss_dir.mkdir(parents=True)
            (loss_dir / "bert-full-seed42.csv").write_text(
                "epoch,train_loss,validation_loss\n1,0.5,0.4\n",
                encoding="utf-8",
            )
            test_run = {
                "run_id": "bert-full-seed42",
                "model": "BERT",
                "architecture": "BERT",
                "checkpoint": "bert-base-cased",
                "seed": 42,
                "token_macro_f1": 0.3,
                "span_strict_micro_f1": 0.2,
                "span_strict_macro_f1": 0.1,
                "span_partial_micro_f1": 0.4,
                "span_partial_macro_f1": 0.3,
                "bootstrap_95_lower": 0.2,
                "bootstrap_95_upper": 0.4,
                "bootstrap_iterations": 1000,
                "train_seconds": 3600,
                "inference_ms_per_contract": 800,
                "training_contracts": 408,
                "test_contracts": 51,
                "training_scope": "full",
                "evaluation_scope": "full",
                "best_validation_macro_f1": 0.35,
                "best_epoch": 2,
                "epochs": 2,
                "learning_rate": 3e-5,
                "effective_batch_size": 16,
                "loss_curve": "artifacts/bert-full-seed42/loss/bert-full-seed42.csv",
                "predictions_path": "artifacts/bert-full-seed42/predictions.jsonl",
                "per_category": {},
            }
            validation_run = {
                **test_run,
                "run_id": "bert-full-seed42-validation",
                "evaluation_split": "validation",
                "test_contracts": None,
                "evaluation_contracts": 51,
            }
            (run_dir / "bert-full-seed42.json").write_text(
                json.dumps(test_run), encoding="utf-8"
            )
            (run_dir / "bert-full-seed42-validation.json").write_text(
                json.dumps(validation_run), encoding="utf-8"
            )

            imported = _import_full_transformer_results(results_dir, artifacts_dir)

            self.assertEqual(
                {run["evaluation_split"] for run in imported}, {"test", "validation"}
            )
            test_result = json.loads(
                (results_dir / "bert-full-seed42.json").read_text(encoding="utf-8")
            )
            self.assertEqual(test_result["evaluation_split"], "test")
            self.assertEqual(test_result["evaluation_contracts"], 51)
            self.assertEqual(test_result["loss_curve"], "loss/bert-full-seed42.csv")
            self.assertTrue((results_dir / test_result["loss_curve"]).is_file())

    def test_loss_figure_reads_utf8_bom_histories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loss_path = root / "bert-seed42.csv"
            loss_path.write_text(
                "\ufeffepoch,train_loss,validation_loss\n1,0.5,0.4\n",
                encoding="utf-8",
            )
            results_dir = root / "results"
            results_dir.mkdir()
            result = _write_loss_figure(
                results_dir,
                [
                    {
                        "run_id": "bert-seed42",
                        "architecture": "BERT",
                        "checkpoint": "bert-base-cased",
                        "loss_curve": str(loss_path),
                    }
                ],
            )

            self.assertEqual(result, ["bert-seed42"])
            self.assertTrue((results_dir / "loss_curves.png").is_file())
            self.assertTrue((results_dir / "loss_curves.pdf").is_file())

    def test_paired_comparisons_skip_missing_prediction_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            (results_dir / "paired_comparison_test.json").write_text(
                json.dumps(
                    {
                        "first_predictions": "artifacts/test_predictions.jsonl",
                        "second_predictions": "results/lookup_predictions.jsonl",
                        "paired_contract_bootstrap_macro_f1_difference_first_minus_second": {
                            "mean_difference": 0.1
                        },
                    }
                ),
                encoding="utf-8",
            )
            runs = [
                {
                    "run_id": "validation-run",
                    "architecture": "BERT",
                    "checkpoint": "bert-base-cased",
                    "predictions_path": None,
                },
                {
                    "run_id": "crf-run",
                    "architecture": "CRF",
                    "checkpoint": "features",
                    "predictions_path": "artifacts/test_predictions.jsonl",
                },
                {
                    "run_id": "lookup-run",
                    "architecture": "lookup",
                    "checkpoint": "none",
                    "predictions_path": "results/lookup_predictions.jsonl",
                },
            ]

            comparisons = _saved_paired_comparisons(results_dir, runs)

            self.assertEqual(comparisons[0]["first"], "CRF | features")
            self.assertEqual(comparisons[0]["second"], "lookup | none")


if __name__ == "__main__":
    unittest.main()
