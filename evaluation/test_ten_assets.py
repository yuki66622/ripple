"""Explicit numerical fixtures and local truth only; no Kronos/network calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from data_pipeline import load_window
from evaluation.comparison import compare_records, competition_top, initial_ranking_baseline
from evaluation.model_comparison import compare_reports
from evaluation.runner import build_manifest, main as runner_main, run_task
from evaluation.scoring import score_forecast
from evaluation.test_comparison import IDENTITY
from evaluation.test_evaluation import FixtureAdapter, fixture_forecast
from evaluation.test_model_comparison import synthetic_report
from evaluation.test_volume_quality import ledger_fixture
from evaluation.universe import DEFAULT_ASSETS, TEN_ASSETS, TEN_PROFILE
from evaluation.volume_quality import summarize_volume_quality
from forecast_metrics.engine import freeze_forecast


class TenAssetEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest(profile_id=TEN_PROFILE, limit=1, horizon=2)
        cls.task = cls.manifest["tasks"][0]
        with tempfile.TemporaryDirectory() as directory:
            cls.artifact = run_task(cls.task, directory, adapter=FixtureAdapter())
        cls.identity = {**copy.deepcopy(IDENTITY),
                        "model_revision": cls.artifact["result"]["forecast"]["model_revision"]}
        cls.artifact["result"]["runtime"]["identity"] = cls.identity

    def compare(self, artifact=None, manifest=None):
        return compare_records(manifest or self.manifest,
                               [{"path": "TEST_ONLY_ten_assets.json", "artifact": artifact or self.artifact}],
                               expected_identity=self.identity, bootstrap_replicates=0)

    def test_full_ten_asset_scoring_and_publication_audits(self):
        artifact = self.artifact
        scored = artifact["evaluation"]
        self.assertEqual(artifact["status"], "scored", artifact.get("error"))
        self.assertEqual(scored["assets"], list(TEN_ASSETS))
        self.assertEqual(len(scored["rows"]), 90)
        self.assertEqual({row["symbol"] for row in scored["rows"]}, set(TEN_ASSETS))
        self.assertEqual(scored["sample_count"], 1)
        self.assertEqual(scored["portfolio_policy"]["weights"], dict.fromkeys(TEN_ASSETS, .1))
        self.assertEqual(artifact["correction_quality"]["total_candles"], 20)
        self.assertEqual(artifact["volume_quality"]["total_candles"], 20)
        self.assertEqual(set(scored["baseline_artifacts"]), set(TEN_ASSETS))

    def test_ten_asset_pnl_uses_explicit_equal_weights(self):
        window = self.task["window"]
        frozen = freeze_forecast(fixture_forecast(window, path_factors=[[1.05, 1.1]]))
        scored = score_forecast(window, frozen.data, frozen.identity)
        self.assertEqual(scored["status"], "scored")
        metrics = scored["prediction_metrics"]["paths"][0]
        for symbol in TEN_ASSETS:
            self.assertAlmostEqual(metrics["assets"][symbol]["pnl"], 100)
        self.assertAlmostEqual(metrics["portfolio"]["pnl"], 1000)

    def test_original_three_asset_output_and_503020_policy_unchanged(self):
        window = load_window()
        frozen = freeze_forecast(fixture_forecast(window, path_factors=[[1.05, 1.1]]))
        scored = score_forecast(window, frozen.data, frozen.identity)
        self.assertEqual(scored["status"], "scored")
        self.assertEqual(len(scored["rows"]), 27)
        self.assertNotIn("assets", scored)
        self.assertNotIn("portfolio_policy", scored)
        metrics = scored["prediction_metrics"]["paths"][0]["assets"]
        for symbol, pnl in zip(DEFAULT_ASSETS, (500, 300, 200)):
            self.assertAlmostEqual(metrics[symbol]["pnl"], pnl)

    def test_strong_local_truth_gate_accepts_all_ten_assets(self):
        report = self.compare()
        self.assertEqual(report["status"], "complete", report["tasks"])
        self.assertEqual(report["assets"], list(TEN_ASSETS))
        self.assertEqual(report["coverage"]["paired_windows"], 1)
        self.assertIn("all 90 scoring rows required", report["method"]["pairing"])
        stats = report["overall"]["errors"]["volatility"]["by_asset"]
        self.assertEqual(set(stats), {*TEN_ASSETS, "equal_weight_assets"})
        self.assertAlmostEqual(stats["equal_weight_assets"]["model_mae"],
                               sum(stats[s]["model_mae"] for s in TEN_ASSETS) / 10)
        ranking = report["overall"]["top2_ranking"]
        self.assertEqual(ranking["principal_baseline"], "historical-volatility")
        self.assertIn("constant-initial-volatility", ranking["strategies"])
        self.assertNotIn("constant-ETH-SOL", ranking["strategies"])

    def test_missing_extra_asset_or_coherent_truth_tamper_cannot_pass(self):
        for case in ("missing_LTC", "wrong_LTC_truth", "wrong_holdings"):
            artifact = copy.deepcopy(self.artifact)
            evaluation = artifact["evaluation"]
            if case == "missing_LTC":
                evaluation["rows"] = [r for r in evaluation["rows"] if r["symbol"] != "LTC"]
            elif case == "wrong_LTC_truth":
                from evaluation.test_comparison import change_metric
                change_metric(artifact, "LTC", "terminal_return", predicted=0, actual=.2, baseline=0)
            else:
                evaluation["portfolio_policy"]["weights"]["LTC"] = .2
            with self.subTest(case=case):
                self.assertEqual(self.compare(artifact)["coverage"]["status_counts"], {"invalid_scored": 1})

    def test_static_baseline_is_bound_to_first_past_input(self):
        expected = initial_ranking_baseline(self.task["window"], 2)
        self.assertEqual(self.manifest["ranking_baseline"], expected)
        for key, value in (("assets", ["BTC", "ETH"]), ("as_of", "2025-01-20T00:00:00Z")):
            altered = copy.deepcopy(self.manifest)
            altered["ranking_baseline"][key] = value
            # Keep a definite difference even if the first synthetic-period top2 happens to match.
            if altered["ranking_baseline"] == expected:
                altered["ranking_baseline"]["assets"] = ["BTC"]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "first input"):
                self.compare(manifest=altered)

    def test_manifest_portfolio_and_declared_universe_are_required(self):
        for key in ("assets", "portfolio_policy"):
            altered = copy.deepcopy(self.manifest)
            del altered[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(manifest=altered)

    def test_ten_profile_plan_only_cli_never_loads_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(runner_main(["--profile", TEN_PROFILE, "--limit", "1", "--horizon", "2",
                                          "--plan-only", "--out", directory]), 0)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
        self.assertEqual(manifest["assets"], list(TEN_ASSETS))
        self.assertEqual(manifest["portfolio_policy"]["weights"], dict.fromkeys(TEN_ASSETS, .1))


def ten_volume_fixture():
    artifact = ledger_fixture(1, total=300)
    artifact["profile_id"] = TEN_PROFILE
    artifact["assets"] = list(TEN_ASSETS)
    rows = []
    for i, asset in enumerate(TEN_ASSETS):
        for metric in ("terminal_return", "volatility"):
            row = copy.deepcopy(artifact["evaluation"]["rows"][0])
            row.update(symbol=asset, metric=metric, actual=0. if i < 3 else 1., absolute_error=0. if i < 3 else 1.)
            rows.append(row)
    artifact["evaluation"]["rows"] = rows
    return artifact


class TenAssetVolumeTests(unittest.TestCase):
    def group(self, artifact):
        return summarize_volume_quality([artifact])["by_group"][0]

    def test_volume_association_uses_every_declared_asset(self):
        group = self.group(ten_volume_fixture())
        self.assertEqual(group["coverage"]["eligible_price_error_windows"], 1)
        for metric in ("terminal_return", "volatility"):
            errors = group["observations"][0]["price_errors"][metric]
            self.assertEqual(set(errors["by_asset"]), set(TEN_ASSETS))
            self.assertAlmostEqual(errors["mean_absolute_error"], .7)

    def test_missing_seven_rows_never_silently_falls_back_to_three(self):
        artifact = ten_volume_fixture()
        artifact["evaluation"]["rows"] = [r for r in artifact["evaluation"]["rows"] if r["symbol"] in DEFAULT_ASSETS]
        group = self.group(artifact)
        self.assertEqual(group["coverage"]["eligible_price_error_windows"], 0)
        self.assertEqual(group["coverage"]["distinct_windows"], 1)
        self.assertIsNone(group["observations"][0]["price_errors"]["volatility"]["mean_absolute_error"])

    def test_conflicting_asset_declaration_is_retained_and_excluded(self):
        artifact = ten_volume_fixture()
        artifact["result"]["forecast"]["spots"] = dict.fromkeys(DEFAULT_ASSETS, 1.)
        group = self.group(artifact)
        self.assertEqual(group["coverage"]["eligible_price_error_windows"], 0)
        self.assertEqual(group["coverage"]["input_records"], 1)
        self.assertIn("asset_universe", group["observations"][0]["exclusion_reasons"][0])


def ten_report(label, error):
    report = synthetic_report(label, error=error)
    report["assets"] = list(TEN_ASSETS)
    for task in report["tasks"]:
        template = copy.deepcopy(task["pairs"]["BTC"])
        task["pairs"] = {asset: copy.deepcopy(template) for asset in TEN_ASSETS}
    return report


class TenAssetModelComparisonTests(unittest.TestCase):
    def test_last_seven_asset_errors_affect_equal_asset_mean(self):
        left, right = ten_report("TEST_ONLY_left", .125), ten_report("TEST_ONLY_right", .125)
        for task in right["tasks"]:
            for asset in TEN_ASSETS[3:]:
                for pair in task["pairs"][asset].values():
                    pair.update(model=.25, model_error=0.)
        result = compare_reports(left, right)
        stat = result["errors"]["terminal_return"]["by_asset"]["equal_weight_assets"]
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(stat["right_mae"], .125 * .3)
        self.assertAlmostEqual(stat["right_vs_left_skill"], .7)
        self.assertEqual(stat["window_count"], 2)

    def test_different_universes_and_different_orders_cannot_pair(self):
        left = ten_report("TEST_ONLY_left", .125)
        for right in (synthetic_report("TEST_ONLY_right"), ten_report("TEST_ONLY_right", .125)):
            right["assets"].reverse()
            result = compare_reports(left, right)
            self.assertEqual(result["coverage"]["matched_windows"], 0)
            self.assertEqual(result["coverage"]["status_counts"], {"incompatible_universe": 2})
            self.assertFalse(result["self_comparison"])

    def test_missing_last_asset_is_invalid_verified_report(self):
        left, right = ten_report("TEST_ONLY_left", .125), ten_report("TEST_ONLY_right", .125)
        del right["tasks"][0]["pairs"]["LTC"]
        with self.assertRaisesRegex(ValueError, "every declared asset"):
            compare_reports(left, right)

    def test_top2_still_uses_competition_rank_not_top_n_minus_one(self):
        values = {asset: float(i) for i, asset in enumerate(TEN_ASSETS)}
        self.assertEqual(competition_top(values), ["LINK", "LTC"])
        values["AVAX"] = values["LINK"]
        self.assertEqual(competition_top(values), ["AVAX", "LINK", "LTC"])


if __name__ == "__main__":
    unittest.main()
