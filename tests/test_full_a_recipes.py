from ashare_signal.backtest.full_a_recipes import _admission_decision, _normalize_study_recipes


def test_full_a_recipe_study_prepends_baseline_recipe() -> None:
    assert _normalize_study_recipes(["combo_v2"]) == ["momentum_core", "combo_v2"]
    assert _normalize_study_recipes(None)[0] == "momentum_core"


def test_full_a_recipe_admission_uses_baseline_controls() -> None:
    baseline = {
        "recipe": "momentum_core",
        "total_return": 1.0,
        "max_drawdown": -0.20,
        "calmar": 2.0,
        "turnover": 10.0,
        "trade_count": 100,
    }
    candidate = {
        "recipe": "combo_v2",
        "total_return": 1.2,
        "max_drawdown": -0.21,
        "calmar": 2.4,
        "turnover": 12.0,
        "trade_count": 120,
    }

    decision = _admission_decision(candidate, baseline)

    assert decision["beats_baseline"] is True
    assert decision["status"] == "candidate"


def test_full_a_recipe_admission_keeps_failed_variant_research_only() -> None:
    baseline = {
        "recipe": "momentum_core",
        "total_return": 1.0,
        "max_drawdown": -0.20,
        "calmar": 2.0,
        "turnover": 10.0,
        "trade_count": 100,
    }
    candidate = {
        "recipe": "combo_v2",
        "total_return": 0.9,
        "max_drawdown": -0.30,
        "calmar": 1.0,
        "turnover": 20.0,
        "trade_count": 250,
    }

    decision = _admission_decision(candidate, baseline)

    assert decision["beats_baseline"] is False
    assert decision["status"] == "research_only"
    assert "total_return" in decision["notes"]
