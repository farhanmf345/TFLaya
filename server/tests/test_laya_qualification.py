"""Offline tests for the Laya factor scoring layer."""
import os
os.environ.setdefault("DB_URL", "sqlite://")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest
from core.laya_qualification import evaluate_laya_factors
from core.scoring import DIMENSION_WEIGHTS


class Obj:
    pass


class FakeBackend:
    def __init__(self, probability=0.8, omit=None):
        self.probability = probability
        self.omit = omit
        self.calls = []

    def ask(self, text, questions):
        self.calls.append((text, dict(questions)))
        return {
            key: self.probability
            for key in questions
            if key != self.omit
        }


def sample_objects():
    company = Obj()
    company.Company_Name = "Example Co"
    company.Annual_Turnover_Cr = 75
    company.Net_Worth_Cr = 20
    company.Current_Ratio = 1.5
    company.Years_Of_Operation = 8
    company.Key_Personnel_Count = 12
    company.Is_MSME = True
    company.Is_Startup = False
    company.Is_Women_Led = False
    company.Valid_From = "2024-01-01"
    company.Valid_To = "2030-01-01"
    dept = Obj()
    dept.Department_Name = "IT"
    return company, dept


def test_laya_scores_all_seven_factors_with_fixed_ceilings_and_evidence():
    company, dept = sample_objects()
    backend = FakeBackend(0.8)
    dimensions = evaluate_laya_factors(
        "IT support tender",
        "Bidder must have annual turnover of INR 50 Cr. ISO 27001 certification required. Similar projects are required.",
        company, dept, [], [], [], ["ISO 9001"], backend=backend,
    )
    assert len(dimensions) == 7
    assert {d.name: d.max_points for d in dimensions} == DIMENSION_WEIGHTS
    for d in dimensions:
        expected = round(d.max_points * 0.8, 2)
        if d.name in {"Domain Experience", "Technical Capability", "Certifications & Registrations"} and "UNVERIFIED" in d.notes:
            expected = min(expected, round(d.max_points * 0.60, 2))
        assert d.earned_points == expected
    assert all("Laya evidence-fit probability" in d.notes for d in dimensions)
    assert any("annual turnover" in d.notes.lower() for d in dimensions)
    assert len(backend.calls) == 7  # factor-specific evidence keeps the classifier context focused
    assert all(len(text) <= 3000 for text, _ in backend.calls)
    assert all(len(questions) == 1 for _, questions in backend.calls)


def test_missing_factor_answer_fails_closed_for_python_fallback():
    company, dept = sample_objects()
    with pytest.raises(ValueError, match="no valid probability"):
        evaluate_laya_factors(
            "Tender", "Minimum turnover required.", company, dept, [], [], [],
            backend=FakeBackend(0.8, omit="factor"),
        )


def test_hybrid_laya_assessment_can_be_limited_to_evidence_factors():
    company, dept = sample_objects()
    backend = FakeBackend(0.7)
    factors = ["Domain Experience", "Technical Capability", "Certifications & Registrations"]
    dimensions = evaluate_laya_factors(
        "IT support tender",
        "Similar projects required. Technical equipment and ISO certification required.",
        company, dept, [], [], [], ["ISO 9001"], backend=backend, factors=factors,
    )
    assert [d.name for d in dimensions] == factors
    assert {d.name: d.max_points for d in dimensions} == {
        "Domain Experience": 40,
        "Technical Capability": 35,
        "Certifications & Registrations": 30,
    }
    assert len(backend.calls) == 3
    assert all("Laya evidence-fit probability" in d.notes for d in dimensions)


def test_unverified_laya_factor_cannot_receive_unsupported_high_score():
    company, dept = sample_objects()
    dimensions = evaluate_laya_factors(
        "Vertiv UPS maintenance contract",
        "Annual maintenance of Vertiv online UPS systems at campus.",
        company, dept, [], [], [], ["ISO 9001", "ISO 27001"],
        backend=FakeBackend(0.95),
        factors=["Domain Experience", "Technical Capability", "Certifications & Registrations"],
    )
    by_name = {d.name: d for d in dimensions}
    assert by_name["Domain Experience"].earned_points <= 24.0
    assert by_name["Technical Capability"].earned_points <= 21.0
    assert by_name["Certifications & Registrations"].earned_points <= 18.0
    assert all("UNVERIFIED" in d.notes for d in dimensions)
    assert all("not a finding of non-compliance" in d.notes for d in dimensions)
