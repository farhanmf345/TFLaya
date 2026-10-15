"""Laya-assisted, evidence-grounded qualification scoring.

Laya's installed Router is a yes/no classifier rather than a text-generating LLM.
This module therefore asks one factor-level assessment question per fixed scoring
factor, maps each validated probability onto that factor's fixed point ceiling,
and generates explanations from source snippets/profile facts in Python.
The caller remains responsible for hard disqualifiers, mandatory rules, verdicts,
and final score aggregation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.scoring import DIMENSION_WEIGHTS, QualificationDimension
from core.tender_requirements import LayaBackend

# Laya assessment instructions aligned to the original Python scoring rubric.
# The Router returns a probability; the application maps it to the factor ceiling.
# These instructions align the evidence considered, but do not replace the original
# deterministic Python formulas with exact formula execution by the classifier.
FACTOR_QUESTIONS: Dict[str, str] = {
    "Financial Strength": (
        "Assess financial strength using the old Python rubric: Financial Strength is 45 points, "
        "made up of turnover vs tender value (25 points), net worth (10 points), and current ratio/liquidity (10 points). "
        "When tender value is known, turnover uses the sigmoid rule 25*sigmoid(2*(turnover/tender_value-1.5)); "
        "when unknown, old Python brackets are 20 points for turnover >=100 Cr, 14 for >=50 Cr, 8 for >=10 Cr, otherwise 3. "
        "Net worth earns up to 10 based on net_worth/max(30% of tender value, 5 Cr), or 20 Cr reference if tender value unknown. "
        "Liquidity earns up to 10 using clamp((current_ratio-1.0)/1.5, 0, 1). "
        "Use only supplied facts, do not infer missing financial data, and give a high yes probability only when the supplied profile supports a strong score under these rules."
    ),
    "Domain Experience": (
        "Does the supplied company evidence demonstrate relevant domain experience for this tender? "
        "Judge relevance to the actual procurement scope, not just broad sector overlap. A related but non-identical project is partial evidence; do not treat it as direct experience with a product the profile does not mention. "
        "Then consider the old Python 40-point rubric: 20 points for relevant matching projects using 20*min(1,sqrt(matching_projects/3)); "
        "10 points for recency using exp(-0.25*years_ago), normalized against 3, with relevant historical bids adding 0.5 each before a cap of 6; "
        "and 10 points for matching-project value relative to tender value, capped at 10. If tender value is unknown, project value uses the old log-scale reference of 50 Cr, or 2 baseline points if no values are supplied. "
        "Count projects as relevant only when their sector, title, description, or keywords match the tender/department. Missing projects are not proof of experience. Return a higher yes probability only when the supplied evidence supports a higher score under this rubric."
    ),
    "Technical Capability": (
        "Does the supplied company evidence demonstrate technical capability relevant to this tender's actual scope? "
        "Use personnel, named equipment, and configured department keywords as evidence; a raw personnel count alone does not prove possession of specialist equipment or experience. "
        "Then consider the old Python 35-point rubric: key personnel earns up to 15 points using log_scale(personnel, reference=30); "
        "equipment count earns up to 10 using log_scale(count, reference=15); and department-keyword overlap earns up to 10 using min(1, 2*matching_keywords/total_department_keywords). "
        "If no department keywords are configured, the old Python rule awards 5 points for that sub-score. Use only supplied personnel, equipment, and keyword evidence; do not invent capabilities. Higher yes probability should correspond to stronger evidence under these rules."
    ),
    "Certifications & Registrations": (
        "Assess the old Python 30-point rubric: mandatory certificates are worth up to 15 points in proportion to required certificates held; optional certificates are worth up to 10, capped at three held optional certificates; GeM registration is worth 5 points if a held certificate name contains 'GeM'. "
        "Use the actual tender-required certificates and company-held certificate list. Missing or expired certificates must not be treated as held/valid. Higher yes probability should correspond to a higher score under this rubric."
    ),
    "Legal Standing": (
        "Assess the old Python 20-point rubric: years of operation earns up to 8 points using log_scale(years, reference=20); no disqualification/blacklisting flag earns 7 points; an active company validity window earns 5 points, otherwise 3 under the scoring formula. "
        "Separately, the application applies a hard disqualifier when company Valid_To has expired. Do not invent legal facts; missing evidence is unknown, not proof of compliance."
    ),
    "Geographic Alignment": (
        "Assess the old Python 15-point rubric. If the department is an IT/remote sector, geography is not applicable and old Python awards all 15 points. Otherwise: exact registered-state match earns 7 points; if there is at least one registered state but no exact match, award 2; region match earns 5; registration in at least three states earns 3. "
        "Only use supplied tender location, department sector, and registered states. Do not invent a geographic restriction when the tender has none."
    ),
    "DEI Bonus": (
        "Assess the old Python 5-point bonus exactly: verified MSME registration = 2 points, verified Startup India registration = 2 points, verified women-led status = 1 point. "
        "Do not award points for preferences mentioned in the tender alone; the company must have the corresponding verified designation. Missing designations are not proof of eligibility."
    ),
}

FACTOR_TERMS: Dict[str, Tuple[str, ...]] = {
    "Financial Strength": ("turnover", "net worth", "financial", "revenue", "liquidity", "current ratio", "solvency", "crore", "cr.", "annual average"),
    "Domain Experience": ("experience", "similar work", "similar project", "completed project", "years of experience", "work order", "past performance"),
    "Technical Capability": ("technical", "personnel", "equipment", "manpower", "qualification", "technical capacity", "infrastructure", "engineer"),
    "Certifications & Registrations": ("certificate", "certification", "registered", "registration", "iso", "cmmi", "license", "licence", "gem"),
    "Legal Standing": ("blacklist", "blacklisted", "legal", "statutory", "solvency", "incorporation", "valid registration", "litigation", "debarred"),
    "Geographic Alignment": ("state", "local office", "local presence", "registered office", "located in", "within the state", "district"),
    "DEI Bonus": ("msme", "startup", "women-owned", "women led", "women-led", "sc/st", "mse preference"),
}

FACTOR_PROFILE_LABELS: Dict[str, Tuple[str, ...]] = {
    "Financial Strength": ("Annual turnover", "Net worth", "Current ratio"),
    "Domain Experience": ("Past projects", "Years of operation"),
    "Technical Capability": ("Key personnel count", "Equipment", "Department"),
    "Certifications & Registrations": ("Company-held certificates",),
    "Legal Standing": ("Validity from", "Validity to", "Years of operation"),
    "Geographic Alignment": ("Registered states",),
    "DEI Bonus": ("MSME registered", "Startup registered", "Women-led"),
}


def _get(obj: Any, *names: str, default=None):
    for name in names:
        try:
            value = getattr(obj, name)
            if value is not None:
                return value
        except Exception:
            pass
    return default


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def _evidence(tender_text: str, factor: str, limit: int = 5) -> List[str]:
    """Find factor-specific clauses across the *entire* supplied tender text.

    Prefer explicit eligibility/threshold language, but do not stop at the first
    few pages or truncate the source before searching. Returned snippets are
    bounded only after all clauses have been scanned.
    """
    terms = FACTOR_TERMS[factor]
    sentences = _sentences(tender_text)
    ranked = []
    seen = set()
    for index, sentence in enumerate(sentences):
        clean = re.sub(r"\s+", " ", sentence).strip()
        lower = clean.lower()
        if not clean or clean in seen or not any(term in lower for term in terms):
            continue
        seen.add(clean)
        score = sum(1 for marker in ("shall", "must", "minimum", "at least", "required", "mandatory", "eligib", "not less than", "within") if marker in lower)
        ranked.append((score, index, clean[:500]))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected = ranked[:limit]
    selected.sort(key=lambda row: row[1])
    return [row[2] for row in selected]


def build_company_profile(company: Any, dept: Any, projects: Iterable[Any], equipment: Iterable[Any], states: Iterable[Any], held_certificates: Iterable[str] = ()) -> str:
    """Serialize known company-side facts; absent values are explicitly marked unknown."""
    fields = [
        ("Company", _get(company, "Company_Name", "Company_NAME", default="Unknown")),
        ("Annual turnover (Cr)", _get(company, "Annual_Turnover_Cr", default="Unknown")),
        ("Net worth (Cr)", _get(company, "Net_Worth_Cr", default="Unknown")),
        ("Current ratio", _get(company, "Current_Ratio", default="Unknown")),
        ("Years of operation", _get(company, "Years_Of_Operation", default="Unknown")),
        ("Key personnel count", _get(company, "Key_Personnel_Count", default="Unknown")),
        ("MSME registered", _get(company, "Is_MSME", default="Unknown")),
        ("Startup registered", _get(company, "Is_Startup", default="Unknown")),
        ("Women-led", _get(company, "Is_Women_Led", default="Unknown")),
        ("Validity from", _get(company, "Valid_From", default="Unknown")),
        ("Validity to", _get(company, "Valid_To", default="Unknown")),
        ("Department", _get(dept, "Department_Name", default="Unknown")),
    ]
    lines = [f"{key}: {value}" for key, value in fields]
    certs = [str(c) for c in held_certificates if c]
    lines.append("Company-held certificates: " + (", ".join(certs[:12]) if certs else "Unknown/not supplied"))
    project_rows = []
    for p in list(projects or [])[:8]:
        title = _get(p, "Project_Title", default="")
        sector = _get(p, "Sector_Name", default="")
        year = _get(p, "Completion_Year", default="")
        value = _get(p, "Value_Cr", default="")
        description = _get(p, "Description", default="")
        keywords = _get(p, "Keywords_JSON", default="")
        if title or sector or description:
            project_rows.append(
                f"{title}; sector={sector}; completion year={year}; value Cr={value}; "
                f"description={str(description)[:180]}; keywords={str(keywords)[:120]}"
            )
    lines.append("Past projects: " + (" | ".join(project_rows) if project_rows else "Unknown/not supplied"))
    eq_rows = []
    for e in list(equipment or [])[:8]:
        label = _get(e, "Equipment_Name", "Name", "Item_Name", default=None)
        if label:
            eq_rows.append(str(label))
    lines.append("Equipment: " + (", ".join(eq_rows) if eq_rows else f"{len(list(equipment or []))} item(s); details unavailable"))
    state_rows = []
    for state in list(states or [])[:10]:
        rel = _get(state, "state", default=None)
        name = _get(rel, "State_Name", default=None) if rel else _get(state, "State_Name", default=None)
        if name:
            state_rows.append(str(name))
    lines.append("Registered states: " + (", ".join(state_rows) if state_rows else "Unknown/not supplied"))
    return "\n".join(lines)


def evaluate_laya_factors(
    tender_name: str,
    tender_description: str,
    company: Any,
    dept: Any,
    projects: Iterable[Any],
    equipment: Iterable[Any],
    states: Iterable[Any],
    held_certificates: Iterable[str] = (),
    backend=None,
    factors: Optional[Iterable[str]] = None,
) -> List[QualificationDimension]:
    """Return Laya-assessed dimensions. `factors` limits calls for hybrid scoring.

    Laya's Router is a classifier, not a generative rubric engine. Its scores for
    qualitative factors are estimates and must be reviewed when evidence is absent.
    Formula-driven factors should be calculated by Python instead.
    """
    tender_text = f"{tender_name or ''}\n{tender_description or ''}".strip()
    company_text = build_company_profile(company, dept, projects, equipment, states, held_certificates)
    profile_lines = company_text.splitlines()
    # Scan the complete tender text first, then make a factor-specific assessment.
    # Laya is a short-context classifier, so passing one giant prompt would risk
    # truncating the later clauses/company facts. Each call gets the strongest
    # clauses found anywhere in the full source plus the company profile.
    requested = list(factors) if factors is not None else list(FACTOR_QUESTIONS)
    unknown = sorted(set(requested) - set(FACTOR_QUESTIONS))
    if unknown:
        raise ValueError(f"Unknown Laya scoring factor(s): {', '.join(unknown)}")
    factor_names = requested
    backend = backend or LayaBackend()
    probs: Dict[str, float] = {}
    for factor in factor_names:
        snippets = _evidence(tender_text, factor, limit=4)
        clause_text = "\n".join(f"- {snippet[:240]}" for snippet in snippets)
        if not clause_text:
            clause_text = "No clear clause found in the supplied tender text. Treat tender-specific conditions as unknown; do not invent a requirement."
        # Keep the prompt compact enough for Laya's classifier context window.
        labels = FACTOR_PROFILE_LABELS.get(factor, ())
        profile_evidence = [
            line for line in profile_lines
            if any(line.lower().startswith(label.lower()) for label in labels)
            and not any(marker in line.lower() for marker in (
                "unknown", "not supplied", "details unavailable"
            ))
        ]
        # Give the classifier the most relevant facts first instead of truncating
        # one generic profile string, which could cut off project/certificate data.
        profile_text = "\n".join(f"- {line[:360]}" for line in profile_evidence)
        if not profile_text:
            profile_text = "No matching company-profile facts were supplied; status is unknown, not automatically non-compliant."
        assessment_text = (
            f"TENDER: {tender_name or 'Unknown'}\n"
            f"FACTOR: {factor}\n"
            f"TENDER CLAUSES FOUND ACROSS FULL DOCUMENT:\n{clause_text}\n"
            f"FACTOR-RELEVANT COMPANY EVIDENCE:\n{profile_text}\n"
            f"ASSESSMENT QUESTION: {FACTOR_QUESTIONS[factor]}\n"
            "Interpret your yes probability as how strongly the supplied evidence supports qualification on this factor, "
            "not as the probability that a requirement clause exists. If no explicit factor-specific clause is found, "
            "assess relevance to the tender scope cautiously and do not treat the missing clause alone as failure. "
            "Distinguish direct evidence from broad/indirect evidence. Do not invent facts; missing information is unknown, not proof of failure."
        )[:3000]
        answer = backend.ask(assessment_text, {"factor": FACTOR_QUESTIONS[factor]})
        if not isinstance(answer, dict):
            raise ValueError(f"Laya returned an invalid assessment for {factor}")
        value = answer.get("factor")
        if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
            probs[factor] = float(value)
    if len(probs) != len(factor_names):
        missing = sorted(set(factor_names) - set(probs))
        raise ValueError(f"Laya returned no valid probability for: {', '.join(missing)}")

    dimensions: List[QualificationDimension] = []
    for factor in factor_names:
        max_points = DIMENSION_WEIGHTS[factor]
        probability = max(0.0, min(1.0, probs[factor]))
        tender_evidence = _evidence(tender_text, factor)
        earned = round(max_points * probability, 2)
        # Laya's Router is a classifier and can return a high probability even
        # when the input contains little evidence relevant to the actual tender.
        # For the three subjective factors, limit unsupported estimates until
        # factor-specific tender requirements are found. This is a conservative
        # evidence ceiling, not a claim that the company failed the requirement.
        evidence_status = "supported"
        if factor in {"Domain Experience", "Technical Capability", "Certifications & Registrations"} and not tender_evidence:
            evidence_status = "unverified"
            evidence_ceiling = round(max_points * 0.60, 2)
            earned = min(earned, evidence_ceiling)
        if tender_evidence:
            evidence_note = "Tender evidence: " + " | ".join(tender_evidence)
        else:
            evidence_note = (
                "Tender evidence: no clear factor-specific clause found in supplied text; "
                "factor score estimates evidence-based fit to the tender scope and does not prove a mandatory condition was met or failed"
            )
        # This probability is used only for qualitative factors in hybrid mode.
        # It remains an estimate, not proof that the company meets a mandatory rule.
        confidence_note = (
            f"Laya evidence-fit probability {probability:.3f}; Python score = probability × factor maximum "
            f"= {probability:.3f} × {max_points} = {earned:.2f}/{max_points}. This is an uncalibrated classifier estimate, "
            "not a calibrated real-world probability; verify evidence before relying on score"
        )
        profile_note = (
            "Company profile: " + " | ".join(profile_evidence)
            if profile_evidence else "Company profile: no matching profile fields supplied; assessment remains uncertain"
        )
        if evidence_status == "unverified":
            confidence_note += (
                f" Evidence status=UNVERIFIED because no factor-specific tender clause was found; "
                f"the score is capped at 60% of the factor maximum ({max_points * 0.60:.2f}) "
                "to prevent an unsupported high score. This is not a finding of non-compliance."
            )
        dimensions.append(QualificationDimension(
            name=factor,
            max_points=max_points,
            earned_points=earned,
            notes=f"{confidence_note}. {evidence_note}. {profile_note}. Verify source records before relying on this assessment.",
        ))
    return dimensions
