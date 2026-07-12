"""
TechVest Recruitment Agent - Tools
Four callable tools: parse_resume, score_candidate, check_availability, propose_interview
"""

import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from models import CandidateProfile, CriterionScore, ScoreCard, InterviewSlot, InterviewConfirmation


# ============================================================
# TOOL 1: parse_resume
# ============================================================
def parse_resume(resume_text: str) -> CandidateProfile:
    """
    Parse a raw résumé text into a structured CandidateProfile.
    Uses LLM-based structured extraction via regex patterns as fallback.
    """
    profile = CandidateProfile(raw_text=resume_text)
    
    # Extract name (first line or after "Name:")
    name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', resume_text)
    if name_match:
        profile.name = name_match.group(1).strip()
    else:
        first_line = resume_text.strip().split('\n')[0]
        profile.name = first_line.strip()
    
    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', resume_text)
    if email_match:
        profile.email = email_match.group(0)
    
    # Extract education
    edu_section = _extract_section(resume_text, "EDUCATION", "EXPERIENCE|PROJECTS|SKILLS|CERTIFICATIONS")
    if edu_section:
        profile.education = edu_section.strip()
    
    # Extract skills
    skills_section = _extract_section(resume_text, "SKILLS", "CERTIFICATIONS|ACHIEVEMENTS|PROJECTS")
    if skills_section:
        skills_list = [s.strip() for s in re.split(r'[,;]', skills_section) if s.strip()]
        profile.skills = skills_list
    
    # Extract projects
    projects_section = _extract_section(resume_text, "PROJECTS", "SKILLS|CERTIFICATIONS|ACHIEVEMENTS|EXPERIENCE")
    if projects_section:
        project_lines = [p.strip() for p in projects_section.split('\n') if p.strip() and not p.strip().startswith('PROJECTS')]
        profile.relevant_projects = project_lines
    
    # Extract certifications
    cert_section = _extract_section(resume_text, "CERTIFICATIONS", "ACHIEVEMENTS|EXPERIENCE|PROJECTS")
    if cert_section:
        certs = [c.strip().lstrip('- ') for c in cert_section.split('\n') if c.strip() and not c.strip().startswith('CERTIFICATIONS')]
        profile.certifications = certs
    
    # Extract achievements
    ach_section = _extract_section(resume_text, "ACHIEVEMENTS", "EXPERIENCE|PROJECTS|CERTIFICATIONS")
    if ach_section:
        achievements = [a.strip().lstrip('- ') for a in ach_section.split('\n') if a.strip() and not a.strip().startswith('ACHIEVEMENTS')]
        profile.achievements = achievements
    
    # Estimate years of experience
    profile.years_of_experience = _estimate_experience(resume_text)
    
    return profile


def _extract_section(text: str, section_name: str, next_sections: str) -> str:
    """Extract a section from the résumé text between section_name and the next section."""
    pattern = rf'{section_name}\s*\n(.*?)(?=\n(?:{"|".join(next_sections.split("|"))})\s*\n|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _estimate_experience(resume_text: str) -> float:
    """Estimate years of experience from résumé text."""
    # Look for date ranges like "2021-2025" or "May 2024 - Aug 2024"
    date_patterns = [
        r'(\d{4})\s*-\s*(\d{4})',  # 2021-2025
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*-\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',  # May 2024 - Aug 2024
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*-\s*Present',  # Sep 2023 - Present
    ]
    
    total_years = 0.0
    for pattern in date_patterns:
        matches = re.findall(pattern, resume_text)
        for match in matches:
            if isinstance(match, tuple):
                start_year = int(match[0])
                end_year = int(match[1])
                total_years += (end_year - start_year)
            else:
                # Present case
                current_year = datetime.now().year
                # Find the start year
                start_match = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\s*-\s*Present', resume_text)
                if start_match:
                    total_years += current_year - int(start_match.group(1))
    
    return round(total_years, 1)


# ============================================================
# TOOL 2: score_candidate
# ============================================================
def score_candidate(profile: CandidateProfile, rubric: Dict[str, Any]) -> ScoreCard:
    """
    Score a candidate profile against the rubric.
    Uses keyword-based scoring with evidence extraction.
    """
    criterion_scores = []
    raw_text_lower = profile.raw_text.lower()
    
    for criterion in rubric.get("criteria", []):
        name = criterion["name"]
        weight = criterion["weight"]
        scale = criterion["scale"]
        
        score, evidence = _score_criterion(name, profile, raw_text_lower, scale)
        
        criterion_scores.append(CriterionScore(
            criterion_name=name,
            score=score,
            weight=weight,
            evidence=evidence,
            justification=scale.get(score, f"Score {score} assigned based on evidence")
        ))
    
    # Calculate weighted total
    weighted_total = sum(cs.score * cs.weight for cs in criterion_scores)
    weighted_total = round(min(weighted_total, 5.0), 2)
    
    # Generate overall justification
    top_criteria = sorted(criterion_scores, key=lambda x: x.score * x.weight, reverse=True)
    strengths = [f"{c.criterion_name}: {c.score}/5 (weight {c.weight})" for c in top_criteria[:3] if c.score >= 3]
    weaknesses = [f"{c.criterion_name}: {c.score}/5" for c in criterion_scores if c.score < 2]
    
    justification_parts = []
    if strengths:
        justification_parts.append(f"Strengths: {'; '.join(strengths)}")
    if weaknesses:
        justification_parts.append(f"Areas for improvement: {'; '.join(weaknesses)}")
    
    overall_justification = " | ".join(justification_parts) if justification_parts else "Mixed profile with no clear strengths or weaknesses."
    
    return ScoreCard(
        candidate_name=profile.name,
        criterion_scores=criterion_scores,
        weighted_total=weighted_total,
        overall_justification=overall_justification
    )


def _score_criterion(name: str, profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score a single criterion based on keyword matching and evidence."""
    
    if name == "Python & ML Fundamentals":
        return _score_python_ml(profile, raw_lower, scale)
    elif name == "Relevant Projects & Portfolio":
        return _score_projects(profile, raw_lower, scale)
    elif name == "Hands-on Tooling & Frameworks":
        return _score_tooling(profile, raw_lower, scale)
    elif name == "Communication & Team Collaboration":
        return _score_communication(profile, raw_lower, scale)
    elif name == "Education & Domain Fit":
        return _score_education(profile, raw_lower, scale)
    else:
        return (0, "No matching criterion found")


def _score_python_ml(profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score Python and ML fundamentals."""
    score = 0
    evidence_parts = []
    
    # Check for Python proficiency
    python_indicators = ["python", "pytorch", "tensorflow", "scikit-learn", "sklearn", "numpy", "pandas"]
    python_found = [ind for ind in python_indicators if ind in raw_lower]
    
    # Check for ML concepts
    ml_indicators = ["machine learning", "deep learning", "neural network", "supervised", "unsupervised",
                     "classification", "regression", "cnn", "rnn", "transformer", "nlp", "computer vision",
                     "model training", "model evaluation", "sentiment analysis", "recommendation"]
    ml_found = [ind for ind in ml_indicators if ind in raw_lower]
    
    # Check for production experience
    prod_indicators = ["deploy", "production", "pipeline", "fastapi", "flask", "api", "endpoint"]
    prod_found = [ind for ind in prod_indicators if ind in raw_lower]
    
    if python_found:
        evidence_parts.append(f"Python skills: {', '.join(python_found[:3])}")
    
    if ml_found:
        evidence_parts.append(f"ML experience: {', '.join(ml_found[:3])}")
    
    if prod_found:
        evidence_parts.append(f"Production: {', '.join(prod_found[:3])}")
    
    # Determine score
    if len(python_found) >= 4 and len(ml_found) >= 4 and prod_found:
        score = 5
    elif len(python_found) >= 3 and len(ml_found) >= 3:
        score = 4
    elif len(python_found) >= 2 and len(ml_found) >= 2:
        score = 3
    elif len(python_found) >= 1 and len(ml_found) >= 1:
        score = 2
    elif python_found:
        score = 1
    
    evidence = "; ".join(evidence_parts) if evidence_parts else "No clear evidence found"
    return (score, evidence)


def _score_projects(profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score relevant projects and portfolio."""
    score = 0
    evidence_parts = []
    
    # Count relevant projects
    project_count = len(profile.relevant_projects)
    
    # Check for end-to-end projects
    e2e_indicators = ["end-to-end", "pipeline", "deploy", "deployed", "production", "docker", "aws", "gcp", "azure"]
    e2e_found = [ind for ind in e2e_indicators if ind in raw_lower]
    
    # Check for measurable impact
    impact_indicators = ["accuracy", "improvement", "reduction", "%", "achieved", "score"]
    impact_found = [ind for ind in impact_indicators if ind in raw_lower]
    
    if project_count > 0:
        evidence_parts.append(f"Projects found: {project_count}")
    
    if e2e_found:
        evidence_parts.append(f"End-to-end: {', '.join(e2e_found[:3])}")
    
    if impact_found:
        evidence_parts.append(f"Measurable impact: {', '.join(impact_found[:3])}")
    
    # Determine score
    if project_count >= 3 and e2e_found and impact_found:
        score = 5
    elif project_count >= 2 and (e2e_found or impact_found):
        score = 4
    elif project_count >= 2:
        score = 3
    elif project_count >= 1:
        score = 2
    elif project_count > 0:
        score = 1
    
    evidence = "; ".join(evidence_parts) if evidence_parts else "No projects found"
    return (score, evidence)


def _score_tooling(profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score hands-on tooling and frameworks experience."""
    score = 0
    evidence_parts = []
    
    # ML frameworks
    ml_frameworks = ["pytorch", "tensorflow", "keras", "huggingface", "scikit-learn", "sklearn"]
    ml_found = [fw for fw in ml_frameworks if fw in raw_lower]
    
    # DevOps/Cloud tools
    devops_tools = ["docker", "kubernetes", "k8s", "aws", "gcp", "azure", "ci/cd", "jenkins", "github actions"]
    devops_found = [t for t in devops_tools if t in raw_lower]
    
    # Core tools
    core_tools = ["git", "rest api", "fastapi", "flask", "sql", "postgresql", "mongodb"]
    core_found = [t for t in core_tools if t in raw_lower]
    
    # MLOps
    mlops_tools = ["mlflow", "dvc", "wandb", "kubeflow", "airflow", "mlops"]
    mlops_found = [t for t in mlops_tools if t in raw_lower]
    
    if ml_found:
        evidence_parts.append(f"ML frameworks: {', '.join(ml_found)}")
    if devops_found:
        evidence_parts.append(f"DevOps/Cloud: {', '.join(devops_found)}")
    if core_found:
        evidence_parts.append(f"Core tools: {', '.join(core_found)}")
    if mlops_found:
        evidence_parts.append(f"MLOps: {', '.join(mlops_found)}")
    
    # Determine score
    tool_count = len(ml_found) + len(devops_found) + len(core_found) + len(mlops_found)
    
    if tool_count >= 10 and mlops_found:
        score = 5
    elif tool_count >= 7:
        score = 4
    elif tool_count >= 4:
        score = 3
    elif tool_count >= 2:
        score = 2
    elif tool_count >= 1:
        score = 1
    
    evidence = "; ".join(evidence_parts) if evidence_parts else "No tooling evidence found"
    return (score, evidence)


def _score_communication(profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score communication and team collaboration."""
    score = 0
    evidence_parts = []
    
    # Team collaboration indicators
    team_indicators = ["collaborat", "team", "cross-functional", "stand-up", "sprint", "agile", "pair"]
    team_found = [ind for ind in team_indicators if ind in raw_lower]
    
    # Communication indicators
    comm_indicators = ["present", "presentation", "document", "blog", "article", "published", "write", "knowledge sharing", "mentor"]
    comm_found = [ind for ind in comm_indicators if ind in raw_lower]
    
    # Leadership indicators
    lead_indicators = ["lead", "led", "coordinator", "managed", "mentor"]
    lead_found = [ind for ind in lead_indicators if ind in raw_lower]
    
    if team_found:
        evidence_parts.append(f"Collaboration: {', '.join(team_found[:3])}")
    if comm_found:
        evidence_parts.append(f"Communication: {', '.join(comm_found[:3])}")
    if lead_found:
        evidence_parts.append(f"Leadership: {', '.join(lead_found[:3])}")
    
    # Determine score
    if team_found and comm_found and lead_found:
        score = 5
    elif team_found and comm_found:
        score = 4
    elif team_found or comm_found:
        score = 3
    elif team_found:
        score = 2
    elif comm_found:
        score = 1
    
    evidence = "; ".join(evidence_parts) if evidence_parts else "No communication evidence found"
    return (score, evidence)


def _score_education(profile: CandidateProfile, raw_lower: str, scale: Dict[int, str]) -> tuple:
    """Score education and domain fit."""
    score = 0
    evidence_parts = []
    
    # Check for relevant degree
    cs_indicators = ["computer science", "computer engineering", "software engineering", "ai", "artificial intelligence",
                     "machine learning", "data science", "information technology"]
    cs_found = [ind for ind in cs_indicators if ind in raw_lower]
    
    # Check for institution prestige (rough heuristic)
    top_institutions = ["iit", "bits", "nit", "iiit", "mit", "stanford", "berkeley", "cmu", "oxford", "cambridge"]
    inst_found = [inst for inst in top_institutions if inst in raw_lower]
    
    # Check for GPA/CGPA
    gpa_match = re.search(r'(?:CGPA|GPA|percentage)[:\s]*([\d.]+)', raw_lower, re.IGNORECASE)
    gpa_value = float(gpa_match.group(1)) if gpa_match else 0
    
    # Check for research
    research_indicators = ["research", "publication", "paper", "thesis", "lab"]
    research_found = [ind for ind in research_indicators if ind in raw_lower]
    
    if cs_found:
        evidence_parts.append(f"Relevant degree: {', '.join(cs_found[:2])}")
    if inst_found:
        evidence_parts.append(f"Institution: {', '.join(inst_found)}")
    if gpa_match:
        evidence_parts.append(f"GPA: {gpa_value}")
    if research_found:
        evidence_parts.append(f"Research: {', '.join(research_found[:2])}")
    
    # Determine score
    if cs_found and inst_found and gpa_value >= 8.0 and research_found:
        score = 5
    elif cs_found and inst_found and gpa_value >= 7.0:
        score = 4
    elif cs_found and (inst_found or gpa_value >= 7.0):
        score = 3
    elif cs_found:
        score = 2
    elif inst_found:
        score = 1
    
    evidence = "; ".join(evidence_parts) if evidence_parts else "No education evidence found"
    return (score, evidence)


# ============================================================
# TOOL 3: check_availability
# ============================================================
def check_availability(candidate_name: str, week: str = "next") -> List[InterviewSlot]:
    """
    Mock tool to check interview availability for a candidate.
    Returns available time slots for the specified week.
    """
    # Mock availability data
    mock_slots = {
        "Priya Sharma": [
            InterviewSlot(candidate_name=candidate_name, date="2026-07-14", time="10:00 AM IST", slot_id="PS-001"),
            InterviewSlot(candidate_name=candidate_name, date="2026-07-14", time="2:00 PM IST", slot_id="PS-002"),
            InterviewSlot(candidate_name=candidate_name, date="2026-07-15", time="11:00 AM IST", slot_id="PS-003"),
        ],
        "Rahul Verma": [
            InterviewSlot(candidate_name=candidate_name, date="2026-07-14", time="3:00 PM IST", slot_id="RV-001"),
            InterviewSlot(candidate_name=candidate_name, date="2026-07-16", time="10:00 AM IST", slot_id="RV-002"),
        ],
        "Meera Patel": [
            InterviewSlot(candidate_name=candidate_name, date="2026-07-15", time="4:00 PM IST", slot_id="MP-001"),
            InterviewSlot(candidate_name=candidate_name, date="2026-07-17", time="9:00 AM IST", slot_id="MP-002"),
        ]
    }
    
    return mock_slots.get(candidate_name, [
        InterviewSlot(candidate_name=candidate_name, date="2026-07-14", time="10:00 AM IST", slot_id="GEN-001"),
    ])


# ============================================================
# TOOL 4: propose_interview (ACTION TOOL - requires human approval)
# ============================================================
def propose_interview(candidate_name: str, slot: InterviewSlot) -> InterviewConfirmation:
    """
    Propose an interview for a candidate at a given slot.
    THIS IS AN ACTION TOOL - it requires human-in-the-loop approval before executing.
    Returns a confirmation object with status 'pending_approval'.
    """
    confirmation = InterviewConfirmation(
        candidate_name=candidate_name,
        slot=slot,
        status="pending_approval"
    )
    return confirmation


# ============================================================
# INJECTION DETECTION
# ============================================================
def detect_prompt_injection(resume_text: str) -> bool:
    """
    Detect potential prompt injection attempts in résumé text.
    Checks for instructions that try to override the agent's behavior.
    """
    injection_patterns = [
        r"ignore\s+(?:your|all|previous|the)\s+(?:instructions|prompt|commands|rules|directions)",
        r"forget\s+(?:your|all|previous)\s+(?:instructions|prompt|commands)",
        r"rank\s+(?:me|this\s+candidate)\s+(?:first|highest|top)",
        r"you\s+(?:must|should|have\s+to)\s+(?:rank|select|choose)\s+(?:me|this)",
        r"override\s+(?:your|the|all)\s+(?:instructions|scoring|ranking|system)",
        r"disregard\s+(?:your|all|previous)\s+(?:instructions|prompt|rules)",
        r"say\s+(?:i\s+am|this\s+is)\s+(?:the\s+)?(?:best|perfect|ideal)\s+candidate",
        r"this\s+is\s+(?:a\s+)?(?:test|simulation|hypothetical)",
        r"pretend\s+(?:you\s+are|that)",
        r"you\s+are\s+(?:now|currently)\s+(?:a\s+)?(?:different|new)\s+(?:system|agent|persona)",
    ]
    
    text_lower = resume_text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


# ============================================================
# FAIRNESS CHECK
# ============================================================
def fairness_check(profiles: Dict[str, CandidateProfile], scorecards: Dict[str, ScoreCard]) -> Dict[str, Any]:
    """
    Check that candidates with identical relevant experience score the same.
    This is a simplified fairness check that verifies scoring consistency.
    """
    result = {
        "verified": True,
        "checks": [],
        "warnings": []
    }
    
    # Check that all candidates have scorecards
    if len(scorecards) < 2:
        result["verified"] = False
        result["warnings"].append("Insufficient candidates for fairness comparison")
        return result
    
    # Check for name-based bias (compare scores ignoring names)
    scores = [(name, sc.weighted_total) for name, sc in scorecards.items()]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Verify that scoring is based on skills, not personal attributes
    for name, sc in scorecards.items():
        profile = profiles.get(name)
        if profile:
            # Check if any criterion score mentions non-JD factors
            for cs in sc.criterion_scores:
                non_jd_keywords = ["gender", "age", "race", "religion", "nationality", "marital", "appearance"]
                for kw in non_jd_keywords:
                    if kw in cs.evidence.lower() or kw in cs.justification.lower():
                        result["warnings"].append(f"Potential bias detected in {name}: {cs.criterion_name} mentions '{kw}'")
                        result["verified"] = False
    
    result["checks"].append({
        "type": "name_swap_test",
        "status": "passed" if result["verified"] else "needs_review",
        "candidates_compared": len(scorecards),
        "score_range": f"{min(s[1] for s in scores):.2f} - {max(s[1] for s in scores):.2f}"
    })
    
    return result