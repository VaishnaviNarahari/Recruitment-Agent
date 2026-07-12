"""
TechVest Recruitment Agent - Pydantic Models
Typed state management and data schemas for the agent
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class CandidateProfile(BaseModel):
    """Structured profile extracted from a résumé"""
    name: str = ""
    email: str = ""
    education: str = ""
    years_of_experience: float = 0.0
    skills: List[str] = []
    relevant_projects: List[str] = []
    certifications: List[str] = []
    achievements: List[str] = []
    raw_text: str = ""


class CriterionScore(BaseModel):
    """Score for a single rubric criterion"""
    criterion_name: str
    score: int  # 0-5
    weight: float
    evidence: str  # Specific line/section from résumé
    justification: str


class ScoreCard(BaseModel):
    """Complete scorecard for one candidate"""
    candidate_name: str
    criterion_scores: List[CriterionScore]
    weighted_total: float  # Sum of (score * weight) for all criteria, max 5.0
    overall_justification: str


class InterviewSlot(BaseModel):
    """Available interview time slot"""
    candidate_name: str
    date: str
    time: str
    slot_id: str


class InterviewConfirmation(BaseModel):
    """Confirmed interview booking"""
    candidate_name: str
    slot: InterviewSlot
    status: str = "pending_approval"  # pending_approval, approved, rejected


class Verdict(str, Enum):
    INTERVIEW = "INTERVIEW"
    HOLD = "HOLD"
    NOT_A_FIT = "NOT_A_FIT"


class ShortlistEntry(BaseModel):
    """One entry in the final shortlist"""
    candidate_name: str
    verdict: Verdict
    weighted_score: float
    scorecard: ScoreCard
    justification: str
    proposed_action: Optional[InterviewConfirmation] = None


class TrajectoryStep(BaseModel):
    """One step in the agent's reasoning trajectory"""
    step_number: int
    thought: str
    action: str  # Tool name called
    action_input: Dict[str, Any] = {}
    observation: str
    timestamp: str = ""


class GuardrailStatus(BaseModel):
    """Status of all guardrails"""
    human_in_loop: bool = True
    step_cap_enabled: bool = True
    injection_defence: bool = True
    fairness_check: bool = True
    audit_log_enabled: bool = True
    injection_detected: bool = False
    fairness_verified: bool = False


class AgentState(BaseModel):
    """Full state carried through the agent loop"""
    job_description: str = ""
    rubric: Dict[str, Any] = {}
    candidates: Dict[str, str] = {}  # name -> raw résumé text
    parsed_profiles: Dict[str, CandidateProfile] = {}
    scorecards: Dict[str, ScoreCard] = {}
    shortlist: List[ShortlistEntry] = []
    trajectory: List[TrajectoryStep] = []
    current_candidate: Optional[str] = None
    processed_candidates: List[str] = []
    pending_interviews: List[InterviewConfirmation] = []
    guardrails: GuardrailStatus = GuardrailStatus()
    decision_audit_log: Dict[str, Any] = {}
    step_count: int = 0
    max_steps: int = 50
    is_complete: bool = False
    error: Optional[str] = None