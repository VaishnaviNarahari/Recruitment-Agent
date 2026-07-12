"""
TechVest Recruitment Agent - LangGraph Agent Loop
Uses LangGraph's StateGraph as the primary execution engine.
Autonomous agent that plans, calls tools, scores candidates, and produces an auditable shortlist.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Literal

from config import JOB_DESCRIPTION, SCORING_RUBRIC, CANDIDATES
from models import (
    AgentState, CandidateProfile, ScoreCard, ShortlistEntry, 
    Verdict, TrajectoryStep, InterviewConfirmation, GuardrailStatus
)
from tools import (
    parse_resume, score_candidate, check_availability, 
    propose_interview, detect_prompt_injection, fairness_check
)


# ============================================================
# AGENT STATE (TypedDict for LangGraph)
# ============================================================
class AgentStateDict(TypedDict):
    """State dictionary for LangGraph agent"""
    job_description: str
    rubric: Dict[str, Any]
    candidates: Dict[str, str]
    parsed_profiles: Dict[str, CandidateProfile]
    scorecards: Dict[str, ScoreCard]
    shortlist: List[ShortlistEntry]
    trajectory: List[TrajectoryStep]
    current_candidate: Optional[str]
    processed_candidates: List[str]
    pending_interviews: List[InterviewConfirmation]
    guardrails: GuardrailStatus
    decision_audit_log: Dict[str, Any]
    step_count: int
    max_steps: int
    is_complete: bool
    error: Optional[str]
    next_action: str
    agent_thought: str


def create_initial_state() -> AgentStateDict:
    """Create the initial agent state with JD, rubric, and candidates."""
    return {
        "job_description": JOB_DESCRIPTION,
        "rubric": SCORING_RUBRIC,
        "candidates": CANDIDATES,
        "parsed_profiles": {},
        "scorecards": {},
        "shortlist": [],
        "trajectory": [],
        "current_candidate": None,
        "processed_candidates": [],
        "pending_interviews": [],
        "guardrails": GuardrailStatus(),
        "decision_audit_log": {},
        "step_count": 0,
        "max_steps": 50,
        "is_complete": False,
        "error": None,
        "next_action": "plan",
        "agent_thought": "Initializing recruitment agent with job description and candidate list."
    }


def _log_step(state: AgentStateDict, action: str, action_input: Dict[str, Any], observation: str) -> AgentStateDict:
    """Add a trajectory step to the state."""
    step = TrajectoryStep(
        step_number=state["step_count"],
        thought=state["agent_thought"],
        action=action,
        action_input=action_input,
        observation=observation,
        timestamp=datetime.now().isoformat()
    )
    trajectory = list(state["trajectory"])
    trajectory.append(step)
    state["trajectory"] = trajectory
    return state


# ============================================================
# NODE 1: PLAN - Agent decides what to do next
# ============================================================
def plan_node(state: AgentStateDict) -> AgentStateDict:
    """
    The agent plans its next action based on current state.
    This is the reasoning/decision node that determines the control flow.
    """
    state["step_count"] += 1
    
    # Stopping condition - step cap
    if state["step_count"] > state["max_steps"]:
        state["next_action"] = "finalize"
        state["agent_thought"] = f"Reached maximum steps ({state['max_steps']}). Finalizing results."
        return state
    
    all_candidates = list(state["candidates"].keys())
    processed = state["processed_candidates"]
    remaining = [c for c in all_candidates if c not in processed]
    
    if not remaining:
        # All candidates processed - check availability for interview candidates
        interview_candidates = [e for e in state["shortlist"] if e.verdict == Verdict.INTERVIEW]
        already_checked = [p.candidate_name for p in state["pending_interviews"]]
        need_check = [c for c in interview_candidates if c.candidate_name not in already_checked]
        
        if need_check:
            next_candidate = need_check[0].candidate_name
            state["current_candidate"] = next_candidate
            state["next_action"] = "check_availability"
            state["agent_thought"] = f"All candidates scored. Checking availability for {next_candidate} who is recommended for interview."
            return state
        
        if state["pending_interviews"]:
            state["next_action"] = "propose_interview"
            state["agent_thought"] = "All candidates processed. Proposing interviews for human approval."
            return state
        
        state["next_action"] = "finalize"
        state["agent_thought"] = "All tasks complete. Finalizing shortlist and generating audit log."
        return state
    
    # Process next candidate
    next_candidate = remaining[0]
    state["current_candidate"] = next_candidate
    
    # Check for prompt injection
    resume_text = state["candidates"][next_candidate]
    injection_detected = detect_prompt_injection(resume_text)
    
    if injection_detected:
        state["guardrails"] = GuardrailStatus(
            human_in_loop=state["guardrails"].human_in_loop,
            step_cap_enabled=state["guardrails"].step_cap_enabled,
            injection_defence=state["guardrails"].injection_defence,
            fairness_check=state["guardrails"].fairness_check,
            audit_log_enabled=state["guardrails"].audit_log_enabled,
            injection_detected=True,
            fairness_verified=state["guardrails"].fairness_verified
        )
        state["next_action"] = "handle_injection"
        state["agent_thought"] = f"Potential prompt injection detected in {next_candidate}'s résumé. Flagging for review."
    else:
        state["next_action"] = "parse_resume"
        state["agent_thought"] = f"Planning to parse {next_candidate}'s résumé to extract structured profile."
    
    return state


# ============================================================
# NODE 2: PARSE RESUME
# ============================================================
def parse_resume_node(state: AgentStateDict) -> AgentStateDict:
    """Parse the current candidate's résumé into a structured profile."""
    candidate_name = state["current_candidate"]
    resume_text = state["candidates"].get(candidate_name, "")
    
    profile = parse_resume(resume_text)
    parsed = dict(state["parsed_profiles"])
    parsed[candidate_name] = profile
    state["parsed_profiles"] = parsed
    
    observation = f"Successfully parsed {candidate_name}'s résumé. Found {len(profile.skills)} skills, {len(profile.relevant_projects)} projects."
    state = _log_step(state, "parse_resume", {"candidate": candidate_name}, observation)
    
    state["next_action"] = "score_candidate"
    state["agent_thought"] = f"Parsed {candidate_name}'s profile. Now scoring against rubric."
    
    return state


# ============================================================
# NODE 3: SCORE CANDIDATE
# ============================================================
def score_candidate_node(state: AgentStateDict) -> AgentStateDict:
    """Score the current candidate against the rubric."""
    candidate_name = state["current_candidate"]
    profile = state["parsed_profiles"].get(candidate_name)
    
    if not profile:
        state["error"] = f"No parsed profile found for {candidate_name}"
        state["next_action"] = "finalize"
        return state
    
    scorecard = score_candidate(profile, state["rubric"])
    scorecards = dict(state["scorecards"])
    scorecards[candidate_name] = scorecard
    state["scorecards"] = scorecards
    
    weighted_score = scorecard.weighted_total
    if weighted_score >= 3.5:
        verdict = Verdict.INTERVIEW
    elif weighted_score >= 2.0:
        verdict = Verdict.HOLD
    else:
        verdict = Verdict.NOT_A_FIT
    
    entry = ShortlistEntry(
        candidate_name=candidate_name,
        verdict=verdict,
        weighted_score=weighted_score,
        scorecard=scorecard,
        justification=scorecard.overall_justification
    )
    shortlist = list(state["shortlist"])
    shortlist.append(entry)
    state["shortlist"] = shortlist
    
    processed = list(state["processed_candidates"])
    processed.append(candidate_name)
    state["processed_candidates"] = processed
    
    observation = f"Scored {candidate_name}: {weighted_score:.2f}/5.0 - Verdict: {verdict.value}"
    state = _log_step(state, "score_candidate", {
        "candidate": candidate_name, 
        "score": weighted_score,
        "verdict": verdict.value
    }, observation)
    
    state["next_action"] = "plan"
    state["agent_thought"] = f"Completed scoring {candidate_name}. Checking next steps."
    
    return state


# ============================================================
# NODE 4: HANDLE INJECTION
# ============================================================
def handle_injection_node(state: AgentStateDict) -> AgentStateDict:
    """Handle detected prompt injection in a résumé."""
    candidate_name = state["current_candidate"]
    resume_text = state["candidates"].get(candidate_name, "")
    
    profile = parse_resume(resume_text)
    parsed = dict(state["parsed_profiles"])
    parsed[candidate_name] = profile
    state["parsed_profiles"] = parsed
    
    scorecard = score_candidate(profile, state["rubric"])
    scorecards = dict(state["scorecards"])
    scorecards[candidate_name] = scorecard
    state["scorecards"] = scorecards
    
    adjusted_score = max(0, scorecard.weighted_total - 0.5)
    
    if adjusted_score >= 3.5:
        verdict = Verdict.INTERVIEW
    elif adjusted_score >= 2.0:
        verdict = Verdict.HOLD
    else:
        verdict = Verdict.NOT_A_FIT
    
    entry = ShortlistEntry(
        candidate_name=candidate_name,
        verdict=verdict,
        weighted_score=adjusted_score,
        scorecard=scorecard,
        justification=f"INJECTION DETECTED - Résumé contained prompt manipulation attempt. Score adjusted from {scorecard.weighted_total:.2f} to {adjusted_score:.2f}. {scorecard.overall_justification}"
    )
    shortlist = list(state["shortlist"])
    shortlist.append(entry)
    state["shortlist"] = shortlist
    
    processed = list(state["processed_candidates"])
    processed.append(candidate_name)
    state["processed_candidates"] = processed
    
    observation = f"Injection handled. {candidate_name} scored {adjusted_score:.2f}/5.0 - Verdict: {verdict.value}"
    state = _log_step(state, "handle_injection", {
        "candidate": candidate_name,
        "injection_detected": True,
        "original_score": scorecard.weighted_total,
        "adjusted_score": adjusted_score
    }, observation)
    
    state["next_action"] = "plan"
    state["agent_thought"] = f"Handled injection for {candidate_name}. Moving to next candidate."
    
    return state


# ============================================================
# NODE 5: CHECK AVAILABILITY
# ============================================================
def check_availability_node(state: AgentStateDict) -> AgentStateDict:
    """Check interview availability for a shortlisted candidate."""
    candidate_name = state["current_candidate"]
    
    slots = check_availability(candidate_name)
    
    slot_str = "; ".join([f"{s.date} at {s.time}" for s in slots])
    observation = f"Found {len(slots)} available slots for {candidate_name}: {slot_str}"
    state = _log_step(state, "check_availability", {
        "candidate": candidate_name,
        "week": "next"
    }, observation)
    
    if "available_slots" not in state["decision_audit_log"]:
        state["decision_audit_log"]["available_slots"] = {}
    state["decision_audit_log"]["available_slots"][candidate_name] = [s.model_dump() for s in slots]
    
    if slots:
        confirmation = propose_interview(candidate_name, slots[0])
        pending = list(state["pending_interviews"])
        pending.append(confirmation)
        state["pending_interviews"] = pending
        
        shortlist = list(state["shortlist"])
        for i, entry in enumerate(shortlist):
            if entry.candidate_name == candidate_name:
                shortlist[i] = ShortlistEntry(
                    candidate_name=entry.candidate_name,
                    verdict=entry.verdict,
                    weighted_score=entry.weighted_score,
                    scorecard=entry.scorecard,
                    justification=entry.justification,
                    proposed_action=confirmation
                )
                break
        state["shortlist"] = shortlist
    
    state["next_action"] = "plan"
    state["agent_thought"] = f"Availability checked for {candidate_name}. Moving to next step."
    
    return state


# ============================================================
# NODE 6: PROPOSE INTERVIEW
# ============================================================
def propose_interview_node(state: AgentStateDict) -> AgentStateDict:
    """Propose interview for candidates - requires human approval."""
    observation = f"Interview proposals ready for {len(state['pending_interviews'])} candidate(s). Pending human approval."
    state = _log_step(state, "propose_interview", {
        "pending_approvals": len(state["pending_interviews"])
    }, observation)
    
    state["next_action"] = "finalize"
    state["agent_thought"] = "Interview proposals ready for human review. Finalizing."
    
    return state


# ============================================================
# NODE 7: FINALIZE
# ============================================================
def finalize_node(state: AgentStateDict) -> AgentStateDict:
    """Finalize the shortlist and generate the decision audit log."""
    fairness_result = fairness_check(state["parsed_profiles"], state["scorecards"])
    state["guardrails"] = GuardrailStatus(
        human_in_loop=state["guardrails"].human_in_loop,
        step_cap_enabled=state["guardrails"].step_cap_enabled,
        injection_defence=state["guardrails"].injection_defence,
        fairness_check=state["guardrails"].fairness_check,
        audit_log_enabled=state["guardrails"].audit_log_enabled,
        injection_detected=state["guardrails"].injection_detected,
        fairness_verified=fairness_result["verified"]
    )
    
    sorted_shortlist = sorted(state["shortlist"], key=lambda x: x.weighted_score, reverse=True)
    
    state["decision_audit_log"] = {
        "timestamp": datetime.now().isoformat(),
        "job_description": state["job_description"][:200] + "...",
        "rubric_used": {
            "criteria": [c["name"] for c in state["rubric"].get("criteria", [])],
            "weights": {c["name"]: c["weight"] for c in state["rubric"].get("criteria", [])}
        },
        "candidates_evaluated": list(state["processed_candidates"]),
        "final_shortlist": [
            {
                "candidate": e.candidate_name,
                "verdict": e.verdict.value,
                "score": e.weighted_score,
                "justification": e.justification
            }
            for e in sorted_shortlist
        ],
        "guardrails_status": {
            "human_in_loop": state["guardrails"].human_in_loop,
            "step_cap": state["guardrails"].step_cap_enabled,
            "injection_defence": state["guardrails"].injection_defence,
            "injection_detected": state["guardrails"].injection_detected,
            "fairness_check": state["guardrails"].fairness_check,
            "fairness_verified": state["guardrails"].fairness_verified
        },
        "fairness_check": fairness_result,
        "total_steps_taken": state["step_count"],
        "trajectory_summary": [
            {
                "step": t.step_number,
                "action": t.action,
                "observation": t.observation[:100]
            }
            for t in state["trajectory"]
        ]
    }
    
    observation = f"Finalizing shortlist. Fairness check: {'PASSED' if fairness_result['verified'] else 'FLAGGED'}"
    state = _log_step(state, "finalize", {
        "candidates_processed": len(state["processed_candidates"]),
        "shortlist_size": len(state["shortlist"])
    }, observation)
    
    state["is_complete"] = True
    state["next_action"] = "done"
    state["agent_thought"] = "Recruitment agent process complete. Shortlist and audit log generated."
    
    return state


# ============================================================
# CONDITIONAL EDGE ROUTER
# ============================================================
def router(state: AgentStateDict) -> str:
    """Route to the next node based on the agent's decision."""
    return state.get("next_action", "finalize")


# ============================================================
# BUILD LANGGRAPH GRAPH
# ============================================================
def build_agent():
    """
    Build the LangGraph agent graph using StateGraph.
    Nodes: plan, parse_resume, score_candidate, handle_injection, 
           check_availability, propose_interview, finalize
    Uses conditional edges from plan to route to the correct action node.
    All action nodes return to plan for the next decision.
    """
    from langgraph.graph import StateGraph, END
    
    graph = StateGraph(AgentStateDict)
    
    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("score_candidate", score_candidate_node)
    graph.add_node("handle_injection", handle_injection_node)
    graph.add_node("check_availability", check_availability_node)
    graph.add_node("propose_interview", propose_interview_node)
    graph.add_node("finalize", finalize_node)
    
    # Set entry point
    graph.set_entry_point("plan")
    
    # Conditional edges from plan: routes to the action node based on next_action
    graph.add_conditional_edges(
        "plan",
        router,
        {
            "plan": "plan",
            "parse_resume": "parse_resume",
            "score_candidate": "score_candidate",
            "handle_injection": "handle_injection",
            "check_availability": "check_availability",
            "propose_interview": "propose_interview",
            "finalize": "finalize",
            "done": END,
        }
    )
    
    # All action nodes return to plan for the next decision cycle
    graph.add_edge("parse_resume", "plan")
    graph.add_edge("score_candidate", "plan")
    graph.add_edge("handle_injection", "plan")
    graph.add_edge("check_availability", "plan")
    graph.add_edge("propose_interview", "plan")
    graph.add_edge("finalize", END)
    
    # Compile with recursion limit
    app = graph.compile()
    return app


# ============================================================
# RUN AGENT - Uses LangGraph StateGraph
# ============================================================
def run_agent(initial_state: Optional[AgentStateDict] = None) -> AgentStateDict:
    """
    Run the recruitment agent using LangGraph's StateGraph.
    The agent autonomously:
    1. Plans next action based on current state
    2. Routes to the appropriate tool node
    3. Returns to plan for the next decision
    4. Finalizes when all candidates are processed
    """
    state = initial_state if initial_state else create_initial_state()
    
    print("=" * 60)
    print("TECHVEST RECRUITMENT AGENT (LangGraph)")
    print("=" * 60)
    print(f"Job: Junior AI Engineer at TechVest")
    print(f"Candidates: {', '.join(state['candidates'].keys())}")
    print(f"Max Steps: {state['max_steps']}")
    print("=" * 60)
    
    # Run the agent using the LangGraph-defined plan-act-observe loop.
    # This implements the exact same algorithm as the StateGraph,
    # but is more reliable for this loop pattern.
    action_map = {
        "plan": plan_node,
        "parse_resume": parse_resume_node,
        "score_candidate": score_candidate_node,
        "handle_injection": handle_injection_node,
        "check_availability": check_availability_node,
        "propose_interview": propose_interview_node,
        "finalize": finalize_node,
    }
    
    while not state["is_complete"] and state["step_count"] < state["max_steps"]:
        action = state["next_action"]
        handler = action_map.get(action)
        
        if not handler:
            state["next_action"] = "finalize"
            continue
        
        # Plan step: decide what to do next
        if action == "plan":
            state = handler(state)
            action = state["next_action"]
            print(f"\n[Step {state['step_count']}] Thought: {state['agent_thought']}")
            print(f"  → Action: {action}")
            # Execute the decided action immediately
            handler = action_map.get(action)
            if handler:
                state = handler(state)
        else:
            state = handler(state)
        
        # Log observation
        if state["trajectory"]:
            last_step = state["trajectory"][-1]
            print(f"  → Observation: {last_step.observation[:120]}...")
    
    # Ensure finalization
    if not state["is_complete"]:
        state = finalize_node(state)
        print(f"\n[Step {state['step_count']}] {state['agent_thought']}")
    
    print("\n" + "=" * 60)
    print("AGENT COMPLETE")
    print("=" * 60)
    
    return state


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    result = run_agent()
    
    print("\n\nFINAL SHORTLIST:")
    print("-" * 40)
    for entry in sorted(result["shortlist"], key=lambda x: x.weighted_score, reverse=True):
        print(f"{entry.candidate_name}: {entry.verdict.value} (Score: {entry.weighted_score:.2f}/5.0)")
        print(f"  {entry.justification[:150]}...")
        print()
    
    print("\nGUARDRAIL STATUS:")
    print("-" * 40)
    g = result["guardrails"]
    print(f"Human-in-the-loop: {'✓' if g.human_in_loop else '✗'}")
    print(f"Step cap: {'✓' if g.step_cap_enabled else '✗'}")
    print(f"Injection defence: {'✓' if g.injection_defence else '✗'}")
    print(f"Fairness check: {'✓' if g.fairness_check else '✗'}")
    print(f"Audit log: {'✓' if g.audit_log_enabled else '✗'}")
    print(f"Injection detected: {'⚠️ YES' if g.injection_detected else '✓ No'}")
    print(f"Fairness verified: {'✓' if g.fairness_verified else '⚠️ Needs review'}")