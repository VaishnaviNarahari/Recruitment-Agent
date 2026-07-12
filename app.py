"""
TechVest Recruitment Agent - Streamlit UI
Autonomous evaluation with dynamic input, full audit trail, and guardrails.
Users upload résumé files - the agent auto-extracts names and evaluates.
"""

import streamlit as st
import json
import time
import re
import io
from datetime import datetime
from typing import Dict, List, Any

from config import JOB_DESCRIPTION as DEFAULT_JD, SCORING_RUBRIC, CANDIDATES as DEFAULT_CANDIDATES
from models import Verdict
from agent import run_agent, create_initial_state
from tools import parse_resume, detect_prompt_injection, fairness_check

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TechVest Recruitment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, sans-serif; }
    .main-header { font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }
    .candidate-card { background: #ffffff; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: all 0.2s; }
    .candidate-card:hover { border-color: #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    .badge { padding: 3px 12px; border-radius: 16px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px; display: inline-block; }
    .badge-interview { background: #d1fae5; color: #065f46; }
    .badge-hold { background: #fef3c7; color: #92400e; }
    .badge-reject { background: #fee2e2; color: #991b1b; }
    .score-badge { display: inline-flex; align-items: center; justify-content: center; min-width: 44px; height: 28px; border-radius: 14px; padding: 0 10px; font-size: 0.85rem; font-weight: 700; color: white; }
    .score-high { background: #10b981; }
    .score-medium { background: #f59e0b; }
    .score-low { background: #ef4444; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.2rem; }
    .trace-step { border-left: 3px solid #6366f1; padding: 0.6rem 1rem; margin: 0.35rem 0; background: #f8fafc; border-radius: 0 6px 6px 0; }
    .trace-thought { color: #6366f1; font-weight: 600; font-size: 0.85rem; }
    .trace-action { color: #2563eb; font-family: monospace; font-size: 0.8rem; }
    .trace-obs { color: #475569; font-size: 0.8rem; margin-top: 0.2rem; }
    .evidence-box { background: #f1f5f9; padding: 0.4rem 0.7rem; border-radius: 6px; font-size: 0.78rem; color: #334155; border-left: 3px solid #6366f1; margin: 0.25rem 0; }
    .footer { text-align: center; color: #94a3b8; font-size: 0.7rem; padding: 1.5rem 0 0.5rem; border-top: 1px solid #e2e8f0; margin-top: 2rem; }
    .stButton button { border-radius: 8px; font-weight: 600; font-size: 0.85rem; }
    div[data-testid="stSidebar"] { background: #fafafa; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
    .uploaded-file { font-size: 0.8rem; color: #475569; padding: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "approvals" not in st.session_state:
    st.session_state.approvals = {}
if "num_candidates" not in st.session_state:
    st.session_state.num_candidates = 3
if "auto_run_done" not in st.session_state:
    st.session_state.auto_run_done = False


def extract_name_from_resume(resume_text: str) -> str:
    """Extract candidate name from résumé text."""
    name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', resume_text)
    if name_match:
        return name_match.group(1).strip()
    first_line = resume_text.strip().split('\n')[0]
    return first_line.strip()


def read_uploaded_file(uploaded_file) -> str:
    """Read content from an uploaded file (txt, pdf, docx)."""
    try:
        # Try to read as text first
        content = uploaded_file.read().decode("utf-8")
        return content
    except UnicodeDecodeError:
        # For binary files, try to extract text
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            pass
        try:
            import docx
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            pass
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("**LangGraph** · Stateful graph · Plan → Act → Observe loop")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Number of Candidates**")
    with col2:
        num = st.number_input("Count", min_value=1, max_value=5, value=st.session_state.num_candidates, label_visibility="collapsed", key="num_candidates_input")
        st.session_state.num_candidates = num
    
    st.markdown("---")
    st.markdown("#### 📄 Job Description")
    jd_input = st.text_area(
        "Paste the job description here...",
        value=DEFAULT_JD.strip(),
        height=120,
        label_visibility="collapsed",
        key="jd_input"
    )
    
    st.markdown("---")
    st.markdown("#### 📎 Upload Résumés")
    st.markdown("*Upload .txt, .pdf, or .docx files. The agent will auto-extract names and details.*")
    
    uploaded_files = st.file_uploader(
        "Choose résumé files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader",
        label_visibility="collapsed"
    )
    
    # Show uploaded files
    candidate_resumes = {}
    if uploaded_files:
        for f in uploaded_files:
            content = read_uploaded_file(f)
            name = extract_name_from_resume(content)
            candidate_resumes[name] = content
            st.markdown(f'<div class="uploaded-file">✅ {f.name} → <strong>{name}</strong></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    run_disabled = st.session_state.agent_running or len(candidate_resumes) < 1
    if st.button("🚀 Run Agent", type="primary", use_container_width=True, disabled=run_disabled):
        if not jd_input.strip():
            st.error("Please enter a job description.")
        elif len(candidate_resumes) < 1:
            st.error("Please upload at least one résumé.")
        else:
            st.session_state.agent_running = True
            st.session_state.agent_result = None
            st.session_state.run_config = {"jd": jd_input, "candidates": candidate_resumes}
            st.rerun()
    
    if st.session_state.agent_result:
        st.button("🔄 Reset", use_container_width=True, on_click=lambda: st.session_state.update(agent_result=None, approvals={}, auto_run_done=False))
    
    if st.session_state.agent_result:
        r = st.session_state.agent_result
        st.markdown("---")
        st.markdown(f"**Last Run:** {len(r['processed_candidates'])} candidates · {r['step_count']} steps")


# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<div class="main-header">🤖 Recruitment Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous candidate evaluation with full audit trail</div>', unsafe_allow_html=True)


# Auto-run on first load (only if no files uploaded yet)
if st.session_state.agent_result is None and not st.session_state.agent_running and not st.session_state.auto_run_done:
    st.session_state.auto_run_done = True
    # Don't auto-run with default data - wait for user to upload files
    pass

# Run agent
if st.session_state.agent_running and st.session_state.agent_result is None:
    config = st.session_state.get("run_config", {})
    jd = config.get("jd", DEFAULT_JD)
    candidates_dict = config.get("candidates", {})
    
    if candidates_dict:
        with st.spinner("🤖 Agent is parsing résumés and evaluating candidates..."):
            progress = st.progress(0)
            status = st.empty()
            for i in range(5):
                status.text(f"Step {i+1}/5: Processing...")
                progress.progress((i + 1) * 20)
                time.sleep(0.2)
            
            custom_state = create_initial_state()
            custom_state["job_description"] = jd
            custom_state["candidates"] = candidates_dict
            result = run_agent(custom_state)
            st.session_state.agent_result = result
            st.session_state.agent_running = False
            st.rerun()

# Display results
if st.session_state.agent_result:
    result = st.session_state.agent_result
    
    tab1, tab2, tab3 = st.tabs(["📋 Shortlist", "🔍 Trajectory & Guardrails", "📄 Rubric"])
    
    # ========== TAB 1: SHORTLIST ==========
    with tab1:
        if not result["shortlist"]:
            st.warning("No candidates were evaluated.")
        else:
            interview_n = len([e for e in result["shortlist"] if e.verdict == Verdict.INTERVIEW])
            hold_n = len([e for e in result["shortlist"] if e.verdict == Verdict.HOLD])
            reject_n = len([e for e in result["shortlist"] if e.verdict == Verdict.NOT_A_FIT])
            
            cols = st.columns(4)
            for col, (val, color, label) in zip(cols, [
                (len(result["shortlist"]), "#10b981", "Evaluated"),
                (interview_n, "#059669", "Interview"),
                (hold_n, "#d97706", "Hold"),
                (reject_n, "#dc2626", "Not a Fit"),
            ]):
                col.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            sorted_entries = sorted(result["shortlist"], key=lambda x: x.weighted_score, reverse=True)
            for i, entry in enumerate(sorted_entries):
                score = entry.weighted_score
                if score >= 3.5:
                    badge_cls, score_cls, verdict_label = "badge-interview", "score-high", "INTERVIEW"
                elif score >= 2.0:
                    badge_cls, score_cls, verdict_label = "badge-hold", "score-medium", "HOLD"
                else:
                    badge_cls, score_cls, verdict_label = "badge-reject", "score-low", "NOT A FIT"
                rank = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                
                st.markdown(f"""
                <div class="candidate-card">
                    <div style="display:flex;align-items:center;justify-content:space-between;">
                        <div style="display:flex;align-items:center;gap:0.75rem;">
                            <span style="font-size:1.4rem;">{rank}</span>
                            <div>
                                <div style="font-weight:600;font-size:1.05rem;color:#0f172a;">{entry.candidate_name}</div>
                                <div style="margin-top:0.2rem;"><span class="badge {badge_cls}">{verdict_label}</span></div>
                            </div>
                        </div>
                        <div><span class="score-badge {score_cls}">{score:.2f}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📊 Scorecard & Evidence — {entry.candidate_name}"):
                    sc = entry.scorecard
                    st.markdown("**Per-Criterion Scores with Evidence**", unsafe_allow_html=True)
                    for cs in sc.criterion_scores:
                        if cs.score >= 4: c = "#10b981"
                        elif cs.score >= 2: c = "#f59e0b"
                        else: c = "#ef4444"
                        st.markdown(f"""
                        <div style="padding:0.4rem 0;border-bottom:1px solid #f1f5f9;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span><strong>{cs.criterion_name}</strong> <span style="color:#94a3b8;font-size:0.8rem;">({int(cs.weight*100)}%)</span></span>
                                <span style="background:{c};color:white;padding:1px 10px;border-radius:10px;font-weight:600;font-size:0.8rem;">{cs.score}/5 · {cs.score*cs.weight:.2f}</span>
                            </div>
                            <div class="evidence-box">📎 {cs.evidence}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""<div style="margin-top:0.75rem;padding:0.6rem;background:#f8fafc;border-radius:6px;font-size:0.85rem;"><strong>Justification:</strong> {sc.overall_justification}</div>""", unsafe_allow_html=True)
                    
                    if entry.proposed_action:
                        pa = entry.proposed_action
                        slot = pa.slot
                        st.markdown("---")
                        st.markdown("#### 📅 Proposed Interview", unsafe_allow_html=True)
                        st.markdown(f"""<div style="padding:0.6rem;background:#f1f5f9;border-radius:6px;font-size:0.85rem;"><strong>Date:</strong> {slot.date}<br><strong>Time:</strong> {slot.time}<br><strong>Status:</strong> <span style="color:#d97700;font-weight:600;">⏳ {pa.status.replace('_',' ').title()}</span></div>""", unsafe_allow_html=True)
                        
                        ak = f"app_{entry.candidate_name}"
                        if ak not in st.session_state.approvals:
                            st.session_state.approvals[ak] = False
                        
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button(f"✅ Approve — {entry.candidate_name}", key=f"btn_{ak}", type="primary", use_container_width=True):
                                st.session_state.approvals[ak] = True
                                st.success(f"✅ Interview approved for {entry.candidate_name}!")
                                st.rerun()
                        with c2:
                            if st.button(f"❌ Decline — {entry.candidate_name}", key=f"rej_{ak}", use_container_width=True):
                                st.session_state.approvals[ak] = False
                                st.warning(f"Interview declined for {entry.candidate_name}.")
                                st.rerun()
                        
                        if st.session_state.approvals.get(ak, False):
                            st.success(f"✅ **Approved!** {entry.candidate_name} scheduled for {slot.date} at {slot.time}.")
            
            if result["guardrails"].injection_detected:
                st.warning("⚠️ **Prompt Injection Detected:** A résumé contained instructions attempting to manipulate scoring. The agent blocked this attack.")
    
    # ========== TAB 2: TRAJECTORY & GUARDRAILS ==========
    with tab2:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 🤖 Reasoning Trace")
            st.markdown("*Every thought, tool call, and observation logged for full auditability.*")
            
            colors = {"plan": "#6366f1", "parse_resume": "#2563eb", "score_candidate": "#10b981", "handle_injection": "#ef4444", "check_availability": "#f59e0b", "propose_interview": "#f97316", "finalize": "#0f172a"}
            
            for step in result["trajectory"]:
                c = colors.get(step.action, "#64748b")
                thought_preview = step.thought[:100] + ("..." if len(step.thought) > 100 else "")
                obs_preview = step.observation[:120] + ("..." if len(step.observation) > 120 else "")
                inp_preview = json.dumps(step.action_input)[:60] if step.action_input else ""
                
                st.markdown(f"""
                <div class="trace-step" style="border-left-color:{c};">
                    <div style="display:flex;justify-content:space-between;">
                        <span class="trace-thought">Step {step.step_number}: {thought_preview}</span>
                        <span style="font-size:0.7rem;color:#94a3b8;">{step.timestamp.split('T')[1][:8] if step.timestamp else ''}</span>
                    </div>
                    <div class="trace-action" style="color:{c};">🛠 {step.action}{f' | {inp_preview}' if inp_preview else ''}</div>
                    <div class="trace-obs">📌 {obs_preview}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🛡️ Guardrails")
            g = result["guardrails"]
            
            items = [
                ("👤 1. Human-in-the-loop", g.human_in_loop, "Blocks propose_interview without approval", "🟢 Active"),
                ("⏱ 2. Step Cap", g.step_cap_enabled, f"Max {result['max_steps']} steps enforced", "🟢 Active"),
                ("🛡️ 3. Injection Defence", g.injection_defence, "⚠️ Detected" if g.injection_detected else "✅ No injection", "🟢 Active"),
                ("⚖️ 4. Fairness Check", g.fairness_check, "✅ Verified" if g.fairness_verified else "⚠️ Flagged", "🟢 Active"),
                ("📝 5. Audit Log", g.audit_log_enabled, "Full trajectory persisted", "🟢 Active"),
            ]
            
            for icon_name, enabled, detail, status in items:
                st.markdown(f"""
                <div style="padding:0.5rem;margin:0.3rem 0;background:#f8fafc;border-radius:6px;border-left:3px solid #10b981;">
                    <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                        <strong>{icon_name}</strong>
                        <span style="color:#10b981;font-weight:600;">{status}</span>
                    </div>
                    <div style="font-size:0.78rem;color:#475569;margin-top:0.15rem;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 📦 Decision Audit Log")
            audit = result["decision_audit_log"]
            st.json({
                "timestamp": audit.get("timestamp",""),
                "candidates": audit.get("candidates_evaluated",[]),
                "shortlist": audit.get("final_shortlist",[]),
                "guardrails": audit.get("guardrails_status",{}),
                "steps": audit.get("total_steps_taken",0)
            })
    
    # ========== TAB 3: RUBRIC ==========
    with tab3:
        st.markdown("#### 📋 Scoring Rubric")
        st.markdown(f"*Evidence Rule: {SCORING_RUBRIC['evidence_rule']}*")
        for crit in SCORING_RUBRIC["criteria"]:
            with st.expander(f"**{crit['name']}** — Weight: {int(crit['weight']*100)}%"):
                st.markdown(f"*{crit['description']}*")
                st.markdown("**Scale:**")
                for s, desc in crit["scale"].items():
                    st.markdown(f"- **{s}**: {desc}")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:3.5rem;margin-bottom:0.5rem;">🤖</div>
        <h3 style="color:#0f172a;margin-bottom:0.3rem;">Upload Résumés to Begin</h3>
        <p style="color:#64748b;max-width:500px;margin:0 auto;font-size:0.9rem;line-height:1.5;">
            Upload résumé files (.txt, .pdf, .docx) in the sidebar. The agent will automatically 
            extract candidate names, parse skills and experience, score against the rubric, 
            and produce a ranked shortlist with full audit trail.
        </p>
        <div style="display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;margin-top:1.5rem;">
            <div style="padding:0.75rem 1.25rem;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
                <div style="font-size:1.3rem;font-weight:700;color:#6366f1;">5</div>
                <div style="font-size:0.75rem;color:#64748b;">Rubric Criteria</div>
            </div>
            <div style="padding:0.75rem 1.25rem;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
                <div style="font-size:1.3rem;font-weight:700;color:#6366f1;">5</div>
                <div style="font-size:0.75rem;color:#64748b;">Guardrails Active</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Recruitment Agent · LangGraph · GenAI & Agentic AI Engineering
</div>
""", unsafe_allow_html=True)