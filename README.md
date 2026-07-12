# 🤖 AI Recruitment Agent

An intelligent AI-powered Recruitment Agent that automates candidate screening by parsing resumes, evaluating candidates against a predefined job description, ranking applicants based on weighted criteria, and generating interview recommendations while ensuring transparency, fairness, and human oversight.

## 📌 Features

- 📄 Resume parsing using LLMs
- 🎯 Candidate scoring with a weighted recruitment rubric
- 📊 Automatic candidate ranking
- ✅ Evidence-based decision making
- 👨‍💼 Human approval before interview scheduling
- 🛡️ Prompt injection detection
- ⚖️ Fairness validation using name-swapping
- 📝 Complete audit trail of every decision
- 🔄 Agent workflow powered by LangGraph

## 🏗️ Tech Stack

- Python
- LangGraph
- LangChain
- OpenRouter LLM
- Pydantic
- Streamlit (UI)
- Git & GitHub

## 📂 Project Structure

```
├── app.py                # Streamlit application
├── agent.py              # Recruitment agent logic
├── models.py             # Pydantic models
├── tools.py              # Resume parsing & scoring tools
├── config.py             # Configuration
├── requirements.txt      # Project dependencies
└── test_agent.py         # Test cases
```

## 🚀 Workflow

1. Upload candidate resumes.
2. Parse resumes into structured candidate profiles.
3. Evaluate candidates using a weighted recruitment rubric.
4. Generate evidence-backed scores.
5. Rank candidates based on overall performance.
6. Recommend:
   - Interview
   - Hold
   - Reject
7. Require human approval before scheduling interviews.
8. Record the complete decision trajectory for auditing.

## 🎯 Evaluation Criteria

Candidates are assessed using four weighted criteria:

- Python & Machine Learning Fundamentals
- Relevant Projects
- Hands-on Technical Skills
- Communication Skills

Each score includes evidence extracted directly from the candidate's resume.

## 🔒 Safety Features

- Human-in-the-loop approval
- Prompt injection detection
- Fairness validation through name swapping
- Iteration limits to prevent runaway execution
- Transparent audit logs for every decision

## 📈 Future Enhancements

- Multi-agent recruitment workflow
- Real calendar integration
- Resume upload through drag-and-drop
- Bias analysis dashboard
- Candidate analytics and reporting
- Support for multiple job descriptions

## ▶️ Installation

```bash
git clone <repository-url>
cd recruitment-agent

pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run app.py
```

## 📄 License

This project was developed for learning purposes as part of the **GenAI & Agentic AI Engineering** hands-on lab.
