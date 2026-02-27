import streamlit as st
import hashlib
import pdfplumber
import io
from datetime import date
from sqlalchemy import UniqueConstraint
from sqlalchemy.exc import IntegrityError
from sqlalchemy import Column, Integer, String, Text, Float, LargeBinary, Boolean, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- FULL WIDTH PAGE -----------------
st.set_page_config(
    page_title="AI Grant Management",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Hide top toolbar (Deploy, menu, etc.) */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Remove header spacing */
header {
    visibility: hidden;
    height: 0px;
}

</style>
""", unsafe_allow_html=True)
if st.session_state.get("logged_in", False):

    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {display: none !important;}

    .block-container {
        padding-top: 0rem !important;
    }

    .top-banner {
        background: linear-gradient(135deg, #e4ecff 0%, #d8e4ff 100%);
        padding: 8px 24px;
        height: 60px;
        display: flex;
        align-items: center;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 2000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .top-banner h1 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
        margin-left: 260px;
    }

    </style>

    <div class="top-banner">
        <h1>AI Based Grant Management Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)
    
# ================= MODERN TECH UI STYLE =================
st.markdown("""
<style>

/* Global Background */
.stApp {
    background: linear-gradient(135deg, #f4f8ff 0%, #eef3ff 100%);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #f1f6ff 0%, #eaf1ff 100%);
    color: #1f1f1f;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: black;
}
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #f1f6ff 0%, #eaf1ff 100%);
    color: black;
    border-radius: 10px;
    border: none;
    padding: 10px;
    transition: all 0.3s ease;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background-color: rgba(255,255,255,0.18);
    transform: translateY(-2px);
}

/* Modern Card */
.modern-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(30, 42, 120, 0.08);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.modern-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(30, 42, 120, 0.15);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #3949ab, #5c6bc0);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 8px 16px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(57,73,171,0.4);
}

/* Metric Cards */
.metric-box {
    background: linear-gradient(135deg, #f1f1f1, #f1f1f1);
    padding: 20px;
    border-radius: 14px;
    color: black;
    text-align: center;
    box-shadow: 0 10px 25px rgba(57,73,171,0.3);
}
.metric-number {
    font-size: 28px;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    opacity: 0.9;
}
/* Make all input fields white */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
}

/* Fix selectbox dropdown background */
div[data-baseweb="select"] > div {
    background-color: white !important;
}

/* Fix multiselect */
div[data-baseweb="select"] div[role="option"] {
    background-color: white !important;
}

/* Reviewer Dashboard Right Panel */
/* Reviewer Left Menu Styled Like Dashboard */
.reviewer-menu {
    background: linear-gradient(135deg, #f1f6ff 0%, #eaf1ff 100%);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(30, 42, 120, 0.08);
    min-height: 600px;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)
# ----------------- DATABASE SETUP -----------------
DATABASE_URL = "sqlite:///grant_demo.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ----------------- MODELS -----------------
class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password_hash = Column(String)

class Reviewer(Base):
    __tablename__ = "reviewers"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password_hash = Column(String)
    cv_text = Column(Text)
    cv_pdf = Column(LargeBinary)
    expertise = Column(Text)

class Call(Base):
    __tablename__ = "calls"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    identifier = Column(String)
    background = Column(Text)
    objectives = Column(Text)
    priority_areas = Column(Text)
    scope = Column(Text)
    funding_details = Column(Text)
    eligibility = Column(Text)
    expected_deliverables = Column(Text)
    evaluation_criteria = Column(Text)
    ethics = Column(Text)
    application_requirements = Column(Text)
    timeline = Column(Text)
    reporting_monitoring = Column(Text)

class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True)
    proposal_id = Column(Integer, ForeignKey("proposals.id"))
    reviewer_id = Column(Integer, ForeignKey("reviewers.id"))
    similarity_score = Column(Float)
    explanation = Column(Text)   # ✅ store reason
    anonymized = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("proposal_id", "reviewer_id", name="unique_proposal_reviewer"),
    )
class ReviewCriteria(Base):
    __tablename__ = "review_criteria"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    area = Column(Text)          # Optional: priority area
    criteria = Column(Text)       # Criteria list (bullet or separated text)

class ReviewScore(Base):
    __tablename__ = "review_scores"
    id = Column(Integer, primary_key=True)
    proposal_id = Column(Integer)
    reviewer_id = Column(Integer)
    originality = Column(Float)
    methodology = Column(Float)
    impact = Column(Float)
    feasibility = Column(Float)
    overall = Column(Float)
    comments = Column(Text)

class Proposal(Base):
    __tablename__ = "proposals"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    abstract = Column(Text)
    keywords = Column(Text)
    selected_area = Column(Text)  # area from call
    proposal_text = Column(Text)
    proposal_pdf = Column(LargeBinary)

    call_id = Column(Integer, ForeignKey("calls.id"))
    status = Column(String, default="Under Review")
    submitted_by = Column(Integer)  # researcher id
    __table_args__ = (
        UniqueConstraint("title", "call_id", name="unique_proposal_per_call"),
    )
class ConflictOfInterest(Base):
    __tablename__ = "conflicts"
    id = Column(Integer, primary_key=True)
    reviewer_id = Column(Integer)
    proposal_id = Column(Integer)
    reason = Column(Text)

class Researcher(Base):
    __tablename__ = "researchers"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password_hash = Column(String)
    expertise = Column(Text)
    cv_text = Column(Text)
    cv_pdf = Column(LargeBinary)

# ----------------- INITIAL SETUP -----------------
Base.metadata.create_all(bind=engine)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def create_demo_admin():
    db = SessionLocal()
    if not db.query(Admin).first():
        db.add(Admin(username="admin", password_hash=hash_password("admin123")))
        db.commit()
    db.close()

create_demo_admin()
def get_db():
    db = SessionLocal()
    return db
# ----------------- SESSION STATE -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Dashboard"


# ----------------- LOGIN / REGISTER -----------------
def register_selection():
    st.markdown("### Register As")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Register as Reviewer", use_container_width=True):
            st.session_state.selected_page = "reviewer_register"
            st.rerun()

    with col2:
        if st.button("Register as Researcher", use_container_width=True):
            st.session_state.selected_page = "researcher_register"
            st.rerun()


def login_page():

    # ---------------- STYLE ----------------
    st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important;
    }
    div[data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
    }
    main > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # -------- LOGIN PAGE TITLE --------
    st.markdown("""
    <div style="
        text-align: center;
        margin-top: 40px;
        margin-bottom: 20px;
    ">
        <h1 style="
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(90deg, #3949ab, #5c6bc0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        ">
            AI Based Grant Management System
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CENTER LOGIN ----------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        email_or_username = st.text_input(
            "Email / Username",
            placeholder="Enter email or username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        # ---------------- LOGIN + REGISTER BUTTONS ----------------
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            login_clicked = st.button("Login", use_container_width=True)

        with col_btn2:
            register_clicked = st.button("Register", use_container_width=True)

        # ---------------- LOGIN LOGIC ----------------
        if login_clicked:

            db = get_db()

            admin = db.query(Admin).filter_by(
                username=email_or_username,
                password_hash=hash_password(password)
            ).first()

            reviewer = db.query(Reviewer).filter_by(
                email=email_or_username,
                password_hash=hash_password(password)
            ).first()

            researcher = db.query(Researcher).filter_by(
                email=email_or_username,
                password_hash=hash_password(password)
            ).first()

            db.close()

            if admin:
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.rerun()

            elif reviewer:
                st.session_state.logged_in = True
                st.session_state.role = "reviewer"
                st.session_state.user_id = reviewer.id
                st.rerun()

            elif researcher:
                st.session_state.logged_in = True
                st.session_state.role = "researcher"
                st.session_state.user_id = researcher.id
                st.rerun()

            else:
                st.error("Invalid credentials")

        # ---------------- REGISTER ACTION ----------------
        if register_clicked:
            st.session_state.selected_page = "register_selection"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

def dashboard():

    db = SessionLocal()

    total_calls = db.query(Call).count()
    total_props = db.query(Proposal).count()
    total_reviewers = db.query(Reviewer).count()
    total_assignments = db.query(Assignment).count()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{total_calls}</div>
            <div class="metric-label">Active Calls</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{total_props}</div>
            <div class="metric-label">Proposals</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{total_reviewers}</div>
            <div class="metric-label">Reviewers</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{total_assignments}</div>
            <div class="metric-label">Assignments</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Calls Overview")

    calls = db.query(Call).all()
    if calls:
        for c in calls:
            with st.expander(f"{c.title} ({c.identifier})"):
                st.markdown(f"""
                <div class="modern-card">     
                    <h3>{c.title} ({c.identifier})</h3>
                    <p><strong>Priority Areas:</strong> {c.priority_areas}</p>
                    <p><strong>Objectives:</strong> {c.objectives}</p>
                </div>
                """, unsafe_allow_html=True)
                num_props = db.query(Proposal).filter(Proposal.call_id == c.id).count()
                st.write(f"**Number of Proposals:** {num_props}")
    else:
        st.info("No calls in the database yet.")
    db.close()
# ----------------- CALLS PAGE -----------------
def calls_page():
    db = SessionLocal()

    # Existing calls
    st.subheader("Existing Calls")
    calls = db.query(Call).all()
    if calls:
        for c in calls:
            with st.expander(f"{c.title} ({c.identifier})"):
                format_bullet_list("Background / Rationale", c.background)
                format_bullet_list("Objectives", c.objectives)
                format_bullet_list("Priority Areas / Thematic Areas", c.priority_areas)
                format_bullet_list("Scope of Funding", c.scope)
                format_bullet_list("Funding Details", c.funding_details)
                format_bullet_list("Eligibility Criteria", c.eligibility)
                format_bullet_list("Expected Deliverables", c.expected_deliverables)
                format_bullet_list("Evaluation Criteria", c.evaluation_criteria)
                format_bullet_list("Ethical & Regulatory Requirements", c.ethics)
                format_bullet_list("Application Requirements", c.application_requirements)
                format_bullet_list("Timeline", c.timeline)
                format_bullet_list("Reporting & Monitoring", c.reporting_monitoring)

                num_props = db.query(Proposal).filter(Proposal.call_id == c.id).count()
                st.write(f"**Number of Proposals:** {num_props}")

    else:
        st.info("No calls in the database yet.")

    st.markdown("---")

    # Add new call
    st.subheader("Add New Call")
    with st.form("new_call_form"):
        title = st.text_input("Call Title")
        identifier = st.text_input("Identifier / Reference Number")
        background = st.text_area("Background / Rationale")
        objectives = st.text_area("Objectives")
        priority_areas = st.text_area("Priority Areas / Thematic Areas")
        scope = st.text_area("Scope of Funding")
        funding_details = st.text_area("Funding Details")
        eligibility = st.text_area("Eligibility Criteria")
        expected_deliverables = st.text_area("Expected Deliverables")
        evaluation_criteria = st.text_area("Evaluation Criteria")
        ethics = st.text_area("Ethical & Regulatory Requirements")
        application_requirements = st.text_area("Application Requirements")
        timeline = st.text_area("Timeline")
        reporting_monitoring = st.text_area("Reporting & Monitoring")

        submitted = st.form_submit_button("Save Call")
        if submitted:
            if title and identifier:
                db.add(Call(
                    title=title,
                    identifier=identifier,
                    background=background,
                    objectives=objectives,
                    priority_areas=priority_areas,
                    scope=scope,
                    funding_details=funding_details,
                    eligibility=eligibility,
                    expected_deliverables=expected_deliverables,
                    evaluation_criteria=evaluation_criteria,
                    ethics=ethics,
                    application_requirements=application_requirements,
                    timeline=timeline,
                    reporting_monitoring=reporting_monitoring
                ))
                db.commit()
                st.success(f"Call '{title}' created successfully")
                st.rerun()
            else:
                st.warning("Title and Identifier are required")
    db.close()
# ----------------- REVIEWERS PAGE -----------------
def reviewers_page():
    st.header("Reviewers")
    db = SessionLocal()
    reviewers = db.query(Reviewer).all()
    if not reviewers:
        st.info("No reviewers found")
    for r in reviewers:
        with st.expander(r.name):
            st.write(f"**Email:** {r.email}")
            st.write(f"**Expertise:** {r.expertise}")
            st.text_area("CV Text", r.cv_text, height=200)
            st.download_button("Download CV PDF", data=r.cv_pdf, file_name=f"{r.name}.pdf", mime="application/pdf")
    db.close()

def delete_proposal(proposal_id):
    db = SessionLocal()
    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

    if proposal:
        db.delete(proposal)

        # Also delete related assignments and reviews (important for consistency)
        db.query(Assignment).filter(Assignment.proposal_id == proposal_id).delete()
        db.query(ReviewScore).filter(ReviewScore.proposal_id == proposal_id).delete()

        db.commit()

    db.close()
# ----------------- PROPOSALS PAGE -----------------
def proposals_page():

    st.header("📄 Submitted Proposals")

    db = SessionLocal()

    calls = db.query(Call).all()

    if not calls:
        st.info("No calls available")
        db.close()
        return

    call_dict = {f"{c.title} ({c.identifier})": c for c in calls}
    selected_call_name = st.selectbox(
        "Select Call",
        list(call_dict.keys())
    )

    selected_call = call_dict[selected_call_name]

    # ---------------- SELECT AREA ----------------
    if selected_call.priority_areas:
        areas = [
            a.strip()
            for a in selected_call.priority_areas
            .replace("\n", ",")
            .split(",")
            if a.strip() != ""
        ]
    else:
        areas = []

    if areas:
        selected_area = st.selectbox(
            "Select Area",
            ["All Areas"] + areas
        )
    else:
        selected_area = "All Areas"
        st.warning("No areas defined for this call")

    # ---------------- FILTER PROPOSALS ----------------
    query = db.query(Proposal).filter(
        Proposal.call_id == selected_call.id
    )

    if selected_area != "All Areas":
        query = query.filter(
            Proposal.selected_area == selected_area
        )

    proposals = query.all()

    st.markdown("---")

    if not proposals:
        st.info("No proposals found for this area.")
        db.close()
        return

    st.subheader("📌 Proposal List")

    for p in proposals:

        st.markdown(f"""
        <div class="modern-card">
            <h3 style="margin-bottom:5px;">{p.title}</h3>
            <p><strong>Area:</strong> {p.selected_area}</p>
            <p><strong>Status:</strong> {p.status}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Open", key=f"open_{p.id}"):
                st.session_state.selected_proposal = p.id

        # ---------------- ADMIN DELETE BUTTON ----------------
        if st.session_state.role == "admin":
            with col2:
                if st.button("🗑 Delete", key=f"delete_{p.id}"):
                    delete_proposal(p.id)
                    st.success("Proposal deleted successfully.")
                    st.rerun()

        # ---------------- DETAILS VIEW ----------------
        if st.session_state.get("selected_proposal") == p.id:

            st.markdown("### Proposal Details")

            st.write(f"**Title:** {p.title}")
            st.write(f"**Abstract:** {p.abstract}")
            st.write(f"**Keywords:** {p.keywords}")
            st.write(f"**Selected Area:** {p.selected_area}")
            st.write(f"**Status:** {p.status}")
            st.write(f"**Submitted By:** {p.submitted_by}")

            st.text_area(
                "Full Proposal Text",
                p.proposal_text,
                height=250
            )

            st.download_button(
                "⬇ Download Proposal PDF",
                data=p.proposal_pdf,
                file_name=f"{p.title}.pdf",
                mime="application/pdf"
            )

            st.divider()

    db.close()
# ----------------- ASSIGNMENTS PAGE -----------------
def assignments_page():

    st.header("🔗 Assign Reviewers")

    db = SessionLocal()

    calls = db.query(Call).all()
    reviewers = db.query(Reviewer).all()

    if not calls:
        st.info("No calls available.")
        db.close()
        return

    if not reviewers:
        st.info("No reviewers available.")
        db.close()
        return

    # ---------------- SELECT CALL ----------------
    call_options = {
        f"{c.title} ({c.identifier})": c.id for c in calls
    }

    selected_call_name = st.selectbox(
        "Select Call",
        list(call_options.keys())
    )

    selected_call_id = call_options[selected_call_name]
    selected_call = db.query(Call).get(selected_call_id)

    # ---------------- SELECT PROPOSAL ----------------
    proposals = db.query(Proposal).filter(
        Proposal.call_id == selected_call_id
    ).all()

    if not proposals:
        st.info("No proposals for this call.")
        db.close()
        return

    proposal_dict = {p.title: p for p in proposals}

    selected_prop_title = st.selectbox(
        "Select Proposal",
        list(proposal_dict.keys())
    )

    selected_prop = proposal_dict[selected_prop_title]

    st.divider()

    # ---------------- GENERATE SUGGESTIONS ----------------
    if st.button("📊 Generate Reviewer Suggestions"):

        prop_text = (
            (selected_prop.proposal_text or "")
            + " "
            + (selected_call.priority_areas or "")
        )

        prop_emb = model.encode(prop_text)
        rev_embs = model.encode([r.cv_text or "" for r in reviewers])

        scores = cosine_similarity([prop_emb], rev_embs)[0]
        ranked_indices = scores.argsort()[::-1]

        suggestions = []
        count = 0

        for idx in ranked_indices:

            if count >= 3:
                break

            reviewer = reviewers[idx]
            score = float(scores[idx])

            existing = db.query(Assignment).filter(
                Assignment.proposal_id == selected_prop.id,
                Assignment.reviewer_id == reviewer.id
            ).first()

            if existing:
                continue

            matched_areas = []

            if selected_call.priority_areas and reviewer.expertise:
                for area in selected_call.priority_areas.split(","):
                    if area.strip().lower() in reviewer.expertise.lower():
                        matched_areas.append(area.strip())

            explanation = f"""
Similarity Score:
- Text embedding similarity
- Priority area matching

Matched Areas:
{', '.join(matched_areas) if matched_areas else 'None'}

Reviewer Expertise:
{reviewer.expertise}
"""

            suggestions.append(
                {
                    "reviewer_id": reviewer.id,
                    "name": reviewer.name,
                    "score": score,
                    "explanation": explanation
                }
            )

            count += 1

        st.session_state["suggestions"] = suggestions
        st.rerun()

    # ---------------- SHOW SUGGESTIONS ----------------
    if "suggestions" in st.session_state:

        suggestions = st.session_state["suggestions"]

        if suggestions:

            st.subheader("✅ Suggested Reviewers (Max 3)")

            for s in suggestions:

                st.markdown(f"""
                <div class="modern-card">
                    <h3 style="margin-bottom:8px;">👤 {s['name']}</h3>
                    <p><strong>Similarity Score:</strong> {s['score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Explainability Details"):
                    st.code(s["explanation"])

            st.divider()

            # ---------------- CONFIRM BUTTON ----------------
            if st.button("✔ Confirm & Save Assignments"):

                for s in suggestions:

                    db.add(
                        Assignment(
                            proposal_id=selected_prop.id,
                            reviewer_id=s["reviewer_id"],
                            similarity_score=s["score"],
                            anonymized=True
                        )
                    )

                db.commit()

                st.success("Assignments saved successfully.")

                st.session_state.pop("suggestions", None)

                st.rerun()

        else:
            st.info("No new reviewer suggestions.")

    db.close()# REGISTRATION PAGES
# =====================================================
def criteria_page():
    st.header("📝 Reviewing Criteria Management")

    db = SessionLocal()

    calls = db.query(Call).all()

    if not calls:
        st.info("No calls available.")
        db.close()
        return

    call_dict = {f"{c.title} ({c.identifier})": c for c in calls}
    selected_call_name = st.selectbox("Select Call", list(call_dict.keys()))

    selected_call = call_dict[selected_call_name]

    # Extract priority areas from call
    areas = []
    if selected_call.priority_areas:
        areas = [
            a.strip()
            for a in selected_call.priority_areas.split(",")
            if a.strip()
        ]

    selected_area = st.selectbox(
        "Select Area (Optional)",
        ["General"] + areas
    )

    criteria_text = st.text_area(
        "Enter Reviewing Criteria",
        placeholder="Example:\n- Originality\n- Methodology\n- Impact"
    )

    if st.button("Save Criteria"):
        db.add(ReviewCriteria(
            call_id=selected_call.id,
            area=selected_area if selected_area != "General" else None,
            criteria=criteria_text
        ))
        db.commit()
        st.success("Criteria saved successfully")

    st.markdown("---")
    st.subheader("Existing Criteria")

    saved = db.query(ReviewCriteria).filter(
        ReviewCriteria.call_id == selected_call.id
    ).all()

    for c in saved:
        st.markdown(f"""
        <div class="modern-card">
            <p><strong>Area:</strong> {c.area or "General"}</p>
            <p>{c.criteria}</p>
        </div>
        """, unsafe_allow_html=True)

    db.close()
def reviewer_register_page():

    st.title("Reviewer Registration")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    expertise = st.text_area("Expertise")
    cv = st.file_uploader("Upload CV", type=["pdf"])

    if st.button("Register"):
        if cv is None:
            st.warning("Upload CV")
        else:
            pdf_bytes = cv.read()
            text = ""

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text()

            db = get_db()
            db.add(Reviewer(
                name=name,
                email=email,
                password_hash=hash_password(password),
                expertise=expertise,
                cv_text=text,
                cv_pdf=pdf_bytes
            ))
            db.commit()
            db.close()

            st.success("Registered Successfully")
            st.info("Go back and login")


def researcher_register_page():

    st.title("Researcher Registration")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    expertise = st.text_area("Research Area")
    cv = st.file_uploader("Upload CV", type=["pdf"])

    if st.button("Register"):
        if cv is None:
            st.warning("Upload CV")
        else:
            pdf_bytes = cv.read()
            text = ""

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text()

            db = get_db()
            db.add(Researcher(
                name=name,
                email=email,
                password_hash=hash_password(password),
                expertise=expertise,
                cv_text=text,
                cv_pdf=pdf_bytes
            ))
            db.commit()
            db.close()

            st.success("Registered Successfully")
            st.info("Go back and login")

def researcher_dashboard():

    col1, col2 = st.columns([1, 4])

    # ---------------- MENU ----------------
    with col1:

        st.markdown("### Navigation")

        if st.button("📢 View Calls", use_container_width=True):
            st.session_state.researcher_page = "view_calls"

        if st.button("📤 Submit Proposal", use_container_width=True):
            st.session_state.researcher_page = "submit_proposal"

        if st.button("📊 Proposal Status", use_container_width=True):
            st.session_state.researcher_page = "proposal_status"

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # ---------------- CONTENT ----------------
    with col2:

        page = st.session_state.get("researcher_page", "view_calls")

        db = get_db()

        # ================= VIEW CALLS =================
        if page == "view_calls":

            st.subheader("📢 Available Calls")

            calls = db.query(Call).all()

            if not calls:
                st.info("No calls available")

            for c in calls:

                st.markdown(f"""
                <div class="modern-card">
                    <h3>{c.title} ({c.identifier})</h3>
                    <p><strong>Priority Areas:</strong> {c.priority_areas}</p>
                    <p><strong>Objectives:</strong> {c.objectives}</p>
                </div>
                """, unsafe_allow_html=True)

                count = db.query(Proposal).filter(
                    Proposal.call_id == c.id
                ).count()

                st.write(f"📄 Proposals Submitted: {count}")
                st.divider()

        # ================= SUBMIT PROPOSAL =================
        elif page == "submit_proposal":

            st.subheader("📤 Submit Proposal")

            calls = db.query(Call).all()

            if not calls:
                st.info("No calls available")
            else:

                call_dict = {
                    f"{c.title} ({c.identifier})": c
                    for c in calls
                }

                selected_call_name = st.selectbox(
                    "Select Call",
                    list(call_dict.keys())
                )

                selected_call = call_dict[selected_call_name]

                title = st.text_input("Proposal Title")
                abstract = st.text_area("Abstract")
                keywords = st.text_input("Keywords")
                pdf = st.file_uploader("Upload Proposal PDF", type=["pdf"])

                if st.button("Submit Proposal"):

                    if not title:
                        st.warning("Title required")
                    elif pdf is None:
                        st.warning("Upload PDF")
                    else:

                        pdf_bytes = pdf.read()

                        new_prop = Proposal(
                            title=title,
                            abstract=abstract,
                            keywords=keywords,
                            proposal_text="",
                            proposal_pdf=pdf_bytes,
                            call_id=selected_call.id,
                            submitted_by=st.session_state.user_id,
                            status="Under Review"
                        )

                        db.add(new_prop)
                        db.commit()

                        st.success("Proposal submitted successfully")
                        st.rerun()

        # ================= PROPOSAL STATUS =================
        elif page == "proposal_status":

            st.subheader("📊 My Proposals")

            proposals = db.query(Proposal).filter(
                Proposal.submitted_by == st.session_state.user_id
            ).all()

            if not proposals:
                st.info("No proposals submitted yet")

            for p in proposals:

                st.markdown(f"""
                <div class="modern-card">
                    <h3>{p.title}</h3>
                    <p><strong>Status:</strong> {p.status}</p>
                    <p><strong>Area:</strong> {p.selected_area}</p>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "Download PDF",
                    data=p.proposal_pdf,
                    file_name=f"{p.title}.pdf",
                    mime="application/pdf"
                )

        db.close()
def reviewer_dashboard():

    db = SessionLocal()

    # ---------------- SIDEBAR MENU ----------------
    with st.sidebar:
        st.title("Reviewer Menu")

        if st.button("📢 View Calls"):
            st.session_state.reviewer_page = "view_calls"

        if st.button("📄 Assigned Proposals"):
            st.session_state.reviewer_page = "assigned"

        if st.button("📝 Submit Review"):
            st.session_state.reviewer_page = "review"

        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # ---------------- MAIN CONTENT ----------------
    page = st.session_state.get("reviewer_page", "view_calls")

    # =====================================================
    # VIEW CALLS
    # =====================================================
    if page == "view_calls":

        st.subheader("📢 Available Calls")

        calls = db.query(Call).all()

        if not calls:
            st.info("No calls available.")
        else:
            for c in calls:
                with st.expander(f"{c.title} ({c.identifier})"):
                    st.write("**Priority Areas:**")
                    st.write(c.priority_areas)
                    st.write("**Objectives:**")
                    st.write(c.objectives)

    # =====================================================
    # ASSIGNED PROPOSALS
    # =====================================================
    elif page == "assigned":

        st.subheader("📄 Assigned Proposals")

        reviewer_id = st.session_state.user_id

        assignments = db.query(Assignment).filter(
            Assignment.reviewer_id == reviewer_id
        ).all()

        if not assignments:
            st.info("No proposals assigned to you.")
        else:
            for a in assignments:

                proposal = db.query(Proposal).get(a.proposal_id)

                with st.expander(proposal.title):

                    st.write("**Abstract:**")
                    st.write(proposal.abstract)

                    st.write("**Keywords:**")
                    st.write(proposal.keywords)

                    st.download_button(
                        "Download Proposal PDF",
                        data=proposal.proposal_pdf,
                        file_name=f"{proposal.title}.pdf",
                        mime="application/pdf"
                    )

    # =====================================================
    # SUBMIT REVIEW
    # =====================================================
    elif page == "review":

        st.subheader("📝 Submit Review")

        reviewer_id = st.session_state.user_id

        assignments = db.query(Assignment).filter(
            Assignment.reviewer_id == reviewer_id
        ).all()

        if not assignments:
            st.info("No assigned proposals to review.")
        else:

            proposal_options = {}

            for a in assignments:
                proposal = db.query(Proposal).get(a.proposal_id)
                proposal_options[proposal.title] = proposal.id

            selected_title = st.selectbox(
                "Select Proposal to Review",
                list(proposal_options.keys())
            )

            proposal_id = proposal_options[selected_title]

            originality = st.slider("Originality", 1, 10)
            methodology = st.slider("Methodology", 1, 10)
            impact = st.slider("Impact", 1, 10)
            feasibility = st.slider("Feasibility", 1, 10)
            comments = st.text_area("Comments")

            if st.button("Submit Review"):

                overall = (originality + methodology + impact + feasibility) / 4

                review = ReviewScore(
                    proposal_id=proposal_id,
                    reviewer_id=reviewer_id,
                    originality=originality,
                    methodology=methodology,
                    impact=impact,
                    feasibility=feasibility,
                    overall=overall,
                    comments=comments
                )

                db.add(review)
                db.commit()

                st.success("Review submitted successfully!")

    db.close()
def format_bullet_list(title, text):
    """Display multi-line text from DB as a bullet list with a heading."""
    if text:
        st.write(f"**{title}:**")
        for line in text.split("\n"):
            if line.strip():
                st.markdown(f"- {line.strip()}")

# ----------------- MAIN -----------------
# ----------------- MAIN -----------------

if not st.session_state.logged_in:

    page = st.session_state.get("selected_page")

    if page == "reviewer_register":
        reviewer_register_page()

    elif page == "researcher_register":
        researcher_register_page()

    elif page == "register_selection":
        register_selection()

    else:
        login_page()

else:

    if st.session_state.role == "admin":
        with st.sidebar:
            st.title("Admin Menu")

            if st.button("🏠 Dashboard"):
                st.session_state.selected_page = "Dashboard"

            if st.button("📣 Calls"):
                st.session_state.selected_page = "Calls"

            if st.button("📁 Reviewers"):
                st.session_state.selected_page = "Reviewers"

            if st.button("📄 Proposals"):
                st.session_state.selected_page = "Proposals"

            if st.button("🔗 Assignments"):
                st.session_state.selected_page = "Assignments"

            if st.button("📝 Review Criteria"):
                st.session_state.selected_page = "Criteria"

            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

        page = st.session_state.selected_page

        if page == "Dashboard":
            dashboard()

        elif page == "Calls":
            calls_page()

        elif page == "Reviewers":
            reviewers_page()

        elif page == "Proposals":
            proposals_page()

        elif page == "Assignments":
            assignments_page()

        elif page == "Criteria":
            criteria_page()

    elif st.session_state.role == "researcher":
        researcher_dashboard()

    elif st.session_state.role == "reviewer":
        reviewer_dashboard()
