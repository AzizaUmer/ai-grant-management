from sqlalchemy import Column, Integer, String, Text, Float, LargeBinary, Boolean, ForeignKey
from database import Base

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

class Researcher(Base):
    __tablename__ = "researchers"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password_hash = Column(String)

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

class Researcher(Base):
    __tablename__ = "researchers"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password_hash = Column(String)
    expertise = Column(Text)
    cv_text = Column(Text)
    cv_pdf = Column(LargeBinary)

class Proposal(Base):
    __tablename__ = "proposals"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    proposal_text = Column(Text)
    proposal_pdf = Column(LargeBinary)
    call_id = Column(Integer, ForeignKey("calls.id"))
    status = Column(String, default="Under Review")
    submitted_by = Column(Integer)  # researcher id

class Assignment(Base):
    __tablename__ = "assignments"
    id = Column(Integer, primary_key=True)
    proposal_id = Column(Integer, ForeignKey("proposals.id"))
    reviewer_id = Column(Integer, ForeignKey("reviewers.id"))
    similarity_score = Column(Float)
    anonymized = Column(Boolean, default=True)

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

class ConflictOfInterest(Base):
    __tablename__ = "conflicts"
    id = Column(Integer, primary_key=True)
    reviewer_id = Column(Integer)
    proposal_id = Column(Integer)
    reason = Column(Text)