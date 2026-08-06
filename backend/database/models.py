from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="investigator")
    
    evidence = relationship("Evidence", back_populates="investigator")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    file_size_bytes = Column(Integer)
    duration_seconds = Column(Float)
    upload_time = Column(DateTime, default=datetime.utcnow)
    
    # User / Chain of Custody
    user_id = Column(Integer, ForeignKey("users.id"))
    investigator = relationship("User", back_populates="evidence")

    # Chain of custody
    sha256_hash = Column(String, nullable=True)

    # Audio properties
    sample_rate = Column(Integer, default=0)
    channels = Column(Integer, default=0)
    bitrate = Column(Integer, default=0)

    # Analysis results
    ai_manipulation_score = Column(Float, default=0.0)  # 0.0 to 1.0 (1.0 = deepfake)
    ai_label = Column(String, default="UNKNOWN")        # ORGANIC | SYNTHETIC | UNKNOWN
    ai_details = Column(Text, nullable=True)            # JSON string for full AI model details
    hex_anomalies_detected = Column(Boolean, default=False)
    detected_format = Column(String, nullable=True)
    header_hex = Column(String, nullable=True)
    declared_extension = Column(String, nullable=True)
    extension_mismatch = Column(Boolean, default=False)

    # ENF
    enf_variance = Column(Float, default=0.0)
    splicing_detected = Column(Boolean, default=False)
    enf_anomaly_score = Column(Float, default=0.0)
    enf_target_freq = Column(Integer, default=50)
    enf_track_preview = Column(Text, nullable=True)  # JSON list of float values

    # Fingerprint
    acoustic_fingerprint = Column(String, nullable=True)
    pitch_block_count = Column(Integer, default=0)

    # ENF splice detail
    splice_jump_count = Column(Integer, default=0)

    # Embedded tags (JSON)
    embedded_tags = Column(Text, nullable=True)

    # NLP
    nlp_anomaly_detected = Column(Boolean, default=False)
    suspicious_keywords = Column(Text, nullable=True)  # JSON list

    status = Column(String, default="pending")  # pending | processing | completed | error
    error_message = Column(Text, nullable=True)

