from sqlalchemy import (
    ForeignKey,
    create_engine,
    Column,
    Integer,
    Float,
    DateTime,
    String,
)
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_version_id = Column(
        Integer, ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    zone_id = Column(Integer)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    week = Column(Integer)
    predicted_trips = Column(Float)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    name = Column(String)
    mae = Column(Float)
    mape = Column(Float)
    n_estimators = Column(Integer)
    learning_rate = Column(Float)


def get_engine(db_url: str):
    return create_engine(db_url)


def init_db(db_url: str):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
