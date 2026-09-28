from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import ForeignKey, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    model_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"))
    zone_id: Mapped[Optional[int]] = mapped_column()
    hour: Mapped[Optional[int]] = mapped_column()
    day_of_week: Mapped[Optional[int]] = mapped_column()
    week: Mapped[Optional[int]] = mapped_column()
    predicted_trips: Mapped[Optional[float]] = mapped_column()


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    name: Mapped[Optional[str]] = mapped_column()
    mae: Mapped[Optional[float]] = mapped_column()
    mape: Mapped[Optional[float]] = mapped_column()
    n_estimators: Mapped[Optional[int]] = mapped_column()
    learning_rate: Mapped[Optional[float]] = mapped_column()


class DemandHistory(Base):
    __tablename__ = "demand_history"
    __table_args__ = (UniqueConstraint("zone_id", "pickup_hour_ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[int] = mapped_column(index=True)
    pickup_hour_ts: Mapped[datetime] = mapped_column()
    trip_count: Mapped[float] = mapped_column()
    pickup_hour: Mapped[int] = mapped_column()
    pickup_dow: Mapped[int] = mapped_column()
    temperature_2m: Mapped[Optional[float]] = mapped_column()
    precipitation: Mapped[Optional[float]] = mapped_column()
    snowfall: Mapped[Optional[float]] = mapped_column()
    is_holiday: Mapped[Optional[int]] = mapped_column(default=0)


def get_engine(db_url: str):
    return create_engine(db_url)


def init_db(db_url: str):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
