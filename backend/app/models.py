import datetime
import enum
from typing import List
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ProjectStatus(str, enum.Enum):
    created = "created"
    parsed = "parsed"
    synthesizing = "synthesizing"
    done = "done"


class AudioSource(str, enum.Enum):
    upload = "upload"
    preset = "preset"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[ProjectStatus] = mapped_column(SAEnum(ProjectStatus), default=ProjectStatus.created)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.now)

    characters: Mapped[List["Character"]] = relationship("Character", back_populates="project", cascade="all, delete-orphan")
    dialogues: Mapped[List["Dialogue"]] = relationship("Dialogue", back_populates="project", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    gender: Mapped[str] = mapped_column(String(10), default="unknown")
    dialogue_count: Mapped[int] = mapped_column(Integer, default=0)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"))

    project: Mapped["Project"] = relationship("Project", back_populates="characters")
    audio_files: Mapped[List["AudioFile"]] = relationship("AudioFile", back_populates="character", cascade="all, delete-orphan")


class Dialogue(Base):
    __tablename__ = "dialogues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chapter: Mapped[str] = mapped_column(String(255), default="")
    text: Mapped[str] = mapped_column(Text)
    speaker: Mapped[str] = mapped_column(String(255))
    order: Mapped[int] = mapped_column(Integer)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"))

    project: Mapped["Project"] = relationship("Project", back_populates="dialogues")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(512))
    source: Mapped[AudioSource] = mapped_column(SAEnum(AudioSource))
    character_id: Mapped[int] = mapped_column(Integer, ForeignKey("characters.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.now)

    character: Mapped["Character"] = relationship("Character", back_populates="audio_files")
