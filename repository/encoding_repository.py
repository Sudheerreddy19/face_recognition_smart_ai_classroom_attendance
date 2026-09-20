"""
repository/encoding_repository.py – Student CRUD operations (database layer).

Named "encoding_repository" because it was originally the home for encoding
metadata, but has grown into the general student repository.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from models.student import Student
from utils.logger import get_logger

logger = get_logger(__name__)


class EncodingRepository:
    """Data access layer for the :class:`~models.student.Student` model."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_student_by_id(
        self, db: Session, student_id: str
    ) -> Optional[Student]:
        """Fetch a student by their unique *student_id*.

        Args:
            db:         Active SQLAlchemy session.
            student_id: Student identifier.

        Returns:
            :class:`~models.student.Student` instance, or ``None`` if absent.
        """
        return (
            db.query(Student)
            .filter(Student.student_id == student_id, Student.is_active == True)  # noqa: E712
            .first()
        )

    def get_all_students(self, db: Session) -> list[Student]:
        """Return all active students ordered by registration date.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of :class:`~models.student.Student` objects.
        """
        return (
            db.query(Student)
            .filter(Student.is_active == True)  # noqa: E712
            .order_by(Student.registered_at.desc())
            .all()
        )

    def student_exists(self, db: Session, student_id: str) -> bool:
        """Return ``True`` if an active student with *student_id* exists.

        Args:
            db:         Active SQLAlchemy session.
            student_id: Student identifier.
        """
        return (
            db.query(Student.id)
            .filter(Student.student_id == student_id, Student.is_active == True)  # noqa: E712
            .first()
            is not None
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def create_student(
        self,
        db: Session,
        *,
        student_id: str,
        name: str,
        department: str,
        semester: str,
        section: str,
        teacher_id: str,
        encoding_path: Optional[str] = None,
        image_count: int = 0,
    ) -> Student:
        """Insert a new student record.

        Args:
            db:            Active SQLAlchemy session.
            student_id:    Unique student identifier.
            name:          Full name.
            department:    Department / faculty.
            semester:      Current semester.
            section:       Class section.
            teacher_id:    Associated teacher.
            encoding_path: Path to the encoding pickle file.
            image_count:   Number of training images captured.

        Returns:
            Newly created :class:`~models.student.Student` instance.
        """
        student = Student(
            student_id=student_id,
            name=name,
            department=department,
            semester=semester,
            section=section,
            teacher_id=teacher_id,
            encoding_path=encoding_path,
            image_count=image_count,
            is_active=True,
        )
        db.add(student)
        db.flush()  # flush to get PK without full commit
        logger.info(
            "Created student record: id=%s name=%s dept=%s",
            student_id,
            name,
            department,
        )
        return student

    def update_encoding_info(
        self,
        db: Session,
        student_id: str,
        encoding_path: str,
        image_count: int,
    ) -> Optional[Student]:
        """Update encoding path and image count for an existing student.

        Args:
            db:            Active SQLAlchemy session.
            student_id:    Student identifier.
            encoding_path: New encoding file path.
            image_count:   Updated image count.

        Returns:
            Updated :class:`~models.student.Student`, or ``None`` if not found.
        """
        student = self.get_student_by_id(db, student_id)
        if student:
            student.encoding_path = encoding_path
            student.image_count = image_count
            db.flush()
            logger.info(
                "Updated encoding info for student %s (images=%d)",
                student_id,
                image_count,
            )
        return student

    def soft_delete_student(self, db: Session, student_id: str) -> bool:
        """Soft-delete a student by setting ``is_active = False``.

        Args:
            db:         Active SQLAlchemy session.
            student_id: Student identifier.

        Returns:
            ``True`` if the student was found and deactivated, ``False`` otherwise.
        """
        student = self.get_student_by_id(db, student_id)
        if student:
            student.is_active = False
            db.flush()
            logger.info("Soft-deleted student %s", student_id)
            return True
        logger.warning(
            "Attempted to delete non-existent student %s", student_id
        )
        return False
