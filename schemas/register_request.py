"""
schemas/register_request.py – Pydantic request schema for face registration.
"""

from pydantic import BaseModel, Field, field_validator


class RegisterFaceRequest(BaseModel):
    """Validated request body for POST /register-face."""

    student_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique student identifier",
        examples=["STU001"],
    )
    student_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Full name of the student",
        examples=["John Doe"],
    )
    department: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Department name",
        examples=["Computer Science"],
    )
    semester: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Current semester",
        examples=["3"],
    )
    section: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Class section",
        examples=["A"],
    )
    teacher_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Teacher / faculty identifier",
        examples=["TEACH001"],
    )

    @field_validator("student_id", "teacher_id", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading / trailing whitespace from ID fields."""
        return v.strip()

    @field_validator("student_name", "department", "section", mode="before")
    @classmethod
    def strip_and_title(cls, v: str) -> str:
        """Strip whitespace from text fields."""
        return v.strip()
