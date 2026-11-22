"""
scripts/register_student.py - Standalone CLI to register a student's face.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/register_student.py --id "24FE1A0499" --name "John Doe"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import SessionLocal
from repository.encoding_repository import EncodingRepository
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from services.enrollment_service import EnrollmentService


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face and register a student.")
    parser.add_argument("--id", required=True, help="Student register number (e.g. 24FE1A0499)")
    parser.add_argument("--name", default="", help="Student full name")
    parser.add_argument("--dept", default="CSE", help="Department")
    parser.add_argument("--sem", default="1", help="Semester")
    parser.add_argument("--sec", default="A", help="Section")
    parser.add_argument("--teacher", default="TEACHER01", help="Teacher ID")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")

    args = parser.parse_args()
    student_id = args.id.strip()
    student_name = args.name.strip() or student_id

    print("=" * 60)
    print("  STUDENT FACE ENROLLMENT")
    print("=" * 60)
    print(f"  Register ID : {student_id}")
    print(f"  Name        : {student_name}")
    print(f"  Department  : {args.dept}")
    print(f"  Section     : {args.sec}")
    print(f"  Camera Index: {args.camera}")
    print("=" * 60)
    print("\n[INFO] Starting camera... Look directly at the webcam.")
    print("[INFO] Keep your face visible and still.\n")

    camera_svc = CameraService()
    detector = DetectionService()
    encoder = EncodingService()
    repo = EncodingRepository()

    enrollment_svc = EnrollmentService(
        camera_service=camera_svc,
        detection_service=detector,
        encoding_service=encoder,
        show_preview=True,
    )

    try:
        result = enrollment_svc.enroll(student_id)
    except Exception as exc:
        print(f"\n[ERROR] Enrollment failed with exception: {exc}")
        sys.exit(1)

    if not result.success:
        print(f"\n[FAILED] Enrollment could not complete: {result.error_message} (code: {result.error_code})")
        sys.exit(1)

    # Save encodings
    try:
        enc_path = encoder.save_encodings(student_id, result.encodings)
        print(f"[OK] Saved {len(result.encodings)} face encodings to {enc_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to save encodings file: {exc}")
        sys.exit(1)

    # Save to local database
    with SessionLocal() as db:
        if repo.student_exists(db, student_id):
            print(f"[INFO] Student {student_id} already exists in database. Updating record...")
        else:
            repo.create_student(
                db,
                student_id=student_id,
                name=student_name,
                department=args.dept,
                semester=args.sem,
                section=args.sec,
                teacher_id=args.teacher,
                encoding_path=str(enc_path) if enc_path else None,
                image_count=result.images_captured,
            )
            db.commit()
            print(f"[OK] Student record saved in local database.")

    print("\n" + "=" * 60)
    print(f"  ENROLLMENT COMPLETED SUCCESSFULLY FOR: {student_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
