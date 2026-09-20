"""One-shot script: patch the attendance_session_test.py bug fix."""
import os, sys

path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "tests", "attendance_session_test.py")

with open(path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find the lock check line
LOCK_LINE = 'if result.status in ("RECOGNISED", "ALREADY_PRESENT"):\n                    continue'

if LOCK_LINE not in content:
    print("ERROR: could not find target. Printing nearby text:")
    idx = content.find("RECOGNISED")
    print(repr(content[idx:idx+300]))
    sys.exit(1)

INSERT = (
    '# -- Attendance: mark already-confirmed faces BEFORE the lock check --\n'
    '                # Faces confirmed during MONITORING are locked (continue below).\n'
    '                # We must mark them when the window opens, so this runs first.\n'
    '                if phase == "ATTENDANCE" \\\n'
    '                        and result.status in ("RECOGNISED", "ALREADY_PRESENT") \\\n'
    '                        and result.student_id \\\n'
    '                        and result.student_id not in attendance_marked:\n'
    '                    _sid = result.student_id\n'
    '                    saved = mark_local(_sid, result.student_name, args.subject,\n'
    '                                       args.teacher, args.section, result.distance)\n'
    '                    if saved:\n'
    '                        attendance_marked.add(_sid)\n'
    '                        att_time = datetime.now()\n'
    '                        att_records.append({\n'
    '                            "student_id":   _sid,\n'
    '                            "student_name": result.student_name,\n'
    '                            "att_date":     att_time.date(),\n'
    '                            "att_time":     att_time.time(),\n'
    '                            "distance":     result.distance,\n'
    '                        })\n'
    '                        print(f"  [ATTENDANCE] Marked: {_sid} - "\n'
    '                              f"{result.student_name}  d={result.distance:.3f}")\n\n'
    '                # Lock confirmed faces -- skip re-recognition\n'
)

# Replace only the FIRST occurrence of the lock check
content = content.replace(
    '                ' + LOCK_LINE,
    '                ' + INSERT + LOCK_LINE,
    1
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("PATCHED OK")
print("Verifying syntax ...")
import py_compile
py_compile.compile(path, doraise=True)
print("SYNTAX OK -- ready to run!")
