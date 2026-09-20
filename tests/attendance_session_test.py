"""
tests/attendance_session_test.py
SMART CLASSROOM - Phase 4: Attendance Session

PHASES:
  MONITORING  - camera runs, faces recognised, NO attendance saved yet
  ATTENDANCE  - last N minutes: save each recognised face to SQLite once
  COMPLETE    - session ends, sync to Spring Boot

USAGE:
  # 2-minute demo (1 min monitoring + 1 min attendance):
  python tests/attendance_session_test.py --subject "Demo" --section "CSE-A" --teacher "TEACHER01" --duration 2 --window 1

  # Real 50-minute class, last 10 minutes:
  python tests/attendance_session_test.py --subject "Data Structures" --section "CSE-A" --teacher "TEACHER01" --duration 50 --window 10

KEYBOARD: Q or ESC to quit early.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime

# -- Project root on sys.path so all project modules are importable ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    def _ram_mb():
        return _PROC.memory_info().rss / (1024 * 1024)
except ImportError:
    def _ram_mb():
        return 0.0

try:
    import face_recognition as _fr
except ImportError:
    print("[ERROR] face_recognition not installed. Activate .venv.")
    sys.exit(1)


# =============================================================================
# CONFIG
# =============================================================================
CAMERA_INDEX    = 0
FRAME_W         = 640
FRAME_H         = 480
DETECT_SCALE    = 1.0
UPSAMPLE        = 1
DETECT_EVERY_N  = 3
RECOG_INTERVAL  = 3.0
RECOG_THRESHOLD = 0.50
QUALITY_MIN_PX  = 60
BLUR_THRESHOLD  = 40.0
MIN_TRACK_PX    = 40
MAX_TRACK_PX    = 380
MAX_DISAPPEARED = 20
IOU_THRESHOLD   = 0.30

# BGR colours
C_GREEN  = (0, 220, 0)
C_TEAL   = (200, 180, 0)
C_RED    = (0, 0, 220)
C_AMBER  = (0, 165, 255)
C_YELLOW = (180, 180, 0)
C_WHITE  = (255, 255, 255)
C_DARK   = (20, 20, 20)
FONT     = cv2.FONT_HERSHEY_SIMPLEX
WIN      = "Smart Classroom - Attendance Session  (Q/ESC to quit)"


# =============================================================================
# IoU TRACKER
# =============================================================================
class IouTracker:
    def __init__(self):
        self._next_id = 1
        self._tracks = {}

    def update(self, detections):
        for t in self._tracks.values():
            t["new"] = False
        if not detections:
            dead = [i for i, t in self._tracks.items() if self._tick(t)]
            for i in dead:
                del self._tracks[i]
            return {i: t["box"] for i, t in self._tracks.items()}
        if not self._tracks:
            for b in detections:
                self._reg(b)
            return {i: t["box"] for i, t in self._tracks.items()}

        ids = list(self._tracks.keys())
        tbs = [self._tracks[i]["box"] for i in ids]
        mat = np.zeros((len(tbs), len(detections)), dtype=np.float32)
        for ti, tb in enumerate(tbs):
            for di, db in enumerate(detections):
                mat[ti, di] = self._iou(tb, db)

        mt, md = set(), set()
        for v, ti, di in sorted(
                [(mat[ti, di], ti, di) for ti in range(len(ids))
                 for di in range(len(detections))], key=lambda x: -x[0]):
            if v < IOU_THRESHOLD:
                break
            if ti in mt or di in md:
                continue
            self._tracks[ids[ti]]["box"] = detections[di]
            self._tracks[ids[ti]]["dis"] = 0
            mt.add(ti); md.add(di)

        dead = [ids[ti] for ti in range(len(ids))
                if ti not in mt and self._tick(self._tracks[ids[ti]])]
        for i in dead:
            del self._tracks[i]
        for di, b in enumerate(detections):
            if di not in md:
                self._reg(b)
        return {i: t["box"] for i, t in self._tracks.items()}

    def _reg(self, box):
        self._tracks[self._next_id] = {"box": box, "dis": 0, "new": True}
        self._next_id += 1

    def _tick(self, t):
        t["dis"] += 1
        return t["dis"] > MAX_DISAPPEARED

    @staticmethod
    def _iou(a, b):
        at, ar, ab, al = a
        bt, br, bb, bl = b
        iw = max(0, min(ar, br) - max(al, bl))
        ih = max(0, min(ab, bb) - max(at, bt))
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = (ar-al)*(ab-at) + (br-bl)*(bb-bt) - inter
        return inter / union if union > 0 else 0.0


# =============================================================================
# NMS
# =============================================================================
def nms(locs, thr=0.30):
    if len(locs) <= 1:
        return locs
    def area(b): return max(0, b[1]-b[3]) * max(0, b[2]-b[0])
    def iou(a, b):
        iw = max(0, min(a[1],b[1]) - max(a[3],b[3]))
        ih = max(0, min(a[2],b[2]) - max(a[0],b[0]))
        inter = iw * ih
        if inter == 0: return 0.0
        u = area(a) + area(b) - inter
        return inter/u if u > 0 else 0.0
    sl = sorted(locs, key=area, reverse=True)
    kept, sup = [], set()
    for i, ba in enumerate(sl):
        if i in sup: continue
        kept.append(ba)
        for j in range(i+1, len(sl)):
            if j not in sup and iou(ba, sl[j]) > thr:
                sup.add(j)
    return kept


# =============================================================================
# TRACK RESULT
# =============================================================================
@dataclass
class TR:
    status: str       = "PENDING"
    sid: str          = ""
    name: str         = ""
    dept: str         = ""
    dist: float       = 1.0
    t_recog: float    = 0.0


# =============================================================================
# STUDENT DATABASE
# =============================================================================
def load_db():
    import pickle
    enc_db = {}
    s_info = {}
    enc_dir = os.path.join(_PROJECT_ROOT, "encodings")
    if os.path.isdir(enc_dir):
        for fname in os.listdir(enc_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = fname[:-7]
            try:
                with open(os.path.join(enc_dir, fname), "rb") as f:
                    encs = pickle.load(f)
                if encs:
                    enc_db[sid] = encs
                    print(f"  Loaded {len(encs):2d} encodings: {sid}")
            except Exception as e:
                print(f"  [WARN] {fname}: {e}")

    try:
        from models.attendance import AttendanceRecord  # noqa - fixes mapper
        from models.student import Student
        from database import SessionLocal
        db = SessionLocal()
        try:
            rows = db.query(Student).filter(Student.is_active == True).all()  # noqa
            for s in rows:
                s_info[s.student_id] = {
                    "name": s.name, "dept": s.department,
                    "sem": s.semester, "sec": s.section,
                }
        finally:
            db.close()
        print(f"  Metadata: {len(s_info)} student(s).")
    except Exception as e:
        print(f"  [WARN] Metadata not loaded: {e}")
    return enc_db, s_info


# =============================================================================
# SAVE TO SQLITE
# =============================================================================
def mark_local(sid, sname, subject, teacher, section, dist):
    try:
        from models.attendance import AttendanceRecord  # noqa
        from models.student import Student              # noqa
        from database import SessionLocal
        from repository.attendance_repository import AttendanceRepository
        repo = AttendanceRepository()
        today = date.today()
        db = SessionLocal()
        try:
            if repo.is_already_marked(db, student_id=sid,
                                       subject=subject, attendance_date=today):
                return False
            repo.create(db=db, student_id=sid, subject=subject,
                        teacher_id=teacher, session=section,
                        attendance_date=today,
                        attendance_time=datetime.now().time(),
                        confidence=dist, status="PRESENT")
            db.commit()
            return True
        finally:
            db.close()
    except Exception as e:
        print(f"  [WARN] SQLite save failed ({sid}): {e}")
        return False


# =============================================================================
# SPRING BOOT SYNC
# =============================================================================
def sync_sb(records, subject, teacher, section):
    try:
        from services.springboot_service import SpringBootService
        sb = SpringBootService()
        ok = 0
        for r in records:
            if sb.sync_attendance(student_id=r["sid"], student_name=r["name"],
                                   attendance_date=r["d"], attendance_time=r["t"],
                                   subject=subject, teacher_id=teacher,
                                   session=section, confidence=r["dist"],
                                   status="PRESENT"):
                ok += 1
        return ok
    except Exception as e:
        print(f"  [WARN] Spring Boot sync: {e}")
        return 0


# =============================================================================
# FACE HELPERS
# =============================================================================
def detect(frame):
    rgb = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    locs = _fr.face_locations(rgb, number_of_times_to_upsample=UPSAMPLE, model="hog")
    del rgb
    locs = nms(locs)
    return [b for b in locs
            if MIN_TRACK_PX <= (b[1]-b[3]) <= MAX_TRACK_PX
            and MIN_TRACK_PX <= (b[2]-b[0]) <= MAX_TRACK_PX]


def quality_ok(frame, box):
    t, r, b, l = box
    if (r-l) < QUALITY_MIN_PX or (b-t) < QUALITY_MIN_PX:
        return False
    h, w = frame.shape[:2]
    crop = frame[max(0,t):min(h,b), max(0,l):min(w,r)]
    if crop.size == 0:
        return False
    return float(cv2.Laplacian(
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) >= BLUR_THRESHOLD


def recognise(frame, box, enc_db, s_info):
    t, r, b, l = box
    rgb = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    try:
        encs = _fr.face_encodings(rgb, known_face_locations=[(t,r,b,l)],
                                   num_jitters=1, model="small")
    except Exception:
        return "FAIL", "", "", 1.0
    if not encs:
        return "FAIL", "", "", 1.0
    live = np.array(encs[0], dtype=np.float64)
    best_id, best_d = "", float("inf")
    for sid, known in enc_db.items():
        d = float(np.min(np.linalg.norm(
            np.array(known, dtype=np.float64) - live, axis=1)))
        if d < best_d:
            best_d, best_id = d, sid
    if best_d <= RECOG_THRESHOLD:
        info = s_info.get(best_id, {})
        return "RECOGNISED", best_id, info.get("name", best_id), best_d
    return "UNKNOWN", "", "", best_d


# =============================================================================
# DRAWING
# =============================================================================
def fmt_t(s):
    s = max(0, int(s))
    return f"{s//60:02d}:{s%60:02d}"


def draw_box(frame, tid, box, tr):
    t, r, b, l = box
    fw, fh = r-l, b-t
    if tr.status == "RECOGNISED":
        col = C_GREEN
        l1 = f"{tr.name}  OK COMPLETED"
        l2 = f"Reg: {tr.sid}  d={tr.dist:.3f}"
    elif tr.status == "ALREADY_PRESENT":
        col = C_TEAL
        l1 = f"{tr.name}  Already Present"
        l2 = f"Reg: {tr.sid}"
    elif tr.status == "UNKNOWN":
        col = C_RED
        l1 = "UNKNOWN"
        l2 = f"d={tr.dist:.3f}"
    elif tr.status == "LOW_QUALITY":
        col = C_AMBER
        l1 = f"Face #{tid}  LOW QUALITY"
        l2 = f"{fw}x{fh}px"
    else:
        col = C_YELLOW
        l1 = f"Face #{tid}"
        l2 = "Analysing..."
    cv2.rectangle(frame, (l, t), (r, b), col, 2)
    (lw, lh), base = cv2.getTextSize(l1, FONT, 0.48, 1)
    ly = max(t-6, lh+4)
    cv2.rectangle(frame, (l, ly-lh-4), (l+lw+4, ly+base), col, -1)
    cv2.putText(frame, l1, (l+2, ly-2), FONT, 0.48, (0,0,0), 1, cv2.LINE_AA)
    if l2:
        cv2.putText(frame, l2, (l, b+16), FONT, 0.40, col, 1, cv2.LINE_AA)


def draw_hud(frame, phase, n_faces, n_marked, win_rem, fps, ram, subject, section):
    h, w = frame.shape[:2]
    pc = {"MONITORING": (180,100,0), "ATTENDANCE": (0,0,200), "COMPLETE": (0,200,0)}.get(phase, C_WHITE)
    cv2.rectangle(frame, (5,5), (w-5,h-5), pc, 5)
    cv2.rectangle(frame, (0,0), (w,42), C_DARK, -1)
    if phase == "MONITORING":
        lt = f"MONITORING | {subject} [{section}]"
        rt = f"Attendance opens in: {fmt_t(win_rem)}  Faces:{n_faces}  FPS:{fps:.0f}  RAM:{ram:.0f}MB"
    elif phase == "ATTENDANCE":
        lt = f"TAKING ATTENDANCE | {subject} [{section}]"
        rt = f"Window closes in: {fmt_t(win_rem)}  Marked:{n_marked}  Faces:{n_faces}  FPS:{fps:.0f}  RAM:{ram:.0f}MB"
    else:
        lt = f"COMPLETE | {subject} [{section}]"
        rt = f"Marked:{n_marked}  FPS:{fps:.0f}  RAM:{ram:.0f}MB"
    cv2.putText(frame, lt, (8,28), FONT, 0.50, pc, 1, cv2.LINE_AA)
    (sw,_),_ = cv2.getTextSize(rt, FONT, 0.42, 1)
    cv2.putText(frame, rt, (w-sw-8,28), FONT, 0.42, (0,230,180), 1, cv2.LINE_AA)
    if phase == "ATTENDANCE" and int(time.time()*2) % 2 == 0:
        msg = "  TAKING ATTENDANCE  "
        (bw,bh),_ = cv2.getTextSize(msg, FONT, 0.72, 2)
        bx, by = (w-bw)//2, h-48
        cv2.rectangle(frame, (bx-4,by-bh-4), (bx+bw+4,by+8), (0,0,180), -1)
        cv2.putText(frame, msg, (bx,by), FONT, 0.72, (255,255,255), 2, cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  required=True)
    ap.add_argument("--section",  required=True)
    ap.add_argument("--teacher",  required=True)
    ap.add_argument("--duration", type=int, default=50)
    ap.add_argument("--window",   type=int, default=10)
    args = ap.parse_args()

    if args.window >= args.duration:
        print("[ERROR] --window must be less than --duration"); sys.exit(1)

    dur_s = args.duration * 60
    win_s = args.window   * 60
    mon_s = dur_s - win_s

    print()
    print("=" * 60)
    print("  SMART CLASSROOM - ATTENDANCE SESSION")
    print("=" * 60)
    print(f"  Subject  : {args.subject}")
    print(f"  Section  : {args.section}")
    print(f"  Teacher  : {args.teacher}")
    print(f"  Duration : {args.duration} min | Window: last {args.window} min")
    print(f"  Window opens at: {fmt_t(mon_s)} into class")
    print(f"  RAM start: {_ram_mb():.1f} MB")
    print()

    print("[INFO] Loading student database ...")
    enc_db, s_info = load_db()
    if not enc_db:
        print("[WARN] No encodings. Register students first.")
    else:
        print(f"[INFO] {len(enc_db)} student(s) ready.")
    print()

    print(f"[INFO] Opening camera {CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Camera not available."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print(f"[INFO] Camera: {int(cap.get(3))}x{int(cap.get(4))}")
    print("[INFO] Session started. Press Q/ESC to quit.")
    print()

    tracker  = IouTracker()
    tr_map   = {}              # tid -> TR
    confirmed = set()          # sids recognised at any point (monitoring + window)
    marked    = set()          # sids saved to SQLite (only during window)
    records   = []             # list of dicts for Spring Boot sync

    fps_p  = time.perf_counter()
    fps_v  = 0.0
    fc     = 0
    mxf    = 0
    ramc   = _ram_mb()
    locs   = []
    t0     = time.perf_counter()
    phase  = "MONITORING"
    win_logged = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05); continue

            fc += 1
            now    = time.perf_counter()
            fps_v  = 1.0 / max(now - fps_p, 1e-6)
            fps_p  = now
            if fc % 30 == 0:
                ramc = _ram_mb()

            elapsed = now - t0
            remain  = max(0.0, dur_s - elapsed)

            # Phase transitions
            if elapsed >= dur_s:
                phase = "COMPLETE"
                draw_hud(frame, phase, 0, len(marked), 0, fps_v, ramc,
                         args.subject, args.section)
                cv2.imshow(WIN, frame)
                cv2.waitKey(2000)
                break
            if elapsed >= mon_s and phase == "MONITORING":
                phase = "ATTENDANCE"
                if not win_logged:
                    print(f"\n[INFO] *** ATTENDANCE WINDOW OPENED at "
                          f"{datetime.now().strftime('%H:%M:%S')} ***")
                    win_logged = True

            win_rem = remain if phase == "ATTENDANCE" else max(0.0, mon_s - elapsed)

            # Detection
            if fc % DETECT_EVERY_N == 0:
                locs = detect(frame)

            # Tracker
            active = tracker.update(locs)
            mxf    = max(mxf, len(active))

            for tid in list(tr_map.keys()):
                if tid not in active:
                    del tr_map[tid]

            # --- Per face: recognition + attendance ---
            for tid, box in active.items():
                if tid not in tr_map:
                    tr_map[tid] = TR()
                tr = tr_map[tid]

                # STEP 1: Mark attendance for already-confirmed faces
                # IMPORTANT: This runs BEFORE the lock/continue below.
                # Faces confirmed during MONITORING are locked; when the window
                # opens we still need to mark them - so check happens here.
                if (phase == "ATTENDANCE"
                        and tr.status in ("RECOGNISED", "ALREADY_PRESENT")
                        and tr.sid
                        and tr.sid not in marked):
                    saved = mark_local(tr.sid, tr.name, args.subject,
                                       args.teacher, args.section, tr.dist)
                    if saved:
                        marked.add(tr.sid)
                        at = datetime.now()
                        records.append({"sid": tr.sid, "name": tr.name,
                                        "d": at.date(), "t": at.time(),
                                        "dist": tr.dist})
                        print(f"  [ATTENDANCE] Marked: {tr.sid} - {tr.name}"
                              f"  d={tr.dist:.3f}")

                # STEP 2: Lock confirmed faces - skip re-recognition
                if tr.status in ("RECOGNISED", "ALREADY_PRESENT"):
                    continue

                if not enc_db:
                    continue

                if (now - tr.t_recog) < RECOG_INTERVAL and tr.status != "PENDING":
                    continue

                if not quality_ok(frame, box):
                    tr.status = "LOW_QUALITY"; continue

                # STEP 3: Recognise
                status, sid, name, dist = recognise(frame, box, enc_db, s_info)

                if status == "RECOGNISED":
                    if sid in confirmed:
                        tr.status = "ALREADY_PRESENT"
                    else:
                        tr.status = "RECOGNISED"
                        confirmed.add(sid)
                    tr.sid  = sid
                    tr.name = name
                    tr.dept = s_info.get(sid, {}).get("dept", "")
                    tr.dist = dist
                else:
                    tr.status = status
                    tr.dist   = dist
                tr.t_recog = now

                # STEP 4: Mark newly recognised faces during window
                if (phase == "ATTENDANCE"
                        and tr.status == "RECOGNISED"
                        and sid
                        and sid not in marked):
                    saved = mark_local(sid, name, args.subject,
                                       args.teacher, args.section, dist)
                    if saved:
                        marked.add(sid)
                        at = datetime.now()
                        records.append({"sid": sid, "name": name,
                                        "d": at.date(), "t": at.time(),
                                        "dist": dist})
                        print(f"  [ATTENDANCE] Marked: {sid} - {name}  d={dist:.3f}")

            # Draw
            for tid, box in active.items():
                draw_box(frame, tid, box, tr_map.get(tid, TR()))
            draw_hud(frame, phase, len(active), len(marked),
                     win_rem, fps_v, ramc, args.subject, args.section)
            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print("[INFO] Quit - session ended early.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("[INFO] Syncing to Spring Boot ...")
    synced = sync_sb(records, args.subject, args.teacher, args.section)

    print()
    print("=" * 60)
    print("  SESSION ATTENDANCE REPORT")
    print("=" * 60)
    print(f"  Subject        : {args.subject}")
    print(f"  Section        : {args.section}")
    print(f"  Teacher        : {args.teacher}")
    print(f"  Date           : {date.today()}")
    print(f"  Students marked: {len(marked)}")
    print(f"  Synced to SB   : {synced} / {len(records)}")
    print(f"  RAM final      : {_ram_mb():.1f} MB")
    print()
    if records:
        print("  PRESENT:")
        for r in records:
            print(f"    OK  {r['sid']:20s}  {r['name']:25s}  {r['t'].strftime('%H:%M:%S')}")
    else:
        print("  No students were marked present.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
