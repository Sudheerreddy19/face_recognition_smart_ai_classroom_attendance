"""
agent/ws_agent.py — Outbound WebSocket agent connecting to Spring Boot on Render.

ARCHITECTURE
------------
This module runs as a persistent background asyncio task alongside the FastAPI
server.  It:

  1. Connects OUTBOUND to wss://<render-host>/ws/agent (never exposes port 8000).
  2. Sends an AUTH message immediately after connecting.
  3. Sends periodic HEARTBEAT pings so Spring Boot knows the agent is alive.
  4. Receives CAPTURE_JOB messages from Spring Boot.
  5. Calls the existing EnrollmentService to capture + encode the face.
  6. Sends a CAPTURE_RESULT message with the outcome (success or failure).
  7. Reconnects automatically on any disconnect, with exponential-backoff cap.

MESSAGE PROTOCOL (JSON over raw WebSocket)
------------------------------------------
Agent  → Server:  { "type": "AUTH",           "agentId": "...", "token": "..." }
Server → Agent:   { "type": "AUTH_OK" }
Server → Agent:   { "type": "AUTH_FAIL",       "reason": "..." }
Agent  → Server:  { "type": "HEARTBEAT",       "agentId": "...", "ts": <epoch_ms> }
Server → Agent:   { "type": "HEARTBEAT_ACK",   "ts": <epoch_ms> }
Server → Agent:   { "type": "CAPTURE_JOB",     "jobId": "...", "studentDbId": 42,
                                                "registerNumber": "23FE1A0424",
                                                "studentName": "Sudheer Byreddy",
                                                "department": "CSE", "semester": "4",
                                                "teacherId": "7" }
Agent  → Server:  { "type": "CAPTURE_RESULT",  "jobId": "...", "success": true,
                                                "registerNumber": "23FE1A0424",
                                                "imagesCaptured": 20,
                                                "encodingPath": "encodings/23FE1A0424.pkl",
                                                "message": "Face captured successfully." }
Agent  → Server:  { "type": "CAPTURE_RESULT",  "jobId": "...", "success": false,
                                                "registerNumber": "23FE1A0424",
                                                "errorCode": "CAMERA_ERROR",
                                                "message": "Camera not available." }

DEDUPLICATION
-------------
Processed jobIds are kept in a bounded in-memory set.  Duplicate CAPTURE_JOB
messages (same jobId) are rejected immediately without touching the camera.

THREAD SAFETY
-------------
The enrollment pipeline (CameraService, EnrollmentService) is CPU-bound and
blocking.  It is executed in a thread pool via asyncio.to_thread() so the
asyncio event loop is never blocked.

DEPENDENCIES
------------
  websockets >= 12.0   (already in requirements.txt via uvicorn[standard])
  All other imports are from existing project modules.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Deque, Optional

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Deduplication window ──────────────────────────────────────────────────────
# Keep the last 200 processed job IDs to reject duplicates.
_MAX_SEEN_JOBS = 200
_seen_job_ids: Deque[str] = deque(maxlen=_MAX_SEEN_JOBS)
_seen_job_set: set[str] = set()

# ── Agent state (read by health endpoint) ─────────────────────────────────────
agent_connected: bool = False
agent_last_heartbeat: Optional[float] = None


def _mark_seen(job_id: str) -> None:
    if len(_seen_job_ids) >= _MAX_SEEN_JOBS:
        evicted = _seen_job_ids[0]  # oldest
        _seen_job_set.discard(evicted)
    _seen_job_ids.append(job_id)
    _seen_job_set.add(job_id)


def _already_seen(job_id: str) -> bool:
    return job_id in _seen_job_set


# ── Capture job handler ───────────────────────────────────────────────────────

async def _handle_capture_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the face-capture pipeline for one job.

    Runs the blocking EnrollmentService in a thread-pool executor so the
    asyncio event loop stays responsive.

    Returns a CAPTURE_RESULT dict ready to send over the WebSocket.
    """
    job_id          = job.get("jobId", "")
    student_db_id   = job.get("studentDbId")
    register_number = job.get("registerNumber", "")
    student_name    = job.get("studentName", "")
    department      = job.get("department", "")
    semester        = job.get("semester", "")
    teacher_id      = job.get("teacherId")
    camera_index    = job.get("cameraIndex")

    logger.info(
        "[WS-AGENT] CAPTURE_JOB received — jobId=%s student=%s name=%s",
        job_id, register_number, student_name,
    )

    if not register_number:
        logger.warning("[WS-AGENT] CAPTURE_JOB missing registerNumber — jobId=%s", job_id)
        return {
            "type": "CAPTURE_RESULT",
            "jobId": job_id,
            "success": False,
            "registerNumber": register_number,
            "errorCode": "INVALID_JOB",
            "message": "registerNumber is required.",
        }

    if _already_seen(job_id):
        logger.warning("[WS-AGENT] Duplicate CAPTURE_JOB ignored — jobId=%s", job_id)
        return {
            "type": "CAPTURE_RESULT",
            "jobId": job_id,
            "success": False,
            "registerNumber": register_number,
            "errorCode": "DUPLICATE_JOB",
            "message": f"Job {job_id} was already processed.",
        }

    _mark_seen(job_id)

    # Run blocking enrollment in thread pool ─────────────────────────────────
    try:
        result = await asyncio.to_thread(
            _run_enrollment_sync,
            register_number=register_number,
            student_name=student_name,
            department=department,
            semester=semester,
        camera_index=camera_index,
        )
    except Exception as exc:
        logger.exception("[WS-AGENT] Enrollment raised unexpected exception: %s", exc)
        return {
            "type": "CAPTURE_RESULT",
            "jobId": job_id,
            "success": False,
            "registerNumber": register_number,
            "errorCode": "INTERNAL_ERROR",
            "message": str(exc),
        }

    if not result["success"]:
        # Build the failure result — pass existingStudent if DUPLICATE_FACE
        failure = {
            "type": "CAPTURE_RESULT",
            "jobId": job_id,
            "success": False,
            "registerNumber": register_number,
            "errorCode": result.get("errorCode", "CAPTURE_FAILED"),
            "message": result.get("message", "Face capture failed."),
        }
        # Pass existingStudent details so the frontend can display them
        if result.get("existingStudent"):
            failure["existingStudent"] = result["existingStudent"]
        return failure

    return {
        "type":           "CAPTURE_RESULT",
        "jobId":          job_id,
        "success":        True,
        "registerNumber": register_number,
        "imagesCaptured": result["imagesCaptured"],
        "encodingPath":   result.get("encodingPath", ""),
        "message":        result.get("message", "Face captured successfully."),
    }


def _run_enrollment_sync(
    register_number: str,
    student_name: str,
    department: str,
    semester: str,
    camera_index: Optional[int] = None,
) -> dict[str, Any]:
    """
    Synchronous enrollment pipeline — runs in a thread pool.

    Reuses the existing EnrollmentService, EncodingService, and
    EncodingRepository without any modification to those modules.
    """
    from database import db_context
    from repository.encoding_repository import EncodingRepository
    from services.camera_service import CameraService
    from services.detection_service import DetectionService
    from services.encoding_service import EncodingService
    from services.enrollment_service import EnrollmentService
    from services.recognition_service import RecognitionService
    from utils.constants import DuplicateCheckOutcome

    camera_svc  = CameraService(camera_index=camera_index)
    detector    = DetectionService()
    encoder_svc = EncodingService()
    recognition_svc = RecognitionService(
        camera_service=camera_svc,
        detection_service=detector,
        encoding_service=encoder_svc,
    )
    enrollment  = EnrollmentService(
        camera_service=camera_svc,
        detection_service=detector,
        encoding_service=encoder_svc,
        show_preview=False,  # headless capture, preview is in browser modal
    )

    logger.info("[WS-AGENT] Starting enrollment pipeline for student=%s", register_number)

    try:
        enroll_result = enrollment.enroll(
            student_id=register_number,
        )
    except Exception as exc:
        logger.exception("[WS-AGENT] EnrollmentService.enroll raised: %s", exc)
        return {
            "success": False,
            "errorCode": "CAMERA_ERROR",
            "message": f"Camera/capture error: {exc}",
        }

    if not enroll_result.success:
        return {
            "success": False,
            "errorCode": enroll_result.error_code or "CAPTURE_FAILED",
            "message": enroll_result.error_message or "Enrollment failed.",
        }

    # ── Duplicate Face Verification ─────────────────────────────────────────
    # Reject if this face is already registered under a different roll number!
    try:
        dup_result = recognition_svc.check_duplicate_faces(
            new_encodings=enroll_result.encodings,
            target_student_id=register_number,
        )

        outcome = dup_result.get("outcome")

        if outcome == DuplicateCheckOutcome.DUPLICATE_FOUND:
            other_sid = dup_result.get("matched_student_id")
            min_dist = dup_result.get("min_distance", 0.0)
            matched_count = dup_result.get("matched_count", 0)

            # Fetch details of the already-registered student from SQLite
            existing_name = other_sid
            existing_dept = ""
            existing_sem = ""
            try:
                with db_context() as db:
                    student_repo = EncodingRepository()
                    existing_s = student_repo.get_student_by_id(db, other_sid)
                    if existing_s:
                        existing_name = existing_s.name or other_sid
                        existing_dept = existing_s.department or ""
                        existing_sem = existing_s.semester or ""
            except Exception as db_err:
                logger.debug("Could not query student details for %s: %s", other_sid, db_err)

            dept_str = f" | Dept: {existing_dept}" if existing_dept else ""
            sem_str  = f" | Sem: {existing_sem}" if existing_sem else ""

            logger.warning(
                "[WS-AGENT] DUPLICATE_FOUND: New roll '%s' matches existing student '%s' (%s, min_dist=%.3f, matches=%d/%d)",
                register_number, other_sid, existing_name, min_dist, matched_count, len(enroll_result.encodings),
            )

            return {
                "success": False,
                "errorCode": "DUPLICATE_FACE",
                "message": (
                    f"DUPLICATE FACE DETECTED! This face is ALREADY registered to "
                    f"{existing_name} (Roll: {other_sid}{dept_str}{sem_str}). "
                    "A face cannot be registered under multiple roll numbers."
                ),
                "existingStudent": {
                    "rollNumber": other_sid,
                    "name": existing_name,
                    "department": existing_dept,
                    "semester": existing_sem,
                },
            }

        elif outcome == DuplicateCheckOutcome.UNCERTAIN_MATCH:
            logger.info(
                "[WS-AGENT] UNCERTAIN_MATCH for student %s: %s",
                register_number,
                dup_result.get("message"),
            )
            return {
                "success": False,
                "errorCode": "UNCERTAIN_MATCH",
                "message": dup_result.get("message", "Match confidence is insufficient. Please adjust camera position, improve lighting, and try again."),
            }

        # Otherwise NO_DUPLICATE -> proceed to persist
    except Exception as dup_err:
        logger.exception("[WS-AGENT] Duplicate face verification error: %s", dup_err)

    # Persist encoding ────────────────────────────────────────────────────────
    try:
        encoding_path = encoder_svc.save_encodings(register_number, enroll_result.encodings)
    except Exception as exc:
        logger.warning("[WS-AGENT] Could not save encoding file for %s: %s", register_number, exc)
        encoding_path = None

    # Persist student metadata to local SQLite ─────────────────────────────────
    try:
        with db_context() as db:
            student_repo = EncodingRepository()
            if not student_repo.student_exists(db, register_number):
                student_repo.create_student(
                    db,
                    student_id=register_number,
                    name=student_name or register_number,
                    department=department or "",
                    semester=str(semester or ""),
                    section="",
                    teacher_id=str(teacher_id or ""),
                    encoding_path=str(encoding_path) if encoding_path else "",
                    image_count=enroll_result.images_captured,
                )
                db.commit()
            else:
                student_repo.update_encoding_info(
                    db,
                    student_id=register_number,
                    encoding_path=str(encoding_path) if encoding_path else "",
                    image_count=enroll_result.images_captured,
                )
                db.commit()
    except Exception as exc:
        logger.warning(
            "[WS-AGENT] DB write failed for student %s (non-fatal): %s",
            register_number, exc,
        )

    logger.info(
        "[WS-AGENT] Enrollment SUCCESS — student=%s images=%d encodings=%d",
        register_number,
        enroll_result.images_captured,
        len(enroll_result.encodings),
    )

    return {
        "success":        True,
        "imagesCaptured": enroll_result.images_captured,
        "encodingPath":   str(encoding_path) if encoding_path else "",
        "message":        f"Face captured successfully for {register_number}. "
                          f"{enroll_result.images_captured} images captured.",
    }


# ── WebSocket connection loop ─────────────────────────────────────────────────


# ── Attendance Session State & Loop ──────────────────────────────────────────
_active_attendance_task: Optional[asyncio.Task] = None
_attendance_running: bool = False
_attendance_paused: bool = False
_active_session_id: Optional[int] = None
_active_job_id: Optional[str] = None
_recognized_students: set[str] = set()


async def _run_attendance_loop(ws, session_id: int, job_id: str) -> None:
    """
    Continuous live attendance recognition loop.
    Captures frames from camera, detects faces, compares with enrolled pickle encodings,
    and sends RECOGNITION_EVENT or UNKNOWN_FACE_EVENT to Spring Boot.
    """
    global _attendance_running, _attendance_paused
    from services.camera_service import CameraService
    from services.recognition_service import RecognitionService
    from services.detection_service import DetectionService
    from services.encoding_service import EncodingService
    from utils.constants import RecognitionStatus
    import numpy as np

    logger.info("[ATTENDANCE-LOOP] Starting live attendance recognition for session=%s", session_id)
    camera_svc = CameraService()
    detector = DetectionService()
    encoder_svc = EncodingService()
    recognizer = RecognitionService(
        camera_service=camera_svc,
        detection_service=detector,
        encoding_service=encoder_svc,
    )
    recognizer.reload_encodings()

    if not camera_svc.open():
        logger.error("[ATTENDANCE-LOOP] Failed to open camera for attendance")
        try:
            await ws.send(json.dumps({
                "type": "CAMERA_STATUS",
                "status": "ERROR",
                "message": "Webcam could not be opened by attendance agent."
            }))
        except Exception:
            pass
        return

    try:
        # Notify backend camera is live
        await ws.send(json.dumps({
            "type": "CAMERA_STATUS",
            "status": "OPEN",
            "message": "Attendance camera scanning active."
        }))
    except Exception:
        pass

    frame_count = 0
    unknown_reported_time = 0.0

    try:
        while _attendance_running:
            if _attendance_paused:
                await asyncio.sleep(0.5)
                continue

            frame_res = await asyncio.to_thread(camera_svc.read_frame)
            frame = frame_res[1] if isinstance(frame_res, tuple) else frame_res
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            frame_count += 1
            # Run recognition every 3 frames for high responsiveness and low CPU load
            if frame_count % 3 != 0:
                await asyncio.sleep(0.01)
                continue

            from utils.image_utils import bgr_to_rgb
            rgb_frame = bgr_to_rgb(frame)

            # Recognize all visible faces in this frame
            try:
                results = await asyncio.to_thread(recognizer.recognize_multiple_from_frame, rgb_frame)
            except Exception as e:
                logger.debug("[ATTENDANCE-LOOP] Error during recognition: %s", e)
                results = []

            for dto in results:
                if dto.status == RecognitionStatus.RECOGNIZED and dto.student_id:
                    roll = str(dto.student_id)
                    if roll not in _recognized_students:
                        _recognized_students.add(roll)
                        logger.info("[ATTENDANCE-LOOP] RECOGNIZED student: %s (dist=%.3f)", roll, dto.distance or 0.0)
                        evt = {
                            "type": "RECOGNITION_EVENT",
                            "sessionId": session_id,
                            "jobId": job_id,
                            "rollNumber": roll,
                            "confidence": float(dto.distance or 0.0),
                        }
                        try:
                            await ws.send(json.dumps(evt))
                        except Exception as send_err:
                            logger.error("[ATTENDANCE-LOOP] Error sending RECOGNITION_EVENT: %s", send_err)

                elif dto.status == RecognitionStatus.UNKNOWN:
                    now = time.time()
                    if now - unknown_reported_time > 8.0:
                        unknown_reported_time = now
                        logger.info("[ATTENDANCE-LOOP] UNKNOWN face detected in attendance session %s", session_id)
                        evt = {
                            "type": "UNKNOWN_FACE_EVENT",
                            "sessionId": session_id,
                            "jobId": job_id,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        try:
                            await ws.send(json.dumps(evt))
                        except Exception:
                            pass

            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        logger.info("[ATTENDANCE-LOOP] Attendance task cancelled")
    except Exception as exc:
        logger.exception("[ATTENDANCE-LOOP] Unexpected error in attendance loop: %s", exc)
    finally:
        camera_svc.release()
        _attendance_running = False
        logger.info("[ATTENDANCE-LOOP] Camera released, attendance loop finished")
        try:
            await ws.send(json.dumps({
                "type": "ATTENDANCE_COMPLETE",
                "sessionId": session_id,
                "jobId": job_id,
                "recognizedCount": len(_recognized_students),
            }))
        except Exception:
            pass


async def _run_connection(ws_url: str, agent_id: str, token: str) -> None:
    """
    Manage one WebSocket connection lifecycle:
      1. Connect
      2. Authenticate
      3. Run heartbeat + message loop until disconnect
    """
    global agent_connected, agent_last_heartbeat

    try:
        import websockets
    except ImportError:
        logger.error(
            "[WS-AGENT] 'websockets' package not installed. "
            "Run: pip install websockets>=12.0"
        )
        return

    logger.info("[WS-AGENT] Connecting → %s", ws_url)

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=15,
            ping_timeout=10,
            open_timeout=25,
            close_timeout=10,
            additional_headers={"X-Agent-Id": agent_id},
        ) as ws:

            # ── AUTH ──────────────────────────────────────────────────────────
            await ws.send(json.dumps({
                "type":    "AUTH",
                "agentId": agent_id,
                "token":   token,
            }))

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                logger.error("[WS-AGENT] AUTH response timed out — disconnecting")
                return

            if msg.get("type") == "AUTH_FAIL":
                logger.error(
                    "[WS-AGENT] Authentication REJECTED by server: %s",
                    msg.get("reason", "unknown reason"),
                )
                # Don't reconnect for auth failures — wrong token won't self-heal.
                # Sleep a long time then retry (token may be corrected in .env + restart).
                await asyncio.sleep(60)
                return

            if msg.get("type") != "AUTH_OK":
                logger.error("[WS-AGENT] Unexpected AUTH response: %s", msg)
                return

            agent_connected = True
            logger.info("[WS-AGENT] ✅ Authenticated — agent_id=%s", agent_id)

            # ── Heartbeat task ────────────────────────────────────────────────
            async def heartbeat_loop() -> None:
                global agent_last_heartbeat
                while True:
                    await asyncio.sleep(settings.agent_heartbeat_interval)
                    try:
                        ts = int(time.time() * 1000)
                        await ws.send(json.dumps({
                            "type":    "HEARTBEAT",
                            "agentId": agent_id,
                            "ts":      ts,
                        }))
                        agent_last_heartbeat = time.time()
                        logger.debug("[WS-AGENT] HEARTBEAT sent ts=%d", ts)
                    except Exception:
                        break  # connection broken, exit heartbeat loop

            heartbeat_task = asyncio.create_task(heartbeat_loop())

            # ── Message receive loop ──────────────────────────────────────────
            try:
                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("[WS-AGENT] Non-JSON message received: %r", raw_msg)
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "HEARTBEAT_ACK":
                        logger.debug("[WS-AGENT] HEARTBEAT_ACK received")

                    elif msg_type == "CAPTURE_JOB":
                        # Schedule job handling — don't await inline to keep
                        # the message loop free for more messages.
                        async def _dispatch(job: dict = msg) -> None:
                            result = await _handle_capture_job(job)
                            try:
                                await ws.send(json.dumps(result))
                                logger.info(
                                    "[WS-AGENT] CAPTURE_RESULT sent — jobId=%s success=%s",
                                    result.get("jobId"), result.get("success"),
                                )
                            except Exception as send_err:
                                logger.error(
                                    "[WS-AGENT] Failed to send CAPTURE_RESULT: %s", send_err
                                )

                        asyncio.create_task(_dispatch())

                    elif msg_type == "ATTENDANCE_START":
                        global _active_attendance_task, _attendance_running, _attendance_paused
                        global _active_session_id, _active_job_id, _recognized_students

                        session_id = msg.get("sessionId")
                        job_id = msg.get("jobId", "")
                        logger.info("[WS-AGENT] Received ATTENDANCE_START for session=%s jobId=%s", session_id, job_id)

                        # Stop any previous task if still running
                        if _active_attendance_task and not _active_attendance_task.done():
                            _attendance_running = False
                            _active_attendance_task.cancel()
                            try:
                                await _active_attendance_task
                            except (asyncio.CancelledError, Exception):
                                pass

                        _attendance_running = True
                        _attendance_paused = False
                        _active_session_id = session_id
                        _active_job_id = job_id
                        _recognized_students = set()

                        _active_attendance_task = asyncio.create_task(
                            _run_attendance_loop(ws, session_id, job_id)
                        )

                    elif msg_type in ("ATTENDANCE_PAUSE", "PAUSE"):
                        _attendance_paused = True
                        logger.info("[WS-AGENT] Attendance recognition PAUSED")

                    elif msg_type in ("ATTENDANCE_RESUME", "RESUME"):
                        _attendance_paused = False
                        logger.info("[WS-AGENT] Attendance recognition RESUMED")

                    elif msg_type in ("ATTENDANCE_STOP", "STOP"):
                        logger.info("[WS-AGENT] Received ATTENDANCE_STOP")
                        _attendance_running = False
                        if _active_attendance_task and not _active_attendance_task.done():
                            _active_attendance_task.cancel()

                    elif msg_type == "PING":
                        await ws.send(json.dumps({"type": "PONG", "agentId": agent_id}))

                    else:
                        logger.debug("[WS-AGENT] Unhandled message type: %s", msg_type)

            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        logger.warning("[WS-AGENT] Connection error: %s — will reconnect", exc)
    finally:
        agent_connected = False


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agent() -> None:
    """
    Main agent coroutine — runs for the lifetime of the FastAPI application.

    If CLASSAI_AGENT_WS_URL or CLASSAI_AGENT_TOKEN is not set, the agent
    exits immediately (the face capture flow will fall back to a clear error).

    Reconnect strategy:
      - Attempt 1–5: wait `agent_reconnect_interval` seconds (default 5s)
      - Attempt 6+:  cap at 60 seconds to avoid hammering the server
    """
    ws_url   = settings.classai_agent_ws_url.strip()
    agent_id = settings.classai_agent_id.strip() or "laptop-agent-01"
    token    = settings.classai_agent_token.strip()

    if not ws_url:
        logger.warning(
            "[WS-AGENT] CLASSAI_AGENT_WS_URL not set — WebSocket agent disabled. "
            "Face capture via WebSocket will not work until this is configured."
        )
        return

    if not token:
        logger.warning(
            "[WS-AGENT] CLASSAI_AGENT_TOKEN not set — WebSocket agent disabled. "
            "Set CLASSAI_AGENT_TOKEN in .env to enable the agent."
        )
        return

    logger.info(
        "[WS-AGENT] Starting — url=%s agent_id=%s heartbeat=%ds reconnect=%ds",
        ws_url, agent_id,
        settings.agent_heartbeat_interval,
        settings.agent_reconnect_interval,
    )

    attempt = 0
    while True:
        attempt += 1
        await _run_connection(ws_url, agent_id, token)

        wait = min(
            settings.agent_reconnect_interval * attempt,
            60,
        )
        logger.info(
            "[WS-AGENT] Disconnected (attempt=%d). Reconnecting in %ds …",
            attempt, wait,
        )
        await asyncio.sleep(wait)
