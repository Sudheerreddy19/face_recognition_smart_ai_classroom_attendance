from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from config import get_settings
from services.camera_service import CameraService
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class HardwareService:
    @staticmethod
    def get_all_devices() -> Dict[str, Any]:
        cameras = CameraService.list_available_cameras()
        active_cam_idx = CameraService.get_active_camera_index()
        cmd = ['powershell', '-NoProfile', '-Command', 'Get-PnpDevice | Select-Object Class, FriendlyName, InstanceId, Status, Present | Where-Object Status -eq \'OK\' | Where-Object Class -in \'Ports\', \'USB\', \'Bluetooth\', \'MEDIA\' | ConvertTo-Json']
        pnp_items = []
        try:
            out = subprocess.check_output(cmd, timeout=8).decode('utf-8', errors='ignore').strip()
            if out:
                parsed = json.loads(out)
                pnp_items = parsed if isinstance(parsed, list) else [parsed]
        except Exception as exc:
            logger.debug('PnP query error: %s', exc)
        microphones = []
        serial_ports = []
        bluetooth_devices = []
        usb_devices = []
        seen_mics = set()
        for item in pnp_items:
            cls = (item.get('Class') or '').upper()
            name = item.get('FriendlyName') or ''
            inst_id = item.get('InstanceId') or ''
            if not name: continue
            if cls == 'MEDIA':
                lname = name.lower()
                if ('micro' in lname or 'sound' in lname or 'phocus' in lname or 'audio' in lname) and name not in seen_mics:
                    seen_mics.add(name)
                    is_ext = 'phocus' in lname or 'usb' in lname or 'external' in lname
                    microphones.append({'name': name, 'instanceId': inst_id, 'isExternal': is_ext, 'status': 'CONNECTED', 'type': 'USB Microphone' if is_ext else 'Built-in Microphone Array'})
            elif cls == 'PORTS':
                serial_ports.append({'name': name, 'instanceId': inst_id, 'status': 'CONNECTED', 'type': 'Serial / Microcontroller'})
            elif cls == 'BLUETOOTH':
                bluetooth_devices.append({'name': name, 'instanceId': inst_id, 'status': 'CONNECTED', 'type': 'Bluetooth'})
            elif cls == 'USB':
                if 'composite' not in name.lower() and 'root hub' not in name.lower():
                    usb_devices.append({'name': name, 'instanceId': inst_id, 'status': 'CONNECTED', 'type': 'USB Device'})
        iot_devices = []
        if settings.esp32_camera_url:
            iot_devices.append({'name': 'ESP32-CAM Smart Classroom Camera', 'url': settings.esp32_camera_url, 'status': 'ONLINE' if settings.camera_source.lower() == 'esp32' else 'STANDBY', 'type': 'ESP32 Wi-Fi Node'})
        return {'cameras': cameras, 'activeCameraIndex': active_cam_idx, 'microphones': microphones, 'serialPorts': serial_ports, 'bluetoothDevices': bluetooth_devices, 'usbDevices': usb_devices, 'iotDevices': iot_devices}
