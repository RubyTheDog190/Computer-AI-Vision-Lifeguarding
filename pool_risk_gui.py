"""基于 YOLOv5 的泳池溺水风险预警桌面端（PyQt5）。"""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSlider, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QSplitter,
)

FILE = Path(__file__).resolve()
ROOT = FILE.parent
STARTUP_LOG = ROOT / "pool_risk_gui_startup.log"
USER_DB = ROOT / "pool_users.db"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync


def write_startup_log(status, error=None, append=False):
    """Write and close the log immediately so another launch is never blocked."""
    mode = "a" if append else "w"
    with open(STARTUP_LOG, mode, encoding="utf-8") as log:
        log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {status}\n")
        log.write(f"Python: {sys.executable}\nWorking directory: {Path.cwd()}\n")
        if error is not None:
            log.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))


def show_native_error(message):
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(message), "Pool Risk Alert - Startup Error", 0x10)
    except Exception:
        pass


COLORS = {
    "bg": "#07111f", "panel": "#0d1b2d", "panel_alt": "#102238",
    "line": "#1d3854", "text": "#e7f1fb", "muted": "#8aa0b7",
    "cyan": "#22d3ee", "green": "#2dd4bf", "yellow": "#fbbf24",
    "red": "#fb4d5b", "blue": "#3b82f6",
}
CLASS_COLORS = {"swimming": (52, 211, 153), "tread_water": (0, 196, 255), "drowning": (75, 77, 251)}
CLASS_CN = {"swimming": "正常游泳", "tread_water": "踩水/停留", "drowning": "疑似溺水"}
CLASS_EN = {"swimming": "Swimming", "tread_water": "Treading water", "drowning": "Suspected drowning"}
CLASS_DRAW = {"swimming": "SWIMMING", "tread_water": "TREAD WATER", "drowning": "DROWNING"}

I18N = {
    "zh": {
        "app_title": "蓝盾 · 泳池溺水风险预警系统",
        "app_subtitle": "YOLOv5 智能行为识别与连续帧风险确认",
        "model_loading": "● 正在加载模型", "model_ready": "● 模型就绪 · {device}",
        "model_failed": "● 模型加载失败", "system_ready": "系统就绪",
        "load_model": "加载模型", "choose_image": "选择图片", "choose_video": "选择视频",
        "open_camera": "打开摄像头", "start_resume": "开始 / 继续", "pause": "暂停",
        "stop": "停止", "save_result": "保存结果", "confidence": "置信度",
        "language_button": "English", "waiting_source": "等待选择数据源",
        "source_image": "图片 · {name}", "source_video": "视频 · {name}", "source_camera": "摄像头 · {name}",
        "original": "原始画面", "original_hint": "数据源实时画面", "waiting_frame": "等待画面",
        "analysis_result": "AI 分析结果", "analysis_hint": "检测框与行为类别",
        "risk_situation": "实时风险态势", "system_initializing": "系统初始化中",
        "risk_caption": "/ 100\n综合风险值", "inference_empty": "推理 -- ms", "inference": "推理 {ms:.1f} ms",
        "recent_frames": "最近 90 帧", "counts": "游泳 {swim}  ·  踩水 {tread}  ·  疑似溺水 {drown}",
        "confirm_hint": "红色告警需连续 5 帧检测到疑似溺水（图片立即确认）",
        "analysis_records": "风险分析与事件记录", "current_targets": "当前检测目标",
        "guard_advice": "值守建议", "alert_events": "告警事件",
        "detection_headers": ["序号", "行为", "置信度", "位置"],
        "event_headers": ["时间", "级别", "风险", "事件内容"],
        "choose_prompt": "请选择图片、视频或摄像头开始分析。",
        "risk_danger": "高风险 · 疑似溺水", "risk_checking": "正在复核异常行为",
        "risk_watch": "需要关注", "risk_safe": "泳池状态正常", "paused": "分析已暂停",
        "video_complete": "视频分析完成",
        "advice_danger": "检测到连续多帧疑似溺水行为！\n\n1. 立即查看对应监控区域；\n2. 通知现场救生员快速确认；\n3. 必要时启动救援和急救流程；\n4. 系统告警仅供辅助，必须由人员复核。",
        "advice_checking": "检测到疑似溺水，正在连续帧复核。\n\n当前确认进度：{streak} / 5\n请值守人员立即关注画面。",
        "advice_watch": "检测到踩水、原地停留或异常活动。\n\n建议持续观察是否出现挣扎、下沉或长时间停留。",
        "advice_safe": "当前未发现连续溺水风险。\n\n系统正在监测正常游泳、踩水和疑似溺水三类行为。请保持人工值守。",
        "event_system": "系统", "event_loaded": "已加载 {name}", "event_complete": "完成",
        "event_finished": "{name} 分析结束", "event_danger": "高风险",
        "event_drowning": "疑似溺水，置信度 {confidence:.1%}", "event_save": "保存",
        "event_language": "语言已切换为中文", "model_not_ready": "模型未就绪",
        "wait_model": "请等待模型加载完成。", "select_source": "请选择数据源",
        "select_source_body": "请先选择视频、图片或打开摄像头。", "open_failed": "打开失败",
        "cannot_open": "无法打开数据源：{name}", "read_failed": "读取失败",
        "cannot_read_image": "无法读取该图片。", "analysis_failed": "分析失败",
        "no_result": "暂无结果", "no_result_body": "请先完成一次图片或视频分析。",
        "model_dialog": "选择 YOLOv5 权重", "image_dialog": "选择泳池图片",
        "video_dialog": "选择泳池视频", "save_dialog": "保存分析结果",
        "user_management": "用户管理", "switch_user": "切换用户",
        "current_user": "当前用户：{name}（{role}）", "admin": "管理员", "operator": "操作员",
        "switched_user": "已切换为用户 {name}",
    },
    "en": {
        "app_title": "BlueGuard · Pool Drowning Risk Alert",
        "app_subtitle": "YOLOv5 behavior detection with multi-frame risk confirmation",
        "model_loading": "● Loading model", "model_ready": "● Model ready · {device}",
        "model_failed": "● Model load failed", "system_ready": "System ready",
        "load_model": "Load model", "choose_image": "Open image", "choose_video": "Open video",
        "open_camera": "Open camera", "start_resume": "Start / Resume", "pause": "Pause",
        "stop": "Stop", "save_result": "Save result", "confidence": "Confidence",
        "language_button": "中文", "waiting_source": "Waiting for a source",
        "source_image": "Image · {name}", "source_video": "Video · {name}", "source_camera": "Camera · {name}",
        "original": "Original Feed", "original_hint": "Live source feed", "waiting_frame": "Waiting for video",
        "analysis_result": "AI Analysis", "analysis_hint": "Detection boxes and behavior classes",
        "risk_situation": "Live Risk Status", "system_initializing": "Initializing system",
        "risk_caption": "/ 100\nOverall risk", "inference_empty": "Inference -- ms", "inference": "Inference {ms:.1f} ms",
        "recent_frames": "Last 90 frames", "counts": "Swimming {swim}  ·  Treading {tread}  ·  Drowning {drown}",
        "confirm_hint": "Red alert after 5 consecutive drowning frames (immediate for images)",
        "analysis_records": "Risk Analysis and Event Log", "current_targets": "Current Detections",
        "guard_advice": "Operator Guidance", "alert_events": "Alert Events",
        "detection_headers": ["No.", "Behavior", "Confidence", "Position"],
        "event_headers": ["Time", "Level", "Risk", "Event"],
        "choose_prompt": "Select an image, video, or camera to begin analysis.",
        "risk_danger": "HIGH RISK · DROWNING", "risk_checking": "Reviewing abnormal behavior",
        "risk_watch": "Attention required", "risk_safe": "Pool status normal", "paused": "Analysis paused",
        "video_complete": "Video analysis complete",
        "advice_danger": "Drowning behavior detected across consecutive frames!\n\n1. Check the corresponding camera area immediately.\n2. Notify the on-site lifeguard for rapid confirmation.\n3. Initiate rescue and first-aid procedures when necessary.\n4. This alert is advisory and must be verified by staff.",
        "advice_checking": "Suspected drowning detected; confirming across frames.\n\nConfirmation progress: {streak} / 5\nOperators should inspect the feed immediately.",
        "advice_watch": "Treading water, stationary behavior, or unusual activity detected.\n\nContinue observing for struggling, sinking, or prolonged inactivity.",
        "advice_safe": "No sustained drowning risk detected.\n\nThe system is monitoring swimming, treading water, and suspected drowning. Maintain human supervision.",
        "event_system": "System", "event_loaded": "Loaded {name}", "event_complete": "Complete",
        "event_finished": "Analysis of {name} completed", "event_danger": "High risk",
        "event_drowning": "Suspected drowning, confidence {confidence:.1%}", "event_save": "Saved",
        "event_language": "Language switched to English", "model_not_ready": "Model not ready",
        "wait_model": "Please wait for the model to finish loading.", "select_source": "Select a source",
        "select_source_body": "Open a video, image, or camera first.", "open_failed": "Open failed",
        "cannot_open": "Unable to open source: {name}", "read_failed": "Read failed",
        "cannot_read_image": "Unable to read this image.", "analysis_failed": "Analysis failed",
        "no_result": "No result", "no_result_body": "Run an image or video analysis first.",
        "model_dialog": "Select YOLOv5 weights", "image_dialog": "Select pool image",
        "video_dialog": "Select pool video", "save_dialog": "Save analysis result",
        "user_management": "Users", "switch_user": "Switch user",
        "current_user": "Signed in: {name} ({role})", "admin": "Administrator", "operator": "Operator",
        "switched_user": "Switched to user {name}",
    },
}


AUTH_I18N = {
    "zh": {
        "login_title": "蓝盾 · 用户登录", "login_subtitle": "泳池溺水风险预警系统",
        "username": "用户名", "password": "密码", "login": "登录", "cancel": "退出",
        "language_button": "English", "default_hint": "首次登录：admin / Admin@123",
        "empty_credentials": "请输入用户名和密码。", "login_failed": "用户名、密码错误或账号已停用。",
        "login_error": "登录失败", "user_management": "用户管理", "switch_user": "切换用户",
        "manager_title": "用户管理", "add_user": "新增用户", "edit_user": "编辑用户",
        "reset_password": "重置密码", "enable_user": "启用", "disable_user": "停用",
        "delete_user": "删除用户", "close": "关闭", "refresh": "刷新",
        "id": "编号", "role": "角色", "status": "状态", "created_at": "创建时间",
        "last_login": "最后登录", "admin": "管理员", "operator": "操作员",
        "enabled": "已启用", "disabled": "已停用", "never": "从未登录",
        "select_user": "请先选择一个用户。", "confirm_delete": "确定删除用户“{name}”吗？",
        "delete_title": "确认删除", "cannot_self": "不能删除或停用当前登录账号。",
        "last_admin": "系统必须至少保留一个已启用的管理员账号。",
        "operation_failed": "操作失败", "username_exists": "该用户名已存在。",
        "user_form_add": "新增用户", "user_form_edit": "编辑用户", "confirm_password": "确认密码",
        "password_optional": "新密码（留空则不修改）", "save": "保存",
        "required_fields": "用户名和密码不能为空。", "password_mismatch": "两次输入的密码不一致。",
        "password_short": "密码至少需要6个字符。", "invalid_username": "用户名只能包含字母、数字、下划线、短横线和中文。",
        "new_password": "新密码", "reset_title": "重置用户密码", "reset_success": "密码已成功重置。",
        "current_user": "当前用户：{name}（{role}）", "switched_user": "已切换为用户 {name}",
    },
    "en": {
        "login_title": "BlueGuard · User Login", "login_subtitle": "Pool Drowning Risk Alert System",
        "username": "Username", "password": "Password", "login": "Sign in", "cancel": "Exit",
        "language_button": "中文", "default_hint": "First login: admin / Admin@123",
        "empty_credentials": "Enter a username and password.", "login_failed": "Incorrect credentials or the account is disabled.",
        "login_error": "Sign-in failed", "user_management": "Users", "switch_user": "Switch user",
        "manager_title": "User Management", "add_user": "Add user", "edit_user": "Edit user",
        "reset_password": "Reset password", "enable_user": "Enable", "disable_user": "Disable",
        "delete_user": "Delete user", "close": "Close", "refresh": "Refresh",
        "id": "ID", "role": "Role", "status": "Status", "created_at": "Created",
        "last_login": "Last login", "admin": "Administrator", "operator": "Operator",
        "enabled": "Enabled", "disabled": "Disabled", "never": "Never",
        "select_user": "Select a user first.", "confirm_delete": "Delete user “{name}”?",
        "delete_title": "Confirm deletion", "cannot_self": "The signed-in account cannot be deleted or disabled.",
        "last_admin": "At least one enabled administrator account must remain.",
        "operation_failed": "Operation failed", "username_exists": "That username already exists.",
        "user_form_add": "Add User", "user_form_edit": "Edit User", "confirm_password": "Confirm password",
        "password_optional": "New password (leave blank to keep current)", "save": "Save",
        "required_fields": "Username and password are required.", "password_mismatch": "The passwords do not match.",
        "password_short": "The password must contain at least 6 characters.", "invalid_username": "Use letters, numbers, underscores, hyphens, or Chinese characters only.",
        "new_password": "New password", "reset_title": "Reset User Password", "reset_success": "Password reset successfully.",
        "current_user": "Signed in: {name} ({role})", "switched_user": "Switched to user {name}",
    },
}


class UserDatabase:
    """Small local account store using salted PBKDF2 password hashes."""

    ITERATIONS = 260000

    def __init__(self, path=USER_DB):
        self.path = Path(path)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def _hash_password(cls, password, salt=None):
        salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), cls.ITERATIONS)
        return salt, digest.hex()

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                       password_hash TEXT NOT NULL,
                       salt TEXT NOT NULL,
                       role TEXT NOT NULL CHECK(role IN ('admin', 'operator')),
                       enabled INTEGER NOT NULL DEFAULT 1,
                       created_at TEXT NOT NULL,
                       updated_at TEXT NOT NULL,
                       last_login TEXT
                   )"""
            )
            count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                salt, password_hash = self._hash_password("Admin@123")
                connection.execute(
                    "INSERT INTO users(username,password_hash,salt,role,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    ("admin", password_hash, salt, "admin", 1, now, now),
                )

    def authenticate(self, username, password):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE AND enabled=1", (username.strip(),)
            ).fetchone()
            if row is None:
                return None
            _, candidate = self._hash_password(password, row["salt"])
            if not hmac.compare_digest(candidate, row["password_hash"]):
                return None
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            connection.execute("UPDATE users SET last_login=? WHERE id=?", (now, row["id"]))
            result = dict(row)
            result["last_login"] = now
            return result

    def list_users(self):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id,username,role,enabled,created_at,last_login FROM users ORDER BY id"
            ).fetchall()]

    def get_user(self, user_id):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def create_user(self, username, password, role="operator", enabled=True):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        salt, password_hash = self._hash_password(password)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users(username,password_hash,salt,role,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (username.strip(), password_hash, salt, role, int(enabled), now, now),
            )

    def update_user(self, user_id, username, role, enabled, password=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as connection:
            if password:
                salt, password_hash = self._hash_password(password)
                connection.execute(
                    "UPDATE users SET username=?,role=?,enabled=?,password_hash=?,salt=?,updated_at=? WHERE id=?",
                    (username.strip(), role, int(enabled), password_hash, salt, now, user_id),
                )
            else:
                connection.execute(
                    "UPDATE users SET username=?,role=?,enabled=?,updated_at=? WHERE id=?",
                    (username.strip(), role, int(enabled), now, user_id),
                )

    def reset_password(self, user_id, password):
        salt, password_hash = self._hash_password(password)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as connection:
            connection.execute(
                "UPDATE users SET password_hash=?,salt=?,updated_at=? WHERE id=?",
                (password_hash, salt, now, user_id),
            )

    def set_enabled(self, user_id, enabled):
        with self._connect() as connection:
            connection.execute(
                "UPDATE users SET enabled=?,updated_at=? WHERE id=?",
                (int(enabled), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id),
            )

    def delete_user(self, user_id):
        with self._connect() as connection:
            connection.execute("DELETE FROM users WHERE id=?", (user_id,))

    def enabled_admin_count(self):
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND enabled=1"
            ).fetchone()[0]


def auth_text(language, key, **kwargs):
    text = AUTH_I18N[language][key]
    return text.format(**kwargs) if kwargs else text


def valid_username(username):
    value = username.strip()
    return bool(value) and all(ch.isalnum() or ch in "_-" or "\u4e00" <= ch <= "\u9fff" for ch in value)


class LoginDialog(QDialog):
    def __init__(self, database, language="zh", parent=None):
        super().__init__(parent)
        self.database, self.language, self.user = database, language, None
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(
            f"QDialog{{background:{COLORS['bg']};color:{COLORS['text']};}}"
            f"QLineEdit{{background:{COLORS['panel_alt']};color:{COLORS['text']};border:1px solid {COLORS['line']};padding:9px;border-radius:5px;}}"
            f"QPushButton{{background:#17314d;color:{COLORS['text']};border:0;padding:9px 18px;border-radius:5px;}}"
            f"QPushButton#accent{{background:#087ea4;}} QLabel{{color:{COLORS['text']};}}"
        )
        layout = QVBoxLayout(self); layout.setContentsMargins(36, 30, 36, 30); layout.setSpacing(14)
        self.title_label = QLabel(); self.title_label.setStyleSheet("font-size:26px;font-weight:700;")
        self.subtitle_label = QLabel(); self.subtitle_label.setStyleSheet(f"color:{COLORS['muted']};font-size:14px;")
        layout.addWidget(self.title_label); layout.addWidget(self.subtitle_label)
        form = QFormLayout(); form.setSpacing(12)
        self.username_label, self.password_label = QLabel(), QLabel()
        self.username_edit = QLineEdit(); self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        form.addRow(self.username_label, self.username_edit); form.addRow(self.password_label, self.password_edit)
        layout.addLayout(form)
        self.hint_label = QLabel(); self.hint_label.setStyleSheet(f"color:{COLORS['muted']};")
        layout.addWidget(self.hint_label)
        buttons = QHBoxLayout()
        self.language_button = QPushButton(); self.cancel_button = QPushButton(); self.login_button = QPushButton()
        self.login_button.setObjectName("accent")
        buttons.addWidget(self.language_button); buttons.addStretch(); buttons.addWidget(self.cancel_button); buttons.addWidget(self.login_button)
        layout.addLayout(buttons)
        self.language_button.clicked.connect(self.toggle_language)
        self.cancel_button.clicked.connect(self.reject)
        self.login_button.clicked.connect(self.attempt_login)
        self.password_edit.returnPressed.connect(self.attempt_login)
        self._apply_language(); self.username_edit.setFocus()

    def tr(self, key, **kwargs):
        return auth_text(self.language, key, **kwargs)

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"; self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(self.tr("login_title")); self.title_label.setText(self.tr("login_title"))
        self.subtitle_label.setText(self.tr("login_subtitle")); self.username_label.setText(self.tr("username"))
        self.password_label.setText(self.tr("password")); self.login_button.setText(self.tr("login"))
        self.cancel_button.setText(self.tr("cancel")); self.language_button.setText(self.tr("language_button"))
        self.hint_label.setText(self.tr("default_hint"))

    def attempt_login(self):
        username, password = self.username_edit.text().strip(), self.password_edit.text()
        if not username or not password:
            QMessageBox.warning(self, self.tr("login_error"), self.tr("empty_credentials")); return
        try:
            self.user = self.database.authenticate(username, password)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("login_error"), str(exc)); return
        if self.user is None:
            self.password_edit.clear(); self.password_edit.setFocus()
            QMessageBox.warning(self, self.tr("login_error"), self.tr("login_failed")); return
        self.accept()


class UserEditorDialog(QDialog):
    def __init__(self, language="zh", user=None, protect_current=False, parent=None):
        super().__init__(parent)
        self.language, self.user, self.result_data = language, user, None
        self.setModal(True); self.setMinimumWidth(470)
        layout = QVBoxLayout(self); self.form = QFormLayout(); layout.addLayout(self.form)
        self.username_edit = QLineEdit(user["username"] if user else "")
        self.password_edit, self.confirm_edit = QLineEdit(), QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password); self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.role_combo = QComboBox(); self.role_combo.addItem("", "admin"); self.role_combo.addItem("", "operator")
        if user: self.role_combo.setCurrentIndex(0 if user["role"] == "admin" else 1)
        self.enabled_check = QCheckBox(); self.enabled_check.setChecked(bool(user["enabled"]) if user else True)
        if protect_current: self.role_combo.setEnabled(False); self.enabled_check.setEnabled(False)
        self.labels = [QLabel() for _ in range(5)]
        for label, widget in zip(self.labels, (self.username_edit, self.password_edit, self.confirm_edit, self.role_combo, self.enabled_check)):
            self.form.addRow(label, widget)
        buttons = QHBoxLayout(); self.language_button = QPushButton(); buttons.addWidget(self.language_button); buttons.addStretch()
        self.cancel_button, self.save_button = QPushButton(), QPushButton()
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.save_button); layout.addLayout(buttons)
        self.language_button.clicked.connect(self.toggle_language)
        self.cancel_button.clicked.connect(self.reject); self.save_button.clicked.connect(self.validate_and_accept)
        self._apply_language()

    def tr(self, key, **kwargs):
        return auth_text(self.language, key, **kwargs)

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"; self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(self.tr("user_form_edit" if self.user else "user_form_add"))
        keys = ("username", "password_optional" if self.user else "password", "confirm_password", "role", "status")
        for label, key in zip(self.labels, keys): label.setText(self.tr(key))
        self.role_combo.setItemText(0, self.tr("admin")); self.role_combo.setItemText(1, self.tr("operator"))
        self.enabled_check.setText(self.tr("enabled")); self.language_button.setText(self.tr("language_button"))
        self.cancel_button.setText(self.tr("cancel")); self.save_button.setText(self.tr("save"))

    def validate_and_accept(self):
        username, password, confirm = self.username_edit.text().strip(), self.password_edit.text(), self.confirm_edit.text()
        if not valid_username(username):
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("invalid_username")); return
        if not self.user and not password:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("required_fields")); return
        if password and len(password) < 6:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("password_short")); return
        if password != confirm:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("password_mismatch")); return
        self.result_data = {"username": username, "password": password or None,
                            "role": self.role_combo.currentData(), "enabled": self.enabled_check.isChecked()}
        self.accept()


class PasswordDialog(QDialog):
    def __init__(self, language="zh", parent=None):
        super().__init__(parent); self.language, self.password = language, None; self.setModal(True); self.setMinimumWidth(420)
        layout = QVBoxLayout(self); form = QFormLayout(); layout.addLayout(form)
        self.password_edit, self.confirm_edit = QLineEdit(), QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password); self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.password_label, self.confirm_label = QLabel(), QLabel()
        form.addRow(self.password_label, self.password_edit); form.addRow(self.confirm_label, self.confirm_edit)
        buttons = QHBoxLayout(); self.language_button = QPushButton(); buttons.addWidget(self.language_button); buttons.addStretch(); self.cancel_button, self.save_button = QPushButton(), QPushButton()
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.save_button); layout.addLayout(buttons)
        self.language_button.clicked.connect(self.toggle_language)
        self.cancel_button.clicked.connect(self.reject); self.save_button.clicked.connect(self.validate_and_accept); self._apply_language()

    def tr(self, key): return auth_text(self.language, key)

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"; self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(self.tr("reset_title")); self.password_label.setText(self.tr("new_password"))
        self.confirm_label.setText(self.tr("confirm_password")); self.language_button.setText(self.tr("language_button"))
        self.cancel_button.setText(self.tr("cancel")); self.save_button.setText(self.tr("save"))

    def validate_and_accept(self):
        password = self.password_edit.text()
        if len(password) < 6:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("password_short")); return
        if password != self.confirm_edit.text():
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("password_mismatch")); return
        self.password = password; self.accept()


class UserManagementDialog(QDialog):
    def __init__(self, database, current_user, language="zh", parent=None):
        super().__init__(parent)
        self.database, self.current_user, self.language = database, current_user, language
        self.setModal(True); self.resize(980, 560)
        layout = QVBoxLayout(self)
        top = QHBoxLayout(); self.title_label = QLabel(); self.title_label.setStyleSheet("font-size:22px;font-weight:700;")
        self.language_button = QPushButton(); top.addWidget(self.title_label); top.addStretch(); top.addWidget(self.language_button); layout.addLayout(top)
        self.table = QTableWidget(0, 6); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers); layout.addWidget(self.table)
        actions = QHBoxLayout(); self.buttons = {}
        for key, slot in (("add_user", self.add_user), ("edit_user", self.edit_user),
                          ("reset_password", self.reset_password), ("enable_user", self.enable_user),
                          ("disable_user", self.disable_user), ("delete_user", self.delete_user)):
            button = QPushButton(); button.clicked.connect(slot); self.buttons[key] = button; actions.addWidget(button)
        actions.addStretch(); self.close_button = QPushButton(); self.close_button.clicked.connect(self.accept); actions.addWidget(self.close_button)
        layout.addLayout(actions); self.language_button.clicked.connect(self.toggle_language)
        self._apply_language(); self.refresh()

    def tr(self, key, **kwargs): return auth_text(self.language, key, **kwargs)

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"; self._apply_language(); self.refresh()

    def _apply_language(self):
        self.setWindowTitle(self.tr("manager_title")); self.title_label.setText(self.tr("manager_title"))
        self.language_button.setText(self.tr("language_button")); self.close_button.setText(self.tr("close"))
        for key, button in self.buttons.items(): button.setText(self.tr(key))
        for index, key in enumerate(("id", "username", "role", "status", "created_at", "last_login")):
            self.table.setHorizontalHeaderItem(index, QTableWidgetItem(self.tr(key)))

    def refresh(self):
        users = self.database.list_users(); self.table.setRowCount(len(users))
        for row, user in enumerate(users):
            values = (user["id"], user["username"], self.tr(user["role"]),
                      self.tr("enabled" if user["enabled"] else "disabled"), user["created_at"], user["last_login"] or self.tr("never"))
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setTextAlignment(Qt.AlignCenter); self.table.setItem(row, column, item)

    def selected_user(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, self.tr("manager_title"), self.tr("select_user")); return None
        return self.database.get_user(int(self.table.item(row, 0).text()))

    def _show_database_error(self, exc):
        message = self.tr("username_exists") if isinstance(exc, sqlite3.IntegrityError) else str(exc)
        QMessageBox.critical(self, self.tr("operation_failed"), message)

    def add_user(self):
        dialog = UserEditorDialog(self.language, parent=self)
        if dialog.exec_() != QDialog.Accepted: return
        try:
            self.database.create_user(**dialog.result_data); self.refresh()
        except Exception as exc: self._show_database_error(exc)

    def edit_user(self):
        user = self.selected_user()
        if not user: return
        dialog = UserEditorDialog(self.language, user, user["id"] == self.current_user["id"], self)
        if dialog.exec_() != QDialog.Accepted: return
        data = dialog.result_data
        if user["role"] == "admin" and user["enabled"] and (data["role"] != "admin" or not data["enabled"]) and self.database.enabled_admin_count() <= 1:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("last_admin")); return
        try:
            self.database.update_user(user["id"], **data); self.refresh()
            if user["id"] == self.current_user["id"]: self.current_user["username"] = data["username"]
        except Exception as exc: self._show_database_error(exc)

    def reset_password(self):
        user = self.selected_user()
        if not user: return
        dialog = PasswordDialog(self.language, self)
        if dialog.exec_() != QDialog.Accepted: return
        try:
            self.database.reset_password(user["id"], dialog.password)
            QMessageBox.information(self, self.tr("manager_title"), self.tr("reset_success"))
        except Exception as exc: self._show_database_error(exc)

    def enable_user(self): self._set_selected_enabled(True)
    def disable_user(self): self._set_selected_enabled(False)

    def _set_selected_enabled(self, enabled):
        user = self.selected_user()
        if not user: return
        if not enabled and user["id"] == self.current_user["id"]:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("cannot_self")); return
        if not enabled and user["role"] == "admin" and user["enabled"] and self.database.enabled_admin_count() <= 1:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("last_admin")); return
        try:
            self.database.set_enabled(user["id"], enabled); self.refresh()
        except Exception as exc: self._show_database_error(exc)

    def delete_user(self):
        user = self.selected_user()
        if not user: return
        if user["id"] == self.current_user["id"]:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("cannot_self")); return
        if user["role"] == "admin" and user["enabled"] and self.database.enabled_admin_count() <= 1:
            QMessageBox.warning(self, self.tr("operation_failed"), self.tr("last_admin")); return
        if QMessageBox.question(self, self.tr("delete_title"), self.tr("confirm_delete", name=user["username"])) != QMessageBox.Yes: return
        try:
            self.database.delete_user(user["id"]); self.refresh()
        except Exception as exc: self._show_database_error(exc)


def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box: tuple


@dataclass
class AnalysisResult:
    frame: np.ndarray
    detections: list
    inference_ms: float


class YoloV5Detector:
    def __init__(self, weights, device="", imgsz=640, conf=0.25, iou=0.45):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"未找到模型权重：{self.weights}")
        self.device = select_device(device)
        self.model = attempt_load(str(self.weights), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.half = self.device.type != "cpu"
        self.model.half() if self.half else self.model.float()
        self.model.eval()
        names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.names = names if isinstance(names, (list, tuple)) else [names[i] for i in range(len(names))]
        self.conf, self.iou = conf, iou
        if self.device.type != "cpu":
            warmup = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
            self.model(warmup.half() if self.half else warmup.float())

    @torch.no_grad()
    def infer(self, frame):
        image = letterbox(frame, self.imgsz, stride=self.stride)[0]
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        started = time_sync()
        prediction = self.model(tensor, augment=False)[0]
        prediction = non_max_suppression(prediction, self.conf, self.iou, max_det=100)
        inference_ms = (time_sync() - started) * 1000
        annotated, detections = frame.copy(), []
        det = prediction[0]
        if len(det):
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], annotated.shape).round()
            for x1, y1, x2, y2, confidence, class_id in reversed(det.tolist()):
                class_id = int(class_id)
                raw_name = self.names[class_id]
                name = normalize_class_name(raw_name)
                box = (int(x1), int(y1), int(x2), int(y2))
                detections.append(Detection(class_id, name, float(confidence), box))
                color = CLASS_COLORS.get(name, (238, 179, 34))
                label = f"{CLASS_DRAW.get(name, str(raw_name).upper())} {confidence:.0%}"
                cv2.rectangle(annotated, box[:2], box[2:], color, 3, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                top = max(box[1] - th - 12, 0)
                cv2.rectangle(annotated, (box[0], top), (box[0] + tw + 12, box[1]), color, -1)
                cv2.putText(annotated, label, (box[0] + 6, box[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return AnalysisResult(annotated, detections, inference_ms)


class RiskEngine:
    """将目标检测结果转换为可解释的逐帧风险值。"""

    def __init__(self, confirm_frames=5, history_size=90):
        self.confirm_frames = confirm_frames
        self.history = deque(maxlen=history_size)
        self.drowning_streak = 0

    def reset(self):
        self.history.clear()
        self.drowning_streak = 0

    def update(self, detections, still_image=False):
        grouped = Counter(d.name for d in detections)
        drowning = [d.confidence for d in detections if d.name == "drowning"]
        treading = [d.confidence for d in detections if d.name == "tread_water"]
        swimming = [d.confidence for d in detections if d.name == "swimming"]
        if drowning:
            raw = min(100.0, 68.0 + max(drowning) * 32.0 + (len(drowning) - 1) * 5.0)
            self.drowning_streak += self.confirm_frames if still_image else 1
        elif treading:
            raw = min(69.0, 28.0 + max(treading) * 35.0 + (len(treading) - 1) * 3.0)
            self.drowning_streak = max(0, self.drowning_streak - 1)
        elif swimming:
            raw = min(30.0, 5.0 + max(swimming) * 14.0)
            self.drowning_streak = 0
        else:
            raw = 2.0
            self.drowning_streak = max(0, self.drowning_streak - 1)
        recent = list(self.history)[-4:]
        risk = raw if still_image or not recent else raw * 0.58 + np.mean(recent) * 0.42
        risk = float(np.clip(risk, 0, 100))
        self.history.append(risk)
        confirmed = bool(drowning) and self.drowning_streak >= self.confirm_frames
        if confirmed:
            level, title = "danger", "高风险 · 疑似溺水"
        elif drowning:
            level, title = "checking", "正在复核异常行为"
        elif risk >= 35:
            level, title = "watch", "需要关注"
        else:
            level, title = "safe", "泳池状态正常"
        return {"risk": risk, "level": level, "title": title, "confirmed": confirmed,
                "streak": self.drowning_streak, "counts": dict(grouped),
                "drowning_conf": max(drowning, default=0.0)}


class RiskChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = deque([0.0], maxlen=90)
        self.language = "zh"
        self.setMinimumHeight(170)

    def set_language(self, language):
        self.language = language
        self.update()

    def add(self, value):
        self.values.append(float(value))
        self.update()

    def reset(self):
        self.values.clear()
        self.values.append(0.0)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["panel_alt"]))
        left, top, right, bottom = 38, 10, self.width() - 12, self.height() - 24
        plot_h = max(bottom - top, 1)
        painter.fillRect(left, top, right - left, int(plot_h * .30), QColor("#401a28"))
        painter.fillRect(left, int(top + plot_h * .30), right - left, int(plot_h * .35), QColor("#3b3216"))
        painter.fillRect(left, int(top + plot_h * .65), right - left, int(plot_h * .35), QColor("#123a35"))
        painter.setFont(QFont("Segoe UI", 9))
        for value in (0, 35, 70, 100):
            y = bottom - plot_h * value / 100
            painter.setPen(QPen(QColor("#35506b"), 1, Qt.DashLine))
            painter.drawLine(left, int(y), right, int(y))
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(4, int(y + 4), str(value))
        values = list(self.values)
        if len(values) > 1:
            step = (right - left) / max(len(values) - 1, 1)
            path = QPainterPath(QPointF(left, bottom - plot_h * values[0] / 100))
            for i, value in enumerate(values[1:], 1):
                path.lineTo(QPointF(left + i * step, bottom - plot_h * value / 100))
            painter.setPen(QPen(QColor(COLORS["cyan"]), 3))
            painter.drawPath(path)
        painter.setPen(QColor(COLORS["muted"]))
        painter.drawText(left, self.height() - 6, I18N[self.language]["recent_frames"])


class VideoPanel(QFrame):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.heading = QLabel(title)
        self.heading.setObjectName("panelTitle")
        layout.addWidget(self.heading)
        self.view = QLabel("等待画面")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 160)
        self.view.setStyleSheet("background:#050b13;color:#8aa0b7;border-radius:5px;")
        layout.addWidget(self.view, 1)
        self.hint = QLabel(subtitle)
        self.hint.setObjectName("muted")
        layout.addWidget(self.hint)

    def set_texts(self, title, subtitle, waiting):
        self.heading.setText(title)
        self.hint.setText(subtitle)
        if self.view.pixmap() is None:
            self.view.setText(waiting)


class PoolRiskApp(QMainWindow):
    def __init__(self, weights, device="", imgsz=640, user_database=None, current_user=None, language="zh"):
        super().__init__()
        self.language = language
        self.user_database = user_database
        self.current_user = current_user or {"id": 0, "username": "guest", "role": "operator", "enabled": 1}
        self.weights, self.device_arg, self.imgsz = Path(weights), device, imgsz
        self.detector = None
        self.risk_engine = RiskEngine(confirm_frames=5)
        self.capture = None
        self.source_kind, self.source_name = "", "等待选择数据源"
        self.last_original = self.last_annotated = None
        self.last_result = self.last_risk = None
        self.current_original_pixmap = self.current_result_pixmap = None
        self.last_alert_at = 0.0
        self.media_timer = QTimer(self)
        self.media_timer.timeout.connect(self._next_frame)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(1000)
        self.setWindowTitle("蓝盾 · 泳池溺水风险预警系统")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)
        self._apply_style()
        self._build_ui()
        self._apply_language()
        self._tick_clock()
        QTimer.singleShot(120, lambda: self.load_model(self.weights))

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{COLORS['bg']}; color:{COLORS['text']};
                font-family:'Microsoft YaHei UI','Segoe UI'; font-size:16px; }}
            QFrame#header {{ background:#0a1727; border-bottom:1px solid {COLORS['line']}; }}
            QFrame#panel {{ background:{COLORS['panel']}; border:1px solid {COLORS['line']}; border-radius:8px; }}
            QLabel#panelTitle {{ font-size:18px; font-weight:600; color:{COLORS['text']}; }}
            QLabel#muted {{ color:{COLORS['muted']}; font-size:14px; }}
            QPushButton {{ background:#17314d; color:{COLORS['text']}; border:0; border-radius:5px;
                padding:8px 12px; min-height:20px; }}
            QPushButton:hover {{ background:#22517b; }} QPushButton:pressed {{ background:#0f6d91; }}
            QPushButton#accent {{ background:#087ea4; }} QPushButton#accent:hover {{ background:#0ea5c6; }}
            QPushButton#danger {{ background:#a92839; }} QPushButton#danger:hover {{ background:#d43a4d; }}
            QSlider::groove:horizontal {{ height:5px; background:#24435f; border-radius:2px; }}
            QSlider::handle:horizontal {{ width:15px; margin:-5px 0; background:{COLORS['cyan']}; border-radius:7px; }}
            QTableWidget {{ background:{COLORS['panel_alt']}; alternate-background-color:#0d2034;
                color:{COLORS['text']}; border:0; gridline-color:{COLORS['line']}; selection-background-color:#155e75; }}
            QHeaderView::section {{ background:#17314d; color:{COLORS['text']}; border:0; padding:7px; font-weight:600; }}
            QSplitter::handle {{ background:{COLORS['line']}; }}
            QSplitter::handle:horizontal {{ width:6px; margin:8px 1px; border-radius:3px; }}
            QSplitter::handle:vertical {{ height:6px; margin:1px 8px; border-radius:3px; }}
            QSplitter::handle:hover {{ background:{COLORS['cyan']}; }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_toolbar())
        self.original_panel = VideoPanel("原始画面", "数据源实时画面")
        self.result_panel = VideoPanel("AI 分析结果", "检测框与行为类别")
        self.risk_panel = self._build_risk_panel()
        self.analysis_panel = self._build_analysis_panel()

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.addWidget(self.original_panel)
        self.top_splitter.addWidget(self.result_panel)
        self.top_splitter.addWidget(self.risk_panel)
        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 3)
        self.top_splitter.setStretchFactor(2, 2)
        self.top_splitter.setSizes([560, 560, 380])

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.analysis_panel)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([520, 320])
        outer.addWidget(self.main_splitter, 1)

    def _build_header(self):
        header = QFrame(); header.setObjectName("header"); header.setFixedHeight(92)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 7, 14, 7)
        layout.setSpacing(10)
        icon = QLabel("◉"); icon.setStyleSheet(f"font-size:32px;color:{COLORS['cyan']};")
        layout.addWidget(icon)
        titles = QVBoxLayout()
        self.app_title_label = QLabel("蓝盾 · 泳池溺水风险预警系统")
        self.app_title_label.setStyleSheet("font-size:26px;font-weight:700;")
        titles.addWidget(self.app_title_label)
        self.app_subtitle_label = QLabel("YOLOv5 智能行为识别与连续帧风险确认")
        self.app_subtitle_label.setObjectName("muted")
        titles.addWidget(self.app_subtitle_label); layout.addLayout(titles); layout.addStretch()
        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(2)
        self.model_status = QLabel("● 正在加载模型")
        self.model_status.setStyleSheet(f"color:{COLORS['yellow']};font-weight:600;")
        self.model_status.setMinimumHeight(20)
        self.model_status.setAlignment(Qt.AlignRight); status.addWidget(self.model_status)
        self.user_status = QLabel(); self.user_status.setObjectName("muted"); self.user_status.setAlignment(Qt.AlignRight)
        self.user_status.setMinimumHeight(18)
        status.addWidget(self.user_status)
        self.clock_label = QLabel(); self.clock_label.setObjectName("muted"); self.clock_label.setAlignment(Qt.AlignRight)
        self.clock_label.setMinimumHeight(18)
        status.addWidget(self.clock_label); layout.addLayout(status)
        return header

    def _button(self, text, slot, object_name=""):
        button = QPushButton(text)
        if object_name: button.setObjectName(object_name)
        button.clicked.connect(lambda _checked=False, callback=slot: callback())
        return button

    def _build_toolbar(self):
        bar = QFrame(); bar.setObjectName("panel"); layout = QHBoxLayout(bar)
        self.toolbar_buttons = {}
        for key, text, slot, name in (
            ("load_model", "加载模型", self.choose_model, ""), ("choose_image", "选择图片", self.choose_image, ""),
            ("choose_video", "选择视频", self.choose_video, ""), ("open_camera", "打开摄像头", self.open_camera, ""),
            ("start_resume", "开始 / 继续", self.start_or_resume, "accent"), ("pause", "暂停", self.pause, ""),
            ("stop", "停止", self.stop, "danger"), ("save_result", "保存结果", self.save_result, "")):
            button = self._button(text, slot, name)
            self.toolbar_buttons[key] = button
            layout.addWidget(button)
        layout.addSpacing(10)
        self.confidence_title = QLabel("置信度")
        layout.addWidget(self.confidence_title)
        self.conf_slider = QSlider(Qt.Horizontal); self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25); self.conf_slider.setFixedWidth(130)
        self.conf_slider.valueChanged.connect(self._conf_changed); layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.25")
        self.conf_label.setStyleSheet(f"color:{COLORS['cyan']};font-family:Consolas;font-weight:700;")
        layout.addWidget(self.conf_label)
        self.language_button = self._button("English", self.toggle_language)
        self.language_button.setMinimumWidth(78)
        layout.addWidget(self.language_button)
        self.user_management_button = self._button("用户管理", self.open_user_management)
        self.toolbar_buttons["user_management"] = self.user_management_button
        layout.addWidget(self.user_management_button)
        self.switch_user_button = self._button("切换用户", self.switch_user)
        self.toolbar_buttons["switch_user"] = self.switch_user_button
        layout.addWidget(self.switch_user_button)
        layout.addStretch()
        self.source_label = QLabel(self.source_name); self.source_label.setObjectName("muted")
        layout.addWidget(self.source_label)
        return bar

    def _build_risk_panel(self):
        panel = QFrame(); panel.setObjectName("panel"); layout = QVBoxLayout(panel)
        self.risk_title = QLabel("实时风险态势"); self.risk_title.setObjectName("panelTitle"); layout.addWidget(self.risk_title)
        self.alert_banner = QLabel("系统初始化中"); self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setMinimumHeight(44)
        self.alert_banner.setStyleSheet("background:#1c3449;border-radius:5px;font-size:20px;font-weight:700;")
        layout.addWidget(self.alert_banner)
        score = QHBoxLayout()
        self.risk_score = QLabel("0")
        self.risk_score.setStyleSheet(f"font-size:56px;font-weight:700;color:{COLORS['green']};")
        score.addWidget(self.risk_score)
        self.risk_caption = QLabel("/ 100\n综合风险值")
        score.addWidget(self.risk_caption); score.addStretch()
        self.speed_label = QLabel("推理 -- ms"); self.speed_label.setObjectName("muted"); score.addWidget(self.speed_label)
        layout.addLayout(score)
        self.risk_chart = RiskChart(); layout.addWidget(self.risk_chart, 1)
        self.count_label = QLabel("游泳 0  ·  踩水 0  ·  疑似溺水 0"); self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)
        self.confirm_hint = QLabel("红色告警需连续 5 帧检测到疑似溺水（图片立即确认）")
        self.confirm_hint.setObjectName("muted"); self.confirm_hint.setAlignment(Qt.AlignCenter)
        self.confirm_hint.setWordWrap(True); layout.addWidget(self.confirm_hint)
        return panel

    def _new_table(self, headers):
        table = QTableWidget(0, len(headers)); table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False); table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers); table.setSelectionBehavior(QTableWidget.SelectRows)
        return table

    def _subpanel(self, title, content):
        frame = QFrame(); frame.setStyleSheet(f"background:{COLORS['panel_alt']};border-radius:5px;")
        layout = QVBoxLayout(frame)
        label = QLabel(title); label.setStyleSheet(f"color:{COLORS['cyan']};font-weight:700;")
        frame.title_label = label
        layout.addWidget(label); layout.addWidget(content, 1)
        return frame

    def _build_analysis_panel(self):
        panel = QFrame(); panel.setObjectName("panel"); outer = QVBoxLayout(panel)
        self.analysis_records_title = QLabel("风险分析与事件记录")
        self.analysis_records_title.setObjectName("panelTitle"); outer.addWidget(self.analysis_records_title)
        self.analysis_splitter = QSplitter(Qt.Horizontal)
        self.analysis_splitter.setChildrenCollapsible(False)
        self.detection_table = self._new_table(["序号", "行为", "置信度", "位置"])
        self.detection_group = self._subpanel("当前检测目标", self.detection_table)
        self.analysis_splitter.addWidget(self.detection_group)
        self.detail_label = QLabel("请选择图片、视频或摄像头开始分析。")
        self.detail_label.setAlignment(Qt.AlignLeft | Qt.AlignTop); self.detail_label.setWordWrap(True); self.detail_label.setMargin(8)
        self.advice_group = self._subpanel("值守建议", self.detail_label)
        self.analysis_splitter.addWidget(self.advice_group)
        self.event_table = self._new_table(["时间", "级别", "风险", "事件内容"])
        self.event_group = self._subpanel("告警事件", self.event_table)
        self.analysis_splitter.addWidget(self.event_group)
        self.analysis_splitter.setStretchFactor(0, 3)
        self.analysis_splitter.setStretchFactor(1, 2)
        self.analysis_splitter.setStretchFactor(2, 3)
        self.analysis_splitter.setSizes([520, 360, 520])
        outer.addWidget(self.analysis_splitter, 1)
        return panel

    def _tick_clock(self):
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def tr(self, key, **kwargs):
        text = I18N[self.language][key]
        return text.format(**kwargs) if kwargs else text

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"
        self._apply_language()
        self.event_table.setRowCount(0)
        self._add_event("event_system", 0, self.tr("event_language"))

    def _update_user_access(self):
        role = self.current_user.get("role", "operator")
        self.user_management_button.setVisible(self.user_database is not None and role == "admin")
        self.user_status.setText(self.tr("current_user", name=self.current_user.get("username", "--"), role=self.tr(role)))

    def open_user_management(self):
        if self.user_database is None or self.current_user.get("role") != "admin": return
        dialog = UserManagementDialog(self.user_database, self.current_user, self.language, self)
        dialog.exec_()
        refreshed = self.user_database.get_user(self.current_user["id"])
        if refreshed: self.current_user.update(refreshed)
        self._update_user_access()

    def switch_user(self):
        if self.user_database is None: return
        self.pause()
        dialog = LoginDialog(self.user_database, self.language, self)
        if dialog.exec_() != QDialog.Accepted: return
        self.current_user = dialog.user
        self.language = dialog.language
        self._apply_language()
        self.event_table.setRowCount(0)
        self._add_event("event_system", 0, self.tr("switched_user", name=self.current_user["username"]))

    def _set_table_headers(self, table, headers):
        for column, text in enumerate(headers):
            table.setHorizontalHeaderItem(column, QTableWidgetItem(text))

    def _source_text(self):
        if self.source_kind == "image":
            return self.tr("source_image", name=self.source_name)
        if self.source_kind == "video":
            return self.tr("source_video", name=self.source_name)
        if self.source_kind == "camera":
            return self.tr("source_camera", name=self.source_name)
        return self.tr("waiting_source")

    def _apply_language(self):
        self.setWindowTitle(self.tr("app_title"))
        self.app_title_label.setText(self.tr("app_title"))
        self.app_subtitle_label.setText(self.tr("app_subtitle"))
        for key, button in self.toolbar_buttons.items():
            button.setText(self.tr(key))
        self.language_button.setText(self.tr("language_button"))
        self.confidence_title.setText(self.tr("confidence"))
        self.source_label.setText(self._source_text())
        self._update_user_access()
        self.original_panel.set_texts(self.tr("original"), self.tr("original_hint"), self.tr("waiting_frame"))
        self.result_panel.set_texts(self.tr("analysis_result"), self.tr("analysis_hint"), self.tr("waiting_frame"))
        self.risk_title.setText(self.tr("risk_situation"))
        self.risk_caption.setText(self.tr("risk_caption"))
        self.confirm_hint.setText(self.tr("confirm_hint"))
        self.risk_chart.set_language(self.language)
        self.analysis_records_title.setText(self.tr("analysis_records"))
        self.detection_group.title_label.setText(self.tr("current_targets"))
        self.advice_group.title_label.setText(self.tr("guard_advice"))
        self.event_group.title_label.setText(self.tr("alert_events"))
        self._set_table_headers(self.detection_table, self.tr("detection_headers"))
        self._set_table_headers(self.event_table, self.tr("event_headers"))
        if self.detector is None:
            self.model_status.setText(self.tr("model_loading"))
            self.alert_banner.setText(self.tr("system_initializing"))
            self.detail_label.setText(self.tr("choose_prompt"))
            self.speed_label.setText(self.tr("inference_empty"))
        else:
            device_name = "GPU" if self.detector.device.type != "cpu" else "CPU"
            self.model_status.setText(self.tr("model_ready", device=device_name))
        if self.last_result is not None and self.last_risk is not None:
            self._refresh_analysis_text(self.last_result, self.last_risk)
        elif self.detector is not None:
            self._set_banner(self.tr("system_ready"), "safe")

    def load_model(self, path):
        self.stop(reset_status=False)
        self.model_status.setText(self.tr("model_loading"))
        self.model_status.setStyleSheet(f"color:{COLORS['yellow']};font-weight:600;")
        QApplication.processEvents()
        try:
            self.detector = YoloV5Detector(path, self.device_arg, self.imgsz, self.conf_slider.value() / 100)
            self.weights = Path(path)
            device_name = "GPU" if self.detector.device.type != "cpu" else "CPU"
            self.model_status.setText(self.tr("model_ready", device=device_name))
            self.model_status.setStyleSheet(f"color:{COLORS['green']};font-weight:600;")
            self._set_banner(self.tr("system_ready"), "safe")
            self._add_event("event_system", 0, self.tr("event_loaded", name=self.weights.name))
        except Exception as exc:
            write_startup_log("Model load failed", exc, append=True)
            self.model_status.setText(self.tr("model_failed"))
            self.model_status.setStyleSheet(f"color:{COLORS['red']};font-weight:600;")
            QMessageBox.critical(self, self.tr("model_failed").lstrip("● "), str(exc))

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("model_dialog"), str(ROOT), "PyTorch weights (*.pt)")
        if path: self.load_model(path)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("image_dialog"), str(ROOT / "data/images"),
                                              "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if not path: return
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            QMessageBox.critical(self, self.tr("read_failed"), self.tr("cannot_read_image"))
            return
        self.stop(reset_status=False)
        self.source_kind, self.source_name = "image", Path(path).name
        self.source_label.setText(self._source_text())
        self.risk_engine.reset(); self.risk_chart.reset()
        self._process_frame(frame, still_image=True)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("video_dialog"), str(ROOT),
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)")
        if path: self._open_capture(path, "video", Path(path).name)

    def open_camera(self):
        self._open_capture(0, "camera", "0")

    def _open_capture(self, source, kind, name):
        if self.detector is None:
            QMessageBox.warning(self, self.tr("model_not_ready"), self.tr("wait_model"))
            return
        self.stop(reset_status=False)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            QMessageBox.critical(self, self.tr("open_failed"), self.tr("cannot_open", name=name))
            return
        self.capture, self.source_kind, self.source_name = capture, kind, name
        self.source_label.setText(self._source_text())
        self.risk_engine.reset(); self.risk_chart.reset()
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.media_timer.start(1 if kind == "camera" else max(1, int(1000 / (fps if fps > 0 else 25))))

    def start_or_resume(self):
        if self.capture is not None:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.media_timer.start(max(1, int(1000 / (fps if fps > 0 else 25))))
        elif not self.source_kind:
            QMessageBox.information(self, self.tr("select_source"), self.tr("select_source_body"))

    def pause(self):
        if self.capture is not None:
            self.media_timer.stop(); self._set_banner(self.tr("paused"), "watch")

    def stop(self, reset_status=True):
        self.media_timer.stop()
        if self.capture is not None:
            self.capture.release(); self.capture = None
        if reset_status:
            self.source_kind, self.source_name = "", self.tr("waiting_source")
            if hasattr(self, "source_label"): self.source_label.setText(self._source_text())
            if hasattr(self, "alert_banner"): self._set_banner(self.tr("system_ready"), "safe")

    def _next_frame(self):
        if self.capture is None:
            self.media_timer.stop(); return
        ok, frame = self.capture.read()
        if not ok:
            self.stop(reset_status=False); self._set_banner(self.tr("video_complete"), "safe")
            self._add_event("event_complete", 0, self.tr("event_finished", name=self.source_name)); return
        self._process_frame(frame, still_image=False)

    def _process_frame(self, frame, still_image=False):
        if self.detector is None:
            QMessageBox.warning(self, self.tr("model_not_ready"), self.tr("wait_model")); return
        try:
            self.detector.conf = self.conf_slider.value() / 100
            result = self.detector.infer(frame)
            risk = self.risk_engine.update(result.detections, still_image)
            self.last_original, self.last_annotated = frame.copy(), result.frame.copy()
            self.current_original_pixmap = self._frame_pixmap(frame)
            self.current_result_pixmap = self._frame_pixmap(result.frame)
            self._refresh_video_views(); self._update_analysis(result, risk)
        except Exception as exc:
            self.stop(reset_status=False); QMessageBox.critical(self, self.tr("analysis_failed"), str(exc))

    def _frame_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy())

    def _refresh_video_views(self):
        for view, pixmap in ((self.original_panel.view, self.current_original_pixmap),
                             (self.result_panel.view, self.current_result_pixmap)):
            if pixmap is not None:
                view.setPixmap(pixmap.scaled(view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._refresh_video_views()

    def _set_banner(self, text, level):
        styles = {"safe": ("#123a35", COLORS["green"]), "watch": ("#3b3216", COLORS["yellow"]),
                  "checking": ("#4b2b18", "#fb923c"), "danger": ("#5a1724", COLORS["red"])}
        bg, fg = styles[level]
        self.alert_banner.setText(text)
        self.alert_banner.setStyleSheet(f"background:{bg};color:{fg};border-radius:5px;font-size:20px;font-weight:700;")

    def _update_analysis(self, result, risk):
        self.last_result, self.last_risk = result, risk
        self._set_banner(self.tr(f"risk_{risk['level']}"), risk["level"])
        color = {"safe": COLORS["green"], "watch": COLORS["yellow"],
                 "checking": "#fb923c", "danger": COLORS["red"]}[risk["level"]]
        self.risk_score.setText(f"{risk['risk']:.0f}")
        self.risk_score.setStyleSheet(f"font-size:56px;font-weight:700;color:{color};")
        self.speed_label.setText(self.tr("inference", ms=result.inference_ms))
        self.risk_chart.add(risk["risk"])
        self._refresh_analysis_text(result, risk)
        if risk["confirmed"]:
            self._trigger_alert(risk)

    def _refresh_analysis_text(self, result, risk):
        self._set_banner(self.tr(f"risk_{risk['level']}"), risk["level"])
        self.speed_label.setText(self.tr("inference", ms=result.inference_ms))
        counts = risk["counts"]
        self.count_label.setText(self.tr("counts", swim=counts.get("swimming", 0),
                                         tread=counts.get("tread_water", 0), drown=counts.get("drowning", 0)))
        detections = sorted(result.detections, key=lambda d: d.confidence, reverse=True)
        self.detection_table.setRowCount(len(detections))
        class_names = CLASS_CN if self.language == "zh" else CLASS_EN
        for row, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.box
            values = (str(row + 1), class_names.get(detection.name, detection.name),
                      f"{detection.confidence:.1%}", f"({x1},{y1})-({x2},{y2})")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setTextAlignment(Qt.AlignCenter)
                self.detection_table.setItem(row, column, item)
        if risk["confirmed"]:
            advice = self.tr("advice_danger")
        elif risk["level"] == "checking":
            advice = self.tr("advice_checking", streak=min(risk["streak"], 5))
        elif risk["level"] == "watch":
            advice = self.tr("advice_watch")
        else:
            advice = self.tr("advice_safe")
        self.detail_label.setText(advice)

    def _trigger_alert(self, risk):
        now = time.time()
        if now - self.last_alert_at < 8: return
        self.last_alert_at = now
        self._add_event("event_danger", risk["risk"],
                        self.tr("event_drowning", confidence=risk["drowning_conf"]))
        def sound():
            try:
                if os.name == "nt":
                    import winsound
                    for frequency in (1100, 1350, 1100): winsound.Beep(frequency, 260)
            except Exception: pass
        threading.Thread(target=sound, daemon=True).start()

    def _add_event(self, level, risk, detail):
        self.event_table.insertRow(0)
        display_level = self.tr(level) if level in I18N[self.language] else level
        values = (datetime.now().strftime("%H:%M:%S"), display_level,
                  f"{risk:.0f}" if risk else "--", detail)
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter if column < 3 else Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(0, column, item)
        while self.event_table.rowCount() > 50: self.event_table.removeRow(self.event_table.rowCount() - 1)

    def _conf_changed(self, value):
        self.conf_label.setText(f"{value / 100:.2f}")
        if self.detector is not None: self.detector.conf = value / 100

    def save_result(self):
        if self.last_annotated is None:
            QMessageBox.information(self, self.tr("no_result"), self.tr("no_result_body")); return
        default = str(ROOT / f"pool_alert_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        path, _ = QFileDialog.getSaveFileName(self, self.tr("save_dialog"), default,
                                              "JPEG (*.jpg);;PNG (*.png)")
        if path:
            cv2.imencode(Path(path).suffix or ".jpg", self.last_annotated)[1].tofile(path)
            self._add_event("event_save", 0, Path(path).name)

    def closeEvent(self, event):
        self.stop(); event.accept()


def smoke_test(args):
    detector = YoloV5Detector(args.weights, args.device, args.imgsz, args.conf, args.iou)
    frame = cv2.imdecode(np.fromfile(args.smoke_test, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None: raise FileNotFoundError(f"无法读取测试图片：{args.smoke_test}")
    result = detector.infer(frame)
    risk = RiskEngine(confirm_frames=1).update(result.detections, still_image=True)
    output = Path(args.smoke_output) if args.smoke_output else ROOT / "runs" / "gui-smoke.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(output.suffix or ".jpg", result.frame)[1].tofile(str(output))
    print(json.dumps({"ok": True, "weights": str(args.weights), "image": str(args.smoke_test),
                      "output": str(output), "inference_ms": round(result.inference_ms, 2),
                      "risk": round(risk["risk"], 2), "level": risk["level"],
                      "detections": [{"class": d.name, "confidence": round(d.confidence, 4), "box": d.box}
                                     for d in result.detections]}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="泳池溺水风险预警桌面系统")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/train/exp/weights/best.pt")
    parser.add_argument("--device", default="", help="CUDA 设备，例如 0；留空为自动选择")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--smoke-test", help="不启动界面，只分析指定图片并退出")
    parser.add_argument("--smoke-output", help="冒烟测试结果图片路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test(args); return
    write_startup_log("Starting application")

    def exception_hook(exc_type, exc_value, exc_tb):
        error = exc_value if isinstance(exc_value, Exception) else RuntimeError(str(exc_value))
        write_startup_log("Unhandled application error", error, append=True)
        show_native_error(f"The application encountered an error.\n\n{error}\n\nLog: {STARTUP_LOG}")

    sys.excepthook = exception_hook
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("泳池溺水风险预警系统")
        database = UserDatabase(USER_DB)
        login = LoginDialog(database)
        if login.exec_() != QDialog.Accepted:
            write_startup_log("Login cancelled", append=True)
            return
        window = PoolRiskApp(args.weights, args.device, args.imgsz, database, login.user, login.language)
        window.show()
        write_startup_log("Application window started")
        sys.exit(app.exec_())
    except Exception as exc:
        write_startup_log("Startup failed", exc, append=True)
        show_native_error(f"Unable to start the application.\n\n{exc}\n\nLog: {STARTUP_LOG}")
        raise


if __name__ == "__main__":
    main()
