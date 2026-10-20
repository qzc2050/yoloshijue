# -*- coding: utf-8 -*-
"""Machine and board settings shared by the lab, the grouping script and the trainer.

Only the standard library is imported here so the firmware-side scripts in qc_motion can use it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOST = Path(__file__).resolve().parents[1]
CONFIG_DIR = HOST / "config"
BOARDS_DIR = HOST / "boards"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Machine:
    active_board: str
    scan_root: Path
    yolo_root: Path
    temp_dir: Path
    port: str
    baud: int
    px_per_mm: float
    align_offset_mm: float
    unsure_conf: float


@dataclass(frozen=True)
class Side:
    id: str
    label: str
    gerber: Path
    expect: int
    mirror_y: bool
    rotation_deg: float
    scan_minutes: int
    scan_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Board:
    id: str
    name: str
    dir: Path
    merge_mm: float
    match_mm: float
    column_rotation_deg: float
    gerber_to_machine: tuple[tuple[float, float, float], tuple[float, float, float]]
    sides: dict[str, Side]

    def side(self, side_id: str) -> Side:
        if side_id not in self.sides:
            raise ConfigError(f"板型 {self.id} 没有板面 {side_id}")
        return self.sides[side_id]


_machine_cache: Machine | None = None
_board_cache: dict[str, Board] = {}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"缺少配置文件：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"配置文件不是合法 JSON：{path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是对象：{path}")
    return data


def _need(data: dict, key: str, path: Path) -> Any:
    if key not in data:
        raise ConfigError(f"{path} 缺少字段 {key}")
    return data[key]


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else HOST / path


def _machine_from_file() -> Machine:
    path = CONFIG_DIR / "machine.json"
    data = _read_json(path)
    return Machine(
        active_board=str(_need(data, "active_board", path)),
        scan_root=_path(_need(data, "scan_root", path)),
        yolo_root=_path(_need(data, "yolo_root", path)),
        temp_dir=_path(_need(data, "temp_dir", path)),
        port=str(_need(data, "port", path)),
        baud=int(_need(data, "baud", path)),
        px_per_mm=float(_need(data, "px_per_mm", path)),
        align_offset_mm=float(_need(data, "align_offset_mm", path)),
        unsure_conf=float(_need(data, "unsure_conf", path)),
    )


def _board_from_file(board_id: str) -> Board:
    folder = BOARDS_DIR / board_id
    path = folder / "profile.json"
    data = _read_json(path)
    raw_sides = _need(data, "sides", path)
    if not isinstance(raw_sides, dict) or not raw_sides:
        raise ConfigError(f"{path} 的 sides 必须是非空对象")

    sides: dict[str, Side] = {}
    for side_id, raw in raw_sides.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} 的 sides.{side_id} 必须是对象")
        points = _need(raw, "scan_points", path)
        if not isinstance(points, list) or not points:
            raise ConfigError(f"{path} 的 sides.{side_id}.scan_points 必须是非空数组")
        sides[side_id] = Side(
            id=side_id,
            label=str(_need(raw, "label", path)),
            gerber=folder / str(_need(raw, "gerber", path)),
            expect=int(_need(raw, "expect", path)),
            mirror_y=bool(_need(raw, "mirror_y", path)),
            rotation_deg=float(_need(raw, "rotation_deg", path)),
            scan_minutes=int(raw.get("scan_minutes", 4)),
            scan_points=tuple((float(point[0]), float(point[1])) for point in points),
        )

    affine = _need(data, "gerber_to_machine", path)
    if len(affine) != 2 or any(len(row) != 3 for row in affine):
        raise ConfigError(f"{path} 的 gerber_to_machine 必须是 2x3 数组")
    return Board(
        id=str(_need(data, "id", path)),
        name=str(_need(data, "name", path)),
        dir=folder,
        merge_mm=float(_need(data, "merge_mm", path)),
        match_mm=float(_need(data, "match_mm", path)),
        column_rotation_deg=float(_need(data, "column_rotation_deg", path)),
        gerber_to_machine=tuple(tuple(float(value) for value in row) for row in affine),
        sides=sides,
    )


def reload() -> None:
    global _machine_cache
    _machine_cache = None
    _board_cache.clear()


def machine() -> Machine:
    global _machine_cache
    if _machine_cache is None:
        _machine_cache = _machine_from_file()
    return _machine_cache


def active_board() -> Board:
    return board(machine().active_board)


def board(board_id: str | None = None) -> Board:
    board_id = board_id or machine().active_board
    if board_id not in _board_cache:
        _board_cache[board_id] = _board_from_file(board_id)
    return _board_cache[board_id]


def list_boards() -> list[str]:
    if not BOARDS_DIR.is_dir():
        return []
    return sorted(path.parent.name for path in BOARDS_DIR.glob("*/profile.json"))


def scan_root() -> Path:
    return machine().scan_root
