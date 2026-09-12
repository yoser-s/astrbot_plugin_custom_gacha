"""自定义抽卡插件 —— 支持自定义唤醒词、AI 图片分析、动图、压缩包导入、WebUI 管理。"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.web import (
    PluginUploadFile,
    error_response,
    file_response,
    json_response,
    request,
)

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
except ImportError:
    get_astrbot_plugin_data_path = None

PLUGIN_NAME = "astrbot_plugin_custom_gacha"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_GIF_FRAMES = 4
CANVAS_W = 800
CANVAS_H = 900
IMG_AREA_H = 370
IMG_MAX_W = CANVAS_W - 180


# ==================== 存储工具 ====================

DEFAULT_FOLDER = "Default"


def _data_dir() -> Path:
    if get_astrbot_plugin_data_path:
        base = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
    else:
        base = Path("data") / PLUGIN_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _images_dir() -> Path:
    d = _data_dir() / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cards_dir() -> Path:
    d = _data_dir() / "cards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _data_dir() / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    if not p.exists():
        return {"images": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            data = {"images": []}
    except (OSError, ValueError):
        return {"images": []}
    # 迁移：给没有文件夹的旧图片补上默认文件夹
    changed = False
    for img in data.get("images", []):
        if isinstance(img, dict) and not img.get("folder"):
            img["folder"] = DEFAULT_FOLDER
            changed = True
    if changed:
        try:
            _save_manifest(data)
        except Exception:
            pass
    return data


def _save_manifest(data: dict) -> None:
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _find_in_manifest(file_id: str) -> dict | None:
    for item in _load_manifest().get("images", []):
        if item.get("id") == file_id:
            return item
    return None


# ==================== GIF 帧提取 ====================

def _extract_gif_frames(path: Path, max_frames: int = MAX_GIF_FRAMES) -> list[str]:
    """从动图提取采样帧路径列表，用于 AI 分析。"""
    temp_paths: list[str] = []
    try:
        with Image.open(path) as img:
            n = max(1, int(getattr(img, "n_frames", 1) or 1))
            if n <= 1:
                return [str(path)]
            count = min(n, max_frames)
            indexes = [round(i * (n - 1) / max(1, count - 1)) for i in range(count)] if count > 1 else [0]
            for idx in indexes:
                img.seek(idx)
                frame = img.convert("RGBA")
                tmp = tempfile.NamedTemporaryFile(prefix=f"gacha_frame_{idx}_", suffix=".png", delete=False)
                try:
                    frame.save(tmp, format="PNG")
                finally:
                    tmp.close()
                temp_paths.append(tmp.name)
            return temp_paths
    except Exception:
        for p in temp_paths:
            Path(p).unlink(missing_ok=True)
        return [str(path)]


def _cleanup_temp(paths: list[str]) -> None:
    for p in paths:
        Path(p).unlink(missing_ok=True)


def _is_animated(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            return int(getattr(img, "n_frames", 1) or 1) > 1
    except Exception:
        return False


# ==================== AI 分析 ====================

ANALYSIS_SYSTEM_PROMPT = (
    "你是「自定义抽卡」卡池宇宙的图鉴总编，不是普通的图片描述器。"
    "请完全根据图片画面，为这张卡牌全新创作一个卡牌名和一段图鉴文案："
    "不要复述、模仿或引用原始文件名，也不要在任何结果里出现'文件名''原图'这类字眼，"
    "文件名至多只能作为隐藏灵感，绝不能直接写进输出。"
    "name 必须是 3-8 个汉字，一语道破天机，风趣、有画面感、够创意，不能是泛泛的形容词；"
    "description 必须是 40-120 字的单段文案，带网络梗、风趣感和一点哲学意味，"
    "像图鉴旁白一样自然，可以在结尾反转。"
    "只返回 JSON，不要 Markdown，不要代码块，不要多余文字。"
    '格式：{"name":"3-8字卡牌名","description":"40-120字图鉴文案"}'
)


def _build_analysis_prompt(filename: str, template: str = "") -> str:
    name_hint = Path(filename).stem
    if template and template.strip():
        return template.replace("{filename}", filename).replace("{name}", name_hint)
    return (
        "这里有一张要加入抽卡池的图片，请为它全新创作卡牌名和介绍："
        "name 用 3-8 个汉字，要一语道破天机、风趣有画面感、跳出原有框架想；"
        "description 用 40-120 字一段，带网络梗和一点哲学意味，结尾可以反转。"
        "不要复述原始文件名，也不要在结果里提到'文件名/原图'这样的字眼。"
        '只返回 JSON：{"name":"3-8字卡牌名","description":"40-120字图鉴文案"}'
    )


async def _analyze_image(context: Any, image_path: Path, filename: str, provider_id: str = "",
                         *, vision_model: str = "", system_prompt: str = "",
                         temperature: float | None = None, max_tokens: int = 0) -> dict:
    """调用视觉 LLM 分析图片，返回 {"name":..., "description":...}。"""
    settings = _load_settings()
    visual_paths = _extract_gif_frames(image_path)
    temp_paths = [p for p in visual_paths if p != str(image_path)]
    try:
        prompt = _build_analysis_prompt(filename, settings.get("analysis_prompt_template", ""))
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image_urls": visual_paths,
            "system_prompt": system_prompt or settings.get("analysis_system_prompt") or ANALYSIS_SYSTEM_PROMPT,
            "temperature": temperature if temperature is not None else settings.get("analysis_temperature", 0.8),
            "max_tokens": max_tokens if max_tokens else settings.get("analysis_max_tokens", 512),
        }
        if vision_model:
            kwargs["model"] = vision_model
        if provider_id:
            kwargs["chat_provider_id"] = provider_id
        response = await context.llm_generate(**kwargs)
        text = _extract_response_text(response)
        return _parse_analysis_json(text)
    finally:
        _cleanup_temp(temp_paths)


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("completion_text", "text", "content"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                return val
        if "result" in response:
            return _extract_response_text(response["result"])
        return json.dumps(response, ensure_ascii=False)
    text = getattr(response, "completion_text", None) or getattr(response, "text", None)
    if text:
        return str(text)
    return str(response)


def _parse_analysis_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        data = json.loads(text)
    except ValueError:
        import re
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except ValueError:
                data = {}
        else:
            data = {}
    name = str(data.get("name", "")).strip() if isinstance(data, dict) else ""
    desc = str(data.get("description", "")).strip() if isinstance(data, dict) else ""
    if not name:
        name = "未知卡牌"
    if not desc:
        desc = "一张神秘的图片。"
    return {"name": name, "description": desc}


# ==================== 卡片渲染 ====================

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    res = Path(__file__).resolve().parent / "resource" / "font"
    if res.exists():
        for sub in sorted(res.glob("*.tt*")) + sorted(res.glob("*.ot*")):
            candidates.insert(0, str(sub))
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        test = current + char
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = char
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _ellips(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """截断超宽文本，末尾加省略号。"""
    w = font.getbbox(text)[2]
    if w <= max_width:
        return text
    cut = ""
    for ch in text:
        test = cut + ch
        if font.getbbox(test)[2] > max_width - font.getbbox("…")[2]:
            break
        cut = test
    return cut + "…"


PAPER_BG = (246, 241, 231, 255)
PAPER_BORDER = (205, 184, 146, 255)
TITLE_COLOR = (91, 63, 42, 255)
DESC_COLOR = (72, 64, 52, 255)
_TEXTURE_SPOTS: list = []


def _card_texture(w: int, h: int) -> list:
    global _TEXTURE_SPOTS
    if not _TEXTURE_SPOTS:
        rng = random.Random(20260911)
        _TEXTURE_SPOTS = [
            (rng.randint(0, w - 1), rng.randint(0, h - 1), rng.randint(6, 16))
            for _ in range(max(1, int(w * h / 600)))
        ]
    return _TEXTURE_SPOTS


def _paper_canvas():
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), PAPER_BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((3, 3, CANVAS_W - 4, CANVAS_H - 4), outline=PAPER_BORDER, width=2)
    return canvas, draw


def _card_geometry(new_h: int, name: str, description: str):
    font_name = _load_font(42, bold=True)
    font_desc = _load_font(26)
    desc_lines = _wrap_text(description, font_desc, CANVAS_W - 160)
    n_desc = min(6, len(desc_lines))
    gap_img = 18
    title_h = 58
    line_h = 36
    content_h = new_h + gap_img + title_h + n_desc * line_h
    image_top = max(24, (CANVAS_H - content_h) // 2)
    title_top = image_top + new_h + gap_img
    desc_top = title_top + title_h
    return font_name, font_desc, desc_lines, image_top, title_top, desc_top


def _draw_card_text(draw, font_name, name, title_top, font_desc, desc_lines, desc_top) -> None:
    nb = font_name.getbbox(name)
    draw.text(((CANVAS_W - (nb[2] - nb[0])) // 2, title_top), name, font=font_name, fill=TITLE_COLOR)
    dy = desc_top
    for line in desc_lines[:6]:
        lb = font_desc.getbbox(line)
        draw.text(((CANVAS_W - (lb[2] - lb[0])) // 2, dy), line, font=font_desc, fill=DESC_COLOR)
        dy += 36


def _render_card(image_path: Path, name: str, description: str, is_gif: bool) -> Path:
    """渲染合成卡片图片（名字 + 介绍 叠加在图片上）。"""
    if is_gif:
        return _render_gif_card(image_path, name, description)

    with Image.open(image_path) as src:
        img = src.convert("RGBA")
    out = _cards_dir() / f"{image_path.stem}_card.png"
    if out.exists():
        return out

    canvas, draw = _paper_canvas()

    img_w, img_h = img.size
    scale = min(IMG_MAX_W / img_w, IMG_AREA_H / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    font_name, font_desc, desc_lines, image_top, title_top, desc_top = _card_geometry(new_h, name, description)
    paste_x = (CANVAS_W - new_w) // 2
    canvas.paste(img_resized, (paste_x, image_top), img_resized)
    _draw_card_text(draw, font_name, name, title_top, font_desc, desc_lines, desc_top)

    canvas.convert("RGB").save(out, "PNG")
    return out


def _render_gif_card(image_path: Path, name: str, description: str) -> Path:
    """为 GIF 动图渲染卡片，保留动图效果，叠加文字层。"""
    out = _cards_dir() / f"{image_path.stem}_card.gif"
    with Image.open(image_path) as src:
        n_frames = max(1, int(getattr(src, "n_frames", 1) or 1))
        frames: list[Image.Image] = []
        for i in range(n_frames):
            src.seek(i)
            frame = src.convert("RGBA")
            img_w, img_h = frame.size
            scale = min(IMG_MAX_W / img_w, IMG_AREA_H / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            img_resized = frame.resize((new_w, new_h), Image.LANCZOS)

            canvas, draw = _paper_canvas()
            font_name, font_desc, desc_lines, image_top, title_top, desc_top = _card_geometry(new_h, name, description)
            canvas.paste(img_resized, ((CANVAS_W - new_w) // 2, image_top), img_resized)
            _draw_card_text(draw, font_name, name, title_top, font_desc, desc_lines, desc_top)

            frames.append(canvas)

        durations = []
        for i in range(n_frames):
            try:
                durations.append(max(40, int(src.info.get("duration", 100))))
            except Exception:
                durations.append(100)

        if len(frames) == 1:
            frames[0].convert("RGB").save(out, "PNG")
        else:
            frames[0].save(
                out,
                "GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations[:len(frames)],
                loop=0,
                disposal=2,
            )
        return out


# ==================== 运行时设置 ====================

_BATCH_JOBS: dict[str, dict] = {}


def _batch_jobs_cleanup():
    now = time.time()
    for key, job in list(_BATCH_JOBS.items()):
        if job.get("finished") and now - job.get("end_time", 0) > 600:
            _BATCH_JOBS.pop(key, None)


_PLUGIN_CONFIG: dict = {}

_SETTING_DEFAULTS = {
    "trigger_words": ["抽卡"],
    "match_mode": "exact",
    "allow_direct_trigger": True,
    "vision_provider": "",
    "vision_model": "",
    "analysis_temperature": 0.8,
    "analysis_max_tokens": 512,
    "analysis_system_prompt": ANALYSIS_SYSTEM_PROMPT,
    "analysis_prompt_template": "",
    "history_trigger_words": ["图鉴", "我的图鉴", "抽卡图鉴"],
    "daily_limit_count": 1,
    "super_users": [],
    "default_folder": DEFAULT_FOLDER,
    "folders": [DEFAULT_FOLDER],
    "group_folders": {},
    "board_daily_word": "抽卡日榜",
    "board_monthly_word": "抽卡月榜",
    "board_total_word": "抽卡总榜",
    "help_trigger_words": ["帮助", "菜单", "抽卡帮助"],
    "debug_trigger_word": "自定义抽卡调试",
}


def _settings_path() -> Path:
    return _data_dir() / "settings.json"


def _load_settings() -> dict:
    p = _settings_path()
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
    out = dict(_SETTING_DEFAULTS)
    out.update(data)
    out.update(_PLUGIN_CONFIG)
    return out


def _save_settings(patch: dict) -> None:
    cur = _load_settings()
    cur.update(patch)
    p = _settings_path()
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ==================== 历史记录 ====================

def _history_path() -> Path:
    return _data_dir() / "history.json"


def _load_history() -> dict:
    p = _history_path()
    if not p.exists():
        return {"users": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {"users": {}}
    except (OSError, ValueError):
        return {"users": {}}


def _save_history(data: dict) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _record_draw(user_id: str, file_id: str, name: str, group_id: str = "") -> None:
    history = _load_history()
    users = history.setdefault("users", {})
    record = users.setdefault(user_id, {"total": 0, "cards": {}})
    record["total"] = int(record.get("total", 0)) + 1
    cards = record.setdefault("cards", {})
    entry = cards.get(file_id, {"count": 0, "name": name})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["name"] = name or entry.get("name", "未知卡牌")
    cards[file_id] = entry
    today = time.strftime("%Y-%m-%d")
    record["last_draw_date"] = today
    day = record.setdefault("day", {})
    day_list = day.get(today)
    if not isinstance(day_list, list):
        day_list = []
        day[today] = day_list
    day_list.append({"file_id": file_id, "name": name or entry.get("name", "未知卡牌"), "ts": time.time(), "group_id": str(group_id or "")})
    groups = record.setdefault("groups", {})
    gg = groups.setdefault(str(group_id), {"draws": 0})
    gg["draws"] = int(gg.get("draws", 0)) + 1
    gg["last"] = today
    _save_history(history)


def _get_user_history(user_id: str) -> dict:
    history = _load_history()
    return history.get("users", {}).get(user_id, {"total": 0, "cards": {}})


def _today_history_records(user_id: str, group_id: str | None = None) -> list[dict]:
    record = _load_history().get("users", {}).get(user_id, {})
    day_list = (record.get("day") or {}).get(time.strftime("%Y-%m-%d"))
    if not isinstance(day_list, list):
        return []
    result = [r for r in day_list if isinstance(r, dict) and r.get("file_id")]
    if group_id is not None:
        gid = str(group_id or "")
        result = [r for r in result if str(r.get("group_id") or "") == gid]
    return result


def _check_daily_limit(user_id: str, group_id: str | None = None) -> bool:
    """返回 True 表示今天该群已达抽卡上限。"""
    limit = int(_load_settings().get("daily_limit_count", 1) or 0)
    if limit <= 0:
        return False
    return len(_today_history_records(user_id, group_id)) >= limit


# ==================== 文件夹 / 排行榜 / 调试会话 ====================

_DEBUG_SESSION: dict = {}   # admin_user_id -> {"step": ...}
_DEBUG_POOL: dict = {}      # admin_user_id -> 当前调试卡池文件夹


def _all_folder_names() -> list[str]:
    s = _load_settings()
    names = s.get("folders") or []
    names = [str(n).strip() for n in names if str(n).strip()]
    if DEFAULT_FOLDER not in names:
        names = [DEFAULT_FOLDER] + names
    return names


def _save_folder_names(names: list[str]) -> None:
    names = list(dict.fromkeys(str(n).strip() for n in names if str(n).strip()))
    if DEFAULT_FOLDER not in names:
        names = [DEFAULT_FOLDER] + names
    _save_settings({"folders": names})


def _resolve_folder(group_id) -> str:
    s = _load_settings()
    gf = s.get("group_folders") or {}
    if group_id and gf.get(str(group_id)):
        return str(gf[str(group_id)])
    return DEFAULT_FOLDER


def _set_group_folder(group_id, folder: str) -> None:
    s = _load_settings()
    gf = dict(s.get("group_folders") or {})
    if folder and folder != "未分配":
        gf[str(group_id)] = folder
    else:
        gf.pop(str(group_id), None)
    _save_settings({"group_folders": gf})


def _pool_images(folder=None) -> list[dict]:
    folder = folder or DEFAULT_FOLDER
    return [i for i in _load_manifest().get("images", [])
            if (i.get("folder") or DEFAULT_FOLDER) == folder]


def _boards_path() -> Path:
    return _data_dir() / "boards.json"


def _load_boards() -> dict:
    p = _boards_path()
    if not p.exists():
        return {"groups": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {"groups": {}}
    except (OSError, ValueError):
        return {"groups": {}}


def _save_boards(data: dict) -> None:
    p = _boards_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _board_bump(node: dict, user_id, name, file_id) -> None:
    rec = node.setdefault(str(user_id), {"count": 0, "name": name, "cards": {}})
    rec["count"] = int(rec.get("count", 0)) + 1
    if name:
        rec["name"] = name
    cards = rec.setdefault("cards", {})
    cards[str(file_id)] = int(cards.get(str(file_id), 0)) + 1


def _record_board(group_id, user_id, name, file_id) -> None:
    if not group_id:
        return  # 只记录群内抽卡
    today = time.strftime("%Y-%m-%d")
    month = time.strftime("%Y-%m")
    b = _load_boards()
    g = b.setdefault("groups", {}).setdefault(str(group_id), {})
    _board_bump(g.setdefault("total", {}), user_id, name, file_id)
    _board_bump(g.setdefault("month", {}).setdefault(month, {}), user_id, name, file_id)
    _board_bump(g.setdefault("day", {}).setdefault(today, {}), user_id, name, file_id)
    _save_boards(b)


def _iter_board_buckets(g: dict):
    """遍历一个群的 total / month.* / day.* 各排行桶。"""
    total = g.get("total")
    if isinstance(total, dict):
        yield total
    for key in ("month", "day"):
        sub = g.get(key) or {}
        if isinstance(sub, dict):
            for node in sub.values():
                if isinstance(node, dict):
                    yield node


def _group_cards_in_boards() -> dict:
    """所有在排行/历史中出现过的群号 -> 抽卡次数，用于 WebUI 群卡池分配。"""
    b = _load_boards()
    result: dict[str, int] = {}
    for gid, g in (b.get("groups") or {}).items():
        total = sum(int(r.get("count", 0)) for r in (g.get("total") or {}).values())
        result[str(gid)] = total
    return result


# ==================== 图鉴渲染 ====================

def _fit_thumb(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """将图片缩放裁剪到指定尺寸（cover 模式）。"""
    w, h = img.size
    tw, th = size
    scale = max(tw / w, th / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _mime_for_ext(ext: str) -> str:
    ext = ext.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def _image_to_base64(path: Path, max_side: int = 0) -> tuple[str, str]:
    """返回 (base64, mime_type)。动图和原图直接读字节，静态图可缩为 max_side 缩略图。"""
    ext = Path(path.name).suffix.lower()
    mime = _mime_for_ext(ext)
    if ext == ".gif":
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii"), mime
    try:
        with Image.open(path) as img:
            if max_side and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P", "LA", "PA"):
                img.convert("RGBA").save(buf, format="PNG")
                mime = "image/png"
            else:
                img.convert("RGB").save(buf, format="JPEG", quality=80)
                mime = "image/jpeg"
            return base64.b64encode(buf.getvalue()).decode("ascii"), mime
    except Exception:
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii"), mime


def _render_history_image(user_id: str, total_pool: int = 0) -> Path:
    """渲染用户抽卡历史图鉴。"""
    history = _get_user_history(user_id)
    total = int(history.get("total", 0))
    cards = history.get("cards", {})

    sorted_cards = sorted(
        (c for c in cards.values() if isinstance(c, dict)),
        key=lambda c: c.get("count", 0) or 0, reverse=True,
    )
    unique = len(sorted_cards)

    cols = 4
    thumb_size = 160
    card_gap = 16
    card_w = thumb_size + 24
    card_h = thumb_size + 60
    padding = 32
    title_h = 180

    rows = max(1, (len(sorted_cards) + cols - 1) // cols)
    canvas_w = max(900, padding * 2 + cols * card_w + (cols - 1) * card_gap)
    canvas_h = title_h + rows * card_h + (rows - 1) * card_gap + padding + 40

    canvas = Image.new("RGB", (canvas_w, canvas_h), (28, 25, 35))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(40, bold=True)
    font_stat = _load_font(22)
    font_name = _load_font(16)
    font_count = _load_font(18, bold=True)

    draw.text((padding, 28), "我的卡池 · 抽卡档案", font=font_title, fill=(255, 229, 100))
    stat_text = f"累计抽卡 {total} 次  ·  独立卡牌 {unique} 种"
    draw.text((padding, 82), stat_text, font=font_stat, fill=(180, 180, 200))
    if total_pool > 0:
        collect_text = f"已收集 {unique} / {total_pool} 种"
        # 收集进度比例条
        ratio = min(1.0, unique / total_pool)
        bar_w = 320
        draw.text((padding, 122), collect_text, font=font_stat, fill=(255, 214, 120))
        draw.rounded_rectangle(
            (padding + 230, 126, padding + 230 + bar_w, 126 + 16),
            radius=8, fill=(55, 50, 66),
        )
        draw.rounded_rectangle(
            (padding + 230, 126, padding + 230 + int(bar_w * ratio), 126 + 16),
            radius=8, fill=(255, 190, 70),
        )

    manifest = _load_manifest()
    manifest_map = {img["id"]: img for img in manifest.get("images", [])}

    grid_x = padding
    grid_y = title_h

    for idx, card in enumerate(sorted_cards):
        row, col = divmod(idx, cols)
        x = grid_x + col * (card_w + card_gap)
        y = grid_y + row * (card_h + card_gap)

        draw.rounded_rectangle(
            (x, y, x + card_w, y + card_h),
            radius=16,
            fill=(45, 40, 55),
        )

        file_id = None
        for fid, c in cards.items():
            if c is card:
                file_id = fid
                break

        if file_id and file_id in manifest_map:
            img_path = _images_dir() / manifest_map[file_id]["filename"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as src:
                        thumb = _fit_thumb(src.convert("RGBA"), (thumb_size, thumb_size))
                        canvas.paste(thumb, (x + 12, y + 12), thumb)
                except Exception:
                    draw.rounded_rectangle(
                        (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                        radius=12, fill=(60, 55, 70),
                    )
            else:
                draw.rounded_rectangle(
                    (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                    radius=12, fill=(60, 55, 70),
                )
        else:
            draw.rounded_rectangle(
                (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                radius=12, fill=(60, 55, 70),
            )

        card_name = str(card.get("name", "未知"))
        if len(card_name) > 8:
            card_name = card_name[:7] + "…"
        count = int(card.get("count", 0))
        draw.text((x + 12, y + 12 + thumb_size + 8), card_name, font=font_name, fill=(220, 220, 230))
        draw.text((x + 12, y + 12 + thumb_size + 30), f"x{count}", font=font_count, fill=(255, 200, 80))

    out = _cards_dir() / f"history_{user_id}.png"
    canvas.save(out, "PNG")
    return out


def _render_today_image(user_id: str, records: list[dict]) -> Path:
    """渲染今日抽取的卡牌图片。"""
    records = records or []
    cols = 4
    thumb_size = 160
    card_gap = 16
    card_w = thumb_size + 24
    card_h = thumb_size + 44
    padding = 32
    title_h = 130
    rows = max(1, (len(records) + cols - 1) // cols)
    canvas_w = max(900, padding * 2 + cols * card_w + (cols - 1) * card_gap)
    canvas_h = title_h + rows * card_h + (rows - 1) * card_gap + padding + 24

    canvas = Image.new("RGB", (canvas_w, canvas_h), (28, 25, 35))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(34, bold=True)
    font_tip = _load_font(20)
    font_name = _load_font(16)

    draw.text((padding, 24), "今日抽卡 · " + time.strftime("%Y-%m-%d"), font=font_title, fill=(120, 220, 255))
    draw.text((padding, 76), f"今天共抽取 {len(records)} 次", font=font_tip, fill=(180, 180, 200))

    manifest = _load_manifest()
    manifest_map = {img["id"]: img for img in manifest.get("images", [])}

    grid_x, grid_y = padding, title_h
    for idx, rec in enumerate(records):
        row, col = divmod(idx, cols)
        x = grid_x + col * (card_w + card_gap)
        y = grid_y + row * (card_h + card_gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=16, fill=(45, 40, 55))
        fid = rec.get("file_id")
        if fid and fid in manifest_map:
            img_path = _images_dir() / manifest_map[fid]["filename"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as src:
                        thumb = _fit_thumb(src.convert("RGBA"), (thumb_size, thumb_size))
                    canvas.paste(thumb, (x + 12, y + 12), thumb)
                except Exception:
                    draw.rounded_rectangle(
                        (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                        radius=12, fill=(60, 55, 70),
                    )
            else:
                draw.rounded_rectangle(
                    (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                    radius=12, fill=(60, 55, 70),
                )
        else:
            draw.rounded_rectangle(
                (x + 12, y + 12, x + 12 + thumb_size, y + 12 + thumb_size),
                radius=12, fill=(60, 55, 70),
            )
        cname = str(rec.get("name", "未知"))
        if len(cname) > 8:
            cname = cname[:7] + "…"
        draw.text((x + 12, y + 12 + thumb_size + 6), cname, font=font_name, fill=(220, 220, 230))

    out = _cards_dir() / f"today_{user_id}.png"
    canvas.save(out, "PNG")
    return out


# ==================== 插件主类 ====================

@register(PLUGIN_NAME, "custom", "自定义抽卡", "v0.4.1")
class CustomGachaPlugin(Star):
    """自定义抽卡插件主类。"""

    def __init__(self, context: Context, config):
        super().__init__(context, config)
        self.config = config if hasattr(config, "get") else {}
        global _PLUGIN_CONFIG
        _PLUGIN_CONFIG = dict(self.config)
        self._register_web_apis()

    def _register_web_apis(self):
        p = PLUGIN_NAME
        self.context.register_web_api(f"/{p}/list", self.api_list, ["GET"], "图片列表")
        self.context.register_web_api(f"/{p}/upload", self.api_upload, ["POST"], "上传图片")
        self.context.register_web_api(f"/{p}/upload_zip", self.api_upload_zip, ["POST"], "上传压缩包")
        self.context.register_web_api(f"/{p}/analyze", self.api_analyze, ["POST"], "AI分析图片")
        self.context.register_web_api(f"/{p}/analyze_all", self.api_analyze_all, ["POST"], "批量AI分析")
        self.context.register_web_api(f"/{p}/analyze_all/start", self.api_analyze_all_start, ["POST"], "批量AI分析(异步)")
        self.context.register_web_api(f"/{p}/analyze_all/status/<job>", self.api_analyze_all_status, ["GET"], "批量AI分析进度")
        self.context.register_web_api(f"/{p}/analyze_all/cancel/<job>", self.api_analyze_all_cancel, ["POST"], "停止批量AI分析")
        self.context.register_web_api(f"/{p}/delete", self.api_delete, ["POST"], "删除图片")
        self.context.register_web_api(f"/{p}/update", self.api_update, ["POST"], "保存卡牌名与介绍")
        self.context.register_web_api(f"/{p}/image/<file_id>", self.api_image, ["GET"], "获取图片")
        self.context.register_web_api(f"/{p}/card/<file_id>", self.api_card, ["GET"], "获取卡片图")
        self.context.register_web_api(f"/{p}/overview", self.api_overview, ["GET"], "总览统计")
        self.context.register_web_api(f"/{p}/settings", self.api_settings_get, ["GET"], "读取设置")
        self.context.register_web_api(f"/{p}/settings", self.api_settings_set, ["POST"], "保存设置")
        self.context.register_web_api(f"/{p}/settings/providers", self.api_settings_providers, ["GET"], "官方 Provider 列表")
        self.context.register_web_api(f"/{p}/history/users", self.api_history_users, ["GET"], "用户列表")
        self.context.register_web_api(f"/{p}/history/search/<qq>", self.api_history_search, ["GET"], "按QQ搜索历史")
        self.context.register_web_api(f"/{p}/history/delete", self.api_history_delete, ["POST"], "删除抽卡记录")
        self.context.register_web_api(f"/{p}/history/<user_id>", self.api_history_detail, ["GET"], "用户历史详情")
        self.context.register_web_api(f"/{p}/history_card/<user_id>", self.api_history_card, ["GET"], "用户图鉴图")
        self.context.register_web_api(f"/{p}/folders", self.api_folders, ["GET"], "文件夹与群卡池")
        self.context.register_web_api(f"/{p}/folders/add", self.api_folder_add, ["POST"], "新建文件夹")
        self.context.register_web_api(f"/{p}/folders/move", self.api_folder_move, ["POST"], "图片移动文件夹")
        self.context.register_web_api(f"/{p}/group_folder", self.api_group_folder, ["POST"], "设置群卡池文件夹")

    def _get_triggers(self) -> list[str]:
        s = _load_settings()
        raw = s.get("trigger_words") or self.config.get("trigger_words", ["抽卡"])
        if isinstance(raw, str):
            raw = [raw]
        return [str(w).strip() for w in raw if str(w).strip()]

    def _event_ctx(self, event) -> tuple[str, str, str]:
        """返回 (group_id, user_id, name)；group_id 为空表示私聊。"""
        user_id = str(getattr(event, "get_sender_id", lambda: "")() or "")
        if not user_id:
            user_id = str(getattr(event, "sender_id", "") or "unknown")
        group_id = ""
        try:
            gid = getattr(event, "get_group_id", None)
            if callable(gid):
                group_id = str(gid() or "")
            else:
                group_id = str(getattr(event, "group_id", "") or "")
        except Exception:
            group_id = ""
        name = ""
        try:
            sender = getattr(event, "message_obj", None)
            if sender is not None:
                seg = getattr(sender, "sender", None)
                if seg is not None:
                    name = str(getattr(seg, "card", "") or getattr(seg, "nickname", "") or "")
        except Exception:
            name = ""
        return group_id, user_id, name

    def _is_super_admin(self, user_id) -> bool:
        s = _load_settings()
        su = s.get("super_users") or []
        if isinstance(su, str):
            su = [su]
        su_set = {str(x) for x in su}
        if str(user_id) in su_set:
            return True
        try:
            cfg = self.context.get_config()
            extra = cfg.get("supers_users") if hasattr(cfg, "get") else getattr(cfg, "supers_users", None)
            if extra:
                if isinstance(extra, str):
                    extra = [extra]
                for x in extra:
                    if str(user_id) == str(x):
                        return True
        except Exception:
            pass
        return False

    def _match_message(self, message_str: str) -> bool:
        text = message_str.strip()
        triggers = self._get_triggers()
        if not triggers:
            return False
        s = _load_settings()
        mode = s.get("match_mode") or self.config.get("match_mode", "exact")
        if mode == "starts_with":
            return any(text.startswith(t) for t in triggers)
        return text in triggers

    def _get_vision_provider(self) -> str:
        s = _load_settings()
        return str(s.get("vision_provider") or self.config.get("vision_model_provider", "") or "").strip()

    def _get_vision_model(self) -> str:
        s = _load_settings()
        return str(s.get("vision_model") or self.config.get("vision_model", "") or "").strip()

    async def _ensure_analyzed(self, item: dict) -> None:
        """如果图片尚未经过 AI 分析，则先分析。"""
        if item.get("name") and item.get("description"):
            return
        provider = self._get_vision_provider()
        if not provider:
            return
        img_path = _images_dir() / item["filename"]
        if not img_path.exists():
            return
        try:
            result = await _analyze_image(
                self.context, img_path, item.get("original_name", item["filename"]), provider,
                vision_model=self._get_vision_model(),
            )
            item["name"] = result["name"]
            item["description"] = result["description"]
            manifest = _load_manifest()
            for img in manifest.get("images", []):
                if img.get("id") == item["id"]:
                    img["name"] = result["name"]
                    img["description"] = result["description"]
                    break
            _save_manifest(manifest)
            logger.info(f"[custom_gacha] AI 分析完成: {item['id']} -> {result['name']}")
        except Exception as exc:
            logger.warning(f"[custom_gacha] AI 分析失败 {item.get('id')}: {exc}")

    def _get_or_render_card(self, item: dict) -> Path | None:
        """获取已缓存的卡片图，或现场渲染。"""
        name = item.get("name", "未知卡牌")
        desc = item.get("description", "")
        is_gif = item.get("is_gif", False)
        ext = ".gif" if is_gif else ".png"
        card_path = _cards_dir() / f"{item['id']}_card{ext}"
        if card_path.exists():
            return card_path
        img_path = _images_dir() / item["filename"]
        if not img_path.exists():
            return None
        try:
            return _render_card(img_path, name, desc, is_gif)
        except Exception as exc:
            logger.error(f"[custom_gacha] 渲染卡片失败: {exc}")
            return None

    async def _do_gacha(self, event: AstrMessageEvent, folder: str | None = None, record: bool = True):
        """执行抽卡逻辑。record=False 用于调试抽卡（不计数、不记记录、不设限）。"""
        group_id, user_id, name = self._event_ctx(event)
        if record and _check_daily_limit(user_id, group_id):
            await event.send(event.plain_result("你今天在本群已经抽过卡啦，明天再来吧 👋"))
            today_recs = _today_history_records(user_id, group_id)
            if today_recs:
                try:
                    img_path = _render_today_image(user_id, today_recs)
                    await event.send(event.image_result(str(img_path.absolute())))
                except Exception as exc:
                    logger.error(f"[custom_gacha] 今日卡图渲染失败: {exc}")
            return
        folder = folder or _resolve_folder(group_id)
        images = _pool_images(folder)
        if not images:
            await event.send(event.plain_result(f"当前卡池「{folder}」还没有图片，先去 WebUI / 调试里给它分配图片吧。"))
            return

        pick = random.choice(images)
        file_id = pick["id"]

        await self._ensure_analyzed(pick)

        card_path = self._get_or_render_card(pick)
        card_name = pick.get("name", "未知卡牌")

        if card_path and card_path.exists():
            await event.send(event.image_result(str(card_path.absolute())))
        else:
            img_path = _images_dir() / pick.get("filename", "")
            if img_path.exists():
                await event.send(event.image_result(str(img_path.absolute())))
            else:
                await event.plain_result("图片文件不存在。")

        if record:
            _record_draw(user_id, file_id, card_name, group_id=group_id)
            _record_board(group_id, user_id, name or f"QQ{user_id}", file_id)

    async def _send_history_image(self, event: AstrMessageEvent, folder: str | None = None):
        group_id, user_id, _name = self._event_ctx(event)
        history = _get_user_history(user_id)
        total = int(history.get("total", 0))
        if total == 0:
            await event.send(event.plain_result("你还没有抽过卡，发送唤醒词抽一张吧！"))
            return
        folder = folder or _resolve_folder(group_id)
        total_pool = len(_pool_images(folder))
        try:
            img_path = _render_history_image(user_id, total_pool=total_pool)
            await event.send(event.image_result(str(img_path.absolute())))
        except Exception as exc:
            logger.error(f"[custom_gacha] 渲染历史图鉴失败: {exc}")
            await event.send(event.plain_result(f"渲染图鉴失败: {exc}"))

    def _get_history_triggers(self) -> list[str]:
        s = _load_settings()
        raw = s.get("history_trigger_words") or self.config.get("history_trigger_words", ["图鉴", "我的图鉴", "抽卡图鉴"])
        if isinstance(raw, str):
            raw = [raw]
        return [str(w).strip() for w in raw if str(w).strip()]

    def _match_history_keyword(self, message_str: str) -> bool:
        text = message_str.strip()
        triggers = self._get_history_triggers()
        return any(text == t for t in triggers) if triggers else False

    def _match_board(self, text: str) -> str | None:
        s = _load_settings()
        for period, key in (("daily", "board_daily_word"), ("monthly", "board_monthly_word"), ("total", "board_total_word")):
            w = (s.get(key) or "").strip()
            if w and text == w:
                return period
        return None

    # ---------- 排行榜 ----------

    def _board_bucket(self, group_id: str, period: str) -> list[dict]:
        """返回按单卡被抽次数降序排列的列表，每项 {file_id, count}。"""
        b = _load_boards()
        g = (b.get("groups") or {}).get(str(group_id)) or {}
        if period == "daily":
            node = (g.get("day") or {}).get(time.strftime("%Y-%m-%d")) or {}
        elif period == "monthly":
            node = (g.get("month") or {}).get(time.strftime("%Y-%m")) or {}
        else:
            node = g.get("total") or {}
        card_totals: dict[str, int] = {}
        for _uid, rec in node.items():
            if not isinstance(rec, dict):
                continue
            for fid, cnt in (rec.get("cards") or {}).items():
                card_totals[str(fid)] = card_totals.get(str(fid), 0) + int(cnt or 0)
        rows = [{"file_id": fid, "count": cnt} for fid, cnt in card_totals.items()]
        rows.sort(key=lambda r: int(r.get("count", 0) or 0), reverse=True)
        return rows

    def _render_board_image(self, group_id: str, period: str) -> Path:
        period_names = {"daily": "日榜", "monthly": "月榜", "total": "总榜"}
        rows = self._board_bucket(group_id, period)[:20]
        title = f"抽卡榜单 · {period_names.get(period, period)}"

        W = 900
        HEAD = 120
        ROW_H = 88
        PAD = 40
        thumb = 60
        n = max(1, len(rows))
        H = HEAD + n * ROW_H + 30
        # 空则仍给一点高度
        canvas = Image.new("RGB", (W, H), (26, 24, 34))
        draw = ImageDraw.Draw(canvas)
        font_t = _load_font(42, bold=True)
        font_n = _load_font(26, bold=True)
        font_s = _load_font(20)

        draw.text((PAD, 34), title, font=font_t, fill=(120, 220, 255))
        span = f"{time.strftime('%Y-%m-%d')}"
        if period == "monthly":
            span = time.strftime("%Y-%m")
        draw.text((PAD, 88), f"统计范围:{span} · 本群共 {len(rows)} 种卡牌上榜", font=font_s, fill=(170, 170, 190))

        if not rows:
            draw.text((PAD, HEAD), "该时段群内还没有抽卡记录。", font=font_n, fill=(190, 190, 200))
            out = _cards_dir() / f"board_{group_id}_{period}.png"
            canvas.save(out, "PNG")
            return out

        manifest = _load_manifest()
        mf = {i["id"]: i for i in manifest.get("images", [])}
        for i, rec in enumerate(rows):
            y = HEAD + i * ROW_H
            cx = PAD + 34
            cy = y + ROW_H // 2
            rank = i + 1
            colors = ((255, 215, 0), (205, 205, 210), (205, 130, 70))
            fill = colors[rank - 1] if rank <= 3 else (60, 56, 74)
            draw.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=fill)
            rf = _load_font(26, bold=True)
            rb = rf.getbbox(str(rank))
            draw.text((cx - (rb[2] - rb[0]) // 2, cy - (rb[3] - rb[1]) // 2 - rb[1]), str(rank), font=rf, fill=(30, 28, 40))

            fid = str(rec.get("file_id") or "")
            card_name = "未知卡牌"
            img_path = None
            if fid and fid in mf:
                card_name = str(mf[fid].get("name") or mf[fid].get("filename") or "未知卡牌")
                p = _images_dir() / mf[fid]["filename"]
                if p.exists():
                    img_path = p

            draw.text((PAD + 86, y + 16), _ellips(card_name, font_n, 360), font=font_n, fill=(235, 235, 245))
            cnt = int(rec.get("count", 0) or 0)
            draw.text((PAD + 86, y + 50), f"被抽 {cnt} 次", font=font_s, fill=(170, 170, 190))

            tx = W - PAD - thumb
            if img_path:
                try:
                    with Image.open(img_path) as src:
                        th = _fit_thumb(src.convert("RGBA"), (thumb, thumb))
                    canvas.paste(th, (tx, cy - thumb // 2), th)
                    continue
                except Exception:
                    pass
            draw.rounded_rectangle((tx, cy - thumb // 2, tx + thumb, cy + thumb // 2), radius=10, fill=(50, 46, 64))

        out = _cards_dir() / f"board_{group_id}_{period}.png"
        canvas.save(out, "PNG")
        return out

    async def _send_board(self, event, group_id, period):
        if not group_id:
            await event.send(event.plain_result("排行榜仅在群内使用。"))
            return
        try:
            path = self._render_board_image(group_id, period)
            await event.send(event.image_result(str(path.absolute())))
        except Exception as exc:
            logger.error(f"[custom_gacha] 排行榜渲染失败: {exc}")
            await event.send(event.plain_result(f"生成排行榜失败: {exc}"))

    # ---------- 超级管理员调试会话（仅私聊） ----------

    def _debug_menu_text(self, user_id) -> str:
        folder = _DEBUG_POOL.get(user_id) or DEFAULT_FOLDER
        n = len(_pool_images(folder))
        return (
            "🎴 抽卡调试菜单（仅私聊）\n"
            f"当前调试卡池：{folder}（{n} 张）\n"
            "1️⃣ 切换调试卡池\n"
            "2️⃣ 设置某群的卡池（群号 + 文件夹）\n"
            "3️⃣ 测试抽卡（不计入记录）\n"
            "4️⃣ 查看卡池配置\n"
            "0️⃣ 退出\n"
            "回复数字即可继续。"
        )

    def _get_debug_trigger(self) -> str:
        s = _load_settings()
        return str(s.get("debug_trigger_word") or "自定义抽卡调试").strip() or "自定义抽卡调试"

    def _get_help_triggers(self) -> list[str]:
        s = _load_settings()
        raw = s.get("help_trigger_words") or ["帮助", "菜单", "抽卡帮助"]
        if isinstance(raw, str):
            raw = [raw]
        return [str(w).strip() for w in raw if str(w).strip()]

    def _match_help(self, text: str) -> bool:
        triggers = self._get_help_triggers()
        return bool(triggers) and text in triggers

    def _build_help_text(self) -> str:
        s = _load_settings()
        draws = self._get_triggers()
        hist = self._get_history_triggers()
        daily = str(s.get("board_daily_word") or "抽卡日榜").strip()
        monthly = str(s.get("board_monthly_word") or "抽卡月榜").strip()
        total = str(s.get("board_total_word") or "抽卡总榜").strip()
        lines = ["🎴 自定义抽卡 · 指令帮助", ""]
        if draws:
            lines.append("✨ 抽卡")
            lines.append("   发送「" + "」或「".join(draws[:6]) + "」即可抽卡")
            lines.append("")
        if hist:
            lines.append("📥 图鉴 / 抽卡记录")
            lines.append("   发送「" + "」或「".join(hist[:6]) + "」查看已收集卡牌与收集进度")
            lines.append("")
        board_lines = [x for x in (daily, monthly, total) if x]
        if board_lines:
            lines.append("🏆 排行榜")
            lines.append("   日榜「" + daily + "」 月榜「" + monthly + "」 总榜「" + total + "」")
        return "\n".join(lines)

    async def _debug_dispatch(self, event, text: str, user_id: str) -> bool:
        dbg = self._get_debug_trigger()
        if text in (dbg, "/" + dbg):
            _DEBUG_SESSION[user_id] = {"step": "menu"}
            await event.send(event.plain_result(self._debug_menu_text(user_id)))
            return True
        sess = _DEBUG_SESSION.get(user_id)
        if not sess or not sess.get("step"):
            return False
        await self._debug_handle(event, text.strip(), user_id, sess)
        return True

    async def _debug_handle(self, event, t: str, user_id: str, sess: dict) -> None:
        step = sess.get("step")
        if step == "menu":
            if t in ("1", "1️⃣", "切换卡池", "切换调试卡池"):
                names = _all_folder_names()
                exist = "\n".join(f"· {n}（{len(_pool_images(n))} 张）" for n in names)
                sess["step"] = "pick_folder"
                await event.send(event.plain_result(f"选择调试卡池文件夹（输入文件夹名，回复 0 返回菜单）：\n{exist}"))
                return
            if t in ("2", "2️⃣", "设置群卡池", "分配群卡池"):
                lst = " / ".join(_all_folder_names())
                sess["step"] = "pick_group_folder"
                await event.send(event.plain_result(f"输入「群号 文件夹名」设置该群卡池，例如：\n123456 Default\n可用文件夹：{lst}\n（回复 0 返回菜单）"))
                return
            if t in ("3", "3️⃣", "测试抽卡"):
                await self._do_gacha(event, folder=_DEBUG_POOL.get(user_id), record=False)
                return
            if t in ("4", "4️⃣", "查看配置"):
                s = _load_settings()
                gf = s.get("group_folders") or {}
                lines = [f"· 默认：「{DEFAULT_FOLDER}」（{len(_pool_images(DEFAULT_FOLDER))} 张）"]
                for name in _all_folder_names():
                    if name == DEFAULT_FOLDER:
                        continue
                    lines.append(f"· 「{name}」（{len(_pool_images(name))} 张）")
                for gid, f in gf.items():
                    lines.append(f"· 群 {gid} → 「{f}」（{len(_pool_images(f))} 张）")
                await event.send(event.plain_result("📁 卡池配置\n" + "\n".join(lines) + "\n\n回复 0 返回菜单"))
                return
            if t in ("0", "0️⃣", "退出", "结束"):
                _DEBUG_SESSION.pop(user_id, None)
                await event.send(event.plain_result("已退出抽卡调试。"))
                return
            await event.send(event.plain_result(self._debug_menu_text(user_id)))
            return
        if step == "pick_folder":
            if t in ("0", "返回", "菜单", "0️⃣"):
                sess["step"] = "menu"
                await event.send(event.plain_result(self._debug_menu_text(user_id)))
                return
            if t not in _all_folder_names():
                await event.send(event.plain_result(f"没有「{t}」这个文件夹。可用：{' / '.join(_all_folder_names())}"))
                return
            _DEBUG_POOL[user_id] = t
            sess["step"] = "menu"
            await event.send(event.plain_result(f"调试卡池已切换为「{t}」（{len(_pool_images(t))} 张）。\n" + self._debug_menu_text(user_id)))
            return
        if step == "pick_group_folder":
            if t in ("0", "返回", "菜单", "0️⃣"):
                sess["step"] = "menu"
                await event.send(event.plain_result(self._debug_menu_text(user_id)))
                return
            parts = [p for p in t.replace("，", " ").replace(",", " ").split() if p.strip()]
            gid = parts[0] if parts else ""
            folder_name = " ".join(parts[1:]) if len(parts) >= 2 else ""
            if not gid:
                await event.send(event.plain_result("请按「群号 文件夹名」格式输入，或回复 0 返回。"))
                return
            if folder_name and folder_name not in _all_folder_names():
                await event.send(event.plain_result(f"没有「{folder_name}」文件夹。可用：{' / '.join(_all_folder_names())}"))
                return
            if not folder_name:
                folder_name = DEFAULT_FOLDER
            _set_group_folder(gid, folder_name)
            sess["step"] = "menu"
            await event.send(event.plain_result(f"群 {gid} 的卡池已设为「{folder_name}」。\n" + self._debug_menu_text(user_id)))
            return

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def keyword_trigger(self, event: AstrMessageEvent):
        message_str = str(getattr(event, "message_str", "") or "").strip()
        group_id, user_id, _name = self._event_ctx(event)
        is_private = not bool(group_id)
        if not message_str:
            return
        if message_str.startswith(("!", "！")):
            return
        s = _load_settings()
        allow_direct = s.get("allow_direct_trigger")
        if allow_direct is None:
            allow_direct = self.config.get("allow_direct_trigger", True)

        # 超级管理员私有调试会话优先
        if is_private and self._is_super_admin(user_id):
            if await self._debug_dispatch(event, message_str, user_id):
                return
        if not allow_direct and not message_str.startswith("/"):
            return

        bare = message_str[1:].strip() if message_str.startswith("/") else message_str

        # 帮助（按自定义触发词动态生成）
        if self._match_help(message_str) or (bare and self._match_help(bare)):
            event.stop_event()
            await event.send(event.plain_result(self._build_help_text()))
            return
        # 图鉴
        if self._match_history_keyword(message_str) or bare and self._match_history_keyword(bare):
            event.stop_event()
            await self._send_history_image(event, folder=_resolve_folder(group_id))
            return
        # 排行榜
        period = self._match_board(message_str) or (bare and self._match_board(bare))
        if period:
            event.stop_event()
            await self._send_board(event, group_id, period)
            return
        # 抽卡
        if self._match_message(message_str) or (bare and self._match_message(bare)):
            event.stop_event()
            await self._do_gacha(event, folder=_resolve_folder(group_id), record=True)

    # ==================== Web API ====================

    async def api_list(self):
        manifest = _load_manifest()
        return json_response(manifest)

    async def api_upload(self):
        files = await request.files()
        upload: PluginUploadFile | None = files.get("file")
        if not isinstance(upload, PluginUploadFile):
            return error_response("缺少文件", status_code=400)

        folder = DEFAULT_FOLDER
        try:
            form = await request.form()
            folder = str((form.get("folder") or "").strip() or DEFAULT_FOLDER)
        except Exception:
            folder = DEFAULT_FOLDER
        if folder not in _all_folder_names():
            _save_folder_names(_all_folder_names() + [folder])

        ext = Path(upload.filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            return error_response(f"不支持的格式: {ext}", status_code=400)

        file_id = uuid.uuid4().hex[:12]
        stored_name = f"{file_id}{ext}"
        target = _images_dir() / stored_name
        await upload.save(target)

        is_gif = ext == ".gif" and _is_animated(target)
        manifest = _load_manifest()
        manifest.setdefault("images", []).append({
            "id": file_id,
            "filename": stored_name,
            "original_name": upload.filename,
            "size": target.stat().st_size,
            "is_gif": is_gif,
            "folder": folder,
            "name": "",
            "description": "",
        })
        _save_manifest(manifest)
        logger.info(f"[custom_gacha] 图片已上传: {stored_name} (gif={is_gif}, folder={folder})")
        return json_response({"id": file_id, "filename": stored_name, "is_gif": is_gif, "folder": folder})

    async def api_upload_zip(self):
        files = await request.files()
        upload: PluginUploadFile | None = files.get("file")
        if not isinstance(upload, PluginUploadFile):
            return error_response("缺少压缩包", status_code=400)

        ext = Path(upload.filename).suffix.lower()
        if ext not in (".zip",):
            return error_response("仅支持 .zip 格式", status_code=400)

        tmp_zip = _data_dir() / f"_tmp_{uuid.uuid4().hex[:8]}.zip"
        await upload.save(tmp_zip)
        folder = DEFAULT_FOLDER
        try:
            form = await request.form()
            folder = str((form.get("folder") or "").strip() or DEFAULT_FOLDER)
        except Exception:
            folder = DEFAULT_FOLDER
        if folder not in _all_folder_names():
            _save_folder_names(_all_folder_names() + [folder])
        results: list[dict] = []
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    member_ext = Path(info.filename).suffix.lower()
                    if member_ext not in SUPPORTED_IMAGE_EXTS:
                        continue
                    file_id = uuid.uuid4().hex[:12]
                    stored_name = f"{file_id}{member_ext}"
                    target = _images_dir() / stored_name
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    is_gif = member_ext == ".gif" and _is_animated(target)
                    original = Path(info.filename).name
                    manifest = _load_manifest()
                    manifest.setdefault("images", []).append({
                        "id": file_id,
                        "filename": stored_name,
                        "original_name": original,
                        "size": target.stat().st_size,
                        "is_gif": is_gif,
                        "folder": folder,
                        "name": "",
                        "description": "",
                    })
                    _save_manifest(manifest)
                    results.append({"id": file_id, "filename": stored_name, "original_name": original, "is_gif": is_gif, "folder": folder})
        except zipfile.BadZipFile:
            return error_response("压缩包损坏或格式不正确", status_code=400)
        finally:
            tmp_zip.unlink(missing_ok=True)

        logger.info(f"[custom_gacha] 压缩包导入完成: {len(results)} 张图片")
        return json_response({"imported": len(results), "items": results})

    async def api_analyze(self):
        payload = await request.json(default={})
        file_id = str(payload.get("id", "")).strip()
        if not file_id:
            return error_response("缺少 id", status_code=400)

        item = _find_in_manifest(file_id)
        if not item:
            return error_response("图片不存在", status_code=404)

        provider = self._get_vision_provider()
        if not provider:
            return error_response("未配置视觉模型 Provider，请在插件配置中设置 vision_model_provider", status_code=400)

        img_path = _images_dir() / item["filename"]
        if not img_path.exists():
            return error_response("图片文件不存在", status_code=404)

        try:
            result = await _analyze_image(self.context, img_path, item.get("original_name", item["filename"]), provider)
        except Exception as exc:
            return error_response(f"AI 分析失败: {exc}", status_code=500)

        manifest = _load_manifest()
        for img in manifest.get("images", []):
            if img.get("id") == file_id:
                img["name"] = result["name"]
                img["description"] = result["description"]
                break
        _save_manifest(manifest)
        return json_response(result)

    async def api_delete(self):
        payload = await request.json(default={})
        file_id = str(payload.get("id", "")).strip()
        if not file_id:
            return error_response("缺少 id", status_code=400)

        manifest = _load_manifest()
        images = manifest.get("images", [])
        target_item = None
        for item in images:
            if item.get("id") == file_id:
                target_item = item
                break
        if not target_item:
            return error_response("图片不存在", status_code=404)

        for fn in (target_item.get("filename"),):
            p = _images_dir() / fn
            p.unlink(missing_ok=True)
        for ext in (".gif", ".png"):
            cp = _cards_dir() / f"{file_id}_card{ext}"
            cp.unlink(missing_ok=True)

        manifest["images"] = [i for i in images if i.get("id") != file_id]
        _save_manifest(manifest)
        return json_response({"deleted": file_id})

    async def api_update(self):
        payload = await request.json(default={})
        file_id = str(payload.get("id", "")).strip()
        if not file_id:
            return error_response("缺少 id", status_code=400)
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", "")).strip()
        if not name:
            return error_response("卡牌名称不能为空", status_code=400)
        manifest = _load_manifest()
        for img in manifest.get("images", []):
            if img.get("id") == file_id:
                img["name"] = name
                img["description"] = description
                break
        else:
            return error_response("图片不存在", status_code=404)
        _save_manifest(manifest)
        for ext in (".gif", ".png"):
            (_cards_dir() / f"{file_id}_card{ext}").unlink(missing_ok=True)
        return json_response({"ok": True})

    async def api_image(self, file_id: str):
        item = _find_in_manifest(file_id)
        if not item:
            return error_response("图片不存在", status_code=404)
        img_path = _images_dir() / item["filename"]
        if not img_path.exists():
            return error_response("文件不存在", status_code=404)
        try:
            b64, mime = _image_to_base64(img_path, max_side=512)
            return json_response({"base64": b64, "mime_type": mime})
        except Exception as exc:
            return error_response(f"读取图片失败: {exc}", status_code=500)

    async def api_card(self, file_id: str):
        item = _find_in_manifest(file_id)
        if not item:
            return error_response("卡片不存在", status_code=404)

        card_path = self._get_or_render_card(item)
        if card_path and card_path.exists():
            try:
                b64, mime = _image_to_base64(card_path)
                return json_response({"base64": b64, "mime_type": mime})
            except Exception as exc:
                return error_response(f"读取卡片失败: {exc}", status_code=500)
        return error_response("卡片渲染失败", status_code=500)

    # -------------------- 总览 --------------------

    async def api_overview(self):
        manifest = _load_manifest()
        images = manifest.get("images", [])
        analyzed = sum(1 for i in images if i.get("name"))
        history = _load_history()
        users = history.get("users", {})
        total_draws = sum(int(u.get("total", 0)) for u in users.values())
        return json_response({
            "total_images": len(images),
            "analyzed_images": analyzed,
            "pending_images": len(images) - analyzed,
            "total_users": len(users),
            "total_draws": total_draws,
            "recent_users": sorted(
                [{"user_id": uid, "total": int(u.get("total", 0)),
                  "unique": len(u.get("cards", {})),
                  "last_name": (sorted(u.get("cards", {}).values(), key=lambda c: c.get("count", 0), reverse=True) or [{"name": "-"}])[0].get("name", "-")}
                 for uid, u in users.items()],
                key=lambda x: x["total"], reverse=True)[:10],
        })

    # -------------------- 设置 --------------------

    async def api_settings_get(self):
        s = _load_settings()
        return json_response(s)

    async def api_settings_providers(self):
        """返回 AstrBot 官方已配置的对话 Provider 列表，供设置页下拉选择。"""
        providers: list[dict] = []
        try:
            getter = getattr(self.context, "get_all_providers", None)
            if callable(getter):
                for provider in getter():
                    try:
                        meta = provider.meta()
                        pid = str(getattr(meta, "id", "") or "")
                        model = str(getattr(meta, "model", "") or "")
                    except Exception:
                        pid, model = str(getattr(provider, "provider_config", {}).get("id", "")), ""
                    if pid:
                        providers.append({"id": pid, "model": model})
        except Exception as exc:
            logger.warning(f"[custom_gacha] 获取 Provider 列表失败: {exc}")
        logger.info(f"[custom_gacha] 读取到 {len(providers)} 个官方 Provider")
        return json_response({"chat": providers})

    async def api_settings_set(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("格式错误", status_code=400)

        safe_keys = set(_SETTING_DEFAULTS.keys())
        patch = {}
        for k, v in payload.items():
            if k not in safe_keys:
                continue
            if k == "trigger_words" or k == "history_trigger_words":
                if isinstance(v, list):
                    patch[k] = [str(w).strip() for w in v if str(w).strip()]
                elif isinstance(v, str):
                    patch[k] = [w.strip() for w in v.replace("，", ",").split(",") if w.strip()]
                continue
            if k in ("match_mode", "vision_provider", "vision_model",
                     "analysis_system_prompt", "analysis_prompt_template"):
                if isinstance(v, str):
                    patch[k] = v.strip()
                continue
            if k == "allow_direct_trigger":
                patch[k] = bool(v)
                continue
            if k == "analysis_temperature":
                try:
                    patch[k] = max(0.0, min(2.0, float(v)))
                except (TypeError, ValueError):
                    patch[k] = 0.8
                continue
            if k == "analysis_max_tokens":
                try:
                    patch[k] = max(32, min(16000, int(v)))
                except (TypeError, ValueError):
                    patch[k] = 512
                continue
            if k == "daily_limit_count":
                try:
                    patch[k] = max(0, int(v))
                except (TypeError, ValueError):
                    patch[k] = 1
                continue
            if k == "super_users":
                if isinstance(v, str):
                    patch[k] = [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
                elif isinstance(v, list):
                    patch[k] = [str(x).strip() for x in v if str(x).strip()]
                continue
            if k == "folders":
                if isinstance(v, list):
                    patch[k] = list(dict.fromkeys(str(x).strip() for x in v if str(x).strip()))
                continue
            if k == "group_folders":
                if isinstance(v, dict):
                    patch[k] = {str(g): str(f) for g, f in v.items()}
                continue
            patch[k] = v
        _save_settings(patch)
        logger.info(f"[custom_gacha] 设置已更新")
        return json_response({"ok": True, "settings": _load_settings()})

    # -------------------- 历史记录 --------------------

    def _history_group_list(self) -> list[dict]:
        """按群号分组返回历史用户，避免一个人在多个群抽卡时混淆。"""
        history = _load_history()
        users = history.get("users", {})
        groups: dict[str, dict] = {}
        for gid in _group_cards_in_boards():
            groups.setdefault(str(gid), {"group_id": str(gid), "users": {}})
        for uid, u in users.items():
            cards = u.get("cards", {})
            top = (sorted(cards.values(), key=lambda c: c.get("count", 0), reverse=True) or [{"name": "-"}])[0]
            ugroups = u.get("groups")
            if isinstance(ugroups, dict) and ugroups:
                for gid, info in ugroups.items():
                    gg = groups.setdefault(str(gid), {"group_id": str(gid), "users": {}})
                    gg["users"][str(uid)] = {
                        "user_id": str(uid),
                        "total": int((info or {}).get("draws", 0) if isinstance(info, dict) else 0),
                        "unique": len(cards),
                        "last_name": top.get("name", "-"),
                    }
            else:
                gg = groups.setdefault("", {"group_id": "", "users": {}})
                gg["users"][str(uid)] = {
                    "user_id": str(uid),
                    "total": int(u.get("total", 0)),
                    "unique": len(cards),
                    "last_name": top.get("name", "-"),
                }
        out = []
        for gid, gg in groups.items():
            users_list = sorted(gg["users"].values(), key=lambda x: x["total"], reverse=True)
            out.append({"group_id": gid, "users": users_list})
        out.sort(key=lambda g: str(g["group_id"]))
        return out

    async def api_history_users(self):
        return json_response({"groups": self._history_group_list()})

    async def api_history_search(self, qq: str):
        query = qq.strip()
        if not query:
            return error_response("缺少 QQ", status_code=400)
        groups = self._history_group_list()
        matched = []
        for g in groups:
            us = [u for u in g["users"] if query in str(u["user_id"])]
            if us:
                matched.append({"group_id": g["group_id"], "users": us})
        return json_response({"query": query, "groups": matched})

    async def api_history_delete(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("参数错误", status_code=400)
        user_id = str(payload.get("user_id", "")).strip()
        group_id = str(payload.get("group_id", "")).strip()
        if not user_id:
            return error_response("缺少 user_id", status_code=400)
        history = _load_history()
        users = history.get("users", {})
        existed = user_id in users
        if existed:
            del users[user_id]
            _save_history(history)
        b = _load_boards()
        broot = b.get("groups") or {}
        gids = [group_id] if group_id else list(broot.keys())
        removed_board = False
        for gid in gids:
            g = broot.get(str(gid))
            if not isinstance(g, dict):
                continue
            for node in _iter_board_buckets(g):
                if str(user_id) in node:
                    del node[str(user_id)]
                    removed_board = True
        if removed_board:
            _save_boards(b)
        return json_response({"ok": True, "deleted": existed or removed_board})

    async def api_history_detail(self, user_id: str):
        uid = user_id.strip()
        history = _get_user_history(uid)
        if not history.get("cards"):
            return error_response("该用户暂无记录", status_code=404)
        manifest = _load_manifest()
        manifest_map = {img["id"]: img for img in manifest.get("images", [])}
        cards = []
        for fid, c in history.get("cards", {}).items():
            cards.append({
                "id": fid,
                "name": c.get("name", "未知"),
                "count": int(c.get("count", 0)),
                "has_image": fid in manifest_map,
                "original_name": manifest_map.get(fid, {}).get("original_name", ""),
            })
        cards.sort(key=lambda c: c["count"], reverse=True)
        return json_response({
            "user_id": uid,
            "total": int(history.get("total", 0)),
            "unique": len(cards),
            "cards": cards,
        })

    async def api_history_card(self, user_id: str):
        uid = user_id.strip()
        history = _get_user_history(uid)
        if not history.get("cards"):
            return error_response("该用户暂无记录", status_code=404)
        try:
            img_path = _render_history_image(uid, total_pool=len(_pool_images(DEFAULT_FOLDER)))
            b64, mime = _image_to_base64(img_path)
            return json_response({"base64": b64, "mime_type": mime})
        except Exception as exc:
            return error_response(f"渲染图鉴失败: {exc}", status_code=500)

    # -------------------- 文件夹 / 群卡池 --------------------

    async def api_folders(self):
        manifest = _load_manifest()
        images = manifest.get("images", [])
        per_folder: dict[str, int] = {}
        for i in images:
            f = i.get("folder") or DEFAULT_FOLDER
            per_folder[f] = per_folder.get(f, 0) + 1
        gf = _load_settings().get("group_folders") or {}
        groups = {}
        for gid, f in gf.items():
            groups[gid] = {"folder": f, "images": per_folder.get(f, 0)}
        return json_response({
            "folders": [{"name": n, "images": per_folder.get(n, 0)} for n in _all_folder_names()],
            "group_folders": groups,
            "known_groups": _group_cards_in_boards(),
        })

    async def api_folder_add(self):
        payload = await request.json(default={})
        name = str(payload.get("name", "")).strip()
        if not name:
            return error_response("缺少文件夹名", status_code=400)
        names = _all_folder_names()
        if name in names:
            return json_response({"ok": True, "folders": names})
        _save_folder_names(names + [name])
        return json_response({"ok": True, "folders": _all_folder_names()})

    async def api_folder_move(self):
        payload = await request.json(default={})
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        folder = str(payload.get("folder", "")).strip() if isinstance(payload, dict) else ""
        if not ids or not folder:
            return error_response("缺少参数", status_code=400)
        if folder not in _all_folder_names():
            return error_response("文件夹不存在", status_code=400)
        manifest = _load_manifest()
        moved = 0
        for img in manifest.get("images", []):
            if img.get("id") in ids:
                img["folder"] = folder
                moved += 1
        _save_manifest(manifest)
        return json_response({"ok": True, "moved": moved})

    async def api_group_folder(self):
        payload = await request.json(default={})
        group_id = str(payload.get("group_id", "")).strip() if isinstance(payload, dict) else ""
        folder = str(payload.get("folder", "")).strip() if isinstance(payload, dict) else ""
        if not group_id:
            return error_response("缺少群号", status_code=400)
        if folder and folder not in _all_folder_names():
            return error_response("文件夹不存在", status_code=400)
        _set_group_folder(group_id, folder)
        return json_response({"ok": True, "group_id": group_id, "folder": folder or "未分配"})

    # -------------------- 批量分析 --------------------

    async def api_analyze_all(self):
        """兼容旧的批量分析接口：同步等待全部完成后返回汇总。"""
        payload = await request.json(default={})
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        provider = self._get_vision_provider()
        if not provider:
            return error_response("未配置视觉模型 Provider，请先在设置页配置", status_code=400)
        manifest = _load_manifest()
        targets = [i for i in manifest.get("images", []) if i.get("id") in ids] if ids else [i for i in manifest.get("images", []) if not i.get("name")]
        if not targets:
            return json_response({"ok": 0, "fail": 0, "message": "没有待分析的图片"})
        job_id = uuid.uuid4().hex[:12]
        _BATCH_JOBS[job_id] = {
            "id": job_id, "total": len(targets), "done": 0, "ok": 0, "fail": 0,
            "current": "", "finished": False, "message": "", "results": [],
        }
        await self._run_batch(job_id, targets)
        job = _BATCH_JOBS[job_id]
        return json_response({"ok": job["ok"], "fail": job["fail"], "results": job["results"]})

    async def api_analyze_all_start(self):
        payload = await request.json(default={})
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        provider = self._get_vision_provider()
        if not provider:
            return error_response("未配置视觉模型 Provider，请先在设置页配置", status_code=400)

        manifest = _load_manifest()
        if ids:
            targets = [i for i in manifest.get("images", []) if i.get("id") in ids]
        else:
            targets = [i for i in manifest.get("images", []) if not i.get("name")]
        _batch_jobs_cleanup()
        job_id = uuid.uuid4().hex[:12]
        _BATCH_JOBS[job_id] = {
            "id": job_id,
            "total": len(targets),
            "done": 0,
            "ok": 0,
            "fail": 0,
            "current": "",
            "finished": False,
            "message": "",
            "results": [],
        }
        if not targets:
            job = _BATCH_JOBS[job_id]
            job["finished"] = True
            job["end_time"] = time.time()
            job["message"] = "没有待分析的图片"
            return json_response({"job": job_id})
        asyncio.create_task(self._run_batch(job_id, targets))
        return json_response({"job": job_id, "total": len(targets)})

    async def api_analyze_all_status(self, job: str):
        job_data = _BATCH_JOBS.get(job)
        if not job_data:
            return error_response("任务不存在或已过期", status_code=404)
        return json_response(job_data)

    async def api_analyze_all_cancel(self, job: str):
        job_data = _BATCH_JOBS.get(job)
        if not job_data:
            return error_response("任务不存在或已过期", status_code=404)
        job_data["cancelled"] = True
        return json_response({"ok": True})

    async def _run_batch(self, job_id: str, targets: list[dict]):
        job = _BATCH_JOBS[job_id]
        vision_model = self._get_vision_model()
        provider = self._get_vision_provider()
        manifest = _load_manifest()
        for item in targets:
            if job.get("cancelled"):
                break
            job["current"] = item.get("original_name", item.get("filename", ""))
            img_path = _images_dir() / item["filename"]
            if not img_path.exists():
                job["fail"] += 1
                job["done"] += 1
                job["results"].append({"id": item["id"], "ok": False, "error": "文件缺失"})
                continue
            try:
                result = await _analyze_image(
                    self.context, img_path, item.get("original_name", item["filename"]), provider,
                    vision_model=vision_model,
                )
                for img in manifest.get("images", []):
                    if img.get("id") == item["id"]:
                        img["name"] = result["name"]
                        img["description"] = result["description"]
                        break
                _save_manifest(manifest)
                job["ok"] += 1
                job["results"].append({"id": item["id"], "ok": True, "name": result["name"]})
            except Exception as exc:
                job["fail"] += 1
                job["results"].append({"id": item["id"], "ok": False, "error": str(exc)})
            job["done"] += 1
        job["current"] = ""
        job["finished"] = True
        job["end_time"] = time.time()
        job["message"] = ("已停止" if job.get("cancelled") else "批量分析完成") + \
            f"：成功 {job['ok']}，失败 {job['fail']}"
        logger.info(f"[custom_gacha] 批量分析结束(取消={job.get('cancelled')}): ok={job['ok']} fail={job['fail']}")
