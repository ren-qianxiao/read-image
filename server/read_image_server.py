#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read-image —— 让纯文本主模型"看见"图片。

原理：
    本地图片 --(PIL 缩放/压缩)--> base64 data URL --> 视觉模型(OpenAI 兼容接口)
                                                      |
                              主模型 <-- 文字结果 <----+

主模型（如 deepseek 这类纯文本模型）本身没有视觉通道，本服务器把图片编码后
外包给视觉模型，再把模型的文字回答原样交回主模型使用。

通过 stdio 提供 MCP 工具：read_image / read_image_doctor
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

from PIL import Image, ImageOps

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# ----------------------------------------------------------------- 配置

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 按"新一代 -> 兜底"排序。READ_IMAGE_MODEL 若设置会排到最前面。
FALLBACK_MODELS = [
    "glm-5.3-flash",        # 多模态主力
    "glm-5v-turbo",         # 专用视觉模型
    "glm-4.6v-flash",       # 上一代视觉 flash
    "glm-4.6v",
    "glm-4v-flash",         # 经典免费视觉模型
]

KEY_ENV_NAMES = [
    "READ_IMAGE_API_KEY",
    "ZHIPU_API_KEY",
    "GLM_API_KEY",
    "BIGMODEL_API_KEY",
    "ZHIPUAI_API_KEY",
]

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

# 远程图片下载上限。必须有：否则指向超大文件的 URL 会把进程内存撑爆，
# stdio 服务被 OOM kill 后整个工具就离线了。
MAX_REMOTE_BYTES = 64 * 1024 * 1024

# 单次调用最多几张图（防止把上下文塞爆）
MAX_IMAGES_PER_CALL = 8

MODE_PROMPTS = {
    "describe": (
        "详细描述这张图片的内容。包括：主体对象、场景、构图、颜色、"
        "可辨识的文字、以及任何值得注意的细节。用中文回答，直接描述，不要客套。"
    ),
    "ocr": (
        "提取图片中的全部文字，严格保持原有的阅读顺序和层次结构"
        "（标题/段落/表格/列表用 Markdown 表示）。不要翻译，不要总结，"
        "不要添加图片里没有的内容。"
    ),
    "ui": (
        "这是一个界面截图。描述它的布局结构：有哪些区域、每个区域里有什么控件、"
        "控件的文字标签、排列方式（水平/垂直/网格）、以及配色。"
        "用中文，按从上到下、从左到右的顺序描述。"
    ),
    "data": (
        "这是一张图表或表格的截图。请把其中的数据完整提取出来，"
        "用 Markdown 表格呈现；同时说明图表的类型、坐标轴含义、图例、单位。"
    ),
}


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return default


def _as_int(value, default: int = 0) -> int:
    """宽松取整：网关把字段返回成 {"total": 5} 或 "5" 都不该炸。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _num(name: str, default: float, cast=float, warn: list | None = None):
    """读数字类环境变量：写错格式只警告并回退默认值，绝不让 config() 抛异常。

    config() 是两个工具的第一行，一旦抛异常两个工具会同时彻底失效——
    而 TRUNCATE_NOTICE 恰恰引导用户去调 READ_IMAGE_MAX_TOKENS，
    所以这里必须容错（"8,192" / "90s" / "abc" 都要能活下来）。
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return cast(default)
    cleaned = str(raw).strip().replace(",", "").replace("_", "")
    for suffix in ("s", "秒", "px"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    try:
        return cast(float(cleaned))
    except (TypeError, ValueError):
        if warn is not None:
            warn.append(f"环境变量 {name}={raw!r} 不是合法数字，已回退默认值 {cast(default)}")
        return cast(default)


# 环境变量兜底：有些宿主不会把父进程的环境变量透传给 MCP 子进程，
# 此时从配置文件读。环境变量优先，文件只做后备。
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".claude", "read-image.json")


def file_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_file_config(api_key: str, model: str | None = None,
                     base_url: str | None = None) -> str:
    cfg = file_config()
    cfg["api_key"] = api_key.strip()
    if model:
        cfg["model"] = model.strip()
    if base_url:
        cfg["base_url"] = base_url.strip().rstrip("/")
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    try:                                    # 尽量别让密钥对同机其他用户可读
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass
    return CONFIG_PATH


def config() -> dict:
    """读取运行配置。每次调用都重新读，便于热改。环境变量优先，配置文件兜底。

    本函数**不会抛异常**——它是两个工具的第一行，抛了就等于工具全废。
    数字类变量写错只警告并回退默认值。
    """
    warns: list[str] = []
    fc = file_config()
    key = _env(*KEY_ENV_NAMES) or (str(fc.get("api_key") or "").strip() or None)
    models: list[str] = []
    pinned = _env("READ_IMAGE_MODEL") or (str(fc.get("model") or "").strip() or None)
    if pinned:
        models.append(pinned)
    for m in FALLBACK_MODELS:
        if m not in models:
            models.append(m)
    src = next((n for n in KEY_ENV_NAMES if (os.environ.get(n) or "").strip()), None)
    if not src and key:
        src = f"配置文件 {CONFIG_PATH}"
    return {
        "api_key": key,
        "key_source": src,
        "base_url": (_env("READ_IMAGE_BASE_URL")
                     or (str(fc.get("base_url") or "").strip() or None)
                     or DEFAULT_BASE_URL).rstrip("/"),
        "models": models,
        "timeout": _num("READ_IMAGE_TIMEOUT", 90, float, warns),
        "max_side": _num("READ_IMAGE_MAX_SIDE", 1568, int, warns),
        # 推理模型会先烧掉一截预算在思维链上，默认给宽一点
        "max_tokens": _num("READ_IMAGE_MAX_TOKENS", 4096, int, warns),
        "warnings": warns,
    }


# ----------------------------------------------------------------- 图片编码


def encode_image(source: str, max_side: int, max_raw_bytes: int = 3_500_000) -> tuple[str, str]:
    """把图片编码成 data URL。返回 (data_url, 说明文字)。

    自动处理：超大分辨率降采样、超大体积转 JPEG 并逐步降质，
    确保 base64 之后的体积仍在视觉模型的接收上限内。
    """
    note = []

    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "read-image-mcp/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 必须有上限：否则一个指向超大文件的 URL 能把进程内存一次撑爆，
            # stdio 服务被 OOM kill 后工具会直接离线。
            raw = resp.read(MAX_REMOTE_BYTES + 1)
        if len(raw) > MAX_REMOTE_BYTES:
            raise ValueError(
                f"远程图片超过 {MAX_REMOTE_BYTES // 1024 // 1024}MB 上限，已拒绝下载。"
                f"请先保存到本地再读取。"
            )
        img = Image.open(io.BytesIO(raw))
        note.append(f"来自 URL（{len(raw) / 1024:.0f}KB）")
    else:
        path = os.path.expanduser(source)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到图片文件：{source}")
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in SUPPORTED_EXT:
            note.append(f"扩展名 {ext} 不在常用列表内，仍尝试解析")
        img = Image.open(path)

    orig_size = img.size

    # 动图只取第一帧。
    # 注意顺序：必须在 exif_transpose 之前——它在没有方向标签时会返回 image.copy()，
    # 而 copy() 会丢掉 n_frames，放在后面这个分支就永远不触发了。
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
        note.append("动图仅取首帧")

    # 手机照片靠 EXIF Orientation 标记朝向，PIL 打开时不会自动旋转。
    # 不做这一步，竖拍照片会以横躺的姿态交给视觉模型，OCR 顺序和图表读数全错。
    img = ImageOps.exif_transpose(img)

    # 统一成 RGB（丢掉 alpha 时用白底，避免出现黑块）
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 降采样
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        note.append(f"缩放 {w}x{h} -> {img.size[0]}x{img.size[1]}")

    def _enc(im: Image.Image, image_format: str, **kw) -> bytes:
        buf = io.BytesIO()
        im.save(buf, image_format, **kw)
        return buf.getvalue()

    data = _enc(img, "PNG", optimize=True)
    mime = "image/png"

    if len(data) > max_raw_bytes:
        quality = 88
        used = None
        while True:
            data = _enc(img, "JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"
            used = quality
            if len(data) <= max_raw_bytes or quality <= 45:
                break
            # 还是太大就降一档质量，并把分辨率也缩一档
            nw, nh = max(1, int(img.size[0] * 0.8)), max(1, int(img.size[1] * 0.8))
            img = img.resize((nw, nh), Image.LANCZOS)
            quality -= 12
        # 报告实际用来编码的那一档，否则按 note 排查体积问题会被带偏
        note.append(f"压缩为 JPEG q={used}")

    if orig_size != img.size:
        note.append(f"原图 {orig_size[0]}x{orig_size[1]}")

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", "；".join(note)


# ----------------------------------------------------------------- 调用视觉模型


class VisionError(RuntimeError):
    pass


def _post(url: str, payload: dict, api_key: str, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read()
    except urllib.error.URLError as e:
        raise VisionError(f"网络不通：{e.reason}") from e
    except (OSError, TimeoutError) as e:          # 读一半断流 / 超时
        raise VisionError(f"连接中断：{type(e).__name__}: {e}") from e

    try:
        return status, json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError) as e:
        # 网关/代理可能回一个 200 的 HTML 错误页。统一转成 VisionError，
        # 这样 _call_once 能把它当成"该候选失败"继续下探，而不是穿出去炸掉整条链。
        snippet = raw[:200].decode("utf-8", "replace").replace("\n", " ")
        raise VisionError(f"网关返回了非 JSON 响应（HTTP {status}）：{snippet}") from e


def _extract_text(payload: dict) -> tuple[str, str, int]:
    """解析回答。返回 (正文, finish_reason, 推理消耗的 token 数)。

    注意：GLM-5.3-Flash 这类是推理模型，会先产出 reasoning_content（思维链），
    再产出 content。思维链同样占用 max_tokens，预算不够时 content 会是空串、
    finish_reason 为 'length'——这种情况绝不能当成"模型不可用"。
    """
    choices = payload.get("choices") or []
    if not choices:
        raise VisionError("视觉模型没有返回 choices")
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # 有些实现返回分片数组
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ).strip()
    else:
        text = (content or "").strip()

    finish = str(choice.get("finish_reason") or "")
    details = (payload.get("usage") or {}).get("completion_tokens_details") or {}
    reasoning = _as_int(details.get("reasoning_tokens"))
    return text, finish, reasoning


def _build_content(data_urls: list[str], prompt: str, detail: str, style: str) -> list[dict]:
    """组装消息内容。

    各家网关对 image_url.url 的写法要求不一致：有的要 data:image/png;base64,xxx，
    有的只吃裸 base64。这里按 style 生成两种形态，出错时自动换另一种重试。
    """
    content: list[dict] = []
    for url in data_urls:
        value = url.split(",", 1)[1] if (style == "bare" and "," in url) else url
        item: dict = {"url": value}
        if detail and detail != "auto":
            item["detail"] = detail
        content.append({"type": "image_url", "image_url": item})
    content.append({"type": "text", "text": prompt})
    return content


def _looks_like_format_error(status: int, resp: dict) -> bool:
    """判断是否是"图片字段格式不对"这类可换写法重试的错误。"""
    if status != 400:
        return False
    text = json.dumps(resp, ensure_ascii=False).lower()
    return any(k in text for k in ("base64", "image", "url", "format", "invalid"))


def _describe_response_error(status: int, resp: dict) -> str:
    code = str(resp.get("error", {}).get("code") or resp.get("code") or status)
    message = str(
        resp.get("error", {}).get("message") or resp.get("message") or resp
    )[:200]
    return f"[{status}/{code}] {message}"


TRUNCATE_NOTICE = (
    "\n\n> ⚠️ 输出被 max_tokens 截断，以上内容不完整。"
    "可调大环境变量 READ_IMAGE_MAX_TOKENS 后重试。"
)


def _call_once(cfg: dict, candidate: str, data_urls: list[str], prompt: str,
               detail: str, style: str, max_tokens: int) -> tuple[str, str]:
    """发一次请求。返回 (结果类型, 文本)。

    结果类型：'ok'（文本是回答）/ 'format'（图片字段写法不对，可换 style 重试）
             / 'fail'（文本是错误说明）
    """
    budget = max_tokens
    # 三种"再来一次"的理由各自只生效一次，互不占用配额——
    # 早先共用 attempt 槽时，一次 429 抖动就会把"加大预算"的机会吃掉。
    escalated = backoff_used = clamped = False

    while True:
        payload = {
            "model": candidate,
            "messages": [
                {"role": "user", "content": _build_content(data_urls, prompt, detail, style)}
            ],
            "max_tokens": budget,
            "temperature": 0.2,
        }
        try:
            status, resp = _post(f"{cfg['base_url']}/chat/completions",
                                 payload, cfg["api_key"], cfg["timeout"])
        except VisionError as e:
            # 该候选的网络/非 JSON 问题只淘汰它自己，不断整条链
            return ("fail", str(e))

        if status == 200:
            try:
                text, finish, reasoning = _extract_text(resp)
            except VisionError as e:
                return ("fail", str(e))
            if text:
                # 被截断的结果必须显式标注——静默返回半截内容比报错更危险
                return ("ok", text + TRUNCATE_NOTICE if finish == "length" else text)

            # 正文为空：推理模型把预算全用在思维链上了，加大预算再试。
            # finish=='length' 同样是预算耗尽的信号，即便网关没给 reasoning_tokens 也要认。
            if not escalated and (reasoning or finish == "length"):
                escalated = True
                budget = min(max(budget * 4, 1024), 32768)
                continue
            if finish == "length" or reasoning:
                return ("fail", f"正文为空：{reasoning} tokens 被推理过程占用，"
                                f"预算 {budget} 不够")
            return ("fail", "返回内容为空")

        if status in (429, 500, 502, 503, 504) and not backoff_used:
            backoff_used = True
            time.sleep(2.0)          # 限流 / 服务端抖动 -> 退避后重试一次
            continue

        msg = _describe_response_error(status, resp)

        # 各模型的 max_tokens 上限并不一致（如 glm-4v-flash 只到 1024）。
        # 撞上上限时按网关报的范围裁剪后重试，而不是把这个模型判死。
        if status == 400 and "max_token" in msg.lower() and not clamped:
            clamped = True
            found = re.search(r"\[\s*\d+\s*,\s*(\d+)\s*\]", msg)
            budget = min(budget, int(found.group(1)) if found else 1024)
            continue

        if _looks_like_format_error(status, resp):
            return ("format", msg)
        return ("fail", msg)


def ask_vision(data_urls: list[str], prompt: str, cfg: dict,
               model: str | None = None, detail: str = "auto",
               max_tokens: int | None = None) -> tuple[str, str]:
    """问视觉模型。返回 (回答, 实际使用的模型)。"""
    if max_tokens is None:
        max_tokens = cfg.get("max_tokens", 4096)
    if not cfg["api_key"]:
        raise VisionError(
            "没有配置视觉模型的 API Key。请设置环境变量其中之一："
            + " / ".join(KEY_ENV_NAMES)
            + "\n智谱 key 申请地址：https://open.bigmodel.cn/usercenter/apikeys"
        )

    candidates = [model] if model else list(cfg["models"])
    errors: list[str] = []

    for candidate in candidates:
        for style in ("dataurl", "bare"):
            outcome, text = _call_once(cfg, candidate, data_urls, prompt,
                                       detail, style, max_tokens)
            if outcome == "ok":
                return text, candidate
            if outcome == "format" and style == "dataurl":
                # 该网关不吃 data: 前缀，换裸 base64 再试一次
                errors.append(f"{candidate}[dataurl]: {text} → 改试裸 base64")
                continue
            errors.append(f"{candidate}[{style}]: {text}")
            break                     # 这个模型不行，换下一个

    raise VisionError(
        "所有候选视觉模型都调用失败：\n  "
        + "\n  ".join(errors)
        + "\n可用 READ_IMAGE_MODEL 指定一个确定的模型名；"
        + "或设置 READ_IMAGE_BASE_URL 指向你的网关。"
    )


# ----------------------------------------------------------------- MCP 工具

mcp = FastMCP("read-image")


@mcp.tool(
    title="读取图片内容",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
)
def read_image(
    paths: list[str],
    question: str = "",
    mode: str = "describe",
    detail: str = "auto",
) -> str:
    """读取一张或多张图片的内容，返回文字结果。

    图片会被转成 base64 交给视觉模型识别，因此可以读取当前模型本身无法直接查看的图片。
    支持本地路径和 http(s) 链接；支持 png/jpg/webp/gif/bmp/tiff。

    Args:
        paths: 图片路径列表（Windows 路径如 D:\\a\\b.png 或相对路径均可），也可传 http(s) 链接。
        question: 想针对图片问什么。留空则按 mode 的默认方式描述。例如"图里第三个柱子的数值是多少"。
        mode: 预设提问方式，可选 describe（整体描述）/ ocr（只提取文字）/
              ui（界面布局）/ data（图表取数）。默认 describe。
        detail: 传给视觉模型的清晰度，auto / low / high，默认 auto。细节密集的截图建议 high。

    Returns:
        Markdown 文本，含每张图的识别结果；调用失败时返回可操作的错误说明。
    """
    cfg = config()

    if not paths:
        return "错误：paths 为空，至少要给一个图片路径。"
    if len(paths) > MAX_IMAGES_PER_CALL:
        return (f"错误：一次最多 {MAX_IMAGES_PER_CALL} 张图，收到 {len(paths)} 张。"
                f"请分批调用。")

    mode_key = (mode or "describe").strip().lower()
    if mode_key not in MODE_PROMPTS:
        return f"错误：mode 只能是 {sorted(MODE_PROMPTS)} 之一，收到 {mode!r}。"

    prompt = MODE_PROMPTS[mode_key]
    if question.strip():
        prompt += f"\n\n另外请特别回答这个问题：{question.strip()}"

    data_urls: list[str] = []
    notes: list[str] = []
    for p in paths:
        try:
            url, note = encode_image(p, cfg["max_side"])
            data_urls.append(url)
            notes.append(f"- `{p}`" + (f"（{note}）" if note else ""))
        except Exception as e:  # noqa: BLE001 - 单张失败不应中断整批
            notes.append(f"- `{p}` 读取失败：{e}")

    if not data_urls:
        return "错误：没有任何图片读取成功。\n" + "\n".join(notes)

    try:
        answer, used = ask_vision(data_urls, prompt, cfg, detail=detail)
    except VisionError as e:
        return f"调用视觉模型失败：\n{e}"

    pinned = cfg["models"][0] if cfg["models"] else None
    header = f"（视觉模型 `{used}` 识别 {len(data_urls)} 张图）"
    if pinned and used != pinned:
        # 静默换模型会让"按免费模型配置却按付费模型计费"这类事无从察觉
        header += f" ⚠️ 首选模型 `{pinned}` 未成功，已自动回退到 `{used}`"
    return f"{header}\n\n" + "\n".join(notes) + "\n\n---\n\n" + answer


def _model_error_line(err_text: str, model: str) -> str:
    """从 VisionError 的多行消息里挑出针对该模型的最终错误。

    错误消息格式：
        所有候选视觉模型都调用失败：
          <model>[<style>]: <真正的错误>
          ...
        可用 READ_IMAGE_MODEL 指定...（通用建议，对诊断没价值）

    两个坑：
      · 直接取最后一行永远拿到那句通用建议；
      · 一个模型可能有多条（先 dataurl 后 bare），要取最后一条才是最终结论，
        否则会把"内部换编码重试"的中间记录当成该模型的最终错误报给用户。
    """
    hits = [ln.strip() for ln in err_text.splitlines() if ln.strip().startswith(f"{model}[")]
    if hits:
        rest = hits[-1][len(model):].strip()
        while rest.startswith("["):            # 依次剥掉 [style] [status/code]
            end = rest.find("]")
            if end < 0:
                break
            rest = rest[end + 1:].lstrip(":：").strip()
        return rest[:180] or "失败"
    for line in err_text.splitlines()[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("可用"):
            return stripped[:180]
    return err_text.splitlines()[-1][:180] if err_text.strip() else "失败"


@mcp.tool(
    title="检查 read-image 配置",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
)
def read_image_doctor(probe: bool = True) -> str:
    """诊断 read-image 的配置：检查 API Key、接口地址，并（可选）实测哪些视觉模型可用。

    图片读不出来、或提示模型不存在时，先跑这个。

    实测会按候选顺序逐个试，**遇到第一个可用的就停**（不必把队列全打一遍）。
    探测用的预算与真实调用一致，所以结论能代表 read_image 的实际行为。

    Args:
        probe: 是否真的发一次请求测试各候选模型。默认 True。设为 False 只做静态检查。

    Returns:
        配置检查报告：Key 来源、网关地址、各候选模型的实测结果。
    """
    cfg = config()
    lines = ["# read-image 诊断", ""]

    key = cfg["api_key"]
    if key:
        lines.append(f"- API Key：已配置（{key[:6]}…{key[-4:]}，共 {len(key)} 字符）")
        lines.append(f"  来源：`{cfg['key_source']}`")
    else:
        lines.append("- API Key：**未配置** ✗")
        lines.append(f"  方式一：设置环境变量 {' / '.join(KEY_ENV_NAMES)}")
        lines.append(f"  方式二：调用 read_image_setup 工具写入 {CONFIG_PATH}")
        lines.append("  智谱申请地址：https://open.bigmodel.cn/usercenter/apikeys")
    lines.append(f"- 接口地址：{cfg['base_url']}")
    lines.append(f"- 候选模型：{', '.join('`' + m + '`' for m in cfg['models'])}")
    lines.append(f"- 图片长边上限：{cfg['max_side']}px，单次输出上限 {cfg['max_tokens']} tokens，"
                 f"超时 {cfg['timeout']:.0f}s")
    for w in cfg.get("warnings") or []:
        lines.append(f"- ⚠️ {w}")
    lines.append("")

    if not key:
        lines.append("配置好 Key 后重新运行本诊断。")
        return "\n".join(lines)

    if not probe:
        lines.append("（已跳过实测）")
        return "\n".join(lines)

    # 一张 8x8 的图，够做连通性与模型可用性探测
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 60, 60)).save(buf, "PNG")
    probe_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    lines.append("## 实测结果")
    any_ok = False
    for model in cfg["models"]:
        try:
            # 探测预算与真实调用保持一致，否则会出现"doctor 说可用、read_image 却 400"的
            # 假绿灯（各模型的 max_tokens 上限并不相同，glm-4v-flash 只到 1024）。
            answer, used = ask_vision([probe_url], "这张图是什么颜色？只回答颜色。",
                                      cfg, model=model, max_tokens=cfg["max_tokens"])
            lines.append(f"- `{model}` ✓ 可用 —— 回答：{answer[:40]}")
            any_ok = True
            break  # 第一个可用的就够了，不必把整个队列打一遍
        except VisionError as e:
            detail = _model_error_line(str(e), model)
            lines.append(f"- `{model}` ✗ {detail}")
            # 基础链路不通的话，后面每个候选都会各挂一个 timeout，白等好几分钟
            if "网络不通" in detail or "连接中断" in detail:
                lines.append("")
                lines.append(f"接口地址 `{cfg['base_url']}` 不可达，已跳过对其余候选模型的探测。")
                lines.append("请检查 READ_IMAGE_BASE_URL，或确认当前网络能否访问该网关。")
                return "\n".join(lines)

    lines.append("")
    lines.append("结论：" + ("配置可用，read_image 已就绪。" if any_ok
                          else "没有任何候选模型可用，请用 READ_IMAGE_MODEL 指定正确的模型名。"))
    return "\n".join(lines)


@mcp.tool(
    title="配置 read-image 的 API Key",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def read_image_setup(api_key: str, model: str = "", base_url: str = "") -> str:
    """把视觉模型的 API Key 写进本地配置文件，免去设置系统环境变量。

    适用于宿主没有把环境变量透传给 MCP 子进程、导致 read_image 报"未配置"的情况。
    环境变量优先级仍然更高：一旦设置了环境变量，会覆盖这里的配置。

    Args:
        api_key: 视觉模型平台的 API Key（智谱在 https://open.bigmodel.cn/usercenter/apikeys 申请）。
        model: 可选。指定默认模型，例如 glm-5.3-flash。留空则用内置候选队列。
        base_url: 可选。OpenAI 兼容网关地址，默认 https://open.bigmodel.cn/api/paas/v4。

    Returns:
        写入结果说明；同时会立刻实测一次，确认配置真的可用。
    """
    if not api_key or not api_key.strip():
        return "错误：api_key 为空。"
    path = save_file_config(api_key, model or None, base_url or None)

    lines = [f"已写入 `{path}`", ""]
    cfg = config()
    lines.append(f"- Key 来源：`{cfg['key_source']}`")
    lines.append(f"- 接口地址：{cfg['base_url']}")
    lines.append(f"- 候选模型：{', '.join('`' + m + '`' for m in cfg['models'][:3])} …")
    lines.append("")
    lines.append("## 实测")
    lines.append(read_image_doctor(probe=True).split("## 实测结果")[-1].strip())
    return "\n".join(lines)


def main() -> None:
    print("[read-image] MCP server 已启动（stdio）", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
