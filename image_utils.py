import random
from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image, UnidentifiedImageError

# --- Поддерживаемые форматы Telegram ---
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
SUPPORTED_DOC_EXTS   = {".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".ogg"}
SUPPORTED_MEDIA_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS | SUPPORTED_DOC_EXTS | SUPPORTED_AUDIO_EXTS

# --- Размеры по умолчанию ---
MAX_IMAGE_SIZE = (1280, 1280)  # Максимальный размер для Telegram (по стороне)
MAX_FILE_SIZE_MB = 50  # Ограничение Telegram на размер файла (50 МБ)

def is_safe_media_path(path: Path, media_dir: Path) -> bool:
    """Путь находится строго внутри media_dir и не содержит переходов наверх."""
    try:
        return media_dir.resolve(strict=False) in path.resolve(strict=False).parents or path.resolve() == media_dir.resolve(strict=False)
    except Exception:
        return False

def pick_random_media_file(media_dir: Path, allowed_exts: Optional[set] = None) -> Optional[Path]:
    """
    Случайно выбирает файл из media_dir (включая подпапки) с поддерживаемым расширением.
    """
    if not media_dir.exists():
        return None
    allowed_exts = allowed_exts or SUPPORTED_MEDIA_EXTS
    files = [f for f in media_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]
    if not files:
        return None
    return random.choice(files)

def validate_media_file(path: Path, media_dir: Path = Path("media")) -> Tuple[bool, str]:
    """
    Проверяет валидность медиа-файла:
      - только из папки media (или подпапок)
      - поддерживаемое расширение
      - не превышает лимит размера
      - файл существует
    """
    if not path.exists():
        return False, "Файл не найден"
    if not is_safe_media_path(path, media_dir):
        return False, "Файл вне папки media"
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTS:
        return False, f"Неподдерживаемый формат: {path.suffix}"
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"Файл слишком большой (>{MAX_FILE_SIZE_MB} МБ)"
    return True, "OK"

def get_media_type(path: Path) -> str:
    """
    Определяет тип медиа-файла по расширению.
    """
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    elif ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    elif ext in SUPPORTED_DOC_EXTS:
        return "document"
    elif ext in SUPPORTED_AUDIO_EXTS:
        return "audio"
    return "unknown"

def process_image(path: Path, output_dir: Optional[Path] = None, max_size: Tuple[int,int]=MAX_IMAGE_SIZE) -> Optional[Path]:
    """
    Уменьшает изображение до max_size по большей стороне (если требуется). Возвращает путь к новому файлу.
    """
    try:
        img = Image.open(path)
        img.thumbnail(max_size, Image.ANTIALIAS)
        out_dir = output_dir or path.parent
        out_path = out_dir / f"{path.stem}_resized{path.suffix}"
        img.save(out_path)
        return out_path
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

def get_all_media_files(media_dir: Path, allowed_exts: Optional[set] = None) -> List[Path]:
    """
    Возвращает список всех файлов в media_dir (и подпапках) с нужными расширениями.
    """
    allowed_exts = allowed_exts or SUPPORTED_MEDIA_EXTS
    return [f for f in media_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]

def prepare_media_for_post(media_dir: Path = Path("media")) -> Optional[Path]:
    """
    Выбирает и валидирует случайный файл из media_dir.
    Если файл — изображение, при необходимости уменьшает размер.
    Возвращает путь к подготовленному файлу или None.
    """
    file = pick_random_media_file(media_dir)
    if not file:
        return None
    is_valid, reason = validate_media_file(file, media_dir)
    if not is_valid:
        return None
    media_type = get_media_type(file)
    if media_type == "image":
        # Проверим размер, если большое — уменьшим
        img = Image.open(file)
        if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
            resized = process_image(file)
            if resized is not None:
                return resized
    return file
