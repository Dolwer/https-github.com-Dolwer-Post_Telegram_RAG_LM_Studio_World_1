import requests
import json
from pathlib import Path
from typing import Union, Optional, List
import logging
import time
import html

# Настройка логгера для этого модуля
logger = logging.getLogger("rag_telegram")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_telegram] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def escape_html(text: str) -> str:
    """
    Экранирует HTML-спецсимволы для Telegram (HTML-mode).
    """
    return html.escape(text, quote=False)

class TelegramPublisher:
    """
    Публикация сообщений и файлов в Telegram-канал через Bot API.
    Поддерживает отправку текста, изображений, документов, видео, аудио, предпросмотр ссылок, отложенную публикацию.
    """

    def __init__(
        self,
        bot_token: str,
        channel_id: Union[str, int],
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        enable_preview: bool = True
    ):
        """
        :param bot_token: Токен Telegram-бота
        :param channel_id: ID или username канала (например, @my_channel)
        :param logger: Логгер
        :param max_retries: Количество попыток при ошибках сети/Telegram
        :param retry_delay: Задержка между попытками (сек)
        :param enable_preview: Включить предпросмотр ссылок в постах
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_preview = enable_preview
        self.logger = logger or logging.getLogger("rag_telegram")

    def _post(self, method: str, data: dict, files: dict = None) -> dict:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, data=data, files=files, timeout=20)
                resp.raise_for_status()
                result = resp.json()
                if not result.get("ok"):
                    self.logger.error(f"Telegram API error: {result}")
                    raise Exception(f"Telegram API error: {result}")
                return result
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Telegram API request failed (attempt {attempt}): {e}")
                time.sleep(self.retry_delay)
        self.logger.error(f"Telegram API request failed after {self.max_retries} attempts: {last_exc}")
        raise last_exc

    def send_text(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка текстового сообщения в канал.
        :param html_escape: экранировать HTML-спецсимволы (True по умолчанию)
        :return: message_id отправленного сообщения или None при ошибке
        """
        # Экранирование HTML по требованию
        if html_escape and parse_mode == "HTML":
            text = escape_html(text)
        data = {
            "chat_id": self.channel_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": not (disable_preview if disable_preview is not None else self.enable_preview),
            "disable_notification": silent,
        }
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        try:
            resp = self._post("sendMessage", data)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Message posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send text message: {e}")
            return None

    def send_photo(
        self,
        photo: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        disable_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка фото с подписью.
        :param photo: путь к файлу или URL
        :param html_escape: экранировать HTML в подписи
        :return: message_id или None
        """
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        if isinstance(photo, (str, Path)) and Path(photo).exists():
            files["photo"] = open(photo, "rb")
        else:
            data["photo"] = str(photo)
        try:
            resp = self._post("sendPhoto", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Photo posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send photo: {e}")
            return None
        finally:
            if files:
                files["photo"].close()

    def send_video(
        self,
        video: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка видеофайла.
        """
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        if isinstance(video, (str, Path)) and Path(video).exists():
            files["video"] = open(video, "rb")
        else:
            data["video"] = str(video)
        try:
            resp = self._post("sendVideo", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Video posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send video: {e}")
            return None
        finally:
            if files:
                files["video"].close()

    def send_audio(
        self,
        audio: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка аудиофайла.
        """
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        if isinstance(audio, (str, Path)) and Path(audio).exists():
            files["audio"] = open(audio, "rb")
        else:
            data["audio"] = str(audio)
        try:
            resp = self._post("sendAudio", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Audio posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}")
            return None
        finally:
            if files:
                files["audio"].close()

    def send_document(
        self,
        document: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка файла-документа.
        :param document: путь к файлу или URL
        :param html_escape: экранировать HTML в подписи
        :return: message_id или None
        """
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        if isinstance(document, (str, Path)) and Path(document).exists():
            files["document"] = open(document, "rb")
        else:
            data["document"] = str(document)
        try:
            resp = self._post("sendDocument", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Document posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send document: {e}")
            return None
        finally:
            if files:
                files["document"].close()

    def send_media_group(
        self,
        media: List[dict]
    ) -> Optional[List[int]]:
        """
        Отправка набора медиа (фото/видео) в одном сообщении.
        :param media: список dict с типом ('photo'/'video'), media (file_id/url), caption (optional)
        :return: список message_id или None
        """
        data = {
            "chat_id": self.channel_id,
            "media": json.dumps(media, ensure_ascii=False),
        }
        try:
            resp = self._post("sendMediaGroup", data)
            results = resp.get("result", [])
            msg_ids = [msg.get("message_id") for msg in results if "message_id" in msg]
            self.logger.info(f"Media group posted to Telegram (messages={msg_ids})")
            return msg_ids
        except Exception as e:
            self.logger.error(f"Failed to send media group: {e}")
            return None

    def check_connection(self) -> bool:
        """
        Проверка связи с Telegram Bot API (getMe).
        """
        url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                self.logger.info("Telegram bot connection OK")
                return True
            else:
                self.logger.error("Telegram bot connection failed")
                return False
        except Exception as e:
            self.logger.error(f"Telegram bot connection error: {e}")
            return False

    def delayed_post(
        self,
        text: str,
        delay_sec: float,
        **kwargs
    ) -> Optional[int]:
        """
        Отправка сообщения с задержкой.
        """
        self.logger.info(f"Delaying message post for {delay_sec} seconds...")
        time.sleep(delay_sec)
        return self.send_text(text, **kwargs)
