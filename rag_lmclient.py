import re
import requests
import asyncio
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List

# ВАЖНО: enrich_context_with_tools и get_prompt_parts импортируются явно
from rag_langchain_tools import enrich_context_with_tools
from rag_utils import get_prompt_parts

class LMClient:
    def __init__(
        self,
        retriever,
        data_dir,
        inform_dir,
        logger,
        # a) Все параметры генерации — теперь явные, с дефолтами или обязательные
        model_url: str,
        model_name: str,
        max_tokens: int = 1024,
        max_chars: int = 2600,
        max_attempts: int = 3,
        temperature: float = 0.7,
        timeout: int = 40,
        history_lim: int = 3,
        system_msg: Optional[str] = None
    ):
        self.retriever = retriever
        self.data_dir = Path(data_dir)
        self.inform_dir = Path(inform_dir)
        self.logger = logger

        self.model_url = model_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.timeout = timeout
        self.history_lim = history_lim
        # b) system_msg теперь параметр, дефолтная роль — если не указано
        self.system_msg = system_msg or "Вы — эксперт по бровям и ресницам."

    async def generate(self, topic: str, uploadfile: Optional[str] = None) -> str:
        """
        Генерирует текст по теме с обогащением инструментами (интернет/калькулятор/таблица) при необходимости.
        uploadfile: путь к прикреплённому файлу для Telegram-бота (или None).
        """
        try:
            # 1. Получаем сырой контекст из RAG.
            ctx = self.retriever.retrieve(topic)

            # 2. Обогащаем контекст инструментами, если это нужно.
            ctx = enrich_context_with_tools(topic, ctx, self.inform_dir)

            # 3. Генерируем промт (случайная сборка prompt_1/prompt_2 или fallback на prompt.txt)
            try:
                user_text = get_prompt_parts(self.data_dir, topic, ctx, uploadfile=uploadfile)
            except Exception as e:
                self.logger.error(f"Ошибка генерации промта из prompt_1/prompt_2: {e}")
                prompt_file = self.data_dir / 'prompt.txt'
                if not prompt_file.exists():
                    self.logger.error(f"Prompt file not found: {prompt_file}")
                    return "[Ошибка: файл промпта не найден]"
                prompt_template = prompt_file.read_text(encoding='utf-8')
                user_text = prompt_template.replace('{TOPIC}', topic).replace('{CONTEXT}', ctx)

            # b) system message — теперь параметр, не захардкожен
            system_msg = {"role": "system", "content": self.system_msg}
            user_msg = {"role": "user", "content": user_text}
            messages = [system_msg, user_msg]

            for attempt in range(self.max_attempts):
                try:
                    self.logger.info(f"Generation attempt {attempt + 1} for topic: {topic}")
                    resp = requests.post(
                        self.model_url,
                        json={
                            "model": self.model_name,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens
                        },
                        timeout=self.timeout
                    )
                    resp.raise_for_status()
                    response_data = resp.json()
                    if 'choices' not in response_data or not response_data['choices']:
                        self.logger.error("Invalid LM response format")
                        continue
                    text = response_data['choices'][0]['message']['content'].strip()

                    # f) Форматирование: убираем markdown-заголовки, разделители, промо-тексты, ссылки
                    text = re.sub(r"(?m)^#{2,}.*$", "", text)  # markdown-заголовки
                    text = re.sub(r"(?m)^---+", "", text)      # разделители
                    text = re.sub(r"\[\[.*?\]\]\(.*?\)", "", text)  # markdown-ссылки вида [[1]](url)
                    text = re.sub(r"\n{2,}", "\n", text)       # множественные переводы строк
                    # Удаляем явные фразы LLM ("As an AI language model", "Я искусственный интеллект" и т.п.)
                    text = re.sub(
                        r"(as an ai language model|i am an ai language model|я искусственный интеллект|как искусственный интеллект)[\.,]?\s*",
                        "",
                        text, flags=re.IGNORECASE
                    )
                    text = text.strip()

                    if len(text) <= self.max_chars:
                        self.logger.info(f"Generated text length: {len(text)} chars")
                        return text
                    # e) Улучшенная логика истории сообщений
                    if attempt < self.max_attempts - 1:
                        messages.append({"role": "assistant", "content": text})
                        messages.append({
                            "role": "user",
                            "content": f"Текст слишком длинный ({len(text)}>{self.max_chars}), сократи до {self.max_chars} символов."
                        })
                        sysm, rest = messages[0], messages[1:]
                        # Берем последние self.history_lim*2 сообщений (user/assistant), не нарушая структуру
                        last_msgs = []
                        for m in reversed(rest):
                            if len(last_msgs) >= self.history_lim * 2:
                                break
                            last_msgs.insert(0, m)
                        messages = [sysm] + last_msgs
                    else:
                        self.logger.warning(f"Force truncating text from {len(text)} to {self.max_chars} chars")
                        return text[:self.max_chars-10] + "..."
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"LM request error on attempt {attempt + 1}: {e}")
                    # g) Уведомление о критических ошибках (например, через notify_admin)
                    if attempt == self.max_attempts - 1:
                        # self.notify_admin(f"LM request failed after all attempts: {e}")  # если есть notify_admin
                        return "[Ошибка соединения с языковой моделью]"
                    await asyncio.sleep(5)
                except Exception as e:
                    self.logger.error(f"Unexpected error in generation attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        # self.notify_admin(f"LM unexpected error after all attempts: {e}")
                        return "[Ошибка генерации текста]"
            return "[Ошибка: превышено количество попыток генерации]"
        except Exception as e:
            self.logger.error(f"Critical error in generate: {e}")
            # self.notify_admin(f"Critical error in LMClient.generate: {e}")
            return "[Критическая ошибка генерации]"
