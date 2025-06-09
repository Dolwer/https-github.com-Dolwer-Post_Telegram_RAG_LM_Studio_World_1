import os
import json
import logging
import re
import html
import requests
import aiofiles
import asyncio
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timedelta
import random
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application
from telegram.constants import ParseMode
from telegram.error import RetryAfter, BadRequest, TelegramError

# ====== КОНФИГУРАЦИЯ ======
BOT_TOKEN     = "7995300452:AAEAmaKzR-Qa3XX-RJhIFGuR2FjDkGknXX0"
CHANNEL_ID    = "-1002469401173"
LM_URL        = "http://localhost:1234/v1/chat/completions"
LM_MODEL      = "gemma-3-27b-it"
MAX_CHARS     = 4096
MAX_TOKENS    = 1024
HISTORY_LIM   = 6
LM_TIMEOUT    = 600.0
RETRIES       = 5
RETRY_DELAY   = 15.0
MAX_GENERATION_ATTEMPTS = 3

# Новые параметры для управления разнообразием
CHUNK_USAGE_LIMIT = 10  # Максимальное количество использований чанка за период
USAGE_RESET_DAYS = 7    # Период сброса статистики использования
DIVERSITY_BOOST = 0.3   # Коэффициент буста для редко используемых чанков
TITLE_DIVERSITY_WEIGHT = 0.4  # Вес разнообразия по заголовкам

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
LOG_DIR       = BASE_DIR / "logs"
INFORM_DIR    = BASE_DIR / "inform"

INDEX_FILE    = DATA_DIR / "faiss_index.idx"
CONTEXT_FILE  = DATA_DIR / "faiss_contexts.json"
USAGE_STATS_FILE = DATA_DIR / "usage_statistics.json"
MONITORING_FILE = DATA_DIR / "chunk_monitoring.json"

EMB_MODEL     = "all-MiniLM-L6-v2"
CROSS_MODEL   = "cross-encoder/stsb-roberta-large"
CHUNK_SIZE    = 500
OVERLAP       = 100
TOP_K_TITLE   = 2
TOP_K_FAISS   = 8
TOP_K_FINAL   = 3

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Расширенная статистика использования
class ChunkUsageTracker:
    def __init__(self):
        self.usage_stats = {}
        self.recent_usage = deque(maxlen=100)  # Последние 100 использований
        self.title_usage = Counter()
        self.chunk_usage = Counter()
        self.session_usage = Counter()
        self.load_statistics()
    
    def load_statistics(self):
        """Загружает статистику из файла"""
        try:
            if USAGE_STATS_FILE.exists():
                with open(USAGE_STATS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.usage_stats = data.get('usage_stats', {})
                    self.title_usage = Counter(data.get('title_usage', {}))
                    self.chunk_usage = Counter(data.get('chunk_usage', {}))
                    # Восстанавливаем recent_usage из сохраненных данных
                    recent_data = data.get('recent_usage', [])
                    self.recent_usage = deque(recent_data[-100:], maxlen=100)
                logger.info("Loaded usage statistics successfully")
        except Exception as e:
            logger.warning(f"Failed to load usage statistics: {e}")
            self.usage_stats = {}
    
    def save_statistics(self):
        """Сохраняет статистику в файл"""
        try:
            data = {
                'usage_stats': self.usage_stats,
                'title_usage': dict(self.title_usage),
                'chunk_usage': dict(self.chunk_usage),
                'recent_usage': list(self.recent_usage),
                'last_updated': datetime.now().isoformat()
            }
            with open(USAGE_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage statistics: {e}")
    
    def record_usage(self, chunk_indices, metadata):
        """Записывает использование чанков"""
        timestamp = datetime.now().isoformat()
        
        for idx in chunk_indices:
            if idx < len(metadata):
                title = metadata[idx]['title']
                chunk_id = f"{title}_{idx}"
                
                # Обновляем статистику
                if chunk_id not in self.usage_stats:
                    self.usage_stats[chunk_id] = {
                        'count': 0,
                        'last_used': None,
                        'title': title
                    }
                
                self.usage_stats[chunk_id]['count'] += 1
                self.usage_stats[chunk_id]['last_used'] = timestamp
                
                # Обновляем счетчики
                self.title_usage[title] += 1
                self.chunk_usage[chunk_id] += 1
                self.session_usage[chunk_id] += 1
                
                # Добавляем в recent_usage
                self.recent_usage.append({
                    'chunk_id': chunk_id,
                    'title': title,
                    'timestamp': timestamp
                })
        
        self.save_statistics()
    
    def get_usage_penalty(self, chunk_id, title):
        """Возвращает штраф за частое использование (0.0 - 1.0)"""
        chunk_count = self.usage_stats.get(chunk_id, {}).get('count', 0)
        title_count = self.title_usage.get(title, 0)
        
        # Штраф за частое использование конкретного чанка
        chunk_penalty = min(chunk_count / CHUNK_USAGE_LIMIT, 1.0)
        
        # Штраф за частое использование заголовка
        title_penalty = min(title_count / (CHUNK_USAGE_LIMIT * 2), 0.5)
        
        return chunk_penalty + title_penalty
    
    def get_diversity_boost(self, chunk_id, title):
        """Возвращает буст для редко используемых чанков"""
        chunk_count = self.usage_stats.get(chunk_id, {}).get('count', 0)
        title_count = self.title_usage.get(title, 0)
        
        # Буст для неиспользованных или редко используемых чанков
        if chunk_count == 0:
            return DIVERSITY_BOOST * 2
        elif chunk_count < CHUNK_USAGE_LIMIT // 3:
            return DIVERSITY_BOOST
        else:
            return 0.0
    
    def cleanup_old_stats(self):
        """Очищает старую статистику"""
        cutoff_date = datetime.now() - timedelta(days=USAGE_RESET_DAYS)
        cutoff_str = cutoff_date.isoformat()
        
        cleaned_count = 0
        for chunk_id in list(self.usage_stats.keys()):
            last_used = self.usage_stats[chunk_id].get('last_used')
            if last_used and last_used < cutoff_str:
                # Уменьшаем счетчик, а не удаляем полностью
                self.usage_stats[chunk_id]['count'] = max(0, self.usage_stats[chunk_id]['count'] - 1)
                if self.usage_stats[chunk_id]['count'] == 0:
                    del self.usage_stats[chunk_id]
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old usage statistics entries")
            self.save_statistics()

# Инициализация трекера
usage_counter = Counter()  # Сохраняем для совместимости
usage_tracker = ChunkUsageTracker()

# ====== RAG RETRIEVER ======
class HybridRetriever:
    def __init__(self):
        self.sentencemodel = SentenceTransformer(EMB_MODEL)
        self.crossencoder = CrossEncoder(CROSS_MODEL)
        if INDEX_FILE.exists() and CONTEXT_FILE.exists():
            try:
                self._load_indices()
                logger.info("Successfully loaded existing indices")
            except Exception as e:
                logger.warning(f"Failed to load indices: {e}. Rebuilding...")
                self._build_indices()
        else:
            logger.info("No existing indices found. Building new ones...")
            self._build_indices()

    def _load_indices(self):
        logger.info("Loading indices...")
        try:
            with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            if not isinstance(self.metadata, list) or len(self.metadata) == 0:
                raise ValueError("Metadata must be a non-empty list")
            
            sample = self.metadata[0]
            required_fields = ['title', 'chunk']
            if not all(field in sample for field in required_fields):
                raise ValueError(f"Metadata must contain fields: {required_fields}")
            
            for item in self.metadata:
                if 'tokens' not in item:
                    item['tokens'] = item['chunk'].split()
                if isinstance(item['tokens'], str):
                    item['tokens'] = item['tokens'].split()
            
            self.faiss_index = faiss.read_index(str(INDEX_FILE))
            logger.info(f"Loaded {len(self.metadata)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            raise

    def _build_indices(self):
        logger.info("Building indices...")
        metadata = []
        
        txt_files = list(INFORM_DIR.glob('*.txt'))
        if not txt_files:
            raise RuntimeError(f"No .txt files found in {INFORM_DIR}")
        
        logger.info(f"Found {len(txt_files)} text files to process")
        
        for file in txt_files:
            try:
                title = file.stem.lower()
                text = file.read_text(encoding='utf-8')
                
                if not text.strip():
                    logger.warning(f"Empty file: {file}")
                    continue
                
                words = text.split()
                
                for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                    chunk = ' '.join(words[i:i + CHUNK_SIZE])
                    if len(chunk.strip()) < 10:
                        continue
                    
                    tokens = chunk.split()
                    metadata.append({
                        'title': title, 
                        'chunk': chunk, 
                        'tokens': tokens
                    })
                    
                logger.info(f"Processed {file.name}: {len(words)} words -> {len([m for m in metadata if m['title'] == title])} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                continue
        
        if not metadata:
            raise RuntimeError("No valid chunks created from text files")
        
        logger.info(f"Total chunks created: {len(metadata)}")
        
        try:
            texts = [f"{m['title']}: {m['chunk']}" for m in metadata]
            logger.info("Generating embeddings...")
            embs = self.sentencemodel.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            embs = embs.astype('float32')
            
            dim = embs.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(embs)
            
            faiss.write_index(self.faiss_index, str(INDEX_FILE))
            with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.metadata = metadata
            logger.info(f"Indices built and saved successfully. Dimension: {dim}")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def _apply_diversity_scoring(self, candidates, query):
        """Применяет скоринг разнообразия к кандидатам"""
        scored_candidates = []
        
        for idx in candidates:
            if idx >= len(self.metadata):
                continue
                
            title = self.metadata[idx]['title']
            chunk_id = f"{title}_{idx}"
            
            # Базовый семантический скор (расстояние от FAISS инвертируем)
            q_emb = self.sentencemodel.encode([query], convert_to_tensor=False).astype('float32')
            chunk_text = f"{title}: {self.metadata[idx]['chunk']}"
            c_emb = self.sentencemodel.encode([chunk_text], convert_to_tensor=False).astype('float32')
            
            # Косинусное сходство
            similarity = np.dot(q_emb[0], c_emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(c_emb[0]))
            
            # Применяем штрафы и бусты
            usage_penalty = usage_tracker.get_usage_penalty(chunk_id, title)
            diversity_boost = usage_tracker.get_diversity_boost(chunk_id, title)
            
            # Финальный скор
            final_score = similarity - usage_penalty + diversity_boost
            
            scored_candidates.append((final_score, idx, chunk_id, title))
        
        # Сортируем по итоговому скору
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates

    def retrieve(self, query):
        try:
            title_q = query.lower()
            candidates = []
            
            # 1) Title-based candidates с учетом разнообразия
            title_hits = []
            for i, m in enumerate(self.metadata):
                if m['title'] == title_q:
                    title_hits.append(i)
            
            # Применяем разнообразие к title-based кандидатам
            if title_hits:
                title_scored = self._apply_diversity_scoring(title_hits, query)
                candidates.extend([x[1] for x in title_scored[:TOP_K_TITLE]])
            
            # 2) Semantic FAISS candidates
            q_emb = self.sentencemodel.encode([query], convert_to_tensor=False).astype('float32')
            D, I = self.faiss_index.search(q_emb, TOP_K_FAISS * 2)  # Берем больше для разнообразия
            
            faiss_candidates = [idx for idx in I[0].tolist() if idx != -1 and idx not in candidates]
            
            if faiss_candidates:
                faiss_scored = self._apply_diversity_scoring(faiss_candidates, query)
                for score, idx, chunk_id, title in faiss_scored:
                    if idx not in candidates:
                        candidates.append(idx)
                        if len(candidates) >= TOP_K_FAISS:
                            break
            
            # 3) Lexical scoring fallback с разнообразием
            if len(candidates) < TOP_K_FINAL:
                words = set(query.lower().split())
                lexical_candidates = []
                
                for i, m in enumerate(self.metadata):
                    if i not in candidates:
                        lexical_score = sum(1 for w in m['tokens'] if w.lower() in words)
                        if lexical_score > 0:
                            lexical_candidates.append(i)
                
                if lexical_candidates:
                    lexical_scored = self._apply_diversity_scoring(lexical_candidates, query)
                    for score, idx, chunk_id, title in lexical_scored:
                        if idx not in candidates:
                            candidates.append(idx)
                            if len(candidates) >= TOP_K_FINAL:
                                break
            
            # 4) Cross-encoder reranking с учетом разнообразия
            if len(candidates) > TOP_K_FINAL:
                # Берем топ кандидатов для реранкинга
                rerank_candidates = candidates[:min(len(candidates), TOP_K_FAISS)]
                pairs = [(query, self.metadata[i]['chunk']) for i in rerank_candidates]
                
                if pairs:
                    cross_scores = self.crossencoder.predict(pairs)
                    
                    # Комбинируем cross-encoder скоры с diversity скорами
                    final_candidates = []
                    for i, (cross_score, idx) in enumerate(zip(cross_scores, rerank_candidates)):
                        title = self.metadata[idx]['title']
                        chunk_id = f"{title}_{idx}"
                        
                        usage_penalty = usage_tracker.get_usage_penalty(chunk_id, title)
                        diversity_boost = usage_tracker.get_diversity_boost(chunk_id, title)
                        
                        combined_score = cross_score - usage_penalty + diversity_boost
                        final_candidates.append((combined_score, idx))
                    
                    final_candidates.sort(key=lambda x: x[0], reverse=True)
                    selected_indices = [idx for _, idx in final_candidates[:TOP_K_FINAL]]
                else:
                    selected_indices = candidates[:TOP_K_FINAL]
            else:
                selected_indices = candidates[:TOP_K_FINAL]
            
            # Записываем использование
            usage_tracker.record_usage(selected_indices, self.metadata)
            
            # Обновляем legacy counter для совместимости
            for idx in selected_indices:
                if idx < len(self.metadata):
                    usage_counter[self.metadata[idx]['title']] += 1
            
            # Собираем контекст
            context_parts = []
            for i in selected_indices:
                if i < len(self.metadata):
                    context_parts.append(self.metadata[i]['chunk'])
            
            # Логируем выбранные чанки для мониторинга
            self._log_selection(query, selected_indices)
            
            return ' '.join(context_parts)
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return ""
    
    def _log_selection(self, query, selected_indices):
        """Логирует выбранные чанки для мониторинга"""
        try:
            selection_info = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'selected_chunks': []
            }
            
            for idx in selected_indices:
                if idx < len(self.metadata):
                    title = self.metadata[idx]['title']
                    chunk_id = f"{title}_{idx}"
                    usage_count = usage_tracker.usage_stats.get(chunk_id, {}).get('count', 0)
                    
                    selection_info['selected_chunks'].append({
                        'chunk_id': chunk_id,
                        'title': title,
                        'usage_count': usage_count,
                        'chunk_preview': self.metadata[idx]['chunk'][:100] + "..."
                    })
            
            # Сохраняем в файл мониторинга
            monitoring_data = []
            if MONITORING_FILE.exists():
                try:
                    with open(MONITORING_FILE, 'r', encoding='utf-8') as f:
                        monitoring_data = json.load(f)
                except:
                    monitoring_data = []
            
            monitoring_data.append(selection_info)
            
            # Сохраняем только последние 1000 записей
            monitoring_data = monitoring_data[-1000:]
            
            with open(MONITORING_FILE, 'w', encoding='utf-8') as f:
                json.dump(monitoring_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging selection: {e}")

# ====== LM CLIENT ======
class LMClient:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    async def generate(self, topic):
        try:
            ctx = self.retriever.retrieve(topic)
            
            prompt_file = DATA_DIR / 'prompt.txt'
            if not prompt_file.exists():
                logger.error(f"Prompt file not found: {prompt_file}")
                return "[Ошибка: файл промпта не найден]"
            
            prompt_template = prompt_file.read_text(encoding='utf-8')
            user_text = prompt_template.replace('{TOPIC}', topic).replace('{CONTEXT}', ctx)
            
            system_msg = {"role": "system", "content": "Вы — эксперт по бровям и ресницам."}
            user_msg = {"role": "user", "content": user_text}
            messages = [system_msg, user_msg]
            
            for attempt in range(MAX_GENERATION_ATTEMPTS):
                try:
                    logger.info(f"Generation attempt {attempt + 1} for topic: {topic}")
                    
                    resp = requests.post(
                        LM_URL,
                        json={
                            "model": LM_MODEL, 
                            "messages": messages,
                            "temperature": 0.7, 
                            "max_tokens": MAX_TOKENS
                        },
                        timeout=LM_TIMEOUT
                    )
                    resp.raise_for_status()
                    
                    response_data = resp.json()
                    if 'choices' not in response_data or not response_data['choices']:
                        logger.error("Invalid LM response format")
                        continue
                    
                    text = response_data['choices'][0]['message']['content'].strip()
                    
                    text = re.sub(r"(?m)^#{2,}.*$", "", text)
                    text = re.sub(r"(?m)^---+", "", text)
                    text = text.strip()
                    
                    if len(text) <= MAX_CHARS:
                        logger.info(f"Generated text length: {len(text)} chars")
                        return text
                    
                    if attempt < MAX_GENERATION_ATTEMPTS - 1:
                        messages.append({"role": "assistant", "content": text})
                        messages.append({
                            "role": "user", 
                            "content": f"Текст слишком длинный ({len(text)}>{MAX_CHARS}), сократи до {MAX_CHARS} символов."
                        })
                        
                        sysm, rest = messages[0], messages[1:]
                        pairs = [rest[i:i+2] for i in range(0, len(rest), 2)]
                        messages = [sysm] + sum(pairs[-HISTORY_LIM:], [])
                    else:
                        logger.warning(f"Force truncating text from {len(text)} to {MAX_CHARS} chars")
                        return text[:MAX_CHARS-10] + "..."
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"LM request error on attempt {attempt + 1}: {e}")
                    if attempt == MAX_GENERATION_ATTEMPTS - 1:
                        return "[Ошибка соединения с языковой моделью]"
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Unexpected error in generation attempt {attempt + 1}: {e}")
                    if attempt == MAX_GENERATION_ATTEMPTS - 1:
                        return "[Ошибка генерации текста]"
            
            return "[Ошибка: превышено количество попыток генерации]"
            
        except Exception as e:
            logger.error(f"Critical error in generate: {e}")
            return "[Критическая ошибка генерации]"

# ====== TELEGRAM POSTER ======
class TelegramPoster:
    ALLOWED_TAGS = {"b","strong","i","em","u","ins",
                    "s","strike","del","span","tg-spoiler",
                    "a","tg-emoji","code","pre","blockquote"}
    
    def __init__(self):
        self.app = Application.builder()\
            .token(BOT_TOKEN)\
            .connect_timeout(30)\
            .read_timeout(30)\
            .write_timeout(30)\
            .build()
    
    def to_html(self, text):
        htmlified = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        htmlified = re.sub(r"\*(.+?)\*", r"<i>\1</i>", htmlified)
        
        parts = re.split(r"(<[^>]+>)", htmlified)
        safe = []
        
        for part in parts:
            m = re.match(r"^<(/?\w+)", part)
            if m and m.group(1).strip("/").lower() in TelegramPoster.ALLOWED_TAGS:
                safe.append(part)
            else:
                safe.append(html.escape(part, quote=False))
        
        msg = "".join(safe)
        
        for tag in ("b","i","u","s","code","pre"):
            oc, cc = msg.count(f"<{tag}>"), msg.count(f"</{tag}>")
            if oc > cc: 
                msg += f"</{tag}>" * (oc - cc)
            elif cc > oc: 
                for _ in range(cc - oc):
                    msg = msg.replace(f"</{tag}>", "", 1)
        
        return msg
    
    def make_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("Записаться", url="https://t.me/Adelisva")],
            [
                InlineKeyboardButton("Avito", url="https://www.avito.ru/sankt-peterburg/predlozheniya_uslug/naraschivanie_resnits_brovi_i_laminirovanie_resnits_3786080678"), 
                InlineKeyboardButton("Instagram", url="https://www.instagram.com/qween_brows")
            ],
            [InlineKeyboardButton("Основной канал", url="https://t.me/brovist_piter")]
        ])
    
    async def post(self, text):
        if not text or not text.strip():
            logger.error("Attempt to post empty text")
            return False
        
        try:
            html_msg = self.to_html(text)
            kb = self.make_keyboard()
            
            for attempt in range(1, RETRIES + 1):
                try:
                    await self.app.bot.send_message(
                        chat_id=CHANNEL_ID, 
                        text=html_msg,
                        parse_mode=ParseMode.HTML, 
                        disable_web_page_preview=True,
                        reply_markup=kb
                    )
                    logger.info(f"Message sent successfully on attempt {attempt}")
                    return True
                    
                except RetryAfter as e:
                    logger.warning(f"Rate limited, waiting {e.retry_after + 1} seconds")
                    await asyncio.sleep(e.retry_after + 1)
                    
                except BadRequest as e:
                    logger.error(f"BadRequest error: {e}")
                    if "can't parse entities" in str(e).lower():
                        logger.info("Trying to send as plain text")
                        try:
                            await self.app.bot.send_message(
                                chat_id=CHANNEL_ID, 
                                text=text,
                                disable_web_page_preview=True,
                                reply_markup=kb
                            )
                            return True
                        except Exception as fallback_e:
                            logger.error(f"Fallback send also failed: {fallback_e}")
                    return False
                    
                except TelegramError as e:
                    logger.error(f"TelegramError attempt {attempt}: {e}")
                    if attempt < RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        return False
                        
                except Exception as e:
                    logger.error(f"Unexpected error in post attempt {attempt}: {e}")
                    if attempt < RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Critical error in post: {e}")
            return False

# ====== MONITORING AND STATISTICS ======
def generate_usage_report():
    """Генерирует отчет об использовании чанков"""
    try:
        # Собираем данные для отчета
        report_data = []
        
        # Анализируем статистику использования
        for chunk_id, stats in usage_tracker.usage_stats.items():
            title = stats['title']
            count = stats['count']
            last_used = stats.get('last_used', 'Never')
            
            # Определяем статус использования
            if count == 0:
                status = "Неиспользован"
            elif count < CHUNK_USAGE_LIMIT // 3:
                status = "Редко используется"
            elif count < CHUNK_USAGE_LIMIT:
                status = "Умеренно используется"
            else:
                status = "Часто используется"
            
            report_data.append({
                'chunk_id': chunk_id,
                'title': title,
                'usage_count': count,
                'last_used': last_used,
                'status': status
            })
        
        # Сортируем по количеству использований
        report_data.sort(key=lambda x: x['usage_count'], reverse=True)
        
        # Статистика по заголовкам
        title_stats = []
        for title, count in usage_tracker.title_usage.most_common():
            # Подсчитываем количество чанков для каждого заголовка
            title_chunks = len([chunk_id for chunk_id in usage_tracker.usage_stats.keys() 
                              if usage_tracker.usage_stats[chunk_id]['title'] == title])
            
            avg_usage = count / title_chunks if title_chunks > 0 else 0
            
            title_stats.append({
                'title': title,
                'total_usage': count,
                'chunks_count': title_chunks,
                'avg_usage_per_chunk': round(avg_usage, 2)
            })
        
        # Сохраняем отчет
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_chunks': len(usage_tracker.usage_stats),
            'total_usage_events': sum(stats['count'] for stats in usage_tracker.usage_stats.values()),
            'chunk_details': report_data,
            'title_statistics': title_stats,
            'recent_usage': list(usage_tracker.recent_usage)[-20:],  # Последние 20 использований
            'session_statistics': dict(usage_tracker.session_usage)
        }
        
        report_file = DATA_DIR / "usage_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Usage report generated: {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating usage report: {e}")
        return None

def display_monitoring_table():
    """Отображает таблицу мониторинга в удобном формате"""
    try:
        report = generate_usage_report()
        if not report:
            logger.error("Failed to generate report for display")
            return
        
        # Создаем сводную таблицу
        print("\n" + "="*100)
        print("МОНИТОРИНГ ИСПОЛЬЗОВАНИЯ ЧАНКОВ RAG СИСТЕМЫ")
        print("="*100)
        
        print(f"\nОбщая статистика:")
        print(f"Всего чанков в базе: {report['total_chunks']}")
        print(f"Общее количество использований: {report['total_usage_events']}")
        print(f"Время генерации отчета: {report['generated_at']}")
        
        # Топ используемых заголовков
        print(f"\n{'='*50}")
        print("ТОП-10 НАИБОЛЕЕ ИСПОЛЬЗУЕМЫХ ЗАГОЛОВКОВ:")
        print(f"{'='*50}")
        print(f"{'Заголовок':<30} {'Использований':<12} {'Чанков':<8} {'Среднее':<8}")
        print("-" * 58)
        
        for i, title_stat in enumerate(report['title_statistics'][:10], 1):
            title = title_stat['title'][:28] + ".." if len(title_stat['title']) > 30 else title_stat['title']
            print(f"{i:2}. {title:<28} {title_stat['total_usage']:>8} {title_stat['chunks_count']:>8} {title_stat['avg_usage_per_chunk']:>8}")
        
        # Детальная статистика по чанкам
        print(f"\n{'='*80}")
        print("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ЧАНКАМ:")
        print(f"{'='*80}")
        print(f"{'Статус':<20} {'Количество':<12} {'Процент':<10}")
        print("-" * 42)
        
        status_counts = Counter(chunk['status'] for chunk in report['chunk_details'])
        total_chunks = len(report['chunk_details'])
        
        for status, count in status_counts.items():
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            print(f"{status:<20} {count:>8} {percentage:>8.1f}%")
        
        # Наиболее и наименее используемые чанки
        print(f"\n{'='*80}")
        print("ТОП-10 НАИБОЛЕЕ ИСПОЛЬЗУЕМЫХ ЧАНКОВ:")
        print(f"{'='*80}")
        print(f"{'№':<3} {'Заголовок':<25} {'ID чанка':<20} {'Использований':<12} {'Статус':<15}")
        print("-" * 75)
        
        for i, chunk in enumerate(report['chunk_details'][:10], 1):
            title = chunk['title'][:23] + ".." if len(chunk['title']) > 25 else chunk['title']
            chunk_id = chunk['chunk_id'][:18] + ".." if len(chunk['chunk_id']) > 20 else chunk['chunk_id']
            print(f"{i:2}. {title:<25} {chunk_id:<20} {chunk['usage_count']:>8} {chunk['status']:<15}")
        
        # Неиспользованные чанки
        unused_chunks = [chunk for chunk in report['chunk_details'] if chunk['usage_count'] == 0]
        if unused_chunks:
            print(f"\n{'='*60}")
            print(f"НЕИСПОЛЬЗОВАННЫЕ ЧАНКИ (показано первые 10 из {len(unused_chunks)}):")
            print(f"{'='*60}")
            print(f"{'Заголовок':<30} {'ID чанка':<25}")
            print("-" * 55)
            
            for chunk in unused_chunks[:10]:
                title = chunk['title'][:28] + ".." if len(chunk['title']) > 30 else chunk['title']
                chunk_id = chunk['chunk_id'][:23] + ".." if len(chunk['chunk_id']) > 25 else chunk['chunk_id']
                print(f"{title:<30} {chunk_id:<25}")
        
        # Последние использования
        if report['recent_usage']:
            print(f"\n{'='*70}")
            print("ПОСЛЕДНИЕ 10 ИСПОЛЬЗОВАНИЙ:")
            print(f"{'='*70}")
            print(f"{'Время':<20} {'Заголовок':<25} {'ID чанка':<20}")
            print("-" * 65)
            
            for usage in report['recent_usage'][-10:]:
                timestamp = usage['timestamp'][:19].replace('T', ' ')
                title = usage['title'][:23] + ".." if len(usage['title']) > 25 else usage['title']
                chunk_id = usage['chunk_id'][:18] + ".." if len(usage['chunk_id']) > 20 else usage['chunk_id']
                print(f"{timestamp:<20} {title:<25} {chunk_id:<20}")
        
        print("\n" + "="*100)
        
    except Exception as e:
        logger.error(f"Error displaying monitoring table: {e}")

# ====== MAIN ======
async def main():
    try:
        logger.info("Starting main process")
        
        # Очистка старой статистики
        usage_tracker.cleanup_old_stats()
        
        topics_file = DATA_DIR / 'topics.txt'
        if not topics_file.exists():
            logger.error(f"Topics file not found: {topics_file}")
            return
        
        logger.info("Initializing retriever...")
        retriever = HybridRetriever()
        
        logger.info("Initializing LM client...")
        lm = LMClient(retriever)
        
        logger.info("Initializing Telegram poster...")
        tg = TelegramPoster()
        
        topics_content = topics_file.read_text(encoding='utf-8')
        topics = [t.strip() for t in topics_content.splitlines() if t.strip()]
        
        if not topics:
            logger.info("No topics found in topics.txt")
            # Показываем статистику даже если нет новых тем
            display_monitoring_table()
            return
        
        logger.info(f"Loaded {len(topics)} topics to process")
        
        successful_posts = 0
        failed_posts = 0
        
        for i, topic in enumerate(topics, 1):
            logger.info(f"Processing topic {i}/{len(topics)}: {topic}")
            
            try:
                # Генерируем текст
                text = await lm.generate(topic)
                
                if not text or text.startswith("[Ошибка"):
                    logger.error(f"Failed to generate text for topic: {topic}")
                    failed_posts += 1
                    continue
                
                logger.info(f"Generated text length: {len(text)} characters")
                
                # Отправляем в Telegram
                success = await tg.post(text)
                
                if success:
                    logger.info(f"Successfully posted topic: {topic}")
                    successful_posts += 1
                else:
                    logger.error(f"Failed to post topic: {topic}")
                    failed_posts += 1
                
                # Пауза между постами
                if i < len(topics):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing topic '{topic}': {e}")
                failed_posts += 1
                continue
        
        # Очищаем файл тем после обработки
        topics_file.write_text('', encoding='utf-8')
        logger.info(f"Processing completed. Successful: {successful_posts}, Failed: {failed_posts}")
        
        # Обновляем legacy counter для совместимости
        for title, count in usage_tracker.title_usage.items():
            usage_counter[title] = count
        
        # Генерируем и отображаем отчет
        display_monitoring_table()
        
        # Дополнительная статистика для pandas/ace_tools если доступно
        try:
            import pandas as pd
            from ace_tools import display_dataframe_to_user
            
            report = generate_usage_report()
            if report and report['chunk_details']:
                # Создаем DataFrame для детальной статистики
                df_chunks = pd.DataFrame(report['chunk_details'])
                df_chunks = df_chunks.sort_values('usage_count', ascending=False)
                display_dataframe_to_user("Детальная статистика использования чанков", df_chunks)
                
                # DataFrame для статистики по заголовкам
                df_titles = pd.DataFrame(report['title_statistics'])
                df_titles = df_titles.sort_values('total_usage', ascending=False)
                display_dataframe_to_user("Статистика использования по заголовкам", df_titles)
            else:
                logger.info("No detailed statistics to display in DataFrame format")
                
        except ImportError:
            logger.info("pandas or ace_tools not available for DataFrame display")
        except Exception as e:
            logger.error(f"Error displaying DataFrame statistics: {e}")
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
