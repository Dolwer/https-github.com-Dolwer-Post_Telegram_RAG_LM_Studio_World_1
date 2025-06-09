import json
import hashlib
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple

class ChunkUsageTracker:
    """
    Трекер использования чанков — хранит, обновляет и очищает статистику использования фрагментов знаний/контекста.
    Поддерживает устойчивые идентификаторы чанков (hash чанка), версионность базы, гибкие penalty/boost функции,
    полную очистку по возрасту и расширенные методы аналитики.
    """

    def __init__(
        self,
        usage_stats_file: Path,
        logger,
        chunk_usage_limit: int,
        usage_reset_days: int,
        diversity_boost: float,
        index_version: Optional[str] = None,
        index_hash: Optional[str] = None,
        penalty_func: Optional[Callable[[int, int, int], float]] = None,
        boost_func: Optional[Callable[[int, int], float]] = None,
    ):
        """
        :param usage_stats_file: Путь к файлу статистики использования
        :param logger: Логгер
        :param chunk_usage_limit: Лимит использования для penalty
        :param usage_reset_days: Сколько дней хранить usage перед очисткой
        :param diversity_boost: Базовый коэффициент diversity
        :param index_version: Версия индекса/базы знаний (для сброса статистики при обновлении)
        :param index_hash: Хеш базы знаний (для сброса статистики при обновлении)
        :param penalty_func: Кастомная функция penalty, принимает (chunk_count, title_count, chunk_usage_limit)
        :param boost_func: Кастомная функция diversity boost, принимает (chunk_count, chunk_usage_limit)
        """
        self.usage_stats_file: Path = usage_stats_file
        self.logger = logger
        self.chunk_usage_limit = chunk_usage_limit
        self.usage_reset_days = usage_reset_days
        self.diversity_boost = diversity_boost
        self.index_version = index_version
        self.index_hash = index_hash

        self.penalty_func = penalty_func or self._default_penalty_func
        self.boost_func = boost_func or self._default_boost_func

        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.recent_usage: deque = deque(maxlen=100)
        self.title_usage: Counter = Counter()
        self.chunk_usage: Counter = Counter()
        self.session_usage: Counter = Counter()
        self.loaded_index_version = None
        self.loaded_index_hash = None
        self.load_statistics()

    @staticmethod
    def get_chunk_hash(chunk_text: str, source: Optional[str]=None) -> str:
        """
        Возвращает устойчивый идентификатор чанка (sha1 от текста + source).
        """
        to_hash = (chunk_text or "") + "|" + (source or "")
        return hashlib.sha1(to_hash.encode('utf-8')).hexdigest()

    def load_statistics(self):
        """
        Загружает статистику из файла. Если версия/хеш базы отличается — сбрасывает usage.
        """
        try:
            if self.usage_stats_file.exists():
                with open(self.usage_stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.usage_stats = data.get('usage_stats', {})
                    self.title_usage = Counter(data.get('title_usage', {}))
                    self.chunk_usage = Counter(data.get('chunk_usage', {}))
                    self.recent_usage = deque(data.get('recent_usage', []), maxlen=100)
                    self.loaded_index_version = data.get('index_version')
                    self.loaded_index_hash = data.get('index_hash')
                if (self.index_version and self.loaded_index_version != self.index_version) or \
                   (self.index_hash and self.loaded_index_hash != self.index_hash):
                    self.logger.warning("Knowledge base index version/hash mismatch: usage statistics will be reset.")
                    self.clear_all_statistics()
                else:
                    self.logger.info("Loaded usage statistics successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load usage statistics: {e}")
            self.usage_stats = {}

    def save_statistics(self):
        """
        Сохраняет статистику в файл (atomic save).
        """
        try:
            data = {
                'usage_stats': self.usage_stats,
                'title_usage': dict(self.title_usage),
                'chunk_usage': dict(self.chunk_usage),
                'recent_usage': list(self.recent_usage),
                'last_updated': datetime.now().isoformat(),
                'index_version': self.index_version,
                'index_hash': self.index_hash
            }
            tmp_file = self.usage_stats_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_file.replace(self.usage_stats_file)
        except Exception as e:
            self.logger.error(f"Failed to save usage statistics: {e}")

    def record_usage(self, chunk_hashes: List[str], titles: List[str], metadata: List[Dict[str, Any]]):
        """
        Записывает использование чанков (по hash'ам).
        :param chunk_hashes: Список хешей чанков (уникальные id)
        :param titles: Список тайтлов источников (соответствует chunk_hashes)
        :param metadata: Массив метадаты (может быть пустой, используется для расширения)
        """
        timestamp = datetime.now().isoformat()
        for i, chunk_hash in enumerate(chunk_hashes):
            title = titles[i] if i < len(titles) else "unknown"
            if chunk_hash not in self.usage_stats:
                self.usage_stats[chunk_hash] = {
                    'count': 0,
                    'last_used': None,
                    'title': title
                }
            self.usage_stats[chunk_hash]['count'] += 1
            self.usage_stats[chunk_hash]['last_used'] = timestamp
            self.title_usage[title] += 1
            self.chunk_usage[chunk_hash] += 1
            self.session_usage[chunk_hash] += 1
            self.recent_usage.append({
                'chunk_id': chunk_hash,
                'title': title,
                'timestamp': timestamp
            })
        self.save_statistics()

    def get_usage_penalty(self, chunk_hash: str, title: str) -> float:
        """
        Возвращает штраф за частое использование (0.0 - 1.5), с учетом настраиваемой penalty-функции.
        """
        chunk_count = self.usage_stats.get(chunk_hash, {}).get('count', 0)
        title_count = self.title_usage.get(title, 0)
        return self.penalty_func(chunk_count, title_count, self.chunk_usage_limit)

    def get_diversity_boost(self, chunk_hash: str, title: str) -> float:
        """
        Возвращает буст для редко используемых чанков, с учетом настраиваемой boost-функции.
        """
        chunk_count = self.usage_stats.get(chunk_hash, {}).get('count', 0)
        return self.boost_func(chunk_count, self.chunk_usage_limit)

    @staticmethod
    def _default_penalty_func(chunk_count: int, title_count: int, chunk_usage_limit: int) -> float:
        """
        Стандартная penalty-функция (можно заменить через __init__).
        """
        chunk_penalty = min(chunk_count / chunk_usage_limit, 1.0)
        title_penalty = min(title_count / (chunk_usage_limit * 2), 0.5)
        return chunk_penalty + title_penalty

    @staticmethod
    def _default_boost_func(chunk_count: int, chunk_usage_limit: int) -> float:
        """
        Стандартная функция diversity boost (можно заменить через __init__).
        """
        if chunk_count == 0:
            return 2.0
        elif chunk_count < chunk_usage_limit // 3:
            return 1.0
        else:
            return 0.0

    def cleanup_old_stats(self, full_reset: bool = False):
        """
        Очищает старую статистику по времени, либо полностью сбрасывает usage по возрасту.
        :param full_reset: Если True — полностью сбрасывает usage у всех чанков, last_used которых старше порога.
        """
        cutoff_date = datetime.now() - timedelta(days=self.usage_reset_days)
        cutoff_str = cutoff_date.isoformat()
        cleaned_count = 0
        for chunk_hash in list(self.usage_stats.keys()):
            last_used = self.usage_stats[chunk_hash].get('last_used')
            if last_used and last_used < cutoff_str:
                if full_reset:
                    del self.usage_stats[chunk_hash]
                    cleaned_count += 1
                else:
                    old_count = self.usage_stats[chunk_hash]['count']
                    self.usage_stats[chunk_hash]['count'] = max(0, old_count - 1)
                    if self.usage_stats[chunk_hash]['count'] == 0:
                        del self.usage_stats[chunk_hash]
                        cleaned_count += 1
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old usage statistics entries{' (full reset)' if full_reset else ''}")
            self.save_statistics()

    def clear_all_statistics(self):
        """
        Полный сброс всей статистики использования.
        """
        self.usage_stats.clear()
        self.title_usage.clear()
        self.chunk_usage.clear()
        self.session_usage.clear()
        self.recent_usage.clear()
        self.save_statistics()
        self.logger.info("All usage statistics cleared.")

    def get_unused_chunks(self, metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Возвращает список hash'ей неиспользованных чанков из metadata.
        """
        all_hashes = set(self.get_chunk_hash(m['chunk'], m.get('source')) for m in metadata)
        used = set(self.usage_stats.keys())
        return list(all_hashes - used)

    def get_top_used_chunks(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Возвращает топ-N наиболее часто используемых чанков (hash, count).
        """
        return self.chunk_usage.most_common(n)

    def get_usage_distribution(self) -> Dict[str, int]:
        """
        Возвращает распределение использования по чанкам (hash -> count).
        """
        return dict(self.chunk_usage)

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Краткая сводка по статистике использования.
        """
        total_chunks = len(self.usage_stats)
        total_titles = len(self.title_usage)
        never_used = sum(1 for data in self.usage_stats.values() if data.get('count', 0) == 0)
        most_used = self.chunk_usage.most_common(1)[0] if self.chunk_usage else ("", 0)
        return {
            "total_chunks": total_chunks,
            "total_titles": total_titles,
            "never_used_chunks": never_used,
            "most_used_chunk": most_used,
        }

    # Документация property-методов
    @property
    def usage_stats_count(self) -> int:
        """Текущее число уникальных чанков с usage-статистикой."""
        return len(self.usage_stats)

    @property
    def title_count(self) -> int:
        """Число различных источников (title) в usage-статистике."""
        return len(self.title_usage)
