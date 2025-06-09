import asyncio
from pathlib import Path
import logging

from rag_chunk_tracker import ChunkUsageTracker
from rag_retriever import HybridRetriever
from rag_telegram import TelegramPoster
from rag_lmclient import LMClient
from rag_utils import extract_text_from_file

# ==== КОНФИГ ====
BOT_TOKEN     = "7995300452:AAEAmaKZ-RJhIFGuR2FjDkGknXX0"
CHANNEL_ID    = "-1002469401173"
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
LOG_DIR       = BASE_DIR / "logs"
INFORM_DIR    = BASE_DIR / "inform"
INDEX_FILE    = DATA_DIR / "faiss_index.idx"
CONTEXT_FILE  = DATA_DIR / "faiss_contexts.json"
USAGE_STATS_FILE = DATA_DIR / "usage_statistics.json"

CHUNK_USAGE_LIMIT = 10
USAGE_RESET_DAYS = 7
DIVERSITY_BOOST = 0.3

EMB_MODEL     = "all-MiniLM-L6-v2"
CROSS_MODEL   = "cross-encoder/stsb-roberta-large"
CHUNK_SIZE    = 500
OVERLAP       = 100
TOP_K_TITLE   = 2
TOP_K_FAISS   = 8
TOP_K_FINAL   = 3

PROMPT1_DIR = DATA_DIR / "prompt_1"
PROMPT2_DIR = DATA_DIR / "prompt_2"
MAX_TELEGRAM_LENGTH = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    # Причина: инициализация всех модулей и параметров, чтобы иметь согласованную среду.
    usage_tracker = ChunkUsageTracker(
        usage_stats_file=USAGE_STATS_FILE,
        logger=logger,
        chunk_usage_limit=CHUNK_USAGE_LIMIT,
        usage_reset_days=USAGE_RESET_DAYS,
        diversity_boost=DIVERSITY_BOOST
    )
    usage_tracker.cleanup_old_stats()
    retriever = HybridRetriever(
        emb_model=EMB_MODEL,
        cross_model=CROSS_MODEL,
        index_file=INDEX_FILE,
        context_file=CONTEXT_FILE,
        inform_dir=INFORM_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k_title=TOP_K_TITLE,
        top_k_faiss=TOP_K_FAISS,
        top_k_final=TOP_K_FINAL,
        usage_tracker=usage_tracker,
        logger=logger
    )
    lm = LMClient(retriever=retriever, data_dir=DATA_DIR, inform_dir=INFORM_DIR, logger=logger)
    tg = TelegramPoster(BOT_TOKEN, CHANNEL_ID, logger)

    prompt1_files = sorted(PROMPT1_DIR.glob("*.txt"))
    prompt2_files = sorted(PROMPT2_DIR.glob("*.txt"))
    if not prompt1_files or not prompt2_files:
        logger.error("Нет файлов в prompt_1 или prompt_2")
        return

    # Причина: перебор всех пар файлов, чтобы темы генерировались не из topics.txt, а из файлов, как требуется.
    for file1 in prompt1_files:
        for file2 in prompt2_files:
            # Получаем TOPIC: причина — назвать его по имени файла1+файла2 или по первому заголовку внутри файла1 (актуализировать под бизнес-логику)
            topic = file1.stem
            # Получаем context через retriever: причина — согласованность с архитектурой RAG
            context = retriever.retrieve(topic)
            # Собираем промт: причина — не генерировать промт в коде, а только по шаблонам из файлов
            prompt_template_1 = file1.read_text(encoding="utf-8")
            prompt_template_2 = file2.read_text(encoding="utf-8")
            prompt_full = (prompt_template_1 + "\n" + prompt_template_2).replace("{TOPIC}", topic).replace("{CONTEXT}", context)
            # Диалог с LM: причина — если слишком длинно, просим сократить, иначе публикуем
            messages = [
                {"role": "system", "content": "Вы — эксперт по бровям и ресницам."},
                {"role": "user", "content": prompt_full}
            ]
            attempt = 0
            max_attempts = 6
            while attempt < max_attempts:
                attempt += 1
                text = await lm.generate_raw(messages)  # generate_raw возвращает только LM-ответ без доп. логики
                if text is None:
                    logger.error(f"LM не дал ответ для темы {topic}")
                    break
                if len(text) <= MAX_TELEGRAM_LENGTH:
                    # Причина: длина в лимите, отправляем
                    await tg.post(text)
                    logger.info(f"Тема '{topic}': успешно отправлено в Telegram (длина {len(text)})")
                    break
                else:
                    # Причина: длина превышает, просим сократить
                    logger.info(f"Тема '{topic}': длина {len(text)}, просим сократить")
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": f"Текст слишком длинный ({len(text)}>4096). Сократи до 4096 символов без потери смысла."})
            else:
                logger.warning(f"Тема '{topic}': не удалось сократить текст до лимита за {max_attempts} попыток.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
