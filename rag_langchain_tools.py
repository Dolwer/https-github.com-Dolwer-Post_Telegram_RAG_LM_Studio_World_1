import logging
from rag_utils import web_search, safe_eval, analyze_table
from pathlib import Path
from typing import Optional, Dict, Any, List

# Причина: Внедрение логгирования для прозрачности и отладки интеллектуального слоя,
# а также для согласованности с остальной системой.
logger = logging.getLogger("rag_langchain_tools")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_langchain_tools] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ключевые слова для инструментов (расширяемый конфиг)
TOOL_KEYWORDS = {
    "web": ["найди", "поиск", "интернет", "lookup", "search", "google", "bing", "duckduckgo"],
    "calc": ["выгод", "посчит", "calculate", "profit", "выбери", "сколько", "рассчитай"],
    "table": ["таблиц", "excel", "csv", "xlsx", "анализируй", "данные", "отчет", "таблица"]
}

def tool_internet_search(query: str, num_results: int = 8) -> str:
    logger.info(f"Вызов интернет-поиска по запросу: {query}")
    results = web_search(query, num_results=num_results)
    if not results:
        logger.warning("Интернет-поиск не дал результатов")
        return "[Интернет-поиск не дал результатов]"
    return "\n".join(results)

def tool_calculator(expr: str, variables: Optional[Dict[str, Any]] = None) -> str:
    logger.info(f"Вызов калькулятора с выражением: {expr}")
    try:
        return str(safe_eval(expr, variables=variables))
    except Exception as e:
        logger.error(f"Ошибка калькуляции: {e}")
        return f"[Ошибка калькуляции]: {e}"

def tool_table_analysis(
    table_filename: str,
    info_query: Optional[dict]=None,
    inform_dir: Optional[str]=None,
    max_rows: int = 18,
    max_cols: int = 10
) -> str:
    logger.info(f"Анализ таблицы: {table_filename}")
    try:
        file_path = Path(inform_dir) / table_filename
        return analyze_table(file_path, info_query, max_rows=max_rows, max_cols=max_cols)
    except Exception as e:
        logger.error(f"Ошибка анализа таблицы: {e}")
        return f"[Ошибка анализа таблицы]: {e}"

def smart_tool_selector(
    topic: str,
    context: str,
    inform_dir: str,
    tool_keywords: Optional[Dict[str, List[str]]] = None,
    tool_log: Optional[List[str]] = None,
    max_tool_results: int = 8
) -> str:
    """
    Интеллектуальный селектор инструментов: анализирует промт, вызывает нужные инструменты,
    поддерживает цепочки, логирует действия, расширяем по ключевым словам.
    """
    tool_keywords = tool_keywords or TOOL_KEYWORDS
    tool_log = tool_log or []
    topic_lc = topic.lower()
    results = []
    used_tools = []

    # Причина: поддержка сложных сценариев — одновременный вызов нескольких инструментов, если встречаются разные ключевые слова.
    # Сначала web search
    if any(x in topic_lc for x in tool_keywords["web"]):
        logger.info("[smart_tool_selector] Web search triggered")
        tool_log.append("web_search")
        results.append("[Интернет]:\n" + tool_internet_search(topic, num_results=max_tool_results))
        used_tools.append("web_search")
    # Затем калькулятор
    if any(x in topic_lc for x in tool_keywords["calc"]):
        import re
        logger.info("[smart_tool_selector] Calculator triggered")
        tool_log.append("calculator")
        m = re.search(r"(посчитай|calculate|выгоднее|выгодность|сколько)[^\d]*(.+)", topic_lc)
        expr = m.group(2) if m else topic
        results.append("[Калькулятор]:\n" + tool_calculator(expr))
        used_tools.append("calculator")
    # Затем таблица
    if any(x in topic_lc for x in tool_keywords["table"]):
        logger.info("[smart_tool_selector] Table analysis triggered")
        tool_log.append("analyze_table")
        table_files = [f.name for f in Path(inform_dir).glob("*.csv")] + [f.name for f in Path(inform_dir).glob("*.xlsx")]
        if table_files:
            results.append("[Таблица]:\n" + tool_table_analysis(table_files[0], None, inform_dir))
            used_tools.append("analyze_table")
        else:
            results.append("[Нет подходящих таблиц для анализа]")

    if used_tools:
        logger.info(f"Вызваны инструменты: {used_tools}")
    if results:
        return "\n\n".join(results)
    else:
        logger.info("Ни один инструмент не был вызван")
        return ""

def enrich_context_with_tools(
    topic: str,
    context: str,
    inform_dir: str,
    max_tool_results: int = 8
) -> str:
    """
    Добавляет к контексту максимум информации из инструментов для нейросети.
    """
    logger.info("Расширение контекста инструментами...")
    tool_result = smart_tool_selector(topic, context, inform_dir, max_tool_results=max_tool_results)
    if tool_result:
        context = context + "\n\n[Инструментальное расширение]:\n" + tool_result
        logger.info("Контекст расширен инструментами.")
    else:
        logger.info("Инструменты не были использованы для расширения контекста.")
    return context
