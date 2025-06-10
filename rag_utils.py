import pandas as pd
from bs4 import BeautifulSoup
import docx
from pathlib import Path
import random
import requests
import html2text
from typing import Optional, Any, Dict, List, Tuple
import logging

try:
    import textract
except ImportError:
    textract = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Настройка логгирования для этого модуля
logger = logging.getLogger("rag_utils")
if not logger.hasHandlers():
    # Позволяет логировать, даже если модуль импортируется отдельно
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def extract_text_from_file(path: Path) -> str:
    """
    Универсальный парсер файлов для RAG-контекста: txt, html, doc, docx, csv, xlsx, pdf.
    Возвращает plain text. Логгирует ошибки.
    """
    ext = path.suffix.lower()
    try:
        if ext == ".txt":
            logger.info(f"Extracting text from TXT file: {path}")
            return path.read_text(encoding="utf-8", errors='ignore')
        elif ext == ".html":
            logger.info(f"Extracting text from HTML file: {path}")
            html_content = path.read_text(encoding="utf-8", errors='ignore')
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ")
        elif ext in [".csv"]:
            logger.info(f"Extracting text from CSV file: {path}")
            try:
                df = pd.read_csv(path)
                return df.to_csv(sep="\t", index=False)
            except Exception as e:
                logger.error(f"Error reading CSV with pandas: {e}")
                return path.read_text(encoding="utf-8", errors='ignore')
        elif ext in [".xlsx", ".xls", ".xlsm"]:
            logger.info(f"Extracting text from Excel file: {path}")
            try:
                df = pd.read_excel(path)
                return df.to_csv(sep="\t", index=False)
            except Exception as e:
                logger.error(f"Error reading Excel with pandas: {e}")
                return ''
        elif ext in [".docx"]:
            logger.info(f"Extracting text from DOCX file: {path}")
            try:
                doc = docx.Document(path)
                return "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                logger.error(f"Error reading DOCX: {e}")
                return ''
        elif ext in [".doc"]:
            logger.info(f"Extracting text from DOC file: {path}")
            if textract is not None:
                try:
                    return textract.process(str(path)).decode("utf-8")
                except Exception as e:
                    logger.error(f"Error extracting DOC with textract: {e}")
                    return ''
            else:
                logger.error("textract is not installed. Cannot parse DOC files.")
                return ''
        elif ext == ".pdf":
            logger.info(f"Extracting text from PDF file: {path}")
            if PyPDF2 is not None:
                try:
                    text = []
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text.append(page_text)
                    return "\n".join(text)
                except Exception as e:
                    logger.error(f"Error extracting PDF with PyPDF2: {e}")
                    return ''
            elif textract is not None:
                try:
                    return textract.process(str(path)).decode("utf-8")
                except Exception as e:
                    logger.error(f"Error extracting PDF with textract: {e}")
                    return ''
            else:
                logger.error("Neither PyPDF2 nor textract installed. Cannot parse PDF files.")
                return ''
        else:
            logger.warning(f"Unsupported file type: {path}")
            return ''
    except Exception as e:
        logger.error(f"Exception extracting text from {path}: {e}")
        return ''
def clean_html_from_cell(cell_value: Any) -> str:
    """Очистить ячейку от HTML-тегов (если строка), иначе привести к строке."""
    if isinstance(cell_value, str):
        return BeautifulSoup(cell_value, "html.parser").get_text(separator=" ")
    return str(cell_value)

def process_table_for_rag(
    file_path: Path,
    columns: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    add_headers: bool = True,
    row_delim: str = "\n"
) -> str:
    """
    Обрабатывает таблицу для RAG: читает только нужные столбцы, фильтрует строки,
    очищает от HTML-тегов, формирует строки в формате "col1: val1 | col2: val2", 
    объединяет их через перенос строки (row_delim), добавляет заголовки при необходимости.
    Не ограничивает объем, если не указано иное.
    """
    ext = file_path.suffix.lower()
    try:
        logger.info(f"Processing table for RAG: {file_path.name}")
        if ext == ".csv":
            df = pd.read_csv(file_path, usecols=columns)
        elif ext in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(file_path, usecols=columns)
        else:
            logger.error("Not a table file")
            raise ValueError("Not a table file")
        if filter_expr:
            df = df.query(filter_expr)
        # Очищаем все ячейки от HTML-тегов
        for col in df.columns:
            df[col] = df[col].apply(clean_html_from_cell)
        # Формируем строки (с подписями столбцов)
        rows = []
        colnames = list(df.columns)
        for idx, row in df.iterrows():
            row_items = [f"{col}: {row[col]}" for col in colnames]
            rows.append(" | ".join(row_items))
        # Добавляем заголовки один раз сверху, если нужно
        result = ""
        if add_headers:
            header = " | ".join(colnames)
            result = header + row_delim
        result += row_delim.join(rows)
        logger.info(f"Table processed for RAG: {file_path.name}, rows: {len(df)}")
        return result
    except Exception as e:
        logger.error(f"process_table_for_rag error: {e}")
        return f"[Ошибка обработки таблицы для RAG]: {e}"

def process_text_file_for_rag(
    file_path: Path,
    chunk_size: int = 1000,
    overlap: int = 0
) -> List[str]:
    """
    Универсальный обработчик txt-файлов для RAG: разбивает файл на блоки/чанки нужного размера.
    Можно добавить очистку от html или предобработку.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # Можно добавить очистку от html, если нужно:
        # text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
        logger.info(f"Text file processed for RAG: {file_path.name}, chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"process_text_file_for_rag error: {e}")
        return [f"[Ошибка обработки txt-файла для RAG]: {e}"]

def get_prompt_parts(
    data_dir: Path,
    topic: str,
    context: str,
    uploadfile: Optional[Union[str, Path]] = None,
    file1: Optional[Path] = None,
    file2: Optional[Path] = None
) -> str:
    """
    Собирает промт из prompt_1/ и prompt_2/ с подстановкой {TOPIC}, {CONTEXT}, {UPLOADFILE}.
    При наличии {UPLOADFILE} и реально переданном файле:
        - {UPLOADFILE} заменяется на имя файла (например, myfile.pdf)
        - context обрезается до 1024 символов
    При отсутствии файла или ошибке:
        - {UPLOADFILE} -> [Файл не передан] или [Файл не найден: ...]
        - context НЕ обрезается
    Если {UPLOADFILE} отсутствует в шаблоне — логика старая, context режется до 4096.
    """

    import random

    def read_template(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading prompt template {path}: {e}")
            return None

    prompt1_dir = Path(data_dir) / "prompt_1"
    prompt2_dir = Path(data_dir) / "prompt_2"
    template = None

    if file1 is not None and file2 is not None:
        logger.info(f"Deterministic prompt: {file1.name} + {file2.name}")
        txt1 = read_template(file1)
        txt2 = read_template(file2)
        if txt1 is not None and txt2 is not None:
            template = txt1 + "\n" + txt2
    elif prompt1_dir.exists() and prompt2_dir.exists():
        prompt1_files = list(prompt1_dir.glob("*.txt"))
        prompt2_files = list(prompt2_dir.glob("*.txt"))
        if prompt1_files and prompt2_files:
            f1 = random.choice(prompt1_files)
            f2 = random.choice(prompt2_files)
            logger.info(f"Random prompt: {f1.name} + {f2.name}")
            txt1 = read_template(f1)
            txt2 = read_template(f2)
            if txt1 is not None and txt2 is not None:
                template = txt1 + "\n" + txt2
    if template is None:
        prompt_file = Path(data_dir) / "prompt.txt"
        if prompt_file.exists():
            logger.warning("Fallback to prompt.txt")
            template = read_template(prompt_file)
        else:
            logger.warning("Fallback to plain topic + context")
            # Здесь нет шаблона, просто подставим topic и context без uploadfile
            return f"{topic}\n\n{context}"

    if template is None:
        # Всё плохо, fallback на topic+context
        return f"{topic}\n\n{context}"

    # Проверяем наличие {UPLOADFILE} в шаблоне
    has_uploadfile = "{UPLOADFILE}" in template

    # Готовим uploadfile_text согласно логике
    uploadfile_text = ""
    if has_uploadfile:
        if uploadfile is not None:
            # Пробуем получить только имя файла (без пути)
            try:
                file_path = Path(uploadfile)
                if file_path.exists():
                    uploadfile_text = file_path.name
                    # Обрезаем context до 1024 символов
                    context = context[:1024]
                else:
                    uploadfile_text = f"[Файл не найден: {file_path.name}]"
                    # context НЕ обрезаем
            except Exception as e:
                uploadfile_text = "[Ошибка с файлом]"
                logger.error(f"Error processing uploadfile: {e}")
                # context НЕ обрезаем
        else:
            uploadfile_text = "[Файл не передан]"
            # context НЕ обрезаем

    # Лимит контекста для обычных промтов (без uploadfile) — 4096 символов
    if not has_uploadfile:
        context = context[:4096]

    # Подстановка плейсхолдеров
    prompt_out = (
        template.replace("{TOPIC}", topic)
                .replace("{CONTEXT}", context)
    )
    if has_uploadfile:
        prompt_out = prompt_out.replace("{UPLOADFILE}", uploadfile_text)
    return prompt_out

def web_search(
    query: str,
    num_results: int = 5,
    search_order: Optional[List[str]] = None
) -> List[str]:
    """
    Делает интернет-поиск по нескольким поисковикам (DuckDuckGo, Bing, Google Custom Search).
    Возвращает список кратких результатов (plain text).
    Если один поисковик не сработал — логирует и использует следующий.
    Для Bing/Google требуется регистрация и API-ключ!
    """
    results = []
    errors = []

    if search_order is None:
        search_order = ["duckduckgo", "bing", "google"]

    # DuckDuckGo
    if "duckduckgo" in search_order:
        try:
            url = "https://api.duckduckgo.com"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            if "RelatedTopics" in data:
                for topic in data["RelatedTopics"]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(topic["Text"])
                        if len(results) >= num_results:
                            break
            if results:
                logger.info("DuckDuckGo search successful.")
                return results[:num_results]
            else:
                logger.warning("DuckDuckGo returned no results.")
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            errors.append("DuckDuckGo")

    # Bing
    if "bing" in search_order:
        # Требуется регистрация на Azure и получение API-ключа!
        BING_API_KEY = ""  # <-- Вставьте ваш API-ключ
        if BING_API_KEY:
            try:
                url = "https://api.bing.microsoft.com/v7.0/search"
                headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
                params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": num_results}
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                data = resp.json()
                if "webPages" in data and "value" in data["webPages"]:
                    for res in data["webPages"]["value"]:
                        snippet = res.get("snippet", "")
                        results.append(snippet)
                        if len(results) >= num_results:
                            break
                if results:
                    logger.info("Bing search successful.")
                    return results[:num_results]
                else:
                    logger.warning("Bing returned no results.")
            except Exception as e:
                logger.error(f"Bing search error: {e}")
                errors.append("Bing")
        else:
            logger.info("Bing API key not provided. Skipping Bing search.")

    # Google Custom Search
    if "google" in search_order:
        # Требуется регистрация в Google Cloud и получение API-ключа + CX!
        GOOGLE_API_KEY = ""  # <-- Вставьте ваш API-ключ
        GOOGLE_CX = ""       # <-- Вставьте ваш CX
        if GOOGLE_API_KEY and GOOGLE_CX:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "q": query,
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX,
                    "num": num_results
                }
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
                if "items" in data:
                    for item in data["items"]:
                        snippet = item.get("snippet", "")
                        results.append(snippet)
                        if len(results) >= num_results:
                            break
                if results:
                    logger.info("Google Custom Search successful.")
                    return results[:num_results]
                else:
                    logger.warning("Google Custom Search returned no results.")
            except Exception as e:
                logger.error(f"Google Custom Search error: {e}")
                errors.append("Google")
        else:
            logger.info("Google API key or CX not provided. Skipping Google search.")

    if not results:
        logger.error(f"All search engines failed: {errors}")
    return results[:num_results]

def safe_eval(expr: str, allowed_names: Optional[Dict[str, Any]] = None, variables: Optional[Dict[str, Any]] = None) -> Any:
    """
    Безопасный калькулятор для арифметических выражений (в том числе с math.*).
    Поддерживает пользовательские переменные.
    """
    import math
    allowed = {
        "abs": abs, "round": round, "pow": pow,
        **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    }
    if allowed_names:
        allowed.update(allowed_names)
    if variables:
        allowed.update(variables)
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed:
            logger.error(f"Use of '{name}' not allowed in calculator")
            raise NameError(f"Use of '{name}' not allowed in calculator")
    try:
        return eval(code, {"__builtins__": {}}, allowed)
    except Exception as e:
        logger.error(f"safe_eval error for '{expr}': {e}")
        raise

def analyze_table(
    file_path: Path,
    query: Optional[Dict[str, Any]] = None,
    max_rows: int = 10,
    max_cols: int = 10
) -> str:
    """
    Анализирует таблицу (csv/xlsx): простое описание, фильтрация, выбор колонок.
    query: dict с ключами filter (строка для pandas.query), columns (список).
    max_rows/max_cols: ограничение вывода, чтобы промт был максимально обширным, но не переполненным.
    """
    ext = file_path.suffix.lower()
    try:
        logger.info(f"Analyzing table: {file_path.name}")
        if ext in [".csv"]:
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(file_path)
        else:
            logger.error("Not a table file")
            raise ValueError("Not table file")
        orig_shape = df.shape
        if query:
            if 'columns' in query:
                df = df[query['columns']]
            if 'filter' in query:
                df = df.query(query['filter'])
        df = df.iloc[:max_rows, :max_cols]
        stats = f"Столбцы: {list(df.columns)}\nСтрок: {len(df)} (изначально {orig_shape[0]})"
        preview = df.head(max_rows).to_string()
        logger.info(f"Table analyzed: {file_path.name}, {stats}")
        return f"{preview}\n\n{stats}"
    except Exception as e:
        logger.error(f"analyze_table error: {e}")
        return f"[Ошибка анализа таблицы]: {e}"

# Конфигурируемые ключевые слова для инструментов
TOOL_KEYWORDS = {
    "web": ["найди", "поиск", "интернет", "lookup", "search", "google", "bing", "duckduckgo"],
    "calc": ["выгод", "посчит", "calculate", "profit", "выбери", "сколько", "рассчитай"],
    "table": ["таблиц", "excel", "csv", "xlsx", "анализируй", "данные", "отчет", "таблица"]
}
TOOL_LOG = []

def smart_tool_selector(
    topic: str,
    context: str,
    inform_dir: Path,
    tool_keywords: Optional[Dict[str, List[str]]] = None,
    tool_log: Optional[List[str]] = None,
    max_tool_results: int = 12
) -> str:
    """
    Автоматически определяет, нужен ли инструмент (интернет, калькулятор, таблица) и вызывает его.
    Ключевые слова настраиваемы.
    Логирует выбор и результат инструментов.
    Поддержка сложных инструментов (цепочки).
    """
    tool_keywords = tool_keywords or TOOL_KEYWORDS
    tool_log = tool_log or TOOL_LOG
    topic_lc = topic.lower()
    context_lc = context.lower()
    info_parts = []

    # Сложные инструменты: если в теме встречается несколько ключевых слов, вызываем цепочку
    used_tools = []

    # Web search
    if any(x in topic_lc for x in tool_keywords["web"]):
        logger.info("[smart_tool_selector] Web search triggered")
        tool_log.append("web_search")
        results = web_search(topic, num_results=max_tool_results)
        if results:
            info_parts.append("[Интернет-поиск]:\n" + "\n".join(results))
            used_tools.append("web_search")

    # Calculator
    if any(x in topic_lc for x in tool_keywords["calc"]):
        import re
        logger.info("[smart_tool_selector] Calculator triggered")
        tool_log.append("calculator")
        m = re.search(r"(посчитай|calculate|выгоднее|выгодность|сколько)[^\d]*(.+)", topic_lc)
        expr = m.group(2) if m else topic
        try:
            calc_result = str(safe_eval(expr))
            info_parts.append(f"[Калькулятор]: {calc_result}")
            used_tools.append("calculator")
        except Exception as e:
            info_parts.append(f"[Ошибка калькуляции]: {e}")

    # Table analyze
    if any(x in topic_lc for x in tool_keywords["table"]):
        logger.info("[smart_tool_selector] Table analysis triggered")
        tool_log.append("analyze_table")
        table_files = list(Path(inform_dir).glob("*.csv")) + list(Path(inform_dir).glob("*.xlsx"))
        if table_files:
            try:
                table_result = analyze_table(table_files[0], max_rows=20, max_cols=15)
                info_parts.append("[Анализ таблицы]:\n" + table_result)
                used_tools.append("analyze_table")
            except Exception as e:
                info_parts.append(f"[Ошибка анализа таблицы]: {e}")
        else:
            info_parts.append("[Нет подходящих таблиц для анализа]")

    # Если была активирована хотя бы одна цепочка инструментов, добавить их список в лог
    if used_tools:
        logger.info(f"[smart_tool_selector] Использованы инструменты: {used_tools}")
    else:
        logger.info("[smart_tool_selector] Ни один инструмент не был вызван")

    return "\n\n".join(info_parts)

def enrich_context_with_tools(
    topic: str,
    context: str,
    inform_dir: Path,
    max_tool_results: int = 12
) -> str:
    """
    Добавляет к контексту результат работы нужного инструмента (если это требуется).
    Стремится дать максимум дополнительной информации для нейросети.
    """
    tool_result = smart_tool_selector(topic, context, inform_dir, max_tool_results=max_tool_results)
    if tool_result:
        context = context + "\n\n[Дополнительная информация из инструментов]:\n" + tool_result
        logger.info("[enrich_context_with_tools] Контекст дополнен инструментами.")
    else:
        logger.info("[enrich_context_with_tools] Инструменты не были использованы.")
    return context
