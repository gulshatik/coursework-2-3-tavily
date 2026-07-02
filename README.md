# Повторный экзамен #2: Сравнительный обзор 3 сущностей (Tavily)

## Описание задания

В этом проекте реализован **LangGraph‑агент**, который по запросу пользователя генерирует сравнительный обзор трёх сущностей (технологий, продуктов, подходов).  
Обзор состоит из:

1. **Плана критериев** – 3–5 вопросов, по которым сравниваются сущности.  
2. **Поиск по каждой сущности и критерию** – для каждой пары `entity × criterion` агент делает реальный веб‑поиск через Tavily и формирует краткую заметку.  
3. **Финальная таблица** – Markdown‑таблица, где строки – критерии, столбцы – сущности, ячейки – найденные заметки.  
4. **Вердикт** – рекомендация, какую сущность выбрать для конкретного кейса.

> **Тема по умолчанию**: *«Сравни 3 векторные БД: Chroma, FAISS, Qdrant»*.

## Стек технологий

| Технология | Версия | Зачем |
|------------|--------|-------|
| Python | 3.10+ | Язык |
| LangGraph | latest | Создание графа состояний |
| LangChain OpenAI / Ollama | latest | LLM‑интеграция |
| LangChain Tavily | latest | Веб‑поиск |
| Tavily SDK | latest | Альтернатива LangChain |
| python-dotenv | latest | Загрузка переменных окружения |

### Установка зависимостей

```bash
# Создайте виртуальное окружение (рекомендовано)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

`requirements.txt`

```text
langgraph
langchain-openai
langchain-tavily
tavily-python
python-dotenv
```

> **Важно**: Если вы используете Ollama вместо OpenAI, замените `langchain-openai` на `langchain-ollama` и настройте соответствующий провайдер в коде.

## Конфигурация

Создайте файл `.env` в корне проекта:

```dotenv
# Ключ API для Tavily
TAVILY_API_KEY=YOUR_TAVILY_KEY

# (Опционально) Ключ OpenAI
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

> **Тавили** – это сервис, который позволяет выполнять быстрый поиск по вебу. Ключ можно получить на https://tavily.com.

## Структура проекта

```
.
├── compare_agent.py   # Определение состояния, узлов и графа
├── main.py            # Точка входа, CLI
├── utils.py           # Вспомогательные функции
├── .env.example       # Шаблон .env
├── requirements.txt
└── README.md
```

### `compare_agent.py`

```python
from typing import TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------- 1. Состояние ----------
class CompareState(TypedDict):
    entities: List[str]                # 3 имени для сравнения
    criteria: List[str]                # 3–5 критериев
    findings: Dict[str, List[str]]     # entity -> список заметок по критериям
    final_table: Optional[str]
    verdict: Optional[str]
    # вспомогательные индексы для цикла
    current_entity_idx: int
    current_criterion_idx: int

# ---------- 2. Узлы ----------
def plan_criteria(state: CompareState) -> CompareState:
    """LLM генерирует критерии сравнения."""
    prompt = ChatPromptTemplate.from_template(
        """
        Сравни следующие сущности: {entities}.
        Сформулируй 3–5 критериев, по которым можно сравнить их.  
        Ответ в формате JSON: ["критерий1", "критерий2", ...]
        """
    )
    llm = ChatOpenAI()
    chain = prompt | llm | JsonOutputParser()
    criteria = chain.invoke({"entities": state["entities"]})
    state["criteria"] = criteria
    # Инициализируем индексы
    state["current_entity_idx"] = 0
    state["current_criterion_idx"] = 0
    state["findings"] = {e: [] for e in state["entities"]}
    return state

def research_entity(state: CompareState) -> CompareState:
    """Одна итерация: берёт текущую пару entity × criterion,
    делает Tavily‑поиск и сохраняет заметку."""
    entity = state["entities"][state["current_entity_idx"]]
    criterion = state["criteria"][state["current_criterion_idx"]]
    query = f"{entity} {criterion}"
    tavily = TavilySearch()
    result = tavily.invoke({"query": query, "max_results": 3})
    # Берём первые 3 фрагмента и формируем краткую заметку
    snippets = [r["content"][:200] for r in result["results"]]
    note = f"{criterion}: " + "; ".join(snippets)
    state["findings"][entity].append(note)

    # Обновляем индексы
    state["current_criterion_idx"] += 1
    if state["current_criterion_idx"] >= len(state["criteria"]):
        state["current_criterion_idx"] = 0
        state["current_entity_idx"] += 1

    return state

def build_table(state: CompareState) -> CompareState:
    """Формирует Markdown‑таблицу из найденных заметок."""
    headers = ["Критерий"] + state["entities"]
    rows = []
    for crit in state["criteria"]:
        row = [crit]
        for ent in state["entities"]:
            # Берём заметку по этому критерию (если есть)
            notes = state["findings"][ent]
            # Находим заметку, начинающуюся с критерия
            note = next((n for n in notes if n.startswith(crit)), "")
            row.append(note.replace("\n", " "))
        rows.append(row)

    # Формируем таблицу
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        table += "| " + " | ".join(r) + " |\n"

    state["final_table"] = table
    return state

def verdict(state: CompareState) -> CompareState:
    """LLM формирует рекомендацию."""
    prompt = ChatPromptTemplate.from_template(
        """
        На основе следующей таблицы сравнения:
        {table}
        Напиши 2–4 предложения, в которых:
        1. Кратко резюмируешь сильные и слабые стороны каждой сущности.
        2. Рекомендовал бы, какую из них выбрать для кейса «{case}».
        """
    )
    llm = ChatOpenAI()
    chain = prompt | llm
    result = chain.invoke({"table": state["final_table"], "case": "выбор векторной БД для небольшого проекта"})
    state["verdict"] = result
    return state

# ---------- 3. Граф ----------
def build_graph() -> StateGraph:
    graph = StateGraph(CompareState)

    graph.add_node("plan_criteria", plan_criteria)
    graph.add_node("research_entity", research_entity)
    graph.add_node("build_table", build_table)
    graph.add_node("verdict", verdict)

    graph.set_entry_point("plan_criteria")
    graph.add_edge("plan_criteria", "research_entity")
    graph.add_conditional_edges(
        "research_entity",
        lambda state: "research_entity" if state["current_entity_idx"] < len(state["entities"])
        else "build_table",
    )
    graph.add_edge("build_table", "verdict")
    graph.add_edge("verdict", "__end__")

    return graph
```

### `main.py`

```python
import os
from dotenv import load_dotenv
from compare_agent import build_graph, CompareState

def main():
    load_dotenv()
    # Опциональный ввод пользовательских сущностей
    default_entities = ["Chroma", "FAISS", "Qdrant"]
    user_input = input(
        f"Введите 3 сущности через запятую (или нажмите Enter для {default_entities}): "
    )
    if user_input.strip():
        entities = [e.strip() for e in user_input.split(",") if e.strip()]
        if len(entities) != 3:
            print("Нужно ровно 3 сущности.")
            return
    else:
        entities = default_entities

    # Инициализируем состояние
    state: CompareState = {
        "entities": entities,
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "current_entity_idx": 0,
        "current_criterion_idx": 0,
    }

    graph = build_graph()
    final_state = graph.invoke(state)

    # Выводим результаты
    print("\n=== Критерии сравнения ===")
    for c in final_state["criteria"]:
        print(f"- {c}")

    print("\n=== Поиск по каждой паре ===")
    for ent in entities:
        for note in final_state["findings"][ent]:
            print(f"[{ent} × {note.split(':')[0]}] найдено: {note.split(':', 1)[1].strip()}")

    print("\n=== Финальная таблица ===")
    print(final_state["final_table"])

    print("\n=== Вердикт ===")
    print(final_state["verdict"])

if __name__ == "__main__":
    main()
```

### `utils.py`

```python
# В этом проекте вспомогательные функции не требуются,
# но можно добавить, например, форматирование таблицы.
```

## Как запустить

```bash
python main.py
```

Вы увидите:

1. Список критериев.  
2. Для каждой пары `entity × criterion` краткое резюме.  
3. Markdown‑таблица сравнения.  
4. Рекомендацию (вердикт).

## Как это работает

1. **Планирование критериев**  
   - `plan_criteria` использует LLM, чтобы сгенерировать релевантные критерии, основываясь на названиях сущностей.  
   - Критерии сохраняются в `state["criteria"]`.

2. **Итеративный поиск**  
   - `research_entity` последовательно обрабатывает каждую пару `entity × criterion`.  
   - Для каждой пары выполняется Tavily‑поиск (3 результата), из которых формируется краткая заметка.  
   - Заметки добавляются в `state["findings"][entity]`.  
   - После каждой итерации индексы обновляются; если все пары обработаны, переход к `build_table`.

3. **Формирование таблицы**  
   - `build_table` строит Markdown‑таблицу: строки – критерии, столбцы – сущности.  
   - Таблица сохраняется в `state["final_table"]`.

4. **Вердикт**  
   - `verdict` передаёт таблицу LLM, который генерирует рекомендации.  
   - Результат сохраняется в `state["verdict"]`.

## Проверка качества

| Компонент | Что проверяется |
|-----------|-----------------|
| `plan_criteria` | LLM генерирует критерии, а не использует хардкод |
| Цикл `research_entity` | Пошаговый обход всех пар, `findings` накапливается |
| Tavily | Реальный веб‑поиск, ссылки не выдуманы |
| `build_table` | Markdown‑таблица полностью заполнена |
| `verdict` | Явная рекомендация, 2–4 предложения |
| CLI | Выводит все этапы, работает с пользовательским вводом |

## Возможные улучшения

- **Кеширование**: чтобы не делать повторный поиск для одинаковых запросов.  
- **Параллелизация**: использовать `async` для ускорения поиска.  
- **Более сложный парсинг**: извлекать ключевые фразы из результатов, а не просто первые 200 символов.  
- **Интерактивный режим**: добавить меню выбора кейса для вердикта.  

## Лицензия

MIT License – свободно используйте и модифицируйте.

--- 

**Удачной работы!**
