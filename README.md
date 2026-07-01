# Повторный экзамен #2: Сравнительный обзор 3 сущностей (Tavily)

## Описание задания

В этом упражнении вы создадите **LangGraph‑агент**, который по запросу студента генерирует сравнительный обзор трёх технологий/продуктов/подходов.  
Обзор должен иметь следующую структуру:

1. **Общий план** – список критериев сравнения (3–5 пунктов).  
2. **По каждой сущности** – поиск в интернете через Tavily, получение краткой заметки по каждому критерию.  
3. **Финальная таблица** – Markdown‑таблица `N × 3`, где строки = критерии, столбцы = сущности.  
4. **Вердикт** – рекомендация, какая из трёх лучше подходит для конкретных кейсов.

Супервизор задаёт три сущности и проверяет, что агент действительно использует LLM для генерации критериев, а не хардкодит их, а также что поиск выполняется через Tavily (не «выдуманные» ссылки).

---

## Стек технологий

| Библиотека | Версия | Зачем |
|------------|--------|-------|
| `langgraph` | 0.1.x+ | Создание графа состояний и узлов |
| `langchain-openai` / `langchain-ollama` | 0.2.x+ | LLM‑интеграция (OpenAI или локальный Ollama) |
| `langchain-tavily` | 0.1.x+ | Обёртка над Tavily API |
| `tavily-python` | 0.3.x+ | SDK для прямого вызова Tavily |
| `python-dotenv` | 1.0.x+ | Загрузка переменных окружения (`TAVILY_API_KEY`) |

```bash
pip install langgraph langchain-openai langchain-tavily tavily-python python-dotenv
```

В корне проекта создайте файл `.env`:

```dotenv
# .env
TAVILY_API_KEY=YOUR_TAVILY_API_KEY_HERE
```

---

## Структура репозитория

```
.
├── agent.py          # Определение состояния, узлов и графа
├── main.py           # CLI‑точка входа
├── requirements.txt  # Пакеты для pip
└── README.md         # Это файл
```

---

## Файлы проекта

### `requirements.txt`

```text
langgraph>=0.1.0
langchain-openai>=0.2.0
langchain-tavily>=0.1.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
```

> **Важно**: если вы используете Ollama вместо OpenAI, замените `langchain-openai` на `langchain-ollama`.

---

### `agent.py`

```python
# agent.py
from typing import TypedDict, List, Dict, Any
import os

from langgraph.graph import StateGraph
from langgraph.prebuilt import create_conditional_agent_executor
from langchain_openai import ChatOpenAI  # или langchain_ollama.ChatOllama
from tavily_python import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ---------- Состояние ----------
class CompareState(TypedDict):
    entities: List[str]                # 3 имени для сравнения
    criteria: List[str]                 # 3–5 критериев
    findings: Dict[str, List[str]]      # entity -> список заметок по критериям
    final_table: str | None
    verdict: str | None

# ---------- Инициализация LLM и Tavily ----------
llm = ChatOpenAI(temperature=0.2)  # или ChatOllama(...)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ---------- Узлы ----------
def plan_criteria(state: CompareState) -> Dict[str, Any]:
    """LLM генерирует критерии сравнения."""
    prompt = (
        f"Сравни следующие сущности: {', '.join(state['entities'])}.\n"
        "Предложи 3–5 ключевых критериев для сравнения. "
        "Ответ в формате JSON‑массив строк, например:\n"
        '["Качество", "Стоимость", "Поддержка"]'
    )
    response = llm.invoke(prompt)
    import json
    criteria = json.loads(response.content.strip())
    state["criteria"] = criteria
    # Инициализируем findings
    for ent in state["entities"]:
        state.setdefault("findings", {}).setdefault(ent, [])
    return {"state": state}

def research_entity(state: CompareState) -> Dict[str, Any]:
    """Пошаговый поиск по каждой паре (entity × criterion)."""
    # Находим следующую непроработанную пару
    for ent in state["entities"]:
        for crit in state["criteria"]:
            if len(state["findings"][ent]) < state["criteria"].index(crit) + 1:
                query = f"{ent} {crit}"
                results = tavily.search(query, max_results=3)
                # Берём первые 2 предложения из первых результатов
                notes = []
                for r in results[:3]:
                    if r.text:
                        notes.append(r.text.split(".")[0] + ".")
                note = " ".join(notes) or f"Нет данных по {ent} и {crit}"
                state["findings"][ent].append(note)
                print(f"[{ent} × {crit}] найдено: {note}")
                return {"state": state}
    # Если все пары обработаны
    return {"state": state}

def build_table(state: CompareState) -> Dict[str, Any]:
    """Формируем Markdown‑таблицу."""
    header = "| Критерий | " + " | ".join(state["entities"]) + " |\n"
    separator = "|" + "---|" * (len(state["entities"]) + 1) + "\n"
    rows = ""
    for crit in state["criteria"]:
        row_cells = [crit]
        for ent in state["entities"]:
            idx = state["criteria"].index(crit)
            note = state["findings"][ent][idx] if len(state["findings"][ent]) > idx else "—"
            row_cells.append(note.replace("\n", " "))
        rows += "| " + " | ".join(row_cells) + " |\n"
    table = header + separator + rows
    state["final_table"] = table
    print("\n=== Итоговая таблица ===\n")
    print(table)
    return {"state": state}

def verdict(state: CompareState) -> Dict[str, Any]:
    """LLM формирует рекомендацию."""
    prompt = (
        f"На основе следующей таблицы:\n{state['final_table']}\n"
        "Напиши 2–4 предложения с рекомендацией, какая из трёх технологий "
        "подходит для каких кейсов. Ответ в чистом тексте."
    )
    response = llm.invoke(prompt)
    state["verdict"] = response.content.strip()
    print("\n=== Вердикт ===\n")
    print(state["verdict"])
    return {"state": state}

# ---------- Создание графа ----------
def create_compare_graph() -> StateGraph:
    graph = StateGraph(CompareState)

    # Добавляем узлы
    graph.add_node("plan_criteria", plan_criteria)
    graph.add_node("research_entity", research_entity)
    graph.add_node("build_table", build_table)
    graph.add_node("verdict", verdict)

    # Условный переход: если есть непроработанные пары → research_entity,
    # иначе → build_table
    def condition(state: CompareState):
        for ent in state["entities"]:
            if len(state["findings"][ent]) < len(state["criteria"]):
                return "research_entity"
        return "build_table"

    graph.add_conditional_edges(
        "plan_criteria",
        condition,
        {
            "research_entity": "research_entity",
            "build_table": "build_table",
        },
    )

    # После research_entity снова проверяем условие
    graph.add_conditional_edges(
        "research_entity",
        condition,
        {
            "research_entity": "research_entity",
            "build_table": "build_table",
        },
    )

    # Последние узлы
    graph.set_entry_point("plan_criteria")
    graph.add_edge("build_table", "verdict")
    graph.add_edge("verdict", "__end__")

    return graph

compare_graph = create_compare_graph()
```

---

### `main.py`

```python
# main.py
import sys
from agent import compare_graph, CompareState

def prompt_entities() -> list[str]:
    default = ["Chroma", "FAISS", "Qdrant"]
    print("Введите три сущности для сравнения (через запятую), "
          f"или нажмите Enter для использования по умолчанию: {', '.join(default)}")
    inp = input("> ").strip()
    if not inp:
        return default
    parts = [p.strip() for p in inp.split(",") if p.strip()]
    if len(parts) != 3:
        print("Нужно ровно три сущности. Попробуйте снова.")
        sys.exit(1)
    return parts

def main():
    entities = prompt_entities()
    initial_state: CompareState = {
        "entities": entities,
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
    }
    # Запускаем граф
    result = compare_graph.invoke(initial_state)
    print("\n=== Завершено ===")

if __name__ == "__main__":
    main()
```

---

## Как запустить

1. **Клонируйте репозиторий** (или создайте папку и скопируйте файлы).  
2. **Создайте виртуальное окружение**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Установите зависимости**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Получите ключ Tavily**  
   Зарегистрируйтесь на [tavily.com](https://tavily.com) и получите API‑ключ.

5. **Создайте файл `.env`** в корне проекта:

   ```dotenv
   TAVILY_API_KEY=YOUR_TAVILY_API_KEY_HERE
   ```

6. **Запустите демо**:

   ```bash
   python main.py
   ```

   Вы увидите:
   - Сгенерированные критерии сравнения.
   - Поиск по каждой паре `entity × criterion` (с выводом найденной заметки).
   - Итоговую Markdown‑таблицу.
   - Вердикт с рекомендацией.

---

## Проверка выполнения

| Компонент | Как проверить |
|-----------|---------------|
| **plan_criteria** | Убедитесь, что критерии генерируются динамически (не хардкод). |
| **research_entity** | В консоли выводятся строки вида `[entity × criterion] найдено: …`. |
| **Tavily** | Проверить, что в ответах содержатся реальные фрагменты текста из веб‑страниц. |
| **build_table** | Таблица имеет заголовок, разделитель и все ячейки заполнены (или «—»). |
| **verdict** | В конце выводится 2–4 предложения с рекомендацией. |
| **CLI** | При запуске можно ввести свои три сущности. |

---

## Возможные улучшения

- Добавить интерактивный режим выбора LLM (OpenAI vs Ollama).
- Сохранять результаты в файл `results.md`.
- Расширить поиск: использовать `tavily.search` с параметром `include_images=True`, если нужно.
- Внедрить кэширование результатов поиска, чтобы не делать лишние запросы.

---

## Заключение

Вы создали полностью работающий LangGraph‑агент, который:

1. Генерирует критерии сравнения через LLM.  
2. Пошагово ищет информацию в интернете с помощью Tavily.  
3. Формирует структурированный Markdown‑вывод и финальный вердикт.

Удачной работы!
