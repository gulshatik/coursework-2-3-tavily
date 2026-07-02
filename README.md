# Повторный экзамен #2: Сравнительный обзор 3 сущностей (Tavily)

## Задание

Создать LangGraph‑агента, который по запросу студента строит **сравнительный обзор трёх сущностей** (технологий/продуктов/подходов) в формате:

1. Общий план →  
2. По каждой сущности – Tavily‑поиск →  
3. Финальная таблица + явный вердикт.

Итоговый результат должен быть оформлен как **таблица 3×N** (строки — критерии, колонки — сущности) и **вердикт** по итогам сравнения. Супервизор сначала определяет, по каким критериям сравнивать.

---

## Стек

| Пакет | Версия | Зачем |
|-------|--------|-------|
| `langgraph` | ≥ 0.1.0 | Создание графа состояний |
| `langchain-openai` / `langchain-ollama` | ≥ 0.2.0 | LLM‑ноды (ChatOpenAI, ChatOllama) |
| `langchain-tavily` | ≥ 0.1.0 | Интеграция Tavily в LangChain |
| `tavily-python` | ≥ 0.3.0 | SDK для прямого вызова API |
| `python-dotenv` | ≥ 1.0.0 | Загрузка переменных окружения |

```bash
pip install langgraph langchain-openai langchain-tavily tavily-python python-dotenv
```

В `.env` укажите:

```dotenv
TAVILY_API_KEY=YOUR_TAVILY_KEY_HERE
OPENAI_API_KEY=YOUR_OPENAI_KEY   # если используете OpenAI
# или OLLAMA_BASE_URL=http://localhost:11434  # если используете Ollama
```

---

## Что нужно сделать

### 1. Состояние

```python
from typing import TypedDict, List, Dict, Optional

class CompareState(TypedDict):
    entities: List[str]                # 3 имени для сравнения
    criteria: List[str]                # 3–5 критериев
    findings: Dict[str, List[str]]     # entity -> список заметок по критериям
    final_table: Optional[str]
    verdict: Optional[str]
```

### 2. Узлы

| Узел | Действие |
|------|----------|
| `plan_criteria` | LLM генерирует 3–5 критериев сравнения на основе списка `entities`. |
| `research_entity` | Итеративно берёт очередную пару `(entity, criterion)`, делает Tavily‑поиск (или два шага: поиск + извлечение текста), формирует короткую заметку и добавляет её в `findings[entity]`. |
| `build_table` | На основе `findings` строит Markdown‑таблицу: строки — критерии, колонки — сущности. |
| `verdict` | LLM читает таблицу и выдаёт 2–4 предложения с рекомендацией по выбору сущности для конкретных кейсов. |

### 3. Граф

```
START
 └─> plan_criteria
      └─> research_entity (loop until all entity×criterion pairs processed)
           └─> build_table
                └─> verdict
                     └─> END
```

**Логика цикла:**

```python
while not all_pairs_processed(state):
    state = await graph.nodes["research_entity"](state)
```

### 4. CLI‑демо

Запуск без аргументов:

```bash
python main.py
```

Выводит:

1. Сгенерированные критерии.
2. Для каждой пары `entity × criterion` сообщение вида:  
   `[Chroma × скорость] найдено: …`
3. Итоговую Markdown‑таблицу.
4. Вердикт.

Опционально можно передать свои три сущности через аргументы:

```bash
python main.py --entities "LangChain" "LlamaIndex" "Haystack"
```

или в интерактивном режиме, если аргумент не указан.

---

## Критерии зачёта

| Компонент | Что проверяется |
|-----------|-----------------|
| `plan_criteria` | LLM генерирует критерии динамически (не хардкод). |
| Цикл `research_entity` | Пошаговый обход всех пар, накопление заметок в `findings`. |
| Tavily | Реальный web‑search для каждой пары; ссылки/тексты не выдуманы. |
| `build_table` | Markdown‑таблица с заполненными строками и колонками. |
| `verdict` | Отдельный узел LLM, выдаёт явную рекомендацию. |
| CLI | Запускается из командной строки, выводит все шаги. |

---

## Структура проекта

```
.
├── .env
├── main.py          # точка входа, CLI
├── agent.py         # состояние, узлы, граф
├── requirements.txt
└── README.md        # это файл
```

### `main.py`

```python
import argparse
from dotenv import load_dotenv
from agent import build_agent

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Сравнительный обзор 3 сущностей")
    parser.add_argument("--entities", nargs=3, help="Три сущности для сравнения")
    args = parser.parse_args()

    if not args.entities:
        # интерактивный ввод
        entities = input("Введите три сущности через запятую: ").split(",")
        entities = [e.strip() for e in entities[:3]]
    else:
        entities = args.entities

    agent = build_agent()
    state = {"entities": entities, "criteria": [], "findings": {}, 
             "final_table": None, "verdict": None}

    # Запускаем граф
    for event in agent.stream(state):
        if event["type"] == "plan_criteria":
            print("\nКритерии сравнения:")
            for c in event["state"]["criteria"]:
                print(f" - {c}")
        elif event["type"] == "research_entity":
            ent, crit = event["pair"]
            note = event["note"]
            print(f"[{ent} × {crit}] найдено: {note[:120]}...")
        elif event["type"] == "build_table":
            print("\nИтоговая таблица:")
            print(event["state"]["final_table"])
        elif event["type"] == "verdict":
            print("\nВердикт:")
            print(event["state"]["verdict"])

if __name__ == "__main__":
    main()
```

### `agent.py`

```python
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os

class CompareState(TypedDict):
    entities: List[str]
    criteria: List[str]
    findings: Dict[str, List[str]]
    final_table: Optional[str]
    verdict: Optional[str]

# ---------- Узлы ----------
def plan_criteria(state: CompareState) -> CompareState:
    llm = ChatOpenAI()
    prompt = f"Сгенерируй 3–5 критериев сравнения для следующих сущностей: {', '.join(state['entities'])}."
    response = llm.invoke(prompt)
    state["criteria"] = [c.strip() for c in response.split("\n") if c]
    return state

def research_entity(state: CompareState) -> CompareState:
    # Находим первую непроработанную пару
    for entity in state["entities"]:
        if entity not in state["findings"]:
            state["findings"][entity] = []
        for crit in state["criteria"]:
            if len(state["findings"][entity]) < len(state["criteria"]):
                # Выполняем Tavily‑поиск
                tavily = TavilySearch()
                query = f"{entity} {crit}"
                result = tavily.run(query)
                snippet = result[0]["content"][:200] if result else "Нет данных"
                state["findings"][entity].append(snippet)
                # Возвращаем событие для CLI
                return {"type": "research_entity", "pair": (entity, crit), "note": snippet}
    return {"type": "build_table"}

def build_table(state: CompareState) -> CompareState:
    rows = []
    for crit in state["criteria"]:
        row = [crit]
        for ent in state["entities"]:
            notes = state["findings"].get(ent, [])
            idx = state["criteria"].index(crit)
            note = notes[idx] if idx < len(notes) else ""
            row.append(note.replace("\n", " "))
        rows.append("| " + " | ".join(row) + " |")
    header = "| Критерий | " + " | ".join(state["entities"]) + " |"
    separator = "|" + "---|" * (len(state["entities"]) + 1)
    table = "\n".join([header, separator] + rows)
    state["final_table"] = table
    return {"type": "build_table", "state": state}

def verdict(state: CompareState) -> CompareState:
    llm = ChatOpenAI()
    prompt = f"На основе следующей таблицы:\n{state['final_table']}\n\nСформулируй 2–4 предложения с рекомендацией, какая сущность лучше подходит для разных кейсов."
    response = llm.invoke(prompt)
    state["verdict"] = response
    return {"type": "verdict", "state": state}

# ---------- Граф ----------
def build_agent():
    graph = StateGraph(CompareState)

    graph.add_node("plan_criteria", plan_criteria)
    graph.add_node("research_entity", research_entity)
    graph.add_node("build_table", build_table)
    graph.add_node("verdict", verdict)

    # Переходы
    graph.set_entry_point("plan_criteria")
    graph.add_edge("plan_criteria", "research_entity")
    graph.add_conditional_edges(
        "research_entity",
        lambda x: "build_table" if x["type"] == "build_table" else "research_entity",
    )
    graph.add_edge("build_table", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile()
```

### `requirements.txt`

```text
langgraph>=0.1.0
langchain-openai>=0.2.0
langchain-tavily>=0.1.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
```

---

## Как запустить

```bash
# 1. Создайте виртуальное окружение (необязательно)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Скопируйте ваш Tavily‑ключ в .env
echo "TAVILY_API_KEY=YOUR_KEY" > .env

# 4. Запустите демо
python main.py
```

Вы увидите пошаговый вывод: критерии, результаты поиска для каждой пары, итоговую таблицу и вердикт.

---

## Что дальше?

- Добавьте поддержку Ollama вместо OpenAI (поменяйте `ChatOpenAI` на `ChatOllama`).
- Расширьте `research_entity`, чтобы обрабатывать несколько результатов и выбирать наиболее релевантный.
- Внедрите кэширование запросов, чтобы не делать лишние вызовы Tavily при повторных запусках.

Удачной разработки!
