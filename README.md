# Повторный экзамен #2: Сравнительный обзор 3 сущностей (Tavily)

## Практическое задание: Сравнительный обзор в LangGraph (Tavily)

### Цель

Собрать **LangGraph**-агента, который по запросу студента строит **сравнительный обзор 3 сущностей** (технологии / продукта / подхода) в формате «общий план → по каждой сущности — Tavily-поиск → финальная таблица + вердикт». Итоговый результат должен быть оформлен как **таблица 3×N + явный вердикт** по итогам сравнения, а супервизор сначала определяет, по каким критериям сравнивать.

---

### Стек

- Python 3.10+
- `langgraph`, `langchain-openai` или `langchain-ollama`, `langchain-tavily` (или Tavily SDK)

#### Установка зависимостей

```bash
pip install langgraph langchain-openai langchain-tavily tavily-python python-dotenv
```

#### Настройка API ключей

Создайте файл `.env` в корневой директории проекта и добавьте в него ваш API ключ для Tavily:

```dotenv
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here # Или OLLAMA_BASE_URL для Ollama
```

---

### Что нужно сделать

#### 1. Состояние (State)

Определите класс состояния для графа, используя `TypedDict`. Это позволит строго типизировать данные, передаваемые между узлами графа.

```python
from typing import TypedDict, List, Dict, Optional

class CompareState(TypedDict):
    entities: List[str]                # 3 имени для сравнения (например, ["Chroma", "FAISS", "Qdrant"])
    criteria: List[str]                # 3–5 критериев, по которым будет проводиться сравнение
    findings: Dict[str, List[str]]     # Словарь: entity -> список заметок по каждому критерию
    final_table: Optional[str]         # Строка с финальной Markdown-таблицей
    verdict: Optional[str]             # Строка с итоговым вердиктом LLM
    # Дополнительные поля для управления циклом research_entity
    current_entity_idx: int            # Индекс текущей сущности, которую исследуем
    current_criterion_idx: int         # Индекс текущего критерия для исследования
```

#### 2. Узлы (Nodes)

Каждый узел представляет собой функцию, которая принимает текущее состояние графа, выполняет определенное действие и возвращает обновленное состояние.

| Узел | Действие | Описание |
|------|----------|----------|
| `plan_criteria` | LLM по `entities` формулирует 3–5 критериев сравнения | Этот узел использует LLM для генерации списка критериев, основываясь на предоставленных сущностях. Он инициализирует `findings` и индексы для последующего исследования. |
| `research_entity` | Одна итерация: берёт `entity` + каждый `criterion` → 1 Tavily-вызов (или 1 на пару) → короткая заметка | Этот узел выполняет поиск информации для одной пары "сущность-критерий" с помощью Tavily, затем суммаризирует найденное в короткую заметку с помощью LLM. Заметки сохраняются в `findings`. Узел также управляет индексами для перехода к следующей паре. |
| `build_table` | Из `findings` собирает markdown-таблицу: строки = критерии, колонки = сущности | После завершения всех исследований, этот узел формирует итоговую сравнительную таблицу в формате Markdown, используя накопленные заметки. |
| `verdict` | LLM: 2–4 предложения «какую сущность для какого кейса выбрать» | Финальный узел, который использует LLM для анализа сгенерированной таблицы и формулирования краткого вердикта с рекомендациями по выбору сущности для различных сценариев. |

#### 3. Граф (Graph)

Логика выполнения агента определяется структурой графа.

```text
START → plan_criteria → research_entity (×N: пока не все пары entity×criterion) → build_table → verdict → END
```

**Условие перехода:** После каждого выполнения `research_entity` граф должен проверять, остались ли необработанные пары "сущность × критерий". Если да, то снова вызывается `research_entity`. В противном случае, граф переходит к узлу `build_table`.

#### 4. Демонстрация (CLI)

Реализуйте простой интерфейс командной строки для запуска агента.

- **Тема по умолчанию:** *«Сравни 3 векторные БД: Chroma, FAISS, Qdrant»*.
- **Вывод в консоль:**
    - План критериев, сгенерированный LLM.
    - Сообщения о ходе исследования: `[entity × criterion] найдено: ...` для каждой итерации `research_entity`.
    - Итоговая сравнительная таблица в Markdown.
    - Финальный вердикт LLM.
- **Опционально:** Предоставьте возможность пользователю ввести свои 3 сущности для сравнения.

---

### Критерии зачёта

Ваша реализация будет оцениваться по следующим пунктам:

| Компонент | Проверяется |
|-----------|-------------|
| `plan_criteria` | LLM генерирует критерии по `entities`, а не хардкод. Критерии должны быть релевантными. |
| Цикл `research_entity` | Пошаговый обход пар entity×criterion. `findings` корректно накапливается и обновляется в состоянии. |
| Tavily | Реальный web search для каждой пары (или хотя бы для большинства). Результаты поиска должны быть использованы для формирования заметок, а не выдуманные ссылки или информация. |
| `build_table` | Markdown-таблица, все строки (критерии) и колонки (сущности) заполнены соответствующими заметками из `findings`. Формат таблицы должен быть корректным. |
| `verdict` | Отдельный узел с LLM. Явная рекомендация по кейсам, основанная на данных таблицы. Вердикт должен быть кратким (2-4 предложения). |
| Сдача | CLI запускается без ошибок. Вывод в консоль соответствует требованиям. |
| Стек | Использование `langgraph`, `langchain-openai` (или `langchain-ollama`), `langchain-tavily` (или Tavily SDK). |
| Переменные окружения | `TAVILY_API_KEY` и `OPENAI_API_KEY` (или `OLLAMA_BASE_URL`) загружаются из `.env`. |

---

### Ссылки

- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangChain Tavily Integration](https://docs.langchain.com/oss/python/integrations/providers/tavily)

---

### Ожидания преподавателя

**TASK_TYPE:** code
**STACK:** Python 3.10+, langgraph, langchain-openai, langchain-ollama, langchain-tavily, tavily-python, python-dotenv
**FILES:**
- `main.py`: Основной скрипт с реализацией LangGraph агента.
- `.env`: Файл для хранения `TAVILY_API_KEY` и `OPENAI_API_KEY` (или `OLLAMA_BASE_URL`).
**RUN_COMMAND:** `python main.py`

**REQUIREMENTS_CHECKLIST:**
- [x] **Состояние (CompareState)**: Определить класс `CompareState` как `TypedDict` со следующими полями:
    - [x] `entities: list[str]` (3 имени для сравнения)
    - [x] `criteria: list[str]` (3–5 критериев)
    - [x] `findings: dict[str, list[str]]` (entity -> список заметок по критериям)
    - [x] `final_table: str | None`
    - [x] `verdict: str | None`
    - [x] Дополнительные поля для управления циклом `research_entity` (`current_entity_idx`, `current_criterion_idx`).
- [x] **Узел `plan_criteria`**: Реализовать узел, где LLM по `entities` формулирует 3–5 критериев сравнения.
    - [x] LLM генерирует критерии по `entities`, а не хардкод.
- [x] **Узел `research_entity`**: Реализовать узел, который за одну итерацию:
    - [x] Берёт `entity` + каждый `criterion`.
    - [x] Выполняет 1 Tavily-вызов (или 1 на пару).
    - [x] Генерирует короткую заметку.
    - [x] Цикл `research_entity`: Пошаговый обход пар entity×criterion, `findings` накапливается.
    - [x] Tavily: Реальный web search для каждой пары (или хотя бы для большинства), не выдуманные ссылки.
- [x] **Узел `build_table`**: Реализовать узел, который из `findings` собирает markdown-таблицу.
    - [x] Строки = критерии, колонки = сущности.
    - [x] Markdown-таблица, все строки и колонки заполнены.
- [x] **Узел `verdict`**: Реализовать узел, где LLM генерирует 2–4 предложения «какую сущность для какого кейса выбрать».
    - [x] Отдельный узел с LLM.
    - [x] Явная рекомендация по кейсам.
- [x] **Граф**: Построить граф со следующей логикой:
    - [x] `START → plan_criteria → research_entity (×N: пока не все пары entity×criterion) → build_table → verdict → END`
    - [x] Условие: пока есть необработанные пары — снова `research_entity` (с обновлением индекса в state), иначе → `build_table`.
- [x] **Демо (CLI)**: Реализовать CLI для запуска агента.
    - [x] Тема по умолчанию: *«Сравни 3 векторные БД: Chroma, FAISS, Qdrant»*.
    - [x] Вывод в консоль: план критериев.
    - [x] Вывод в консоль: `[entity × criterion] найдено: ...` (для каждой итерации `research_entity`).
    - [x] Вывод в консоль: итоговая таблица + вердикт.
    - [ ] Опционально: интерактивный режим — ввести свои 3 сущности. (Будет реализовано)
- [x] **Сдача**:
    - [x] CLI работает.
    - [ ] README (данный документ).
    - [x] Запускается.
- [x] **Стек**: Использовать `langgraph`, `langchain-openai` или `langchain-ollama`, `langchain-tavily` (или Tavily SDK).
- [x] **Переменные окружения**: `TAVILY_API_KEY` загружается из `.env`.

**PLAN:**

1.  **Настройка окружения и зависимостей**:
    *   Создать `main.py` и `.env`.
    *   Установить необходимые пакеты: `pip install langgraph langchain-openai langchain-tavily tavily-python python-dotenv`.
    *   Добавить `TAVILY_API_KEY=...` и `OPENAI_API_KEY=...` (или `OLLAMA_BASE_URL=...`) в `.env`.
    *   Импортировать `load_dotenv` и загрузить переменные окружения.
    *   Инициализировать LLM (например, `ChatOpenAI` или `ChatOllama`) и `TavilySearch`.

2.  **Определение состояния (CompareState)**:
    *   В `main.py` определить `TypedDict` `CompareState` согласно заданию, включая `current_entity_idx` и `current_criterion_idx`.

3.  **Реализация узлов (Nodes)**:

    *   **`plan_criteria(state: CompareState) -> CompareState`**:
        *   Использовать LLM с промптом: "Ты — эксперт по технологиям. Сформулируй 3-5 ключевых критериев для сравнения следующих сущностей: {state['entities']}. Выведи только список критериев, разделенных запятыми."
        *   Парсить ответ LLM в `list[str]` и обновить `state['criteria']`.
        *   Инициализировать `state['findings']` как пустой словарь, `state['current_entity_idx'] = 0`, `state['current_criterion_idx'] = 0`.

    *   **`research_entity(state: CompareState) -> CompareState`**:
        *   Получить текущую сущность и критерий по `state['current_entity_idx']` и `state['current_criterion_idx']`.
        *   Сформировать поисковый запрос для Tavily: "Что такое {entity} и как оно относится к {criterion}?" или "Сравнение {entity} по критерию {criterion}".
        *   Выполнить `TavilySearch().invoke(query)`.
        *   Использовать LLM для суммаризации результатов поиска в короткую заметку: "На основе следующей информации: {tavily_results}, сформулируй короткую заметку о {entity} по критерию {criterion}. Ответь кратко, 1-2 предложения."
        *   Добавить заметку в `state['findings'][entity]` (создать список, если его нет).
        *   Обновить `state['current_criterion_idx']`. Если все критерии для текущей сущности обработаны, сбросить `current_criterion_idx = 0` и инкрементировать `current_entity_idx`.

    *   **`build_table(state: CompareState) -> CompareState`**:
        *   Сформировать Markdown-таблицу.
        *   Заголовок: `| Критерий | Сущность 1 | Сущность 2 | Сущность 3 |`
        *   Разделитель: `|---|---|---|---|`
        *   Для каждого критерия: `| {criterion} | {finding_entity1_criterionX} | {finding_entity2_criterionX} | {finding_entity3_criterionX} |`
        *   Сохранить таблицу в `state['final_table']`.

    *   **`verdict(state: CompareState) -> CompareState`**:
        *   Использовать LLM с промптом: "На основе следующей сравнительной таблицы: {state['final_table']}, дай краткий вердикт (2-4 предложения), какую сущность для какого кейса лучше выбрать. Укажи преимущества и недостатки каждой сущности."
        *   Сохранить вердикт в `state['verdict']`.

4.  **Определение условных переходов (Conditional Edges)**:

    *   **`should_continue_research(state: CompareState) -> str`**:
        *   Проверить, остались ли необработанные пары `entity × criterion`.
        *   Если `state['current_entity_idx'] < len(state['entities'])` (или `state['current_entity_idx'] == len(state['entities']) - 1` и `state['current_criterion_idx'] < len(state['criteria'])`), вернуть `"research_entity"`.
        *   Иначе, вернуть `"build_table"`.

5.  **Построение графа (LangGraph)**:

    *   Инициализировать `StateGraph(CompareState)`.
    *   Добавить узлы: `plan_criteria`, `research_entity`, `build_table`, `verdict`.
    *   Установить начальный узел: `set_entry_point("plan_criteria")`.
    *   Добавить ребра:
        *   `add_edge("plan_criteria", "research_entity")`
        *   `add_conditional_edges("research_entity", should_continue_research)`
        *   `add_edge("build_table", "verdict")`
    *   Установить конечный узел: `set_finish_point("verdict")`.
    *   Скомпилировать граф: `app = graph.compile()`.

6.  **Реализация CLI**:

    *   В `main.py` добавить блок `if __name__ == "__main__":`.
    *   Предложить пользователю ввести 3 сущности через запятую или нажать Enter для использования дефолтных (`Chroma, FAISS, Qdrant`).
    *   Инициализировать `initial_state = CompareState(...)`.
    *   Запустить граф: `for s in app.stream(initial_state): print(s)`.
    *   Выводить промежуточные результаты (критерии, заметки по `[entity × criterion] найдено: ...`).
    *   В конце вывести `final_table` и `verdict`.
