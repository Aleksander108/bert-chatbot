# BERT Semantic ChatBot / Семантический Чат-Бот на BERT

---

## English

A semantic chatbot that provides answers to user queries by finding the most semantically similar questions in an Excel database. The bot uses Sentence Transformer models to convert questions into numerical vectors and employs cosine similarity to find the closest match in the database. It features both a command-line interface (CLI) and a web interface built with FastAPI.

### Prerequisites

- Python 3.10 or higher
- Git

### Local Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aleksander108/bert-chatbot.git
    cd bert-chatbot
    ```

2.  **Create and activate a virtual environment:**
    (Using `venv`, standard Python virtual environment manager)
    ```bash
    python3 -m venv .venv
    ```
    Activate the environment:
    -   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .venv\\Scripts\\activate
        ```

3.  **Install dependencies:**
    The project uses `requirements.txt` to manage dependencies. You can use `uv` (if installed, faster) or `pip`.
    ```bash
    # Using uv (recommended if installed)
    uv pip install -r requirements.txt
    ```
    Alternatively, using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web Interface (FastAPI)

The main way to interact with the chatbot is via its web interface.

1.  **Ensure the database file is present:**
    The application expects the Excel database file named `DATA_CHAT_2.xlsx` to be in the root directory of the project. Make sure it's there.

2.  **Run the FastAPI application:**
    From the project root directory, run the following command:
    ```bash
    uvicorn main:app --reload
    ```
    This command starts the Uvicorn server, and `--reload` enables auto-reloading when code changes, which is useful for development.

3.  **Open in your browser:**
    Once the server is running, open your web browser and go to:
    ```
    http://127.0.0.1:8000
    ```

4.  **Note on Vector Cache:**
    On the first run, the application will generate a `vector_cache.pkl` file. This file stores the precomputed embeddings (vectors) for the questions in your database, making subsequent startups and queries much faster. This process might take a few minutes depending on the size of your database. This file is gitignored and should not be committed to the repository.

### (Optional) Running the Command-Line Interface (CLI)

The bot also offers a command-line interface.

1.  **Ensure the `bert-chatbot` package is recognized:**
    If you installed dependencies using `uv pip install -r requirements.txt` or `pip install -r requirements.txt` without `-e .` (editable install of the local package), the `bert-chatbot` command might not be directly available. For CLI usage, an editable install might be needed, or you can run modules directly.

    A simpler way to ensure CLI scripts work is to ensure your virtual environment is active and you are in the project root. The CLI is primarily defined in `bert_chatbot/cli.py`.

2.  **Interactive Chat (CLI):**
    ```bash
    python -m bert_chatbot chat interactive DATA_CHAT_2.xlsx
    ```
    -   Replace `DATA_CHAT_2.xlsx` with the actual path to your database file if it's different or not in the root.
    -   You can use options like `--threshold 0.4` or `--cache-file my_cache.pkl`.

3.  **Single Question Mode (CLI):**
    ```bash
    python -m bert_chatbot chat ask DATA_CHAT_2.xlsx "Your question here?"
    ```
    -   Use `--debug` to see more detailed output.

---

## Русский

Семантический чат-бот, который отвечает на запросы пользователей, находя наиболее семантически похожие вопросы в базе данных Excel. Бот использует модели Sentence Transformer для преобразования вопросов в числовые векторы и применяет косинусное сходство для поиска наиболее близкого совпадения в базе данных. Имеет как интерфейс командной строки (CLI), так и веб-интерфейс, созданный с помощью FastAPI.

### Предварительные требования

- Python 3.10 или выше
- Git

### Локальная Настройка и Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/Aleksander108/bert-chatbot.git
    cd bert-chatbot
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    (Используя `venv`, стандартный менеджер виртуальных окружений Python)
    ```bash
    python3 -m venv .venv
    ```
    Активируйте окружение:
    -   На macOS и Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   На Windows:
        ```bash
        .venv\\Scripts\\activate
        ```

3.  **Установите зависимости:**
    Проект использует `requirements.txt` для управления зависимостями. Вы можете использовать `uv` (если установлен, работает быстрее) или `pip`.
    ```bash
    # Используя uv (рекомендуется, если установлен)
    uv pip install -r requirements.txt
    ```
    Как альтернатива, используя `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Запуск Веб-интерфейса (FastAPI)

Основной способ взаимодействия с чат-ботом — через его веб-интерфейс.

1.  **Убедитесь, что файл базы данных на месте:**
    Приложение ожидает, что файл базы данных Excel с именем `DATA_CHAT_2.xlsx` будет находиться в корневой директории проекта. Убедитесь, что он там.

2.  **Запустите приложение FastAPI:**
    Из корневой директории проекта выполните следующую команду:
    ```bash
    uvicorn main:app --reload
    ```
    Эта команда запускает сервер Uvicorn, а флаг `--reload` включает автоматическую перезагрузку при изменении кода, что удобно для разработки.

3.  **Откройте в браузере:**
    После запуска сервера откройте ваш веб-браузер и перейдите по адресу:
    ```
    http://127.0.0.1:8000
    ```

4.  **Примечание о кеше векторов:**
    При первом запуске приложение сгенерирует файл `vector_cache.pkl`. Этот файл хранит предварительно вычисленные эмбеддинги (векторы) для вопросов из вашей базы данных, что значительно ускоряет последующие запуски и обработку запросов. Этот процесс может занять несколько минут в зависимости от размера вашей базы данных. Этот файл добавлен в `.gitignore` и не должен коммититься в репозиторий.

### (Опционально) Запуск через Интерфейс Командной Строки (CLI)

Бот также предлагает интерфейс командной строки.

1.  **Убедитесь, что пакет `bert-chatbot` распознается:**
    Если вы устанавливали зависимости с помощью `uv pip install -r requirements.txt` или `pip install -r requirements.txt` без флага `-e .` (редактируемая установка локального пакета), команда `bert-chatbot` может быть недоступна напрямую. Для использования CLI может потребоваться редактируемая установка или вы можете запускать модули напрямую.

    Более простой способ обеспечить работу CLI-скриптов — убедиться, что ваше виртуальное окружение активировано и вы находитесь в корне проекта. Логика CLI в основном определена в `bert_chatbot/cli.py`.

2.  **Интерактивный чат (CLI):**
    ```bash
    python -m bert_chatbot chat interactive DATA_CHAT_2.xlsx
    ```
    -   Замените `DATA_CHAT_2.xlsx` на фактический путь к вашему файлу базы данных, если он отличается или находится не в корне.
    -   Вы можете использовать опции, такие как `--threshold 0.4` или `--cache-file my_cache.pkl`.

3.  **Режим одного вопроса (CLI):**
    ```bash
    python -m bert_chatbot chat ask DATA_CHAT_2.xlsx "Ваш вопрос сюда?"
    ```
    -   Используйте флаг `--debug` для получения более детального вывода.

---

## Features

- Semantic matching using TF-IDF vectorization
- Caching system for faster subsequent runs
- Interactive command-line interface
- Single question mode via command line
- Debug mode to show similarity scores and matched questions
- Configurable similarity threshold

## How It Works

1. **TF-IDF Vectorization**: Converts text questions into numerical vectors
2. **Cosine Similarity**: Measures the similarity between the user query and all questions in the database
3. **Threshold Filtering**: Returns the most similar answer if it's above the configured threshold

## Future Improvements

- Advanced NLP models like BERT for better semantic understanding
- Multi-language support
- REST API interface
- Web or desktop user interface
- Database management tools
- Analytics for tracking common questions