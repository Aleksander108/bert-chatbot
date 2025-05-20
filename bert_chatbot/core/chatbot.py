"""Core implementation of the semantic chatbot."""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, List, TypeVar, Optional, ClassVar, Any, Set, cast
import re # For tokenization

import numpy as np
import pandas as pd # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# sentence_transformers
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ActualSentenceTransformer # type: ignore
    import numpy.typing as npt
    NDArrayFloat32 = npt.NDArray[np.float32]
else:
    # Directly import the real SentenceTransformer;
    # Python will raise ImportError if it's not found at runtime.
    from sentence_transformers import SentenceTransformer as _ActualSentenceTransformer # type: ignore
    NDArrayFloat32 = np.ndarray

# FAISS removed
# _faiss_module: Optional[Any] = None
# _FaissIndexTypeHint: Any 
# try:
#     import faiss as _imported_faiss_module 
#     _faiss_module = _imported_faiss_module
#     if TYPE_CHECKING:
#         _FaissIndexTypeHint = _faiss_module.Index 
#     else:
#         _FaissIndexTypeHint = getattr(_faiss_module, "Index", object)
# except ImportError:
#     if TYPE_CHECKING:
#         _FaissIndexTypeHint = TypeVar('_FaissIndexTypeHint') 
#     else:
#         _FaissIndexTypeHint = object

# spaCy for linguistic processing (noun extraction)
# _spacy_module: Optional[types.ModuleType]
# _SpacyLanguageTypeHint = Any # Placeholder for spacy.language.Language

# if TYPE_CHECKING:
#     # Make it a forward reference string if direct import is problematic for linter
#     _SpacyLanguageTypeHintStr = "spacy.language.Language"
#     from spacy.language import Language as _ActualSpacyLanguageInTypeCheck
#     _SpacyLanguageTypeHint = _ActualSpacyLanguageInTypeCheck
# elif _spacy_module is not None:
#     _SpacyLanguageTypeHint = getattr(_spacy_module, "Language", object)
# else:
#     _SpacyLanguageTypeHint = object # Fallback if spacy is not even imported

# try:
#     import spacy as _imported_spacy_module # type: ignore
#     _spacy_module = _imported_spacy_module
#     if not TYPE_CHECKING and _spacy_module:
#         _SpacyLanguageTypeHint = getattr(_spacy_module, "Language", _SpacyLanguageType)
# except ImportError:
#     if not TYPE_CHECKING:
#         _SpacyLanguageTypeHint = TypeVar("_SpacyLanguageTypeHint_fallback") # Fallback if spacy not found
#     # _spacy_module remains None


if TYPE_CHECKING:
    import numpy.typing as npt
    T = TypeVar('T', bound=np.generic)
else:
    pass


class SemanticChatBot:
    """A chatbot that finds answers using semantic similarity with BERT transformer model."""

    model: _ActualSentenceTransformer
    database_path: str
    # faiss_index: Optional[_FaissIndexTypeHint] # Removed
    # nlp: Optional[_SpacyLanguageTypeHint]       
    similarity_threshold: float
    cache_file: str
    model_name: str
    # keyword_match_threshold: float # REMOVED
    # synonyms: Dict[str, List[str]] # REMOVED
    # topic_groups: Dict[str, Set[str]] # REMOVED
    df: pd.DataFrame
    questions: List[str]
    answers: List[str]
    sources: List[str] # Added sources
    question_vectors: NDArrayFloat32
    # question_keywords: List[Set[str]] # REMOVED
    # question_topics: List[Dict[str, float]] # REMOVED
    # stemmer: SnowballStemmer # REMOVED as keyword logic is removed
    # stemmed_stop_words: Set[str] # REMOVED
    link_col: Optional[str]
    db_vocabulary: Set[str]

    # For problematic word filtering - REMOVED as per user request
    # OFFENSIVE_OR_LOW_INFO_STEMS: Set[str] = {"хуй", "пизд", "гомик", "жоп"} # stem "жопа"
    
    FAIL_RESPONSES: ClassVar[List[str]] = [
        "Ответа по вашему вопросу в базе нет. Проверьте формулировку.",
        "Данных по такой формулировке нет. Попробуйте спросить по-другому.",
        "Информацию по вашему запросу найти не удалось, уточните вопрос.",
    ]

    RUSSIAN_STOPWORDS: ClassVar[Set[str]] = {
        "и", "в", "на", "с", "о", "не", "бы", "но", "из", "по", "за", "для", "до", "же", "ли", "быть", 
        "мой", "твой", "свой", "наш", "ваш", "их", "это", "тот", "этот", "так", "вот", "как", 
        "разве", "что", "где", "когда", "почему", "зачем", "какой", "который", "или", "если", "то",
        "а", "без", "более", "бы", "был", "была", "были", "было", "вам", "вас", "весь", "во", "вот",
        "все", "всего", "всех", "вы", "где", "да", "даже", "для", "до", "его", "ее", "ей", "ею",
        "если", "есть", "еще", "же", "за", "здесь", "из", "из-за", "или", "им", "ими", "их", "к",
        "как", "ко", "когда", "кто", "ли", "либо", "мне", "может", "мы", "на", "надо", "наш", "не",
        "него", "нее", "нет", "ни", "них", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно",
        "от", "очень", "по", "под", "при", "с", "со", "так", "также", "такой", "там", "те", "тем",
        "то", "того", "тоже", "той", "только", "том", "ты", "у", "уже", "хорошо", "хоть", "чего", "чем",
        "что", "чтобы", "чье", "чья", "эта", "эти", "это", "я"
    }

    def __init__(
        self,
        database_path: str,
        similarity_threshold: float = 0.7,
        cache_file: str = "vector_cache.pkl",
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        # keyword_match_threshold: float = 0.2, # REMOVED
    ) -> None:
        """Initialize the chatbot with the database and BERT model.

        Args:
            database_path: Path to the Excel database file
            similarity_threshold: Minimum combined score to consider a strong match (0-1)
            cache_file: File to cache the vectorized questions
            model_name: Name of the sentence-transformers model to use
            # keyword_match_threshold: Minimum percentage of keywords that should match (0-1) # REMOVED

        """
        # print(f"DEBUG SemanticChatBot __init__: Загрузка с параметрами:")
        # print(f"DEBUG __init__: database_path='{database_path}'")
        # print(f"DEBUG __init__: similarity_threshold={similarity_threshold}")
        # print(f"DEBUG __init__: cache_file='{cache_file}'")
        # print(f"DEBUG __init__: model_name='{model_name}'")

        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.model_name = model_name
        self.link_col = None 
        self.db_vocabulary: Set[str] = set()
        
        # FAISS checks removed
        # if _faiss_module is None:
        #     raise ImportError("FAISS library is not installed...")
        # if _spacy_module is None:
        #     print("DEBUG __init__: spaCy library is not installed or _spacy_module is None. Noun extraction will be disabled.")
        #     self.nlp = None # Explicitly set to None if module not available
        # else:
        #     print("DEBUG __init__: spaCy library _spacy_module найден.")
        #     try:
        #         print(f"DEBUG __init__: Попытка загрузки spaCy модели 'ru_core_news_sm'")
        #         self.nlp = cast(_SpacyLanguageTypeHint, _spacy_module.load("ru_core_news_sm"))
        #         print(f"DEBUG __init__: spaCy модель 'ru_core_news_sm' успешно загружена. self.nlp: {self.nlp}")
        #     except OSError:
        #         print("DEBUG __init__: spaCy Russian model 'ru_core_news_sm' not found. Попытка загрузить blank('ru').")
        #         try:
        #             self.nlp = cast(_SpacyLanguageTypeHint, _spacy_module.blank("ru")) 
        #             print(f"DEBUG __init__: spaCy blank('ru') модель загружена. self.nlp: {self.nlp}")
        #         except Exception as e_blank:
        #             print(f"DEBUG __init__: Не удалось загрузить blank spaCy модель: {e_blank}. Noun extraction disabled.")
        #             self.nlp = None
        #     except Exception as e_load:
        #         print(f"DEBUG __init__: Ошибка загрузки spaCy модели: {e_load}. Noun extraction disabled.")
        #         self.nlp = None

        # REMOVED Stemmer initialization
        # try:
        #     self.stemmer = SnowballStemmer("russian")
        # except ImportError:
        #     raise ImportError("NLTK's SnowballStemmer is required...")

        try:
            # print(f"DEBUG __init__: Попытка загрузки SentenceTransformer модели: '{self.model_name}'")
            self.model = _ActualSentenceTransformer(self.model_name)
            # print(f"DEBUG __init__: SentenceTransformer модель '{self.model_name}' успешно загружена. self.model: {self.model}")
        except Exception as e:
            # print(f"DEBUG __init__: Ошибка загрузки SentenceTransformer модели '{self.model_name}': {str(e)}")
            raise ValueError(f"Failed to initialize semantic model '{self.model_name}': {str(e)}")
            
        # REMOVED keyword_match_threshold initialization
        # self.keyword_match_threshold = keyword_match_threshold
        
        # REMOVED synonyms initialization
        # self.synonyms = { ... } # Content of synonyms dictionary removed for brevity
        
        # REMOVED topic_groups initialization
        # raw_topic_groups = { ... } # Content of topic_groups dictionary removed for brevity
        # self.topic_groups = {}
        # if self.nlp: 
        #     for topic, keywords in raw_topic_groups.items():
        #         stemmed_keywords = {self._stem_word(kw) for kw in keywords if self._stem_word(kw)}
        #         if stemmed_keywords: 
        #             self.topic_groups[topic] = stemmed_keywords
        # else: 
        #     self.topic_groups = raw_topic_groups

        # REMOVED stemmed_stop_words initialization
        # stop_words_russian = [...] 
        # self.stemmed_stop_words = {self.stemmer.stem(word) for word in stop_words_russian}

        try:    
            self.df = pd.read_excel(database_path) # type: ignore[attr-defined]
            # print(f"DEBUG __init__: Excel файл '{database_path}' успешно загружен. Строк: {len(self.df)}")
        except FileNotFoundError:
            # print(f"DEBUG __init__: Excel файл '{database_path}' не найден. Инициализация с пустым DataFrame.")
            self.df = pd.DataFrame(columns=["вопрос", "ответ", "ссылка"])
        except Exception as e:
            # print(f"DEBUG __init__: Ошибка чтения Excel файла '{database_path}': {e}. Инициализация с пустым DataFrame.")
            self.df = pd.DataFrame(columns=["вопрос", "ответ", "ссылка"])

        question_col_name = "вопрос"
        answer_col_name = "ответ"
        link_col_name = "ссылка"

        question_col = self._get_column_case_insensitive(list(self.df.columns), question_col_name)
        answer_col = self._get_column_case_insensitive(list(self.df.columns), answer_col_name)
        self.link_col = self._get_column_case_insensitive(list(self.df.columns), link_col_name, required=False)
        # print(f"DEBUG __init__: Имена колонок: вопрос='{question_col}', ответ='{answer_col}', ссылка='{self.link_col}'")

        if self.df.empty or question_col is None or answer_col is None:
            if self.df.empty:
                 # print(f"DEBUG __init__: DataFrame пуст. Вопросы, ответы, источники будут пустыми.")
                 pass # Let it proceed with empty lists
            else: # Колонки не найдены
                 # print(f"DEBUG __init__: ВНИМАНИЕ! Колонки '{question_col_name}' ({question_col}) или '{answer_col_name}' ({answer_col}) не найдены. Бот может работать некорректно.")
                 pass # Let it proceed with empty lists
            self.questions = []
            self.answers = []
            self.sources = []
        else: 
            self.questions = self.df[question_col].astype(str).tolist()
            self.answers = self.df[answer_col].astype(str).tolist()
            if self.link_col and self.link_col in self.df.columns:
                self.sources = self.df[self.link_col].fillna("").astype(str).tolist() # type: ignore[attr-defined]
            else:
                self.sources = [""] * len(self.questions)
        
        # print(f"DEBUG __init__: Извлечено {len(self.questions)} вопросов, {len(self.answers)} ответов, {len(self.sources)} источников.")

        embedding_dim_val_init = self.model.get_sentence_embedding_dimension()
        embedding_dim_init: int = embedding_dim_val_init if isinstance(embedding_dim_val_init, int) else 384
        # print(f"DEBUG __init__: Определена размерность векторов: {embedding_dim_init}")
        
        self.question_vectors = np.array([], dtype=np.float32).reshape(0, embedding_dim_init)
        self._load_or_generate_vectors()
        self._build_db_vocabulary() # ADD THIS LINE: Build vocabulary after vectors and questions/answers are finalized
        
        # REMOVED question_keywords and question_topics
        # self.question_keywords = [self._extract_keywords(q) for q in self.questions]
        # self.question_topics = [self._identify_topics(keywords) for keywords in self.question_keywords]

        # self.db_questions_processed was for FAISS, remove
        # self._build_faiss_index() # Removed
        # print(f"DEBUG __init__: Chatbot initialized. Loaded {len(self.questions)} questions.")
        if hasattr(self, 'question_vectors') and self.question_vectors.size > 0:
            # print(f"DEBUG __init__: Loaded {self.question_vectors.shape[0]} question vectors with dimension {self.question_vectors.shape[1]}. Type: {self.question_vectors.dtype}")
            pass
        else:
            # print("DEBUG __init__: Warning: Question vectors are not loaded or are empty after _load_or_generate_vectors!")
            pass

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split by non-alphanumeric, filter empty strings."""
        # The type hint `text: str` already ensures text is a string.
        # An explicit isinstance check is redundant here if type checking is enforced.
        # if not isinstance(text, str):
        #     return []
        # Use regex to find sequences of Cyrillic/Latin letters and numbers
        words = re.findall(r'[а-яa-z0-9]+', text.lower())
        return words

    def _build_db_vocabulary(self) -> None:
        """Builds a vocabulary from all questions in the database."""
        # print("DEBUG _build_db_vocabulary: Начало построения словаря базы данных.")
        vocab_set: Set[str] = set()
        for q_text in self.questions:
            vocab_set.update(self._tokenize_text(q_text))
        # Removed: Looping through self.answers to build vocabulary
        # for a_text in self.answers:
        #     vocab_set.update(self._tokenize_text(a_text))
        
        self.db_vocabulary = vocab_set - self.RUSSIAN_STOPWORDS # Remove stopwords from final vocab
        # print(f"DEBUG _build_db_vocabulary: Словарь построен. Уникальных слов (из ВОПРОСОВ, без стоп-слов): {len(self.db_vocabulary)}")
        if len(self.db_vocabulary) < 50: # Print some examples if vocab is very small
             # print(f"DEBUG _build_db_vocabulary: Пример слов из словаря: {list(self.db_vocabulary)[:20]}")
             pass

    def _get_column_case_insensitive(self, df_columns: List[str], column_name: str, required: bool = True) -> Optional[str]:
        """Get the actual column name regardless of case.

        Args:
            df_columns: List of column names to search in
            column_name: The column name to find (case-insensitive)

        Returns:
            The actual column name in the dataframe

        Raises:
            KeyError: If no matching column is found

        """
        actual_col_name = None
        for col in df_columns:
            if col.strip().lower() == column_name.lower():
                actual_col_name = col
                break
        
        if required and actual_col_name is None:
            # print(f"DEBUG _get_column_case_insensitive: Обязательная колонка '{column_name}' не найдена. Доступные: {', '.join(df_columns)}")
            # Не выбрасываем исключение здесь, позволяем __init__ обработать
            pass
        elif not required and actual_col_name is None:
            # print(f"DEBUG _get_column_case_insensitive: Необязательная колонка '{column_name}' не найдена.")
            pass
        return actual_col_name

    def _load_or_generate_vectors(self) -> None:
        """Load question vectors from cache or generate them if cache is missing/stale."""
        # print("DEBUG _load_or_generate_vectors: Начало загрузки/генерации векторов.")
        cache_path = Path(self.cache_file)
        db_path = Path(self.database_path)

        if not db_path.exists():
            # print(f"DEBUG _load_or_generate_vectors: Файл базы данных '{db_path}' НЕ НАЙДЕН. Генерация/загрузка векторов невозможна.")
            # Получаем embedding_dim здесь, чтобы задать правильную форму для пустого массива
            embedding_dim_val = self.model.get_sentence_embedding_dimension()
            current_embedding_dim: int = embedding_dim_val if isinstance(embedding_dim_val, int) else 384
            self.question_vectors = np.array([], dtype=np.float32).reshape(0, current_embedding_dim)
            return

        if cache_path.exists():
            # print(f"DEBUG _load_or_generate_vectors: Файл кеша '{cache_path}' существует.")
            if cache_path.stat().st_mtime > db_path.stat().st_mtime:
                # print(f"DEBUG _load_or_generate_vectors: Кеш ('{cache_path}') новее базы данных ('{db_path}'). Попытка загрузки из кеша.")
                try:
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                    
                    if not all(k in data for k in ["questions", "answers", "vectors"]):
                        # print("DEBUG _load_or_generate_vectors: В кеше отсутствуют необходимые ключи. Перегенерация.")
                        self._generate_and_cache_vectors()
                        return

                    loaded_vectors = data["vectors"]
                    if not isinstance(loaded_vectors, np.ndarray):
                        # print(f"DEBUG _load_or_generate_vectors: Векторы в кеше не np.ndarray. Перегенерация.")
                        self._generate_and_cache_vectors()
                        return
                    if loaded_vectors.dtype != np.float32: # type: ignore[attr-defined]
                        # print(f"DEBUG _load_or_generate_vectors: Тип векторов из кеша ({loaded_vectors.dtype}) не np.float32. Конвертация.") # type: ignore[attr-defined]
                        loaded_vectors = loaded_vectors.astype(np.float32)
                        data["vectors"] = loaded_vectors
                        try:
                            with open(cache_path, "wb") as f_new_cache:
                                pickle.dump(data, f_new_cache)
                            # print("DEBUG _load_or_generate_vectors: Кеш обновлен с float32.")
                        except Exception as _e_cache_save:
                             # print(f"DEBUG _load_or_generate_vectors: Ошибка пересохранения кеша: {_e_cache_save}")
                             pass # Non-critical if re-save fails, will regen next time
                    
                    if len(data["questions"]) != loaded_vectors.shape[0]:
                        # print(f"DEBUG _load_or_generate_vectors: Несоответствие кол-ва вопросов и векторов в кеше. Перегенерация.")
                        self._generate_and_cache_vectors()
                        return

                    self.questions = data["questions"]
                    self.answers = data["answers"]
                    self.sources = data.get("sources", [""] * len(self.questions))
                    self.question_vectors = loaded_vectors
                    # print(f"DEBUG _load_or_generate_vectors: Успешно загружено из кеша. Вопросов: {len(self.questions)}, Векторов: {self.question_vectors.shape[0]}x{self.question_vectors.shape[1]}, Тип: {self.question_vectors.dtype}")
                    return
                except Exception as _e:
                    # print(f"DEBUG _load_or_generate_vectors: Ошибка загрузки из кеша '{cache_path}': {_e}. Перегенерация.")
                    pass # Fall through to regeneration
            else:
                # print(f"DEBUG _load_or_generate_vectors: База данных ('{db_path}') новее кеша. Перегенерация.")
                pass # Fall through to regeneration
        else:
            # print(f"DEBUG _load_or_generate_vectors: Файл кеша '{cache_path}' не найден. Перегенерация.")
            pass # Fall through to regeneration
            
        self._generate_and_cache_vectors()

    def _generate_and_cache_vectors(self) -> None:
        """Generate and cache question vectors from the database."""
        # print("DEBUG _generate_and_cache_vectors: Начало генерации векторов.")
        
        embedding_dim_val_gen = self.model.get_sentence_embedding_dimension()
        current_embedding_dim_gen: int = embedding_dim_val_gen if isinstance(embedding_dim_val_gen, int) else 384

        if not self.questions:
            # print("DEBUG _generate_and_cache_vectors: Список вопросов пуст. Векторы не будут сгенерированы.")
            self.question_vectors = np.array([], dtype=np.float32).reshape(0, current_embedding_dim_gen)
            return

        # print(f"DEBUG _generate_and_cache_vectors: Генерируем векторы для {len(self.questions)} вопросов с моделью '{self.model_name}'.")
        if self.questions:
            # print(f"DEBUG _generate_and_cache_vectors: Пример первого вопроса для векторизации: '{self.questions[0][:100]}...'")
            pass
        
        encoded_output = self.model.encode(self.questions, convert_to_numpy=True, show_progress_bar=True) # type: ignore[no-untyped-call]
        
        # Assuming convert_to_numpy=True guarantees an np.ndarray or raises an error from the library.
        # If it can return other types despite the flag, more complex error handling might be needed.
        try:
            self.question_vectors = np.array(encoded_output, dtype=np.float32) # type: ignore[arg-type]
        except Exception as _e_vector_conv:
            # print(f"DEBUG _generate_and_cache_vectors: КРИТ. ОШИБКА конвертации вывода self.model.encode в np.ndarray(float32): {_e_vector_conv}. Тип был: {type(encoded_output)}. Векторы будут пустыми.")
            embedding_dim_val_gen = self.model.get_sentence_embedding_dimension()
            current_embedding_dim_gen_exc: int = embedding_dim_val_gen if isinstance(embedding_dim_val_gen, int) else 384
            self.question_vectors = np.array([], dtype=np.float32).reshape(0, current_embedding_dim_gen_exc)
            return

        # print(f"DEBUG _generate_and_cache_vectors: Векторы сгенерированы. Форма: {self.question_vectors.shape}. Тип: {self.question_vectors.dtype}")
        
        cache_path = Path(self.cache_file)
        try:
            # print(f"DEBUG _generate_and_cache_vectors: Попытка сохранения векторов в кеш '{cache_path}'.")
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "questions": self.questions,
                        "answers": self.answers,
                        "sources": self.sources,
                        "vectors": self.question_vectors,
                    },
                    f,
                )
            # print(f"DEBUG _generate_and_cache_vectors: Векторы успешно сохранены в кеш.")
        except Exception as _e:
            # print(f"DEBUG _generate_and_cache_vectors: Ошибка сохранения векторов в кеш: {_e}")
            pass # Cache saving is optional for operation

    # get_query_nouns method was integrated into _extract_nouns_spacy or can be kept if used elsewhere
    # def get_query_nouns(self, query: str) -> Set[str]:
    #     """Extracts nouns from the query using spaCy."""
    #     # ... (implementation was similar to _extract_nouns_spacy)
    #     return self._extract_nouns_spacy(query)

    def find_answer(self, query: str) -> dict[str, Any]:
        """
        Find the most relevant answer to the query using semantic similarity.
        """
        # print(f"DEBUG find_answer: Получен запрос: '{query}'")

        if not self.questions or self.question_vectors.size == 0:
            # print("DEBUG find_answer: База вопросов пуста или векторы не загружены.")
            return {
                "answer": "База данных пуста или не загружена.",
                "source": "",
                "matched_question": "",
                "score": 0.0,
                "above_threshold": False,
                "top_matches": [],
                "match_type": "no_data",
                "unmatched_query_terms": [],
                "relevant_questions_list": []
            }

        query_vector = self.model.encode(query, convert_to_numpy=True) # type: ignore[no-untyped-call]
        query_vector = query_vector.astype(np.float32).reshape(1, -1) # type: ignore[no-any-return, union-attr]
        # print(f"DEBUG find_answer: Вектор запроса создан. Форма: {query_vector.shape}, Тип: {query_vector.dtype}")

        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        # print(f"DEBUG find_answer: Рассчитаны схожести (до переранжирования). Всего: {len(similarities)}. Макс: {np.max(similarities):.4f}, Мин: {np.min(similarities):.4f}, Среднее: {np.mean(similarities):.4f}")

        # Re-ranking logic for critical keywords (example: "статин")
        query_lower = query.lower()
        critical_keyword_statin = "статин" 
        penalty_factor = 0.5

        # General re-ranking based on keyword presence can be expanded
        # For now, specific handling for "статин"
        if critical_keyword_statin in query_lower:
            # print(f"DEBUG find_answer: Обнаружен критический ключ '{critical_keyword_statin}' в запросе. Применяется переранжирование.")
            for i in range(len(self.questions)):
                if critical_keyword_statin not in self.questions[i].lower():
                    _original_score = similarities[i]
                    similarities[i] *= penalty_factor
                    # if i < 5 or (self.questions[i].lower().startswith("продлевает ли жизнь спермидин")):
                        # print(f"DEBUG find_answer: Вопрос ID {i} ('{self.questions[i][:50]}...') НЕ содержит '{critical_keyword_statin}'. Схожесть понижена с {_original_score:.4f} до {similarities[i]:.4f}")
            # print(f"DEBUG find_answer: Схожести (после переранжирования для '{critical_keyword_statin}'). Макс: {np.max(similarities):.4f}, Мин: {np.min(similarities):.4f}, Среднее: {np.mean(similarities):.4f}")
        
        # print(f"[WEB_DEBUG] find_answer received query: '{query}'") # Web debug
        tokenized_query = self._tokenize_text(query)
        # print(f"[WEB_DEBUG] tokenized_query: {tokenized_query}") # Web debug
        # print(f"[WEB_DEBUG] Size of db_vocabulary: {len(self.db_vocabulary)}") # Web debug
        # if self.db_vocabulary:
            # print(f"[WEB_DEBUG] Sample of db_vocabulary: {list(self.db_vocabulary)[:10]}") # Web debug
        
        oov_query_words = {
            word for word in tokenized_query 
            if word not in self.db_vocabulary and word not in self.RUSSIAN_STOPWORDS
        }
        # print(f"[WEB_DEBUG] oov_query_words: {oov_query_words}") # Web debug

        # Prepare default response structure
        response_data: dict[str, Any] = {
            "answer": "", # Will be populated or changed based on logic
            "source": "",
            "matched_question": "",
            "score": 0.0,
            "above_threshold": False,
            "top_matches": [], # For detailed debug output in CLI
            "match_type": "no_match", # Default, will be updated
            "unmatched_query_terms": list(oov_query_words), # Initialize with current OOV words
            "clarification_prefix": "",
            "relevant_questions_list": [] # New field
        }

        # Early exit if ALL query terms are OOV (and not stopwords)
        tokenized_query_significant = [word for word in tokenized_query if word not in self.RUSSIAN_STOPWORDS]
        if tokenized_query_significant and all(word in oov_query_words for word in tokenized_query_significant):
            # print(f"DEBUG find_answer: Все значимые слова в запросе '{query}' являются OOV. Возвращаем 'информация отсутствует'.")
            response_data["answer"] = f"По вашему запросу ('{query}') информация в базе отсутствует, так как все ключевые слова не найдены в словаре. Пожалуйста, переформулируйте."
            response_data["match_type"] = "all_oov"
            response_data["above_threshold"] = False
            response_data["score"] = 0.0
            # print(f"DEBUG find_answer: Итоговые возвращаемые данные (all_oov): {response_data}")
            return response_data

        top_indices_for_list = np.argsort(similarities)[-10:][::-1] # Get more for potential list
        
        # Populate top_matches for debugging (always, based on current similarities)
        # This part can be moved after OOV logic if we don't want to show matches if OOV terms dominate.
        # For now, keeping it here to see what initial matches look like.
        for i in top_indices_for_list:
            if i < len(self.questions) and similarities[i] > 0.1: # Basic filter for debug list
                response_data["top_matches"].append({
                    "question": self.questions[i],
                    "score": float(similarities[i]),
                    "answer": self.answers[i], 
                    "source": self.sources[i] if i < len(self.sources) else ""
                })
        response_data["top_matches"] = response_data["top_matches"][:5]
        # print(f"DEBUG find_answer: Top 5 совпадений (для отладки, до OOV-фильтрации): {response_data['top_matches']}")

        best_match_idx = np.argmax(similarities)
        highest_similarity = float(similarities[best_match_idx])
        # print(f"DEBUG find_answer: Лучшее совпадение (индекс): {best_match_idx}, Схожесть (до применения OOV логики): {highest_similarity:.4f}")

        if oov_query_words:
            # print(f"DEBUG find_answer: Обнаружены слова вне словаря БД (OOV): {oov_query_words}")
            response_data["clarification_prefix"] = f"По термину(ам) '{', '.join(oov_query_words)}' информация в базе отсутствует."
            
            tokenized_query_set = set(tokenized_query) 
            known_query_words = tokenized_query_set - oov_query_words
            # print(f"DEBUG find_answer (OOV case): Known query words: {known_query_words}")

            relevant_questions_for_oov_case: List[str] = []
            
            # If there are known words, we search for questions relevant to THEM.
            # The similarities were calculated for the FULL query. This is problematic if OOV words polluted the vector.
            # A better approach for OOV cases:
            # 1. Inform about OOV.
            # 2. If there are known words, perform a NEW search/ranking based ONLY on known words.
            
            if known_query_words:
                # print(f"DEBUG find_answer (OOV case): Есть известные слова: {known_query_words}. Ищем релевантные вопросы по ним.")
                
                # Perform a new, clean search based only on known_query_words
                # Preserve order of known words from the original tokenized_query
                clean_query_tokens_ordered = [
                    word for word in tokenized_query 
                    if word not in oov_query_words and word not in self.RUSSIAN_STOPWORDS
                ]
                clean_query_text = " ".join(clean_query_tokens_ordered)
                
                # print(f"DEBUG find_answer (OOV case): Performing clean search for: '{clean_query_text}'")
                
                if not clean_query_text.strip(): # If only stopwords were known, or known_query_words was empty
                    # print("DEBUG find_answer (OOV case): Clean query text is empty. No relevant questions to find.")
                    pass # Will fall through to "oov_no_related_questions"
                else:
                    clean_query_vector = self.model.encode(clean_query_text, convert_to_numpy=True) # type: ignore[no-untyped-call]
                    clean_query_vector = clean_query_vector.astype(np.float32).reshape(1, -1) # type: ignore[no-any-return, union-attr]
                    clean_similarities = cosine_similarity(clean_query_vector, self.question_vectors)[0]
                    
                    max_clean_similarity = np.max(clean_similarities) if clean_similarities.size > 0 else 0.0
                    MIN_RELEVANCE_FOR_CLEAN_SUGGESTIONS = 0.7 # Changed threshold to 0.7 as per user request

                    if max_clean_similarity < MIN_RELEVANCE_FOR_CLEAN_SUGGESTIONS:
                        # print(f"DEBUG find_answer (OOV - clean search): Max clean similarity {max_clean_similarity:.2f} is below threshold {MIN_RELEVANCE_FOR_CLEAN_SUGGESTIONS}. No suggestions will be made for known part.")
                        # relevant_questions_for_oov_case will remain empty, leading to "oov_no_related_questions" path
                        pass
                    else:
                        # Re-apply statin re-ranking if statin is in known words for the clean search
                        if critical_keyword_statin in known_query_words:
                            # print(f"DEBUG find_answer (OOV - clean search): Re-ranking for '{critical_keyword_statin}' in clean search.")
                            for i in range(len(self.questions)):
                                if critical_keyword_statin not in self.questions[i].lower():
                                    clean_similarities[i] *= penalty_factor # Use the same penalty_factor

                        clean_top_indices = np.argsort(clean_similarities)[-10:][::-1] # Get top 10 for the clean search

                        for i in clean_top_indices: 
                            # Use a threshold for relevance for these cleaned-up suggestions
                            # This threshold should ensure suggestions are genuinely related to the clean_query_text
                            if i < len(self.questions) and clean_similarities[i] >= MIN_RELEVANCE_FOR_CLEAN_SUGGESTIONS: 
                                db_question_text: str = cast(str, self.questions[i])
                                relevant_questions_for_oov_case.append(db_question_text)
                                # print(f"DEBUG find_answer (OOV case - clean search): Добавлен вопрос '{db_question_text[:50]}...' (схожесть {clean_similarities[i]:.2f})")
            
            if relevant_questions_for_oov_case:
                response_data["match_type"] = "oov_with_related_questions"
                response_data["relevant_questions_list"] = relevant_questions_for_oov_case[:5] 
                main_answer_part = "По остальной части вашего запроса в базе найдены следующие вопросы:"
                response_data["answer"] = f"{response_data['clarification_prefix']}\n\n{main_answer_part}"
                response_data["above_threshold"] = False 
                response_data["score"] = 0.0 # No direct answer to the original OOV query
                response_data["matched_question"] = "" # No direct matched question for OOV query
                # print(f"DEBUG find_answer (OOV case): Сформирован ответ с релевантными вопросами. {response_data}")
            else: # OOV words present, but no relevant questions found for the known part (or no known part)
                response_data["match_type"] = "oov_no_related_questions"
                response_data["answer"] = f"{response_data['clarification_prefix']} По остальной части запроса также не найдено достаточно релевантной информации."
                response_data["above_threshold"] = False
                response_data["score"] = 0.0
                response_data["matched_question"] = ""
                # print(f"DEBUG find_answer (OOV case): Нет релевантных вопросов для известных слов. {response_data}")
            return response_data # IMPORTANT: Return here after handling OOV cases
        
        # This block is reached ONLY IF there were NO OOV words
        # print(f"DEBUG find_answer: Нет OOV слов. Продолжаем с обычной логикой совпадения.")
        if highest_similarity >= self.similarity_threshold:
            # Potential full match, but verify term coverage
            matched_db_question_text = self.questions[best_match_idx]
            
            tokenized_user_query = self._tokenize_text(query)
            significant_user_query_tokens = {
                token for token in tokenized_user_query if token not in self.RUSSIAN_STOPWORDS
            }
            
            # Tokenize the matched DB question to check for term presence
            # We use the full set of tokens from the DB question for the .issubset check
            tokenized_db_question_set = set(self._tokenize_text(matched_db_question_text))

            # Check if all significant user query tokens are present in the tokens of the matched DB question
            all_user_terms_in_matched_question = significant_user_query_tokens.issubset(tokenized_db_question_set)

            if all_user_terms_in_matched_question:
                # print(f"DEBUG find_answer: Full match, all significant query terms ('{significant_user_query_tokens}') found in DB question tokens ('{tokenized_db_question_set}').")
                response_data["answer"] = self.answers[best_match_idx]
                response_data["source"] = self.sources[best_match_idx] if best_match_idx < len(self.sources) else ""
                response_data["matched_question"] = matched_db_question_text
                response_data["score"] = highest_similarity
                response_data["above_threshold"] = True
                response_data["match_type"] = "full_match"
            else:
                # print(f"DEBUG find_answer: High similarity ({highest_similarity:.2f}), but not all query terms covered. Missing: {significant_user_query_tokens - tokenized_db_question_set}")
                response_data["answer"] = "По вашему запросу прямого ответа нет. Возможно, вас заинтересуют следующие вопросы по схожим темам:"
                response_data["source"] = "" 
                response_data["score"] = highest_similarity 
                response_data["above_threshold"] = False 
                response_data["match_type"] = "partial_term_coverage_match"
                
                response_data["relevant_questions_list"] = [
                    self.questions[i] for i in top_indices_for_list 
                    if i < len(self.questions) and similarities[i] >= self.similarity_threshold * 0.7
                ][:5]
                
                if not response_data["relevant_questions_list"]:
                    # Fallback if no suggestions meet the criteria, even though similarity was high initially
                    response_data["answer"] = self.FAIL_RESPONSES[0]
                    response_data["match_type"] = "low_confidence_no_oov" # Or a more specific type
                    response_data["matched_question"] = "" # Clear matched question as it wasn't a true fit
                    response_data["score"] = highest_similarity # Still show the best score found before failing here
        
        else:
            # No OOV words, but no strong match found (highest_similarity < self.similarity_threshold)
            response_data["answer"] = self.FAIL_RESPONSES[0] 
            response_data["score"] = highest_similarity 
            if highest_similarity > 0.01 : 
                 response_data["matched_question"] = self.questions[best_match_idx]
            response_data["above_threshold"] = False
            response_data["match_type"] = "low_confidence_no_oov"
            # print(f"DEBUG find_answer: Нет OOV, но совпадение ниже порога. Ответ: '{response_data['answer'][:50]}...'")

        # print(f"DEBUG find_answer: Итоговые возвращаемые данные: {response_data}")
        return response_data

# Example Usage (optional, for direct testing of the class)
if __name__ == '__main__':
    # This basic example assumes an Excel file "test_data.xlsx" exists in the same directory
    # with "вопрос", "ответ", "источники" columns.
    
    # Create a dummy Excel file for testing if it doesn't exist
    dummy_data = {
        'вопрос': [
            "Что такое статины?", 
            "Помогают ли статины сердцу?", 
            "Какие побочные эффекты у статинов?",
            "Солнечный свет полезен?",
            "Что такое Земля?",
            "Статины продлевают жизнь?"
        ],
        'ответ': [
            "Статины - это группа лекарственных препаратов, снижающих уровень холестерина в крови.",
            "Да, статины могут снижать риск сердечно-сосудистых заболеваний.",
            "Побочные эффекты могут включать мышечные боли, проблемы с печенью и др. Обратитесь к врачу.",
            "Умеренное количество солнечного света способствует выработке витамина D, но избыток вреден.",
            "Земля - это третья планета от Солнца.",
            "Некоторые исследования показывают, что статины могут немного продлевать жизнь у определенных групп пациентов, снижая риск смерти от сердечно-сосудистых заболеваний."
        ],
        'источники': [
            "Медицинский справочник", 
            "Исследование AHA", 
            "Инструкция к препарату",
            "ВОЗ",
            "Астрономия для чайников",
            "Обзор исследований по статинам, 2023"
            ]
    }
    dummy_excel_path = "test_data_chatbot_core.xlsx"
    try:
        # Check if file exists to avoid overwriting user's file if they run this directly
        pd.read_excel(dummy_excel_path) # type: ignore[attr-defined]
    except FileNotFoundError:
        print(f"Creating dummy Excel: {dummy_excel_path}")
        pd.DataFrame(dummy_data).to_excel(dummy_excel_path, index=False) # type: ignore[attr-defined]

    print("Initializing chatbot...")
    try:
        # Ensure cache is rebuilt for this test by using a unique cache file name
        bot = SemanticChatBot(database_path=dummy_excel_path, cache_file="test_vector_cache.pkl")
        print("Chatbot initialized.")

        queries = [
            "расскажи про статины",
            "статины вредны для печени?",
            "Солнце полезно или нет?",
            "что известно о Марсе?", # Should yield low score or fail
            "как земля влияет на сатурн", # Should fail
            "статины продлевают жизнь?", # Full
            "статины продлевают жизнь на марсе?", # Partial
            "бег полезен для здоровья на юпитере?" # Partial (no info on jupiter, maybe no info on run)
        ]

        for q_idx, q_text in enumerate(queries):
            print(f"\n--- Query {q_idx + 1}: {q_text} ---")
            result = bot.find_answer(q_text)
            print(f"Match Type: {result.get('match_type', 'N/A')}")
            print(f"Score: {result.get('score', 0.0):.4f}")
            if result.get('matched_question'):
                print(f"Matched Question: {result['matched_question']}")
            if result.get('unanswered_parts'):
                print(f"Unanswered Parts: {result['unanswered_parts']}")
            print(f"Answer: {result['answer']}")
            if result['source']:
                print(f"Source: {result['source']}")
        
    except ImportError as e:
        print(f"Import error during chatbot initialization or usage: {e}")
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up dummy file and cache
        try:
            if Path(dummy_excel_path).exists(): Path(dummy_excel_path).unlink(missing_ok=True)
            if Path("test_vector_cache.pkl").exists(): Path("test_vector_cache.pkl").unlink(missing_ok=True)
            print("Cleaned up test files.")
        except OSError as e:
            print(f"Error cleaning up test files: {e}")
