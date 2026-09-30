Структура проекта
```
disaster-tweets/
├── configs/          # Конфигурации проекта и экспериментов
├── data/             # Исходные и обработанные данные
├── notebooks/        # Jupyter notebooks для EDA
├── scripts/          # Точки запуска типовых операций
├── experiments/      # End-to-end сценарии экспериментов
├── src/              # Основная переиспользуемая логика проекта
├── artifacts/        # Результаты экспериментов
├── checkpoints/      # Сохранённые веса моделей
└── tests/             # Тесты
```


# 1 Выводы   
### Эксперименты classic_nlp :    
0. Валидация только на отложенной выборке не даёт право говорить о стат значимых отличиях между экспериментами, так что дальнейшие выводы сделаны исключительно в рамках наблюдаемых значений и не претендуют на статистическую значимость. 
1. TF-IDF + LogReg — сильнейшая связка. BoW почти не уступает (0.7291) — на коротких твитах TF-IDF и BoW практически эквивалентны, разница в пределах шума. Это связано с небольшой длинной предложений твитов - 13 слов в среднем.  
2. CatBoost с нативным текстом (raw_catboost_native) — худший результат (0.6794). Его примитивная токенизация (split() по пробелам) не конкурирует с полноценным препроцессингом + TF-IDF.
3. Сильной разницы между способами нормализации и филтрацией стоп слов нет, максимальные отличия около 1% 
4. Word2Vec ожидаемо слабее TF-IDF (0.7189): корпус слишком мал (7.6k твитов) для качественных эмбеддингов, а усреднение векторов теряет сигнал.

### Эксперименты embeddings:  
Замена TF-IDF на предобученный sentence encoder (bge-small-en-v1.5) даёт +0.055 F1 на тесте (0.732 → 0.787). Это подтверждает гипотезу о том, что семантические представления превосходят мешок слов на коротких неформальных текстах. Прирост значимо больше порога шума, зафиксированного для classic NLP. Классический препроцессинг (lemma/stem/stopwords) для трансформеров не применялся — он бы только испортил вход.

# Disaster Tweets — классификация твитов о катастрофах

## 1. О задаче

Бинарная классификация твитов: **disaster** (реальное бедствие) vs **not disaster** (метафора, шутка, реклама). Датасет — [Kaggle NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started), 7613 train-примеров с метками, 3263 test без меток.

Задача сложная не из-за технической части, а из-за семантики: `"Forest fire near La Ronge Sask. Canada"` — disaster, а `"This fire is lit 🔥"` — нет. Мешок слов на таких примерах путается.

**Что сделано:** три подхода в порядке возрастания сложности.

1. **Classic NLP** — препроцессинг (стемминг/лемматизация, стоп-слова) + TF-IDF/BoW/Word2Vec + LogReg/GBM/CatBoost.
2. **DL + classic ML** — предобученные эмбеддинги (`bge-small-en-v1.5`) + LogReg поверх.
3. **Fine-tune** — дообучение `roberta-base` под задачу классификации.

Результат: **лучший F1 = 0.787** на отложенном test. Победитель — `bge-small + LogReg`: предобученный sentence encoder без дообучения обошёл полноценный fine-tune RoBERTa и оказался в 60 раз быстрее.

## 2. Данные и выводы по EDA

**Структура:**
- `train.csv` (7613 строк), `test.csv` (3263), `sample_submission.csv`.
- Колонки: `id`, `keyword`, `location`, `text`, `target`.
- Пропуски: `keyword` ~0.8%, `location` ~33%.
- Дисбаланс классов: ~57% not disaster / ~43% disaster. Умеренный, не требует ресемплинга.

**Длина текстов:**
- Медиана ~13 слов, p95 ~25, максимум ~40.
- Вывод: `max_length=48` для трансформеров — с запасом, `max_length=64` — избыточно.

**Дубликаты:**
- Найдены точные дубли текста и противоречивые пары (один текст, разные метки).
- Вывод: обязательная дедупликация **до** сплита, режим `drop_conflict` — удалить все строки с противоречивыми метками.

**`keyword` и `location`:**
- `keyword` — 221 категория, почти категориальный признак.
- `location` — свободный текст, тысячи уникальных значений, требует нормализации и схлопывания редких.
- Вклад в финальную метрику: **+0.02 F1** (см. `stem_stop_tfidf_full_tf_logreg` 0.7308 vs `..._onlytextcol_logreg` 0.7075).

## 3. Результаты и выводы

### 3.1. Сводная таблица

| # | Эксперимент | Test F1 |
|---|---|---|
| 1 | `bge_small_logreg` | **0.7875** |
| 2 | `roberta_base_finetune` | **0.7855** |
| 3 | `stem_stop_tfidf_logreg` | 0.7318 |
| 4 | `stem_stop_tfidf_full_tf_logreg` | 0.7308 |
| 5 | `stem_stop_bow_logreg` | 0.7291 |
| 6 | `lemma_stop_tfidf_lgbm` | 0.7267 |
| 7 | `lemma_stop_tfidf_logreg` | 0.7231 |
| 8 | `lemma_stop_w2v_lgbm` | 0.7196 |
| 9 | `lemma_stop_w2v_logreg` | 0.7189 |
| 10 | `lemma_nostop_tfidf_logreg` | 0.7133 |
| 11 | `stem_stop_tfidf_full_tf_onlytextcol_logreg` | 0.7075 |
| 12 | `lemma_stop_tfidf_catboost` | 0.6995 |
| 13 | `raw_catboost_native` | 0.6794 |
| 14 | `lemma_stop_tfidf_gbm` | 0.6642 |

### 3.2. Выводы по classic_nlp

0. Валидация только на отложенной выборке не даёт право говорить о стат значимых отличиях между экспериментами, так что дальнейшие выводы сделаны исключительно в рамках наблюдаемых значений и не претендуют на статистическую значимость.

1. TF-IDF + LogReg — сильнейшая связка. BoW почти не уступает (0.7291) — на коротких твитах TF-IDF и BoW практически эквивалентны, разница в пределах шума. Это связано с небольшой длиной предложений твитов — 13 слов в среднем.

2. CatBoost с нативным текстом (`raw_catboost_native`) — худший результат (0.6794). Его примитивная токенизация (`split()` по пробелам) не конкурирует с полноценным препроцессингом + TF-IDF.

3. Сильной разницы между способами нормализации и фильтрацией стоп-слов нет, максимальные отличия около 1%.

4. Word2Vec ожидаемо слабее TF-IDF (0.7189): корпус слишком мал (7.6k твитов) для качественных эмбеддингов, а усреднение векторов теряет сигнал.

### 3.3. Выводы по embeddings
bge-small-en-v1.5 выбрана как лучший компромисс между качеством и скоростью среди компактных sentence encoder'ов: на MTEB Classification (12 задач) она даёт 74.14 — один из сильнейших результатов в классе 33M-параметровых моделей, при этом 384-мерные эмбеддинги не создают риска переобучения LogReg на 7.6k примеров

Замена TF-IDF на предобученный sentence encoder (`bge-small-en-v1.5`) даёт **+0.055 F1** на тесте (0.732 → 0.787). Это подтверждает гипотезу о том, что семантические представления превосходят мешок слов на коротких неформальных текстах. Прирост значимо больше порога шума, зафиксированного для classic NLP. Классический препроцессинг (lemma/stem/stopwords) для трансформеров не применялся — он бы только испортил вход.

### 3.4. Выводы по fine-tune

roberta-base выбрана как стандартная модель для finetune, имеет 125M параметров — достаточно ёмкости для адаптации под задачу, но не настолько много, чтобы гарантированно переобучиться на маленьком датасете, и она стабильно обходит BERT и DistilBERT на коротких текстах.

- **`roberta-base` fine-tuned = 0.7855.** Практически на уровне `bge-small`, но в 60 раз дороже (5 минут vs 5 секунд).
- **Ожидалось больше чем bge-small-en-v1.5 **, получено 0.7855. Причина — переобучение на 7.6k примеров: train F1 0.884 vs test F1 0.785.
- **Fine-tune не оправдал себя на этом датасете.** Feature extraction с сильным энкодером обгоняет full fine-tune при малых данных.
- **Победитель по соотношению качество/скорость:** `bge-small + LogReg`.

## 4. Инфраструктура и установка

### 4.1. Требования

- **Python 3.12**
- **uv** — менеджер зависимостей и виртуальных окружений. Установка:
  - Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Документация: https://docs.astral.sh/uv/
- **Git**
- (Опционально) **NVIDIA GPU + драйверы** с поддержкой CUDA 12.1 — только для fine-tune. Все остальные эксперименты работают на CPU.

### 4.2. Установка проекта

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd disaster-tweets

# 2. Установить зависимости (uv создаст .venv и поставит всё из pyproject.toml)
uv sync

# 3. Создать .env (см. следующий пункт)
cp .env.example .env
```
### 4.3. Настройка окружения

Создать `.env` в корне проекта:

```dotenv
KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CLEARML_OFFLINE=1
```

- `KAGGLE_API_TOKEN` — токен Kaggle (получить: https://www.kaggle.com/settings → API → Create New Token).
- `CLEARML_OFFLINE=1` — отключает подключение к ClearML-серверу. Если у тебя настроен свой сервер — убери эту строку.

### 4.4. Загрузка данных и NLTK

```bash
# Один раз: скачать датасет с Kaggle
uv run python scripts/download_data.py

# Один раз: скачать NLTK-ресурсы (stopwords, wordnet)
uv run python scripts/setup_nltk.py
```

## 5. Запуск экспериментов

### 5.1. Classic NLP

```bash
uv run python experiments/classic_nlp/run.py \
    --config configs/experiments/classic_nlp/stem_stop_tfidf_logreg.yaml
```

### 5.2. Embeddings

```bash
uv run python experiments/embeddings/run.py \
    --config configs/experiments/embeddings/bge_small_logreg.yaml
```

### 5.3. Fine-tune

```bash
uv run python experiments/finetune/run.py \
    --config configs/experiments/finetune/roberta_base.yaml
```

### 5.4. Артефакты

Каждый эксперимент сохраняет результаты в `artifacts/<exp_name>/`:

- `data/{train,val,test}.csv` — данные, на которых обучалась модель.
- `{train,val,test}_predictions.csv` — предсказания.
- `metrics.json` — метрики по сплитам.
- `model.joblib` (или `model/` для HF) — сохранённая модель.
- `config.yaml` — копия конфига.

## 6. Структура проекта

```
disaster-tweets/
├── configs/
│   ├── general_config.yaml       # пути, patterns, metrics
│   └── experiments/              # с разделением на тип эксперимента, по одному YAML на эксперимент
├── data/                         # raw, processed
├── scripts/                      # download_data, setup_nltk
├── experiments/                  # end-to-end раннеры (classic_nlp, embeddings, finetune)
├── src/
│   ├── data/                     # loaders, split
│   ├── preprocessing/            # filtering, tokenization, dedup
│   ├── features/                 # vectorizers, categorical, builder, embeddings
│   ├── models/                   # classic, catboost, finetune, factory
│   ├── metrics/                  # compute_metrics
│   ├── experiments/              # runner, artifacts (общая инфраструктура)
│   └── utils/                    # config, clearml_utils
├── artifacts/                    # результаты экспериментов
```