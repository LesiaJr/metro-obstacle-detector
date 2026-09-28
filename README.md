# ros2_obstacle_detector

Пайплайн для поиска препятствий в тоннеле метро по данным лидара
(PointCloud2 в rosbag2 `.db3`), упакованный в Docker.

База образа — **ROS 2 Humble** (`ros:humble`, Ubuntu 22.04,
Python 3.10, numpy 1.x).

Две независимые части:

1. **DynamicGate** (`src/dynamic_gate.py`) — створ: строит range-image,
   вычитает baseline, находит сильные/слабые аномалии, подтверждает их
   N/M-фильтром и пространственно.
2. **Bend tube** (`src/bend_tube.py`) — габарит-труба: находит стены,
   строит θ(Y)-изгиб, extrude-ит трубу, отделяет safe-zone.

`src/gate_tube.py` — комбинированный пайплайн:
`Gate → tube → аномалии внутри трубы = препятствия`.

---

## Структура

```
ros2_obstacle_detector/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── entrypoint.sh
├── README.md
├── .dockerignore
├── src/
│   ├── bend_tube.py
│   ├── dynamic_gate.py
│   ├── db3_to_gate.py
│   ├── gate_tube.py
│   └── db3_inspect.py
└── data/                # сюда кладутся .db3 (монтируется)
```

---

## Сборка

```bash
docker build -t metro-obstacle-detector:latest .
```

Первый билд тянет `ros:humble` (~1 ГБ) — это разово. Дальше
образ кэшируется.

---

## Запуск

`entrypoint.sh` принимает либо путь до `.db3`-файла, либо папку —
во втором случае сам находит первый `.db3` внутри.

### 1. На одном `.db3`-файле

```bash
docker run --rm \
    -v "$PWD/data:/data:ro" \
    metro-obstacle-detector:latest \
    /data/foo.db3 --gate-rotate-deg 90
```

### 2. На папке с bag'ами

```bash
docker run --rm \
    -v "$PWD/data:/data:ro" \
    metro-obstacle-detector:latest \
    /data --gate-rotate-deg 90
```

Если в папке несколько `.db3`, берётся первый.

### 3. Ограничить число кадров (для быстрой проверки)

```bash
docker run --rm \
    -v "$PWD/data:/data:ro" \
    metro-obstacle-detector:latest \
    /data/foo.db3 --gate-rotate-deg 90 --max-frames 20
```

### 4. Через docker-compose

Путь к папке с bag'ами настраивается в `docker-compose.yml`
(секция `volumes`). Затем:

```bash
docker compose up --build
```

---

## Что выводится на экран

На каждом кадре печатается одна строка:

```
[0000] pts= 307200 аном=   0  ----     в_трубе=-        hits=0/6
[0005] pts= 307188 аном=   1  OK       в_трубе=1/1      hits=1/6
[0008] pts= 307187 аном=   5  OK       в_трубе=1/5      hits=4/6  >>> ПРЕПЯТСТВИЕ <<<
[0009] pts= 307194 аном=   5  OK       в_трубе=1/5      hits=5/6  >>> ПРЕПЯТСТВИЕ <<<
[0012] pts= 307188 аном=   5  OK       в_трубе=0/5      hits=5/6  >>> ПРЕПЯТСТВИЕ <<<
[0016] pts= 307188 аном=   0
```

Поля:

| Поле | Что значит |
|---|---|
| `[0008]` | номер кадра |
| `pts=307187` | сколько точек в облаке |
| `аном=5` | сколько ячеек створ подтвердил как аномалии |
| `OK` | режим трубы (см. ниже) |
| `в_трубе=1/5` | 1 из 5 аномалий попала в габарит |
| `hits=4/6` | 4 попадания в скользящем окне из 6 кадров |
| `>>> ПРЕПЯТСТВИЕ <<<` | детекция сработала |

Режимы трубы:

| Режим | Что значит | Учитывается? |
|---|---|---|
| `OK` | стена чистая, труба надёжная | да |
| `WEAK` | мелкие замечания (покрытие или скачок центра) | да |
| `SUSPECT` | плохое покрытие / rms / скачки центра | нет |
| `FALLBACK` | стены не найдены, труба = прямая заглушка | нет |
| `NONE` | трубы нет | нет |

По завершении печатается сводка: сколько кадров с аномалиями,
распределение режимов трубы, список подтверждённых кадров
и группировка их в эпизоды.

---

## Параметры `gate_tube.py`

| Флаг | Смысл | Дефолт |
|---|---|---|
| `bag` | путь к `.db3` или к папке с ним | обязателен |
| `--gate-rotate-deg D` | поворот облака **перед створом** (для forward=-Y: 90) | `0` |
| `--rotate-z-deg D` | поворот облака **перед трубой** (редко нужно) | `0` |
| `--max-frames N` | ограничить число кадров | — |
| `--count-all-modes` | считать «в трубе» и на SUSPECT/FALLBACK/NONE (для отладки) | off |
| `--tube-script PATH` | путь к `bend_tube.py` | рядом с `gate_tube.py` |
| `--no-gate-debug` | отключить debug-печать створа | debug on |

Параметры подтверждения препятствия — константы в
`src/gate_tube.py`:

```python
OBSTACLE_N = 4         # сколько кадров из окна должны быть попадания
OBSTACLE_M = 6         # размер окна
OBSTACLE_MIN_PTS = 1   # минимум точек в трубе для зачёта кадра
```

Изменить их можно, отредактировав файл и пересобрав образ.

---

## Разработка без пересборки образа

Смонтируйте `src/` поверх контейнерного — изменения в коде
подхватятся без `docker build`:

```bash
docker run --rm \
    -v "$PWD/data:/data:ro" \
    -v "$PWD/src:/workspace/src:ro" \
    metro-obstacle-detector:latest \
    /data/foo.db3 --gate-rotate-deg 90 --max-frames 20
```

---

## Отладка

Зайти внутрь контейнера:

```bash
docker run --rm -it \
    -v "$PWD/data:/data:ro" \
    -v "$PWD/src:/workspace/src:ro" \
    --entrypoint /bin/bash \
    metro-obstacle-detector:latest
```

Внутри:

```bash
source /opt/ros/humble/setup.bash
python3 /workspace/src/db3_inspect.py /data/foo.db3
python3 /workspace/src/gate_tube.py /data/foo.db3 --gate-rotate-deg 90
```

`db3_inspect.py` покажет список топиков, типы и число сообщений
в bag'е — полезно для проверки, что топик с облаком называется
так, как ожидает пайплайн.

---

## Как работает алгоритм (кратко)

### Створ (DynamicGate)

1. Облако раскладывается в range image (H×W): по X — азимут,
   по Y — элевация, в ячейке — дальность.
2. Оценивается скорость движения по дальним ячейкам.
3. Считается `drop = predicted − current` — насколько объект
   приблизился относительно предсказания.
4. `drop > 2.5 м` → сильная аномалия; `1.5 < drop ≤ 2.5 м` →
   слабая, требуется ≥3 соседа в радиусе 2 ячейки.
5. Кандидаты накапливаются в буфере длины 6 кадров; ячейка
   становится аномалией, если была кандидатом ≥4 раз.

### Габарит-труба (bend_tube)

1. Ищет вертикальные колонны (кандидаты в стены) в Y-срезах.
2. Кластеризует их, фитит линии, мерджит коллинеарные.
3. Строит θ(Y) — траекторию угла стены по длине тоннеля.
4. Extrude-ит 3D-трубу шириной `2·train_half_width` вдоль θ(Y).
5. Классифицирует трубу: `OK / WEAK / SUSPECT / FALLBACK / NONE`.

### Фильтр препятствий

Точка считается препятствием, если:

- она подтверждена створом как аномалия,
- попала внутрь габарит-трубы (по XY и по Z),
- труба на этом кадре в режиме OK или WEAK,
- за последние 6 кадров таких попаданий было ≥4.

---

## Замечания

- Скрипты в `src/` должны называться **точно** как в `import`:
  `dynamic_gate.py`, `bend_tube.py`, `gate_tube.py`.
- `numpy<2` зафиксирован в `requirements.txt` — ROS 2 Humble
  собирается против numpy 1.x.
- Результат работы пайплайна сейчас — только stdout. Если нужны
  файлы с аномалиями или трубой, их можно добавить, дописав
  сохранение в `gate_tube.py`.

---

## Быстрая проверка после сборки

```bash
# 1. Сборка
docker build -t metro-obstacle-detector:latest .

# 2. Хелп
docker run --rm metro-obstacle-detector:latest --help

# 3. Положите свой .db3 в ./data
mkdir -p data
cp /path/to/your.db3 data/

# 4. Инспекция топиков
docker run --rm -v "$PWD/data:/data:ro" \
    metro-obstacle-detector:latest \
    /data/your.db3 --max-frames 1

# 5. Полноценный прогон
docker run --rm -v "$PWD/data:/data:ro" \
    metro-obstacle-detector:latest \
    /data/your.db3 --gate-rotate-deg 90
```

---

## Сохранение образа для передачи

```bash
docker save metro-obstacle-detector:latest | gzip > metro-obstacle-detector.tar.gz
```

Загрузить на другой машине:

```bash
docker load < metro-obstacle-detector.tar.gz
```
