import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    import aiohttp
except ImportError:  # pragma: no cover - handled by requirements
    aiohttp = None


MONTH_TO_SEASON = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


@dataclass
class CityAnalysis:
    df: pd.DataFrame
    seasonal_stats: pd.DataFrame


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"city", "timestamp", "temperature", "season"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Нет колонок: {', '.join(sorted(missing))}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature", "city", "season"])
    return df


def _analyze_city(city_df: pd.DataFrame, window: int) -> CityAnalysis:
    city_df = city_df.sort_values("timestamp").copy()
    rolling = city_df["temperature"].rolling(window=window, min_periods=1)
    city_df["roll_mean"] = rolling.mean()
    city_df["roll_std"] = rolling.std(ddof=0).fillna(0.0)
    # Долгосрочный тренд: сглаживание на окне ~1 год.
    city_df["trend_mean"] = city_df["temperature"].rolling(window=365, min_periods=1).mean()
    band = 2 * city_df["roll_std"]
    city_df["anomaly"] = (city_df["temperature"] > city_df["roll_mean"] + band) | (
        city_df["temperature"] < city_df["roll_mean"] - band
    )

    seasonal = (
        city_df.groupby("season")["temperature"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "season_mean", "std": "season_std"})
        .reset_index()
    )
    seasonal["season_std"] = seasonal["season_std"].fillna(0.0)
    return CityAnalysis(df=city_df, seasonal_stats=seasonal)


@st.cache_data(show_spinner=False)
def analyze_all(df: pd.DataFrame, window: int) -> Tuple[Dict[str, CityAnalysis], float, float]:
    cities = sorted(df["city"].unique())

    start = time.perf_counter()
    serial_results = {city: _analyze_city(df[df["city"] == city], window) for city in cities}
    serial_time = time.perf_counter() - start

    def _worker(city: str) -> Tuple[str, CityAnalysis]:
        return city, _analyze_city(df[df["city"] == city], window)

    start = time.perf_counter()
    parallel_results: Dict[str, CityAnalysis] = {}
    max_workers = min(8, len(cities)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for city, result in executor.map(_worker, cities):
            parallel_results[city] = result
    parallel_time = time.perf_counter() - start

    # Use the parallel result set for the UI so the benchmark compares to it.
    return parallel_results, serial_time, parallel_time


def plot_time_series(city_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=city_df["timestamp"],
            y=city_df["temperature"],
            mode="lines",
            name="Температура",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=city_df["timestamp"],
            y=city_df["roll_mean"],
            mode="lines",
            name="Скользящее среднее (30 дн.)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=city_df["timestamp"],
            y=city_df["trend_mean"],
            mode="lines",
            name="Долгосрочный тренд (365 дн.)",
        )
    )
    anomalies = city_df[city_df["anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["temperature"],
            mode="markers",
            name="Аномалии",
            marker=dict(color="crimson", size=6),
        )
    )
    fig.update_layout(
        title="Температура по дням и аномалии",
        xaxis_title="Дата",
        yaxis_title="Температура (°C)",
        hovermode="x unified",
    )
    return fig


def plot_seasonal_profile(seasonal_stats: pd.DataFrame) -> go.Figure:
    order = ["winter", "spring", "summer", "autumn"]
    stats = seasonal_stats.set_index("season").reindex(order).reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stats["season"],
            y=stats["season_mean"],
            error_y=dict(type="data", array=stats["season_std"], visible=True),
            name="Среднее по сезонам",
        )
    )
    fig.update_layout(
        title="Сезонный профиль (среднее ± σ)",
        xaxis_title="Сезон",
        yaxis_title="Температура (°C)",
    )
    return fig


def _owm_url(city: str, api_key: str) -> str:
    return (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )


def fetch_current_temp_sync(city: str, api_key: str) -> Tuple[dict, float]:
    start = time.perf_counter()
    response = requests.get(_owm_url(city, api_key), timeout=10)
    elapsed = time.perf_counter() - start
    return response.json(), elapsed


async def _fetch_async(url: str) -> dict:
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for async requests.")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            return await response.json()


def fetch_current_temp_async(city: str, api_key: str) -> Tuple[dict, float]:
    start = time.perf_counter()
    data = asyncio.run(_fetch_async(_owm_url(city, api_key)))
    elapsed = time.perf_counter() - start
    return data, elapsed


def season_for_date(ts: datetime) -> str:
    return MONTH_TO_SEASON[ts.month]


def evaluate_current_temp(city_stats: pd.DataFrame, season: str, temp: float) -> Tuple[bool, float, float]:
    row = city_stats[city_stats["season"] == season]
    if row.empty:
        return True, np.nan, np.nan
    mean = float(row["season_mean"].iloc[0])
    std = float(row["season_std"].iloc[0])
    if std == 0:
        return True, mean, std
    return (mean - 2 * std) <= temp <= (mean + 2 * std), mean, std


st.set_page_config(page_title="Температура: анализ и мониторинг", layout="wide")
st.title("Температура: анализ и мониторинг")

st.sidebar.header("Данные")
uploaded_file = st.sidebar.file_uploader("Загрузить temperature_data.csv", type=["csv"])

data_path = "temperature_data.csv"
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    use_local = st.sidebar.checkbox("Использовать локальный temperature_data.csv", value=True)
    if use_local:
        try:
            raw_df = pd.read_csv(data_path)
        except FileNotFoundError:
            raw_df = None
    else:
        raw_df = None

if raw_df is None:
    st.info("Загрузите CSV-файл с историческими данными, чтобы начать анализ.")
    st.stop()

try:
    df = _validate_columns(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

analysis_results, serial_time, parallel_time = analyze_all(df, window=30)

st.sidebar.header("Выбор")
city = st.sidebar.selectbox("Город", sorted(analysis_results.keys()))
city_analysis = analysis_results[city]

st.subheader("Сравнение: последовательный и параллельный анализ")
st.caption(
    f"Последовательно: {serial_time:.4f} c | Параллельно (потоки): {parallel_time:.4f} c. "
    "Потоки дают выигрыш при больших объёмах или I/O."
)
st.subheader("Выводы по синхронному и асинхронному запросу")
st.caption(
    "Для одного города удобнее sync: проще код и предсказуемое время ответа. "
    "Async оправдан, когда нужно опрашивать много городов одновременно."
)

st.subheader("Описательная статистика")
st.dataframe(city_analysis.df["temperature"].describe(), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_time_series(city_analysis.df), use_container_width=True)
with col2:
    st.plotly_chart(plot_seasonal_profile(city_analysis.seasonal_stats), use_container_width=True)

st.subheader("Текущая температура через OpenWeatherMap")
api_key = st.text_input(
    "API-ключ",
    type="password",
    help="Ключ не сохраняется. Оставьте пустым, если не хотите делать запрос.",
)
mode = st.radio("Режим запроса", ["sync", "async"], horizontal=True)

if st.button("Получить текущую температуру") and api_key:
    if mode == "async" and aiohttp is None:
        st.error("Для async-режима нужен aiohttp. Установите зависимости.")
    else:
        if mode == "sync":
            data, elapsed = fetch_current_temp_sync(city, api_key)
        else:
            data, elapsed = fetch_current_temp_async(city, api_key)

        if str(data.get("cod")) == "401":
            st.error(data.get("message", "Неверный API-ключ."))
        elif "main" not in data or "temp" not in data["main"]:
            st.error(f"Неожиданный ответ API: {data}")
        else:
            current_temp = float(data["main"]["temp"])
            current_season = season_for_date(datetime.utcnow())
            is_normal, mean, std = evaluate_current_temp(
                city_analysis.seasonal_stats, current_season, current_temp
            )
            status = "норма" if is_normal else "аномалия"
            st.success(
                f"{city}: {current_temp:.1f} °C ({status}) | "
                f"Сезон: {current_season}, среднее={mean:.1f}, σ={std:.1f}, "
                f"режим={mode}, время={elapsed:.3f} c"
            )
            st.caption(
                "Для одного города проще sync. Async имеет смысл при множестве запросов."
            )
else:
    st.caption("Введите API-ключ, чтобы получить текущую температуру.")
