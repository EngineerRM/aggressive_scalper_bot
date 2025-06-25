import os
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from telegram.error import BadRequest, NetworkError
from pybit.unified_trading import HTTP, WebSocket as BybitWebSocket
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import optuna
import pandas_ta as ta
import pytz
import matplotlib.pyplot as plt
from io import BytesIO
import json
import redis.asyncio as redis
import requests
import sqlite3
import joblib
import xgboost as xgb
from typing import Optional, Dict, Any
import glob
import time

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Конфигурация --- #
class Config:
    def __init__(self):
        load_dotenv()
        self.API_KEY_BYBIT = os.getenv("BYBIT_API_KEY")
        self.API_SECRET_BYBIT = os.getenv("BYBIT_API_SECRET")
        self.BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        
        self.leverage = 10
        self.takeprofit = 0.5
        self.stoploss = 0.3
        self.trailing_stop = True
        self.trailing_distance = 0.0
        self.symbols = ["XRPUSDT", "SUIUSDT", "SOLUSDT"]
        self.active = False
        self.position_size = 10
        self.confirmation_required = True
        self.max_risk_percent = 0.02
        self.daily_loss_limit = 0.05
        self.daily_profit_limit = 0.1
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.max_trades_per_day = 20
        self.trades_today = 0
        self.last_reset = datetime.now(pytz.utc)
        self.timeframes = ['1', '5', '30', '60']
        self.model_path = 'models'
        self.current_exchange = 'bybit'
        self.current_symbol = 'XRPUSDT'
        self.language = 'ru'
        self.current_model = 'lstm'
        self.test_mode = False
        self.confidence_threshold = 0.80
        
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs('charts', exist_ok=True)
        self.validate_credentials()

    def validate_credentials(self):
        if not self.BOT_TOKEN:
            raise ValueError("Missing Telegram BOT_TOKEN")
        if not all([self.API_KEY_BYBIT, self.API_SECRET_BYBIT]):
            raise ValueError("Missing Bybit credentials")

CONFIG = Config()

# --- Кэширование --- #
class CacheManager:
    def __init__(self):
        self.client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

    async def get(self, key: str, subkey: Optional[str] = None):
        value = await self.client.get(f"{key}:{subkey}" if subkey else key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, subkey: Optional[str] = None, expiry: int = 300):
        await self.client.setex(f"{key}:{subkey}" if subkey else key, expiry, json.dumps(value))

    async def is_valid(self, key: str) -> bool:
        return await self.client.exists(key)

CACHE = CacheManager()

# --- Хранение свечей --- #
class CandleStorage:
    def __init__(self, db_path='models/candles.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    turnover REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            ''')

    def save_candles(self, df: pd.DataFrame, symbol: str, timeframe: str):
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df.to_sql('candles', self.conn, if_exists='append', index=False)
        logger.info(f"Saved {len(df)} candles for {symbol} ({timeframe})")

    def load_candles(self, symbol: str, timeframe: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        query = f"SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        query = "SELECT MAX(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?"
        cursor = self.conn.cursor()
        cursor.execute(query, (symbol, timeframe))
        result = cursor.fetchone()
        return result[0] if result[0] else None

CANDLE_STORAGE = CandleStorage()

# --- API Clients --- #
async def get_fear_greed_index():
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1')
        return response.json()['data'][0]['value']
    except:
        return 50

class ExchangeClient:
    async def get_balance(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    async def get_historical_data(self, symbol: str, limit: int, interval: str) -> pd.DataFrame:
        raise NotImplementedError

    async def get_order_book(self, symbol: str):
        raise NotImplementedError

    async def place_order(self, symbol: str, order_params: Dict):
        raise NotImplementedError

class BybitClient(ExchangeClient):
    def __init__(self):
        self.session = HTTP(
            api_key=CONFIG.API_KEY_BYBIT,
            api_secret=CONFIG.API_SECRET_BYBIT,
            timeout=10,
            recv_window=5000
        )
        self.ws_manager = WebSocketManager(self.session)

    async def get_balance(self, symbol: str) -> Optional[float]:
        if await CACHE.is_valid('balance'):
            return await CACHE.get('balance', 'value')
        try:
            balance = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            available_balance = float(balance["result"]["list"][0]["totalAvailableBalance"])
            await CACHE.set('balance', {'value': available_balance, 'timestamp': str(datetime.now(pytz.utc))})
            return available_balance
        except Exception as e:
            logger.error(f"Bybit balance error: {e}")
            return None

    async def get_historical_data(self, symbol: str, limit: int = 200, interval: str = "1") -> pd.DataFrame:
        cache_key = f"history_{symbol}_{interval}_{limit}"
        if await CACHE.is_valid(cache_key):
            cached_data = await CACHE.get('historical_data', cache_key)
            df = pd.DataFrame(cached_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        latest_ts = CANDLE_STORAGE.get_latest_timestamp(symbol, interval)
        start_time = latest_ts + 1 if latest_ts else int((datetime.now(pytz.utc) - timedelta(days=5*365)).timestamp() * 1000)
        df_list = []
        while True:
            try:
                candles = await asyncio.wait_for(
                    self.session.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        start=start_time
                    ),
                    timeout=10
                )
                if not candles.get("result", {}).get("list"):
                    break
                columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                df = pd.DataFrame(candles["result"]["list"], columns=columns)
                numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
                df[numeric_cols] = df[numeric_cols].astype(float)
                df['timestamp'] = df['timestamp'].astype(int)  # Преобразуем в число для кэширования
                df_list.append(df)
                start_time = df['timestamp'].iloc[-1] + 1
                if len(df) < limit:
                    break
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.error("Таймаут при получении исторических данных")
                break
            except Exception as e:
                logger.error(f"Bybit historical data error: {e}")
                break
        if df_list:
            df = pd.concat(df_list).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            CANDLE_STORAGE.save_candles(df, symbol, interval)
            cache_data = {'data': df.to_dict(orient='records'), 'timestamp': str(datetime.now(pytz.utc))}
            await CACHE.set('historical_data', cache_data, cache_key)
            return df
        return pd.DataFrame()

    async def get_order_book(self, symbol: str):
        try:
            response = self.session.get_orderbook(category="linear", symbol=symbol)
            return response['result']
        except Exception as e:
            logger.error(f"Bybit order book error: {e}")
            return None

    async def place_order(self, symbol: str, order_params: Dict):
        if CONFIG.test_mode:
            logger.info(f"Test mode: Simulated order {order_params}")
            return {'result': {'orderId': f"test_{int(time.time())}"}}
        return self.session.place_order(**order_params)

class WebSocketManager:
    def __init__(self, session):
        self.session = session
        self.ws = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        self.active = False

    async def start(self):
        self.active = True
        while self.active:
            try:
                if not self.ws or not self.ws.is_connected:
                    await self.connect()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self.reconnect()

    async def connect(self):
        try:
            self.ws = BybitWebSocket(
                testnet=False,
                api_key=CONFIG.API_KEY_BYBIT,
                api_secret=CONFIG.API_SECRET_BYBIT,
                channel_type="linear"
            )
            self.ws.kline_stream(
                interval=1,
                symbol=CONFIG.current_symbol,
                callback=self.handle_message
            )
            logger.info("WebSocket подключен")
            self.reconnect_attempts = 0
        except Exception as e:
            logger.error(f"Ошибка подключения WebSocket: {e}")
            await self.reconnect()

    async def reconnect(self):
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Достигнуто максимальное количество попыток переподключения")
            self.active = False
            return
        delay = min(self.reconnect_delay * (self.reconnect_attempts + 1), 60)
        logger.info(f"Повторное подключение через {delay} секунд...")
        await asyncio.sleep(delay)
        self.reconnect_attempts += 1
        await self.connect()

    def handle_message(self, data):
        try:
            if data.get('topic') == f'kline.1::{CONFIG.current_symbol}':
                kline = data['data'][0]
                price = float(kline['close'])
                logger.info(f"WebSocket: Получена цена {CONFIG.current_symbol}: {price}")
                asyncio.run(CACHE.set('price', {'value': price, 'timestamp': str(datetime.now(pytz.utc))}))  # Асинхронный вызов
                self.update_trailing_stop(price)
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения WebSocket: {e}")

    def update_trailing_stop(self, current_price: float):
        positions = CACHE.get('positions', {})
        for position_id, position in positions.items():
            if position['status'] == 'active':
                new_sl = self.calculate_new_stop_loss(position, current_price)
                if new_sl:
                    self.update_position_sl(position_id, new_sl)

    def calculate_new_stop_loss(self, position: Dict, current_price: float) -> Optional[float]:
        if position['side'] == 'long':
            new_sl = current_price * (1 - CONFIG.trailing_distance/100)
            return new_sl if new_sl > position['stop_loss'] else None
        elif position['side'] == 'short':
            new_sl = current_price * (1 + CONFIG.trailing_distance/100)
            return new_sl if new_sl < position['stop_loss'] else None
        return None

    def update_position_sl(self, position_id: str, new_sl: float):
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=CONFIG.current_symbol,
                stopLoss=str(new_sl),
                positionIdx=position_id
            )
            positions = CACHE.get('positions', {})
            positions[position_id]['stop_loss'] = new_sl
            asyncio.run(CACHE.set('positions', positions))  # Асинхронный вызов
            logger.info(f"Обновлен SL для позиции {position_id}: {new_sl}")
        except Exception as e:
            logger.error(f"Ошибка обновления SL: {e}")

# --- Торговая логика --- #
class TradingEngine:
    def __init__(self):
        self.current_client = BybitClient()
        self.lstm_model = self.init_lstm_model()
        self.gboost_model = self.init_gboost_model()
        self.scaler = MinMaxScaler()
        self.cleanup_old_models()

    def init_lstm_model(self):
        class LSTMModel(nn.Module):
            def __init__(self, input_size=3, hidden_size=128, num_layers=2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
                self.dropout = nn.Dropout(0.2)
                self.dense1 = nn.Linear(hidden_size, 32)
                self.dense2 = nn.Linear(32, 2)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x, _ = self.lstm(x)
                x = self.dropout(x[:, -1, :])
                x = torch.relu(self.dense1(x))
                x = self.dense2(x)
                x = self.sigmoid(x)
                return x

        model_files = sorted(glob.glob(os.path.join(CONFIG.model_path, 'lstm_model_*.pth')), key=os.path.getmtime, reverse=True)
        if model_files:
            try:
                model = LSTMModel()
                model.load_state_dict(torch.load(model_files[0]))
                logger.info(f"LSTM-модель загружена: {model_files[0]}")
                return model
            except Exception as e:
                logger.error(f"Ошибка загрузки LSTM: {e}")
        return LSTMModel()

    def init_gboost_model(self):
        model_files = sorted(glob.glob(os.path.join(CONFIG.model_path, 'gboost_model_*.pkl')), key=os.path.getmtime, reverse=True)
        if model_files:
            try:
                model = joblib.load(model_files[0])
                logger.info(f"GBoost-модель загружена: {model_files[0]}")
                return model
            except Exception as e:
                logger.error(f"Ошибка загрузки GBoost: {e}")
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)

    def cleanup_old_models(self):
        now = datetime.now(pytz.utc)
        for model_file in glob.glob(os.path.join(CONFIG.model_path, '*.pth')) + glob.glob(os.path.join(CONFIG.model_path, '*.pkl')):
            mtime = datetime.fromtimestamp(os.path.getmtime(model_file), tz=pytz.utc)
            if (now - mtime).days > 30:
                os.remove(model_file)
                logger.info(f"Удалена старая модель: {model_file}")

    async def save_lstm_model(self):
        if self.lstm_model:
            timestamp = datetime.now(pytz.utc).strftime('%Y%m%d_%H%M')
            model_path = os.path.join(CONFIG.model_path, f'lstm_model_{timestamp}.pth')
            torch.save(self.lstm_model.state_dict(), model_path)
            logger.info(f"LSTM-модель сохранена: {model_path}")

    async def save_gboost_model(self):
        if self.gboost_model:
            timestamp = datetime.now(pytz.utc).strftime('%Y%m%d_%H%M')
            model_path = os.path.join(CONFIG.model_path, f'gboost_model_{timestamp}.pkl')
            joblib.dump(self.gboost_model, model_path)
            logger.info(f"GBoost-модель сохранена: {model_path}")

    async def train_lstm(self, update: Optional[Update] = None):
        try:
            if update:
                await update.effective_message.reply_text("Начато обучение LSTM...")
            df = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=200, interval='1')
            if df.empty:
                raise ValueError("Нет данных для обучения LSTM")
            X, y = self.prepare_data(df)
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            epochs = 50
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.lstm_model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0 and update:
                    await update.effective_message.reply_text(f"LSTM: Эпоха {epoch+1}/{epochs}, Потеря: {loss.item():.4f}")
            accuracy = self.evaluate_lstm(X, y)
            logger.info(f"LSTM обучение завершено: точность={accuracy:.4f}, потеря={loss.item():.4f}")
            if update:
                await update.effective_message.reply_text(f"LSTM обучение завершено: точность={accuracy:.2%}")
            await self.save_lstm_model()
        except Exception as e:
            logger.error(f"Ошибка обучения LSTM: {e}", exc_info=True)
            if update:
                await update.effective_message.reply_text(f"Ошибка обучения LSTM: {str(e)}")

    async def train_gboost(self, update: Optional[Update] = None):
        try:
            if update:
                await update.effective_message.reply_text("Начато обучение GBoost...")
            df = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=200, interval='1')
            if df.empty:
                raise ValueError("Нет данных для обучения GBoost")
            X, y = self.prepare_data_gboost(df)
            self.gboost_model.fit(X, y)
            mse = np.mean((self.gboost_model.predict(X) - y) ** 2)
            variance = np.var(y)
            confidence = 1 - (mse / variance) if variance > 0 else 0
            logger.info(f"GBoost обучение завершено: MSE={mse:.4f}, уверенность={confidence:.4f}")
            if update:
                await update.effective_message.reply_text(f"GBoost обучение завершено: уверенность={confidence:.2%}")
            await self.save_gboost_model()
        except Exception as e:
            logger.error(f"Ошибка обучения GBoost: {e}", exc_info=True)
            if update:
                await update.effective_message.reply_text(f"Ошибка обучения GBoost: {str(e)}")

    async def optimize_gboost(self, update: Optional[Update] = None):
        async def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            df = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=200)
            X, y = self.prepare_data_gboost(df)
            model = xgb.XGBRegressor(**params)
            model.fit(X, y)
            mse = np.mean((model.predict(X) - y) ** 2)
            return mse

        if update:
            await update.effective_message.reply_text("Начата оптимизация GBoost...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: asyncio.run(objective(trial)), n_trials=20)
        self.gboost_model = xgb.XGBRegressor(**study.best_params)
        await self.train_gboost(update)
        logger.info(f"GBoost оптимизация завершена: лучшие параметры={study.best_params}")
        if update:
            await update.effective_message.reply_text(f"GBoost оптимизация завершена: лучшие параметры={study.best_params}")

    def prepare_data(self, df):
        df['rsi'] = ta.rsi(df['close'], length=14)
        df = df.dropna()
        features = df[['close', 'volume', 'rsi']].values
        features_scaled = self.scaler.fit_transform(features)
        X = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor((df['close'].shift(-1) > df['close']).astype(int), dtype=torch.float32).unsqueeze(1)
        y = torch.cat([y, 1-y], dim=1)
        return X, y

    def prepare_data_gboost(self, df):
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ma20'] = ta.sma(df['close'], length=20)
        df['ma50'] = ta.sma(df['close'], length=50)
        df = df.dropna()
        features = df[['close', 'volume', 'rsi', 'ma20', 'ma50']].values
        X = self.scaler.fit_transform(features)
        y = df['close'].shift(-1).fillna(method='ffill').values
        return X, y

    def evaluate_lstm(self, X, y):
        with torch.no_grad():
            outputs = self.lstm_model(X)
            predicted = (outputs[:, 0] > 0.5).float()
            actual = y[:, 0]
            accuracy = (predicted == actual).float().mean().item()
        return accuracy

    async def calculate_indicators(self, df: pd.DataFrame, timeframe: str = '1') -> pd.DataFrame:
        try:
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['ma20'] = ta.sma(df['close'], length=20)
            df['ma50'] = ta.sma(df['close'], length=50)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            bb = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bb], axis=1)
            if timeframe != '1':
                macd = ta.macd(df['close'])
                df = pd.concat([df, macd], axis=1)
            return df
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df

    async def generate_signal(self, update: Optional[Update] = None) -> Optional[tuple[str, float]]:
        try:
            await self.train_if_needed()
            tf1 = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=100, interval='1')
            tf60 = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=100, interval='60')
            if tf1.empty or tf60.empty:
                return None
            tf1 = await self.calculate_indicators(tf1, '1')
            tf60 = await self.calculate_indicators(tf60, '60')
            last_tf60 = tf60.iloc[-1]
            tf60_trend = 'up' if last_tf60['close'] > last_tf60['ma50'] else 'down'
            last_tf1 = tf1.iloc[-1]
            fear_greed = await get_fear_greed_index()
            X_lstm, _ = self.prepare_data(tf1)
            X_gboost = self.prepare_data_gboost(tf1)[-1].reshape(1, -1)
            confidence = 0.0
            signal = None
            if CONFIG.current_model == 'lstm':
                with torch.no_grad():
                    output = self.lstm_model(X_lstm[-1:])
                    confidence = output[0, 0].item() if output[0, 0] > output[0, 1] else output[0, 1].item()
                    signal = 'buy' if output[0, 0] > output[0, 1] else 'sell'
            else:
                pred = self.gboost_model.predict(X_gboost)[0]
                actual = tf1['close'].iloc[-1]
                mse = (pred - actual) ** 2
                variance = np.var(tf1['close'].tail(100))
                confidence = 1 - (mse / variance) if variance > 0 else 0
                signal = 'buy' if pred > actual else 'sell'
            logger.info(f"Сигнал: {signal}, уверенность={confidence:.4f}")
            if confidence < CONFIG.confidence_threshold:
                logger.info(f"Ордер не открыт: уверенность {confidence:.2%} < {CONFIG.confidence_threshold:.2%}")
                if update:
                    await update.effective_message.reply_text(
                        f"❌ Ордер не открыт: уверенность {confidence:.2%} < {CONFIG.confidence_threshold:.2%}"
                    )
                return None
            long_conditions = (
                tf60_trend == 'up' and
                last_tf1['close'] > last_tf1['ma20'] and
                last_tf1['rsi'] < 60 and
                (last_tf1['close'] - last_tf1['low']) > last_tf1['atr'] * 0.5 and
                int(fear_greed) < 75
            )
            short_conditions = (
                tf60_trend == 'down' and
                last_tf1['close'] < last_tf1['ma20'] and
                last_tf1['rsi'] > 40 and
                (last_tf1['high'] - last_tf1['close']) > last_tf1['atr'] * 0.5 and
                int(fear_greed) > 25
            )
            if long_conditions and signal == 'buy':
                return 'buy', confidence
            elif short_conditions and signal == 'sell':
                return 'sell', confidence
            return None
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала: {e}")
            return None

    async def train_if_needed(self):
        now = datetime.now(pytz.utc)
        model_files = sorted(glob.glob(os.path.join(CONFIG.model_path, f'{CONFIG.current_model}_model_*.pth' if CONFIG.current_model == 'lstm' else f'{CONFIG.current_model}_model_*.pkl')), key=os.path.getmtime, reverse=True)
        if not model_files or (now - datetime.fromtimestamp(os.path.getmtime(model_files[0]), tz=pytz.utc)).total_seconds() > 6*3600:
            if CONFIG.current_model == 'lstm':
                await self.train_lstm()
            else:
                await self.train_gboost()

    async def calculate_position_size(self, balance: float) -> float:
        try:
            df = await self.current_client.get_historical_data(CONFIG.current_symbol, limit=14)
            atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
            risk_amount = balance * CONFIG.max_risk_percent
            position_size = risk_amount / (atr * CONFIG.leverage)
            max_position = balance * 0.1
            min_position = 5
            return max(min(position_size, max_position), min_position)
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return CONFIG.position_size

    async def place_order(self, signal: str, confidence: float, update: Optional[Update] = None) -> bool:
        try:
            if not await self.check_limits():
                return False
            await self.train_if_needed()
            balance = await self.current_client.get_balance(CONFIG.current_symbol)
            if not balance:
                return False
            position_size = await self.calculate_position_size(balance)
            price = await CACHE.get('price', 'value')
            if not price:
                logger.error("Цена недоступна для размещения ордера")
                return False
            qty = round(position_size / price, 2)
            if qty <= 0:
                logger.error("Некорректное количество для ордера")
                return False
            order_params = {
                "symbol": CONFIG.current_symbol,
                "orderType": "Market",
                "qty": str(qty),
                "side": "Buy" if signal == 'buy' else "Sell",
                "takeProfit": str(price * (1 + CONFIG.takeprofit/100)) if signal == 'buy' else str(price * (1 - CONFIG.takeprofit/100)),
                "stopLoss": str(price * (1 - CONFIG.stoploss/100)) if signal == 'buy' else str(price * (1 + CONFIG.stoploss/100)),
                "timeInForce": "GoodTillCancel"
            }
            response = await self.current_client.place_order(CONFIG.current_symbol, order_params)
            order_id = response['result']['orderId']
            position_info = {
                'order_id': order_id,
                'side': 'long' if signal == 'buy' else 'short',
                'entry_price': price,
                'qty': qty,
                'take_profit': float(order_params['takeProfit']),
                'stop_loss': float(order_params['stopLoss']),
                'status': 'active',
                'timestamp': str(datetime.now(pytz.utc)),
                'confidence': confidence
            }
            positions = await CACHE.get('positions') or {}
            positions[order_id] = position_info
            await CACHE.set('positions', positions)
            CONFIG.trades_today += 1
            logger.info(f"Открыта позиция: {signal.upper()} {qty} {CONFIG.current_symbol} по {price}, уверенность={confidence:.2%}")
            if update:
                await update.effective_message.reply_text(
                    f"{'📈' if signal == 'buy' else '📉'} Новая позиция:\n"
                    f"• Тип: {signal.upper()}\n"
                    f"• Объем: {qty:.2f} {CONFIG.current_symbol}\n"
                    f"• Цена: {price:.4f}\n"
                    f"• TP: {order_params['takeProfit']}\n"
                    f"• SL: {order_params['stopLoss']}\n"
                    f"• Уверенность: {confidence:.2%}"
                )
            return True
        except Exception as e:
            logger.error(f"Ошибка размещения ордера: {e}")
            if update:
                await update.effective_message.reply_text(f"❌ Ошибка: {str(e)}")
            return False

    async def check_limits(self) -> bool:
        now = datetime.now(pytz.utc)
        if (now - CONFIG.last_reset).days >= 1:
            CONFIG.daily_loss = 0.0
            CONFIG.daily_profit = 0.0
            CONFIG.trades_today = 0
            CONFIG.last_reset = now
            logger.info("Дневные лимиты сброшены")
        if CONFIG.trades_today >= CONFIG.max_trades_per_day:
            logger.error("Достигнут лимит сделок за день")
            return False
        balance = await self.current_client.get_balance(CONFIG.current_symbol)
        if not balance:
            return False
        if CONFIG.daily_loss / balance > CONFIG.daily_loss_limit:
            logger.error("Достигнут лимит дневного убытка")
            CONFIG.active = False
            return False
        if CONFIG.daily_profit / balance > CONFIG.daily_profit_limit:
            logger.error("Достигнут лимит дневной прибыли")
            CONFIG.active = False
            return False
        return True

    async def predict_future_price(self, df: pd.DataFrame) -> float:
        if CONFIG.current_model == 'lstm':
            X, _ = self.prepare_data(df)
            with torch.no_grad():
                output = self.lstm_model(X[-1:])
                return df['close'].iloc[-1] * (1 + (output[0, 0].item() - output[0, 1].item()))
        else:
            X = self.prepare_data_gboost(df)[-1].reshape(1, -1)
            return self.gboost_model.predict(X)[0]

# --- Telegram Interface --- #
class TelegramBot:
    def __init__(self):
        self.application = Application.builder().token(CONFIG.BOT_TOKEN).build()
        self.trading_engine = TradingEngine()
        self.setup_handlers()
        self.schedule_training()

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    def schedule_training(self):
        loop = asyncio.get_event_loop()
        async def periodic_training():
            while True:
                now = datetime.now(pytz.utc)
                if now.hour in [0, 6, 12, 18] and now.minute == 0:
                    await self.trading_engine.train_if_needed()
                await asyncio.sleep(60)
        loop.create_task(periodic_training())

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"👋 Привет, {update.effective_user.first_name}!\n🤖 Скальпер-бот готов!",
            reply_markup=self.get_main_keyboard()
        )

    def get_main_keyboard(self):
        status = "🟢" if CONFIG.active else "🔴"
        model = "LSTM" if CONFIG.current_model == 'lstm' else "GBoost"
        test_mode = "🟢" if CONFIG.test_mode else "🔴"
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{status} Торговля", callback_data='toggle_trading')],
            [InlineKeyboardButton(f"Модель: {model}", callback_data='models_menu'),
             InlineKeyboardButton(f"Тест: {test_mode}", callback_data='toggle_test_mode')],
            [InlineKeyboardButton("⚖️ Плечо", callback_data='leverage_menu'),
             InlineKeyboardButton("🎯 TP/SL", callback_data='tp_sl_menu')],
            [InlineKeyboardButton("💰 Баланс", callback_data='get_balance'),
             InlineKeyboardButton("📊 Инфо", callback_data='get_info')],
            [InlineKeyboardButton("📈 График", callback_data='chart_menu'),
             InlineKeyboardButton("📝 Логи", callback_data='get_logs')],
            [InlineKeyboardButton("⚙️ Настройки", callback_data='settings_menu'),
             InlineKeyboardButton("ℹ️ Помощь", callback_data='help')],
            [InlineKeyboardButton("🔄 Выбрать пару", callback_data='select_symbol')]
        ])

    def get_models_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("LSTM", callback_data='select_lstm'),
             InlineKeyboardButton("GBoost", callback_data='select_gboost')],
            [InlineKeyboardButton("Обучить LSTM", callback_data='train_lstm'),
             InlineKeyboardButton("Обучить GBoost", callback_data='train_gboost')],
            [InlineKeyboardButton("Оптимизировать GBoost", callback_data='optimize_gboost')],
            [InlineKeyboardButton("⬅️ Назад", callback_data='main_menu')]
        ])

    def get_chart_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{tf}m", callback_data=f"chart_{tf}") for tf in CONFIG.timeframes],
            [InlineKeyboardButton("⬅️ Назад", callback_data='main_menu')]
        ])

    def get_symbol_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(symbol, callback_data=f"symbol_{symbol}") for symbol in CONFIG.symbols],
            [InlineKeyboardButton("⬅️ Назад", callback_data='main_menu')]
        ])

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        handlers = {
            'toggle_trading': self.toggle_trading_menu,
            'leverage_menu': self.leverage_menu,
            'tp_sl_menu': self.tp_sl_menu,
            'get_balance': self.show_balance,
            'get_info': self.show_info,
            'get_logs': self.send_logs,
            'settings_menu': self.settings_menu,
            'help': self.show_help,
            'cancel_action': self.cancel_action,
            'toggle_confirmation': self.toggle_confirmation,
            'main_menu': self.main_menu,
            'select_symbol': self.select_symbol_menu,
            'models_menu': self.models_menu,
            'select_lstm': self.select_lstm,
            'select_gboost': self.select_gboost,
            'train_lstm': self.train_lstm,
            'train_gboost': self.train_gboost,
            'optimize_gboost': self.optimize_gboost,
            'chart_menu': self.chart_menu,
            'toggle_test_mode': self.toggle_test_mode
        }
        if query.data in handlers:
            await handlers[query.data](update, context)
        elif query.data.startswith('confirm_'):
            await self.confirm_action(update, context)
        elif query.data.startswith('leverage_'):
            await self.handle_leverage(update, context)
        elif query.data.startswith('tp_'):
            await self.handle_takeprofit(update, context)
        elif query.data.startswith('sl_'):
            await self.handle_stoploss(update, context)
        elif query.data.startswith('symbol_'):
            await self.handle_symbol_selection(update, context)
        elif query.data.startswith('chart_'):
            await self.handle_chart_selection(update, context)

    async def toggle_trading_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        CONFIG.active = not CONFIG.active
        status = "🟢" if CONFIG.active else "🔴"
        await query.edit_message_text(
            f"Торговля: {status}",
            reply_markup=self.get_main_keyboard()
        )
        if CONFIG.active:
            loop = asyncio.get_event_loop()
            loop.create_task(self.trade_loop(context))

    async def leverage_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(str(i), callback_data=f"leverage_{i}") for i in [5, 10, 20]],
            [InlineKeyboardButton("⬅️ Назад", callback_data='main_menu')]
        ])
        await update.callback_query.edit_message_text("Выберите плечо:", reply_markup=keyboard)

    async def tp_sl_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"TP: {i}%", callback_data=f"tp_{i}") for i in [0.5, 1, 2]],
            [InlineKeyboardButton(f"SL: {i}%", callback_data=f"sl_{i}") for i in [0.3, 0.5, 1]],
            [InlineKeyboardButton("⬅️ Назад", callback_data='main_menu')]
        ])
        await update.callback_query.edit_message_text("Установите TP/SL:", reply_markup=keyboard)

    async def show_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        balance = await self.trading_engine.current_client.get_balance(CONFIG.current_symbol)
        await update.callback_query.edit_message_text(
            f"Баланс: {balance:.2f} USDT" if balance else "Баланс недоступен",
            reply_markup=self.get_main_keyboard()
        )

    async def show_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        price = await CACHE.get('price', 'value')
        if price is None:
            await update.callback_query.edit_message_text(
                f"Текущая цена {CONFIG.current_symbol}: не доступна",
                reply_markup=self.get_main_keyboard()
            )
        else:
            await update.callback_query.edit_message_text(
                f"Текущая цена {CONFIG.current_symbol}: {price:.4f}",
                reply_markup=self.get_main_keyboard()
            )

    async def send_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE, timeframe: str = '1'):
        try:
            df = await self.trading_engine.current_client.get_historical_data(CONFIG.current_symbol, limit=100, interval=timeframe)
            if df.empty:
                await update.callback_query.message.reply_text("Нет данных для графика")
                return
            df = await self.trading_engine.calculate_indicators(df, timeframe)
            current_price = df['close'].iloc[-1]
            future_price = await self.trading_engine.predict_future_price(df)
            order_book = await self.trading_engine.current_client.get_order_book(CONFIG.current_symbol)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # График цены
            ax1.plot(df['timestamp'], df['close'], label='Цена', color='blue')
            ax1.plot(df['timestamp'], df['BBM_20_2.0'], label='Bollinger Средняя', color='orange')
            ax1.plot(df['timestamp'], df['BBU_20_2.0'], label='Bollinger Верх', color='green', linestyle='--')
            ax1.plot(df['timestamp'], df['BBL_20_2.0'], label='Bollinger Низ', color='red', linestyle='--')
            ax1.axhline(y=current_price, color='purple', linestyle='-', label=f'Текущая цена: {current_price:.4f}')
            future_time = df['timestamp'].iloc[-1] + timedelta(minutes=60)
            ax1.plot([df['timestamp'].iloc[-1], future_time], [current_price, future_price], 'o-', color='black', label=f'Прогноз: {future_price:.4f}')
            ax1.set_title(f"{CONFIG.current_symbol} - {timeframe}мин")
            ax1.legend()
            ax1.grid(True)
            
            # Торговый стакан
            if order_book:
                bids = [(float(b[0]), float(b[1])) for b in order_book.get('b', [])]
                asks = [(float(a[0]), float(a[1])) for a in order_book.get('a', [])]
                bid_prices, bid_volumes = zip(*bids) if bids else ([], [])
                ask_prices, ask_volumes = zip(*asks) if asks else ([], [])
                ax2.bar(bid_prices, bid_volumes, color='green', alpha=0.5, label='Bids')
                ax2.bar(ask_prices, ask_volumes, color='red', alpha=0.5, label='Asks')
                ax2.set_title('Торговый стакан')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            timestamp = datetime.now(pytz.utc).strftime('%Y%m%d_%H%M')
            chart_path = f"charts/{CONFIG.current_symbol}_{timeframe}m_{timestamp}.png"
            plt.savefig(chart_path, format='png')
            plt.close()
            
            with open(chart_path, 'rb') as f:
                await update.callback_query.message.reply_photo(
                    photo=InputFile(f),
                    caption=f"График {CONFIG.current_symbol} ({timeframe}мин)\nПрогноз через 60 мин: {future_price:.4f}",
                    reply_markup=self.get_chart_keyboard()
                )
            
            for old_chart in glob.glob('charts/*.png'):
                mtime = datetime.fromtimestamp(os.path.getmtime(old_chart), tz=pytz.utc)
                if (datetime.now(pytz.utc) - mtime).days > 7:
                    os.remove(old_chart)
                    logger.info(f"Удалён старый график: {old_chart}")
        except Exception as e:
            logger.error(f"Ошибка построения графика: {e}")
            await update.callback_query.message.reply_text("Ошибка построения графика")

    async def send_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with open('bot.log', 'r') as f:
            logs = f.read()
        await update.callback_query.edit_message_text(f"Логи:\n{logs[-4000:]}...", reply_markup=self.get_main_keyboard())

    async def settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        confirmation_status = "🟢" if CONFIG.confirmation_required else "🔴"
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"Подтверждение: {confirmation_status}", callback_data='toggle_confirmation')],
            [InlineKeyboardButton("⬅ Назад", callback_data='main_menu')]
        ])
        await update.callback_query.edit_message_text("Настройки:", reply_markup=keyboard)

    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text(
            "Команды:\n"
            "/start - Запуск бота\n"
            "Кнопки: Управление торговлей, моделями, графиками и логами.",
            reply_markup=self.get_main_keyboard()
        )

    async def cancel_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text("Действие отменено", reply_markup=self.get_main_keyboard())

    async def toggle_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        CONFIG.confirmation_required = not CONFIG.confirmation_required
        await self.settings_menu(update, context)

    async def main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text("Главное меню:", reply_markup=self.get_main_keyboard())

    async def select_symbol_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text("Выберите торговую пару:", reply_markup=self.get_symbol_keyboard())

    async def models_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text("Меню моделей:", reply_markup=self.get_models_keyboard())

    async def chart_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.edit_message_text("Выберите таймфрейм:", reply_markup=self.get_chart_keyboard())

    async def select_lstm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        CONFIG.current_model = 'lstm'
        await update.callback_query.edit_message_text("Выбрана модель: LSTM", reply_markup=self.get_main_keyboard())

    async def select_gboost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        CONFIG.current_model = 'gboost'
        await update.callback_query.edit_message_text("Выбрана модель: GBoost", reply_markup=self.get_main_keyboard())

    async def train_lstm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trading_engine.train_lstm(update))
        await update.callback_query.edit_message_text("Обучение LSTM запущено в фоновом режиме", reply_markup=self.get_main_keyboard())

    async def train_gboost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trading_engine.train_gboost(update))
        await update.callback_query.edit_message_text("Обучение GBoost запущено в фоновом режиме", reply_markup=self.get_main_keyboard())

    async def optimize_gboost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trading_engine.optimize_gboost(update))
        await update.callback_query.edit_message_text("Оптимизация GBoost запущена в фоновом режиме", reply_markup=self.get_main_keyboard())

    async def toggle_test_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        CONFIG.test_mode = not CONFIG.test_mode
        status = "🟢" if CONFIG.test_mode else "🔴"
        await update.callback_query.edit_message_text(f"Тестовый режим: {status}", reply_markup=self.get_main_keyboard())

    async def handle_symbol_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        symbol = update.callback_query.data.split('_')[1]
        CONFIG.current_symbol = symbol
        await update.callback_query.edit_message_text(f"Выбрана пара: {symbol}", reply_markup=self.get_main_keyboard())

    async def handle_chart_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        timeframe = update.callback_query.data.split('_')[1]
        await self.send_chart(update, context, timeframe=timeframe)

    async def confirm_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        action = update.callback_query.data.split('_')[1]
        await update.callback_query.edit_message_text(f"Действие {action} подтверждено", reply_markup=self.get_main_keyboard())

    async def handle_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        leverage = int(update.callback_query.data.split('_')[1])
        CONFIG.leverage = leverage
        await update.callback_query.edit_message_text(f"Плечо установлено: {leverage}x", reply_markup=self.get_main_keyboard())

    async def handle_takeprofit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tp = float(update.callback_query.data.split('_')[1])
        CONFIG.takeprofit = tp
        await update.callback_query.edit_message_text(f"TP установлен: {tp}%", reply_markup=self.get_main_keyboard())

    async def handle_stoploss(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        sl = float(update.callback_query.data.split('_')[1])
        CONFIG.stoploss = sl
        await update.callback_query.edit_message_text(f"SL установлен: {sl}%", reply_markup=self.get_main_keyboard())

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Пожалуйста, используйте кнопки или команды.")

    async def trade_loop(self, context: ContextTypes.DEFAULT_TYPE):
        while CONFIG.active:
            try:
                signal_confidence = await self.trading_engine.generate_signal()
                if signal_confidence:
                    signal, confidence = signal_confidence
                    await self.trading_engine.place_order(signal, confidence)
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(30)

    def run(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trading_engine.current_client.ws_manager.start())
        if CONFIG.active:
            loop.create_task(self.trade_loop(self.application))
        logger.info("Бот запущен успешно")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    try:
        bot = TelegramBot()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(bot.run())
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        asyncio.run(TradingEngine().save_lstm_model())
        asyncio.run(TradingEngine().save_gboost_model())