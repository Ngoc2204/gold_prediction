import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Flask
from flask import Flask, render_template, request, jsonify
import json
import pickle
import os

class GoldPredictionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.data = pd.DataFrame()
        self.features = []
        self.target = 'Gold_Price'
        
    def collect_data(self, start_date='2020-01-01', end_date=None):
        """Thu thập dữ liệu từ nhiều nguồn - Đã khắc phục lỗi dimension"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print("Đang thu thập dữ liệu...")
        
        try:
            # 1. Giá vàng (GLD ETF hoặc GC=F futures)
            try:
                gold_data = yf.download('GLD', start=start_date, end=end_date, progress=False)
                if gold_data.empty:
                    raise Exception("No data for GLD")
                # FIX: Đảm bảo là Series, không phải DataFrame
                gold_prices = gold_data['Close'].resample('D').last()
                if isinstance(gold_prices, pd.DataFrame):
                    gold_prices = gold_prices.squeeze()  # Chuyển DataFrame 1 cột thành Series
            except:
                # Backup: sử dụng Gold futures
                gold_data = yf.download('GC=F', start=start_date, end=end_date, progress=False)
                if gold_data.empty:
                    raise Exception("No data for gold")
                gold_prices = gold_data['Close'].resample('D').last()
                if isinstance(gold_prices, pd.DataFrame):
                    gold_prices = gold_prices.squeeze()
            
            # 2. Chỉ số USD (DXY)
            try:
                usd_data = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
                if usd_data.empty:
                    raise Exception("No USD data")
                usd_index = usd_data['Close'].resample('D').last()
                if isinstance(usd_index, pd.DataFrame):
                    usd_index = usd_index.squeeze()
            except:
                # Backup: sử dụng EURUSD
                try:
                    eur_data = yf.download('EURUSD=X', start=start_date, end=end_date, progress=False)
                    eur_close = eur_data['Close'].resample('D').last()
                    if isinstance(eur_close, pd.DataFrame):
                        eur_close = eur_close.squeeze()
                    usd_index = (1 / eur_close)
                    usd_index.name = 'USD_Index'
                except:
                    # Fallback: tạo dữ liệu giả
                    usd_index = pd.Series(np.random.normal(100, 5, size=len(gold_prices)), 
                                        index=gold_prices.index, name='USD_Index')
            
            # 3. Giá dầu (WTI)
            try:
                oil_data = yf.download('CL=F', start=start_date, end=end_date, progress=False)
                oil_prices = oil_data['Close'].resample('D').last()
                if isinstance(oil_prices, pd.DataFrame):
                    oil_prices = oil_prices.squeeze()
            except:
                oil_prices = pd.Series(np.random.normal(70, 10, len(gold_prices)), 
                                    index=gold_prices.index, name='Oil_Price')
            
            # 4. Chỉ số chứng khoán S&P 500
            try:
                spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
                spy_prices = spy_data['Close'].resample('D').last()
                if isinstance(spy_prices, pd.DataFrame):
                    spy_prices = spy_prices.squeeze()
            except:
                spy_prices = pd.Series(np.random.normal(400, 50, len(gold_prices)), 
                                    index=gold_prices.index, name='SP500')
            
            # 5. Lãi suất (10-year Treasury)
            try:
                treasury_data = yf.download('^TNX', start=start_date, end=end_date, progress=False)
                treasury_yield = treasury_data['Close'].resample('D').last()
                if isinstance(treasury_yield, pd.DataFrame):
                    treasury_yield = treasury_yield.squeeze()
            except:
                treasury_yield = pd.Series(np.random.normal(3, 0.5, len(gold_prices)), 
                                        index=gold_prices.index, name='Treasury_Yield')
            
            # 6. VIX (Volatility Index)
            try:
                vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                vix_prices = vix_data['Close'].resample('D').last()
                if isinstance(vix_prices, pd.DataFrame):
                    vix_prices = vix_prices.squeeze()
            except:
                vix_prices = pd.Series(np.random.normal(20, 5, len(gold_prices)), 
                                    index=gold_prices.index, name='VIX')
            
            # 7. Bitcoin (as alternative asset)
            try:
                btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
                btc_prices = btc_data['Close'].resample('D').last()
                if isinstance(btc_prices, pd.DataFrame):
                    btc_prices = btc_prices.squeeze()
            except:
                btc_prices = pd.Series(np.random.normal(40000, 10000, len(gold_prices)), 
                                    index=gold_prices.index, name='Bitcoin')
            

            
            # Lấy common index từ gold_prices
            common_index = gold_prices.index
            
            # Reindex tất cả series về cùng index và đảm bảo chúng là Series
            def safe_reindex(series, new_index):
                """Safely reindex a series, ensuring it remains 1-dimensional"""
                if isinstance(series, pd.DataFrame):
                    series = series.squeeze()
                return series.reindex(new_index, method='ffill')
            
            usd_index = safe_reindex(usd_index, common_index)
            oil_prices = safe_reindex(oil_prices, common_index)
            spy_prices = safe_reindex(spy_prices, common_index)
            treasury_yield = safe_reindex(treasury_yield, common_index)
            vix_prices = safe_reindex(vix_prices, common_index)
            btc_prices = safe_reindex(btc_prices, common_index)
            
            # Kiểm tra lại dimensions
            print(f"Gold prices shape: {gold_prices.shape if hasattr(gold_prices, 'shape') else 'Series'}")
            print(f"USD index shape: {usd_index.shape if hasattr(usd_index, 'shape') else 'Series'}")
            
            # Tạo DataFrame - Sử dụng dict thay vì truyền trực tiếp các Series
            data_dict = {
                'Gold_Price': gold_prices.values,  # Lấy values để đảm bảo là array 1D
                'USD_Index': usd_index.values,
                'Oil_Price': oil_prices.values,
                'SP500': spy_prices.values,
                'Treasury_Yield': treasury_yield.values,
                'VIX': vix_prices.values,
                'Bitcoin': btc_prices.values
            }
            
            self.data = pd.DataFrame(data_dict, index=common_index)
            
            
            
            # Loại bỏ dữ liệu thiếu
            self.data = self.data.dropna(subset=['Gold_Price'])

            
            if len(self.data) == 0:
                raise Exception("Không có dữ liệu sau khi xử lý")
            
            print(f"Thu thập được {len(self.data)} điểm dữ liệu từ {start_date} đến {end_date}")
            print(f"Shape của dữ liệu cuối cùng: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Lỗi khi thu thập dữ liệu: {str(e)}")
            # Tạo dữ liệu mẫu nếu không thể thu thập dữ liệu
            return self.create_sample_data(start_date, end_date)
    
    def create_sample_data(self, start_date, end_date):
        """Tạo dữ liệu mẫu khi không thể thu thập dữ liệu thực"""
        print("Tạo dữ liệu mẫu...")
        
        # Tạo date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Loại bỏ weekends
        date_range = date_range[date_range.weekday < 5]
        
        n_days = len(date_range)
        
        # Tạo dữ liệu giả với xu hướng
        np.random.seed(42)
        gold_trend = np.linspace(160, 200, n_days) + np.random.normal(0, 5, n_days)
        
        self.data = pd.DataFrame({
            'Gold_Price': gold_trend,
            'USD_Index': np.random.normal(100, 5, size=n_days),
            'Oil_Price': np.random.normal(70, 10, size=n_days),
            'SP500': np.random.normal(400, 50, size=n_days),
            'Treasury_Yield': np.random.normal(3, 0.5, size=n_days),
            'VIX': np.random.normal(20, 5, size=n_days),
            'Bitcoin': np.random.normal(40000, 10000, size=n_days)
        }, index=date_range)
        
        print(f"Tạo được {len(self.data)} điểm dữ liệu mẫu")
        return self.data
    
    def preprocess_data(self):
        print("Đang tiền xử lý dữ liệu...")

        if self.data.empty:
            raise Exception("Không có dữ liệu để xử lý")

        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        for col in self.data.columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

        self.create_technical_features()
        self.create_lag_features()
        self.create_time_features()

        self.data = self.data[[col for col in self.data.columns if not self.data[col].isna().all()]]

        essential_features = ['Gold_Price'] + [col for col in self.data.columns if 'Gold_Return' in col or 'Gold_Volatility' in col or 'lag' in col]
        self.data = self.data.dropna(subset=essential_features)


        if self.data.tail(1).isna().any(axis=1).values[0]:
            print("Dòng cuối cùng vẫn còn NaN -> đang cố gắng fill dữ liệu...")
            self.data.iloc[-1] = self.data.iloc[-1].fillna(method='ffill').fillna(method='bfill')
            if self.data.tail(1).isna().any(axis=1).values[0]:
                print("Vẫn còn NaN sau khi fill -> loại bỏ dòng cuối")
                self.data = self.data.iloc[:-1]

        self.prepare_features()

        if len(self.data) == 0:
            raise Exception("Không còn dữ liệu sau khi tiền xử lý")

        print(f"Sau tiền xử lý: {len(self.data)} điểm dữ liệu, {len(self.data.columns)} đặc trưng")
        print("Ngày cuối cùng trước khi dropna:", self.data.index.max())
        print("Số lượng dòng:", len(self.data))

        
    def create_technical_features(self):
        """Tạo các đặc trưng kỹ thuật"""
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(self.data) >= window:
                self.data[f'Gold_MA_{window}'] = self.data['Gold_Price'].rolling(window=window).mean()
                self.data[f'Gold_MA_ratio_{window}'] = self.data['Gold_Price'] / self.data[f'Gold_MA_{window}']
        
        # Volatility
        if len(self.data) >= 10:
            self.data['Gold_Volatility_10'] = self.data['Gold_Price'].rolling(window=10).std()
        if len(self.data) >= 20:
            self.data['Gold_Volatility_20'] = self.data['Gold_Price'].rolling(window=20).std()
        
        # Returns
        self.data['Gold_Return_1d'] = self.data['Gold_Price'].pct_change(1)
        if len(self.data) >= 5:
            self.data['Gold_Return_5d'] = self.data['Gold_Price'].pct_change(5)
        if len(self.data) >= 10:
            self.data['Gold_Return_10d'] = self.data['Gold_Price'].pct_change(10)
        
        # RSI (Relative Strength Index)
        if len(self.data) >= 14:
            delta = self.data['Gold_Price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['Gold_RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if len(self.data) >= 20:
            ma20 = self.data['Gold_Price'].rolling(window=20).mean()
            std20 = self.data['Gold_Price'].rolling(window=20).std()
            self.data['BB_upper'] = ma20 + (std20 * 2)
            self.data['BB_lower'] = ma20 - (std20 * 2)
            self.data['BB_width'] = self.data['BB_upper'] - self.data['BB_lower']
            self.data['BB_position'] = (self.data['Gold_Price'] - self.data['BB_lower']) / self.data['BB_width']
        
        # Ratios với các asset khác
        self.data['Gold_Oil_Ratio'] = self.data['Gold_Price'] / self.data['Oil_Price']
        self.data['Gold_SP500_Ratio'] = self.data['Gold_Price'] / self.data['SP500']
        self.data['Gold_Bitcoin_Ratio'] = self.data['Gold_Price'] / self.data['Bitcoin']
        
    def create_lag_features(self):
        """Tạo đặc trưng lag"""
        lag_periods = [1, 2, 3, 5, 10]
        
        for col in ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'VIX']:
            for lag in lag_periods:
                if len(self.data) > lag:
                    self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
    
    def create_time_features(self):
        """Tạo đặc trưng thời gian"""
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        self.data['day_of_year'] = self.data.index.dayofyear
        
        # Cyclical encoding
        self.data['day_of_week_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_of_week_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
    
    def prepare_features(self):
        """Chuẩn bị features cho training"""
        # Loại bỏ target và các cột không cần thiết
        exclude_cols = ['Gold_Price'] + [col for col in self.data.columns if 'Gold_MA_' in col and 'ratio' not in col]
        self.features = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"Số lượng features: {len(self.features)}")
        return self.features
    
    def train_models(self, test_size=0.2):
        """Huấn luyện các mô hình"""
        print("Đang chuẩn bị dữ liệu training...")

        if len(self.data) < 50:
            raise Exception("Không đủ dữ liệu để huấn luyện (cần ít nhất 50 điểm)")

        # Chuẩn bị features
        self.prepare_features()

        X = self.data[self.features]
        y = self.data[self.target].squeeze()

        # Chia train/test theo thời gian
        split_idx = int(len(self.data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # ❗ Loại bỏ các dòng chứa NaN trong X_train/y_train
        train_df = pd.concat([X_train, y_train], axis=1).dropna()
        X_train = train_df[self.features]
        y_train = train_df[self.target]

        test_df = pd.concat([X_test, y_test], axis=1).dropna()
        X_test = test_df[self.features]
        y_test = test_df[self.target]

        # Chuẩn hóa dữ liệu
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Đang huấn luyện các mô hình...")
        
        # 1. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=50,  # Giảm để tăng tốc
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,  # Giảm để tăng tốc
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=50,  # Giảm để tăng tốc
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # 4. Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Lưu models
        self.models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'GradientBoosting': gb_model,
            'LinearRegression': lr_model
        }
        
        # Đánh giá models
        self.evaluate_models(X_test, y_test, X_test_scaled)
        
        return X_test, y_test
    
    def evaluate_models(self, X_test, y_test, X_test_scaled):
        """Đánh giá các mô hình"""
        print("\n=== ĐÁNH GIÁ CÁC MÔ HÌNH ===")
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'LinearRegression':
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        
        # Tìm model tốt nhất
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"\nMô hình tốt nhất: {best_model}")
        
        return results
    
    def predict_next_price(self, days=1):
        if not self.models:
            raise ValueError("Chưa có model nào được train!")

        if len(self.data) == 0:
            raise ValueError("Không có dữ liệu để dự đoán!")

        # ✅ Chỉ lấy dòng gần nhất KHÔNG chứa NaN
        latest_data = self.data[self.features].dropna().iloc[-1:].copy()

        if latest_data.isna().any().any():
            latest_data = latest_data.fillna(method='ffill').fillna(method='bfill')
            if latest_data.isna().any().any():
                print("Cảnh báo: Vẫn còn NaN trong latest_data => fallback")
                raise ValueError("Không thể dự đoán vì dữ liệu đầu vào còn thiếu")

        predictions = {}

        try:
            for name, model in self.models.items():
                if name == 'LinearRegression':
                    latest_scaled = self.scaler.transform(latest_data)
                    pred = model.predict(latest_scaled)[0]
                else:
                    pred = model.predict(latest_data)[0]

                predictions[name] = float(pred)

            weights = {'RandomForest': 0.3, 'XGBoost': 0.3, 'GradientBoosting': 0.25, 'LinearRegression': 0.15}
            ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)
            predictions['Ensemble'] = float(ensemble_pred)

        except Exception as e:
            print(f"Lỗi khi dự đoán: {str(e)}")
            current_price = float(self.data['Gold_Price'].iloc[-1])
            predictions = {
                'RandomForest': current_price * 1.01,
                'XGBoost': current_price * 1.005,
                'GradientBoosting': current_price * 1.008,
                'LinearRegression': current_price * 1.002,
                'Ensemble': current_price * 1.006
            }

        return predictions

    
    def save_model(self, filepath='gold_prediction_model.pkl'):
        """Lưu model"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'features': self.features,
            'data_sample': self.data.tail(100) if len(self.data) > 0 else pd.DataFrame()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model đã được lưu tại: {filepath}")
    
    def load_model(self, filepath='gold_prediction_model.pkl'):
        """Load model mà KHÔNG ghi đè self.data bằng data cũ"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.features = model_data['features']

        print(f"Model đã được load từ: {filepath}")


    def predict_next_7days(self):
        """Dự đoán giá vàng trong 7 ngày tới"""
        if not self.models:
            raise ValueError("Chưa có model nào được train!")
        
        if len(self.data) == 0:
            raise ValueError("Không có dữ liệu để dự đoán!")
        
        print("Đang dự đoán giá vàng trong 7 ngày tới...")
        
        # Tạo dataframe để lưu kết quả
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1), 
            periods=7, 
            freq='D'
        )
        
        # Loại bỏ weekend (chỉ lấy ngày trong tuần)
        future_dates = future_dates[future_dates.weekday < 5][:7]  # Đảm bảo có đủ 7 ngày làm việc
        
        predictions_7days = {}
        
        # Khởi tạo kết quả cho từng model
        for model_name in self.models.keys():
            predictions_7days[model_name] = []
        predictions_7days['Ensemble'] = []
        
        # Copy dữ liệu để mô phỏng
        temp_data = self.data.copy()
        
        try:
            for i, future_date in enumerate(future_dates):
                print(f"Dự đoán ngày {i+1}/7: {future_date.strftime('%Y-%m-%d')}")
                
                # Dự đoán cho ngày tiếp theo
                day_predictions = self.predict_single_day(temp_data)
                
                # Lưu kết quả
                for model_name, pred_value in day_predictions.items():
                    predictions_7days[model_name].append(float(pred_value))
                
                # Cập nhật temp_data với giá dự đoán (sử dụng ensemble prediction)
                predicted_price = day_predictions['Ensemble']
                
                # Tạo row mới cho ngày tương lai
                new_row = self.create_future_row(temp_data, future_date, predicted_price)
                
                # Thêm vào temp_data
                temp_data = pd.concat([temp_data, pd.DataFrame([new_row], index=[future_date])])
                
                # Cập nhật lại features cho ngày mới
                self.data = temp_data.copy()
                self.create_technical_features()
                self.create_lag_features()
                self.create_time_features()
                self.prepare_features()

        
        except Exception as e:
            print(f"Lỗi khi dự đoán 7 ngày: {str(e)}")
            # Fallback: dự đoán đơn giản dựa trên trend
            return self.simple_7day_prediction()
        
        # Tạo kết quả cuối cùng
        result = {
            'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'predictions': predictions_7days,
            'current_price': float(self.data['Gold_Price'].iloc[-1]),
            'price_change_total': {},
            'price_change_percent': {}
        }
        
        # Tính toán thay đổi giá
        current_price = float(self.data['Gold_Price'].iloc[-1])
        for model_name, preds in predictions_7days.items():
            if preds:  # Kiểm tra list không rỗng
                final_price = preds[-1]
                result['price_change_total'][model_name] = final_price - current_price
                result['price_change_percent'][model_name] = ((final_price - current_price) / current_price) * 100
        
        return result

    def predict_single_day(self, data):
        if len(data) == 0:
            raise ValueError("Không có dữ liệu để dự đoán!")

        latest_data = data[self.features].dropna().iloc[-1:].copy()

        if latest_data.isna().any().any():
            latest_data = latest_data.fillna(method='ffill').fillna(method='bfill')
            if latest_data.isna().any().any():
                print("Cảnh báo: Vẫn còn NaN sau fill trong latest_data => fallback.")
                raise ValueError("Không thể dự đoán vì dữ liệu đầu vào còn thiếu")

        predictions = {}

        for name, model in self.models.items():
            try:
                if name == 'LinearRegression':
                    latest_scaled = self.scaler.transform(latest_data)
                    pred = model.predict(latest_scaled)[0]
                else:
                    pred = model.predict(latest_data)[0]
                predictions[name] = float(pred)
            except Exception as e:
                print(f"Lỗi khi dự đoán với model {name}: {str(e)}")
                current_price = float(data['Gold_Price'].iloc[-1])
                predictions[name] = current_price * 1.001

        if predictions:
            weights = {'RandomForest': 0.3, 'XGBoost': 0.3, 'GradientBoosting': 0.25, 'LinearRegression': 0.15}
            ensemble_pred = sum(predictions.get(name, 0) * weight for name, weight in weights.items() if name in predictions)
            predictions['Ensemble'] = float(ensemble_pred)

        return predictions


    def create_future_row(self, data, future_date, predicted_gold_price):
        """Tạo dòng dữ liệu cho ngày tương lai"""
        # Lấy dữ liệu ngày gần nhất
        last_row = data.iloc[-1].copy()
        
        # Cập nhật giá vàng dự đoán
        new_row = last_row.to_dict()
        new_row['Gold_Price'] = predicted_gold_price
        
        # Giả lập các biến khác (có thể cải thiện bằng cách dự đoán riêng)
        # Đây là phương pháp đơn giản - giả sử các biến khác thay đổi nhẹ
        for col in ['USD_Index', 'Oil_Price', 'SP500', 'Treasury_Yield', 'VIX', 'Bitcoin']:
            if col in new_row:
                # Thêm noise nhỏ
                change_factor = np.random.normal(1, 0.01)  # Thay đổi ±1%
                new_row[col] = new_row[col] * change_factor
        
        # Cập nhật time features
        new_row['day_of_week'] = future_date.weekday()
        new_row['month'] = future_date.month
        new_row['quarter'] = future_date.quarter
        new_row['day_of_year'] = future_date.dayofyear
        
        # Cyclical encoding
        new_row['day_of_week_sin'] = np.sin(2 * np.pi * new_row['day_of_week'] / 7)
        new_row['day_of_week_cos'] = np.cos(2 * np.pi * new_row['day_of_week'] / 7)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        
        return new_row

    def update_features_for_prediction(self, data):
        """Cập nhật các features sau khi thêm dữ liệu mới"""
        # Chỉ cập nhật lag features và technical indicators cho dòng cuối
        if len(data) < 2:
            return data
        
        last_idx = len(data) - 1
        
        # Cập nhật lag features
        lag_periods = [1, 2, 3, 5, 10]
        for col in ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'VIX']:
            for lag in lag_periods:
                col_name = f'{col}_lag_{lag}'
                if col_name in data.columns and last_idx >= lag:
                    data.iloc[last_idx, data.columns.get_loc(col_name)] = data.iloc[last_idx - lag][col]
        
        # Cập nhật returns
        if 'Gold_Return_1d' in data.columns and last_idx >= 1:
            prev_price = data.iloc[last_idx - 1]['Gold_Price']
            curr_price = data.iloc[last_idx]['Gold_Price']
            data.iloc[last_idx, data.columns.get_loc('Gold_Return_1d')] = (curr_price - prev_price) / prev_price
        
        # Cập nhật moving averages (đơn giản)
        for window in [5, 10, 20, 50]:
            ma_col = f'Gold_MA_{window}'
            ratio_col = f'Gold_MA_ratio_{window}'
            
            if ma_col in data.columns and last_idx >= window - 1:
                ma_value = data.iloc[last_idx - window + 1:last_idx + 1]['Gold_Price'].mean()
                data.iloc[last_idx, data.columns.get_loc(ma_col)] = ma_value
                
                if ratio_col in data.columns:
                    data.iloc[last_idx, data.columns.get_loc(ratio_col)] = data.iloc[last_idx]['Gold_Price'] / ma_value
        
        return data

    def simple_7day_prediction(self):
        """Dự đoán đơn giản khi có lỗi"""
        current_price = float(self.data['Gold_Price'].iloc[-1])
        
        # Tính trend đơn giản từ 10 ngày gần nhất
        if len(self.data) >= 10:
            recent_prices = self.data['Gold_Price'].tail(10)
            daily_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / 9
        else:
            daily_change = 0
        
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1), 
            periods=7, 
            freq='D'
        )
        future_dates = future_dates[future_dates.weekday < 5][:7]
        
        simple_predictions = []
        for i in range(len(future_dates)):
            pred_price = current_price + (daily_change * (i + 1))
            # Thêm một chút volatility
            pred_price += np.random.normal(0, current_price * 0.005)
            simple_predictions.append(pred_price)
        
        result = {
            'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'predictions': {
                'Simple_Trend': simple_predictions,
                'Ensemble': simple_predictions
            },
            'current_price': current_price,
            'price_change_total': {
                'Simple_Trend': simple_predictions[-1] - current_price,
                'Ensemble': simple_predictions[-1] - current_price
            },
            'price_change_percent': {
                'Simple_Trend': ((simple_predictions[-1] - current_price) / current_price) * 100,
                'Ensemble': ((simple_predictions[-1] - current_price) / current_price) * 100
            }
        }
        
        return result


# Flask Web Application
app = Flask(__name__)
predictor = GoldPredictionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Thu thập và xử lý dữ liệu
        predictor.collect_data()
        predictor.preprocess_data()
        
        # Huấn luyện model
        predictor.train_models()
        
        # Lưu model
        predictor.save_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Model đã được huấn luyện thành công!',
            'data_points': len(predictor.data),
            'features': len(predictor.features)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Load model neu chua co
        if not predictor.models:
            if os.path.exists('gold_prediction_model.pkl'):
                predictor.load_model()
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Chưa có model nào được huấn luyện. Vui lòng train model trước!'
                })

        # Luon thu thap va xu ly lai du lieu moi
        print("Dang cap nhat du lieu moi nhat truoc khi du doan...")
        predictor.collect_data()
        predictor.preprocess_data()
        print("Ngay cuoi cung trong du lieu:", predictor.data.index.max())

        # Du doan
        predictions = predictor.predict_next_price()
        current_price = float(predictor.data['Gold_Price'].iloc[-1])

        return jsonify({
            'status': 'success',
            'current_price': current_price,
            'predictions': predictions,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/data_info')
def get_data_info():
    try:
        if predictor.data.empty:
            return jsonify({
                'status': 'error',
                'message': 'Chưa có dữ liệu'
            })
        
        # Thống kê cơ bản
        stats = {
            'total_records': len(predictor.data),
            'date_range': {
                'start': predictor.data.index.min().strftime('%Y-%m-%d'),
                'end': predictor.data.index.max().strftime('%Y-%m-%d')
            },
            'current_price': float(predictor.data['Gold_Price'].iloc[-1]),
            'price_change_1d': float(predictor.data['Gold_Price'].iloc[-1] - predictor.data['Gold_Price'].iloc[-2]) if len(predictor.data) > 1 else 0,
            'features_count': len(predictor.features) if predictor.features else 0
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    
@app.route('/predict_7days', methods=['POST'])
def predict_7days():
    try:
        if not predictor.models:
            if os.path.exists('gold_prediction_model.pkl'):
                predictor.load_model()
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Chưa có model nào được huấn luyện. Vui lòng train model trước!'
                })

        print("Dang cap nhat du lieu moi nhat truoc khi du doan 7 ngay...")
        predictor.collect_data()
        predictor.preprocess_data()
        print("Ngay cuoi cung trong du lieu:", predictor.data.index.max())

        predictions_7days = predictor.predict_next_7days()

        return jsonify({
            'status': 'success',
            'predictions_7days': predictions_7days,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })



if __name__ == '__main__':
    # Tạo thư mục templates nếu chưa có
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Chạy ứng dụng
    print("hi")
    app.run(debug=True, host='0.0.0.0', port=5000)