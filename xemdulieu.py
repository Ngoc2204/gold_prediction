import pandas as pd
from gold_prediction_system import GoldPredictionSystem

# Khởi tạo hệ thống dự đoán
predictor = GoldPredictionSystem()

# Thu thập, xử lý và huấn luyện (có thể thay đổi thời gian nếu cần)
predictor.collect_data(start_date='2022-01-01')
predictor.preprocess_data()
predictor.train_models()

# Xuất dữ liệu đã xử lý (đã có các feature đầy đủ)
output_file = 'du_lieu_da_train.xlsx'
predictor.data.to_excel(output_file, index=True, engine='openpyxl')

print(f"✅ Đã xuất dữ liệu đã train ra file: {output_file}")
