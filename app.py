import streamlit as st
import pandas as pd
import joblib  # Cần thiết để tải mô hình .joblib

# ----- TẢI MÔ HÌNH ĐÃ ĐƯỢC HUẤN LUYỆN -----
# Chúng ta không huấn luyện lại mô hình trong app
# Thay vào đó, chúng ta tải mô hình 'rf_model_on_full_data' mà bạn đã lưu
# từ file 'mo_hinh_benh.joblib'

@st.cache_resource  # Streamlit sẽ lưu mô hình vào cache, giúp chạy nhanh hơn
def load_model():
    try:
        model = joblib.load('mo_hinh_benh_FINAL.joblib')
        return model
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'mo_hinh_benh_FINAL.joblib'.")
        st.error("Vui lòng đảm bảo file mô hình nằm cùng thư mục với app.py.")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

# Tải mô hình khi ứng dụng khởi động
model = load_model()

# Chỉ hiển thị giao diện nếu mô hình được tải thành công
if model is not None:
    # ----- BẮT ĐẦU GIAO DIỆN WEB STREAMLIT -----
    st.title('🩺 Ứng dụng Dự đoán Nguy cơ Đột quỵ')
    st.markdown('***Nhập các thông số của bạn để dự đoán:***')

    # ----- TẠO CÁC Ô NHẬP LIỆU (Thay thế cho new_row_data) -----
    # Các ô nhập liệu này sẽ lấy thông tin từ người dùng

    # Chúng ta dùng 4 features bạn đã chọn: 'age', 'bmi', 'hypertension', 'heart_disease'
    
    # Tạo 2 cột cho gọn gàng
    col1, col2 = st.columns(2)

    with col1:
        # Giống 'age': [80]
        age = st.number_input('Tuổi (Age)', min_value=1.0, max_value=120.0, value=80.0, step=1.0)
        
        # Giống 'hypertension': [1] (1=Có, 0=Không)
        hypertension_text = st.selectbox('Tiền sử tăng huyết áp?', ('Không', 'Có'), index=1)
        hypertension = 1 if hypertension_text == 'Có' else 0

    with col2:
        # Giống 'bmi': [50]
        bmi = st.number_input('Chỉ số BMI', min_value=10.0, max_value=100.0, value=50.0, step=0.1)

        # Giống 'heart_disease': [1] (1=Có, 0=Không)
        heart_disease_text = st.selectbox('Tiền sử bệnh tim?', ('Không', 'Có'), index=1)
        heart_disease = 1 if heart_disease_text == 'Có' else 0

    st.markdown('---') # Dòng kẻ ngang

    # ----- NÚT DỰ ĐOÁN -----
    # Khi người dùng nhấn nút này, chúng ta sẽ chạy phần dự đoán
    if st.button('Dự đoán Nguy cơ'):
        
        # 1. Tạo DataFrame (giống 'single_test_row' của bạn)
        # Lấy dữ liệu từ các ô nhập liệu ở trên
        new_row_data = {
            'age': [age],
            'bmi': [bmi],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease]
        }
        single_test_row = pd.DataFrame(new_row_data)

        # 2. Dự đoán xác suất (giống code của bạn)
        # Sử dụng mô hình đã được tải (chính là 'rf_model_on_full_data' của bạn)
        probability_predictions = model.predict_proba(single_test_row)
        
        # 3. Lấy xác suất
        probability_of_disease = probability_predictions[0][1]
        percentage_of_disease = probability_of_disease * 100

        # 4. Hiển thị kết quả (thay cho lệnh 'print')
        st.subheader('Kết quả Dự đoán:')
        st.success(f"Xác suất mắc bệnh là: {probability_of_disease:.4f}")
        st.success(f"Phần trăm mắc bệnh dự đoán là: {percentage_of_disease:.2f}%")

        if percentage_of_disease > 20:
             st.warning("Nguy cơ cao. Vui lòng tham khảo ý kiến bác sĩ.", icon="⚠️")
        elif percentage_of_disease > 5:
             st.info("Nguy cơ trung bình. Cần duy trì lối sống lành mạnh.", icon="✨")
        else:
             st.success("Nguy cơ thấp.", icon="✅")
