
import streamlit as st
import pandas as pd
import joblib

# 1. Load Model

@st.cache_resource
def load_model():

    model = joblib.load('stroke_model.pkl') 
    return model

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy file model. Hãy kiểm tra lại tên file .pkl! Chi tiết: {e}")
    st.stop()

# 2. Giao diện Tiêu đề
st.title("🏥 Dự báo Nguy cơ Đột quỵ")
st.write("Nhập thông tin sức khỏe để hệ thống AI phân tích nguy cơ.")
st.write("---")

# 3. Form nhập liệu (Chia 2 cột)
col1, col2 = st.columns(2)

with col1:
    # --- GIỚI TÍNH ---
    st.subheader("Thông tin cá nhân")
    gender_display = st.selectbox("Giới tính:", ["Nam", "Nữ", "Khác"])
    # Từ điển quy đổi: Tiếng Việt -> Tiếng Anh (Model hiểu)
    gender_map = {"Nam": "Male", "Nữ": "Female", "Khác": "Other"}
    
    # --- TUỔI ---
    age = st.number_input("Tuổi:", min_value=1, max_value=120, value=60)
    
    # --- TÌNH TRẠNG HÔN NHÂN ---
    married_display = st.selectbox("Đã từng kết hôn chưa?", ["Rồi", "Chưa"])
    married_map = {"Rồi": "Yes", "Chưa": "No"}
    
    # --- CÔNG VIỆC ---
    work_display = st.selectbox("Loại hình công việc:", 
                                ["Tư nhân / Doanh nghiệp", "Tự kinh doanh", "Nhà nước", "Trẻ nhỏ", "Chưa đi làm"])
    work_map = {
        "Tư nhân / Doanh nghiệp": "Private",
        "Tự kinh doanh": "Self-employed",
        "Nhà nước": "Govt_job",
        "Trẻ nhỏ": "children",
        "Chưa đi làm": "Never_worked"
    }

    # --- NƠI Ở ---
    res_display = st.selectbox("Khu vực sinh sống:", ["Thành thị", "Nông thôn"])
    res_map = {"Thành thị": "Urban", "Nông thôn": "Rural"}

with col2:
    st.subheader("Chỉ số sức khỏe")
    
    # --- BMI ---
    bmi = st.number_input("Chỉ số BMI (Cân nặng/Chiều cao²):", value=22.5)
    
    # --- ĐƯỜNG HUYẾT ---
    avg_glucose_level = st.number_input("Đường huyết trung bình (mg/dL):", value=90.0)
    
    # --- BỆNH NỀN ---
    hypertension_display = st.radio("Có bị Cao huyết áp không?", ["Không", "Có"], horizontal=True)
    hyper_map = {"Không": 0, "Có": 1}
    
    heart_display = st.radio("Có bệnh Tim mạch không?", ["Không", "Có"], horizontal=True)
    heart_map = {"Không": 0, "Có": 1}
    
    # --- HÚT THUỐC ---
    smoke_display = st.selectbox("Tình trạng hút thuốc:", 
                                 ["Chưa bao giờ hút", "Đã bỏ thuốc", "Đang hút thuốc", "Không rõ"])
    smoke_map = {
        "Chưa bao giờ hút": "never smoked",
        "Đã bỏ thuốc": "formerly smoked",
        "Đang hút thuốc": "smokes",
        "Không rõ": "Unknown"
    }

# 4. Xử lý Dự đoán
st.write("---")
if st.button("🔍 PHÂN TÍCH NGAY", type="primary"):
    
    # Tạo dữ liệu đầu vào (Convert từ Tiếng Việt sang Tiếng Anh)
    input_data = {
        'gender': [gender_map[gender_display]],
        'age': [age],
        'hypertension': [hyper_map[hypertension_display]],
        'heart_disease': [heart_map[heart_display]],
        'ever_married': [married_map[married_display]],
        'work_type': [work_map[work_display]],
        'Residence_type': [res_map[res_display]],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoke_map[smoke_display]]
    }
    
    df_input = pd.DataFrame(input_data)
    
    try:
        # Dự đoán
        prediction_prob = pipeline.predict_proba(df_input)
        stroke_risk = prediction_prob[0][1] # Xác suất bị bệnh
        risk_percent = stroke_risk * 100
        
        # Hiển thị kết quả
        st.header("📋 KẾT QUẢ DỰ BÁO")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Tỷ lệ nguy cơ", value=f"{risk_percent:.1f}%")
        
        with col_res2:
            if risk_percent > 50:
                st.error("🚨 CẢNH BÁO: Nguy cơ RẤT CAO. Cần tham khảo ý kiến bác sĩ!")
            elif risk_percent > 20:
                st.warning("⚠️ CẢNH BÁO: Nguy cơ CAO. Cần tầm soát sức khỏe kỹ lưỡng.")
            else:
                st.success("✅ AN TOÀN: Nguy cơ thấp. Hãy tiếp tục duy trì lối sống lành mạnh.")
                
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")