
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 2. THÊM HÀM VẼ BIỂU ĐỒ NÀY VÀO (Đặt trước hàm load_model) ---
def create_gauge_chart(risk_score):
    """
    Hàm vẽ biểu đồ đồng hồ đo nguy cơ
    """
    # Xác định màu sắc dựa trên mức độ nguy hiểm
    if risk_score < 20:
        bar_color = "green"
    elif risk_score < 50:
        bar_color = "orange"
    else:
        bar_color = "red"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Nguy cơ Đột quỵ (%)", 'font': {'size': 24}},
        
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color}, # Màu của thanh hiển thị sẽ đổi theo mức độ
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                # Phân vùng màu nền: Xanh (An toàn) -> Vàng (Cảnh báo) -> Đỏ (Nguy hiểm)
                {'range': [0, 20], 'color': "#ccffcc"}, # Xanh nhạt
                {'range': [20, 50], 'color': "#ffebcc"}, # Cam nhạt
                {'range': [50, 100], 'color': "#ffcccc"}], # Đỏ nhạt
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    # Chỉnh kích thước biểu đồ cho gọn
    fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Arial"})
    return fig




def generate_recommendations(age, bmi, avg_glucose_level, hypertension_display, heart_display, smoke_display):
    """
    Tạo danh sách các khuyến nghị dựa trên các yếu tố nguy cơ của người dùng.
    """
    recommendations = []
    
    # --- 1. KIỂM TRA THÓI QUEN HÚT THUỐC ---
    # Nếu đang hút hoặc không rõ tình trạng, cần dừng/xác minh
    if smoke_display in ["Đang hút thuốc", "Không rõ"]:
        recommendations.append("🚭 Nguy cơ đột quỵ tăng đáng kể khi hút thuốc. Ưu tiên hàng đầu là bỏ thuốc hoặc xác minh tình trạng hút thuốc.")

    # --- 2. KIỂM TRA BMI (Cân nặng) ---
    if bmi >= 30.0:
        recommendations.append("🍏 BMI ở mức Béo phì (>30). Cần tham khảo chuyên gia dinh dưỡng để thiết lập chế độ giảm cân an toàn.")
    elif bmi >= 25.0 and bmi < 30.0:
        recommendations.append("🏃 BMI ở mức Thừa cân. Tăng cường hoạt động thể chất tối thiểu 30 phút mỗi ngày và theo dõi chế độ ăn.")

    # --- 3. KIỂM TRA ĐƯỜNG HUYẾT ---
    # Ngưỡng trung bình/tiền tiểu đường thường là > 100-125 mg/dL
    if avg_glucose_level >= 100.0:
        recommendations.append("🩸 Đường huyết trung bình cao. Cần tầm soát nguy cơ tiểu đường và hạn chế thực phẩm nhiều đường.")

    # --- 4. KIỂM TRA BỆNH NỀN ---
    if hypertension_display == "Có":
        recommendations.append("🩺 Có tiền sử Cao huyết áp. Cần kiểm tra huyết áp thường xuyên và tuân thủ chặt chẽ phác đồ điều trị của bác sĩ.")
        
    if heart_display == "Có":
        recommendations.append("❤️ Có bệnh Tim mạch. Tránh các hoạt động gắng sức không cần thiết và tham khảo ý kiến bác sĩ chuyên khoa tim mạch.")
    
    # --- 5. LỜI KHUYÊN CHUNG (Tuổi) ---
    if age >= 60:
        recommendations.append("🛌 Do tuổi cao (>60), nên duy trì ngủ đủ giấc (7-9 giờ/ngày) và giữ tinh thần thoải mái.")
        
    return recommendations

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
    age = st.number_input("Tuổi:", min_value=1, max_value=120, value=None,placeholder="Nhập tuổi")
    
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
    
    # --- 1. NHẬP CHIỀU CAO & CÂN NẶNG ---
    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Chiều cao (cm)", min_value=50.0, max_value=250.0, value=None, placeholder="Nhập chiều cao")
    with c2:
        weight = st.number_input("Cân nặng (kg)", min_value=20.0, max_value=300.0, value=None, placeholder="Nhập cân nặng")

    # --- 2. TÍNH BMI ---
    if height is not None and weight is not None:
        bmi = weight / ((height / 100) ** 2)
        st.write(f"Chỉ số BMI của bạn: **{bmi:.2f}**")
    else:
        bmi = None
    
    # --- 3. ĐƯỜNG HUYẾT ---
    avg_glucose_level = st.number_input("Đường huyết trung bình (mg/dL):", value=None, placeholder="Nhập đường huyết")
    
    # --- 4. BỆNH NỀN ---
    hypertension_display = st.radio("Có bị Cao huyết áp không?", ["Không", "Có"], horizontal=True)
    hyper_map = {"Không": 0, "Có": 1}
    
    heart_display = st.radio("Có bệnh Tim mạch không?", ["Không", "Có"], horizontal=True)
    heart_map = {"Không": 0, "Có": 1}
    # ----------------------------------------

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
    
    # --- KIỂM TRA DỮ LIỆU ĐẦU VÀO (VALIDATION) ---
    # Nếu thiếu 1 trong các chỉ số quan trọng thì báo lỗi và DỪNG LẠI
    if age is None:
        st.error("Vui lòng nhập Tuổi!")
    elif bmi is None:
        st.error("Vui lòng nhập Chiều cao và Cân nặng!")
    elif avg_glucose_level is None:
        st.error("Vui lòng nhập chỉ số Đường huyết!")
    else:
        # Khi đã nhập đủ hết thì mới chạy đoạn code bên dưới
        
        # Tạo dữ liệu đầu vào
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
            stroke_risk = prediction_prob[0][1]
            risk_percent = stroke_risk * 100
            
            # Lấy khuyến nghị
            recommendations = generate_recommendations(
                age, bmi, avg_glucose_level, hypertension_display, heart_display, smoke_display
            )
            
           # --- HIỂN THỊ KẾT QUẢ VÀ HÀNH ĐỘNG ---
            st.header("📋 KẾT QUẢ DỰ BÁO VÀ HÀNH ĐỘNG")
            
            # Chia cột: Cột 1 hiện biểu đồ, Cột 2 hiện lời cảnh báo
            col_chart, col_text = st.columns([1, 1]) 
            
            with col_chart:
                # Gọi hàm vẽ biểu đồ vừa viết ở Bước 2
                fig_gauge = create_gauge_chart(risk_percent)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_text:
                st.subheader("Đánh giá chi tiết:")
                if risk_percent > 50:
                    st.error(f"🚨 **NGUY CƠ RẤT CAO ({risk_percent:.1f}%)**")
                    st.write("Bạn nằm trong nhóm báo động đỏ. Các chỉ số cho thấy khả năng đột quỵ rất lớn.")
                    st.write("**Hành động:** Đi khám bác sĩ ngay lập tức!")
                elif risk_percent > 20:
                    st.warning(f"⚠️ **NGUY CƠ CAO ({risk_percent:.1f}%)**")
                    st.write("Bạn có nguy cơ cao hơn người bình thường. Cần điều chỉnh lối sống ngay.")
                else:
                    st.success(f"✅ **AN TOÀN ({risk_percent:.1f}%)**")
                    st.write("Các chỉ số của bạn đang ở mức tốt. Hãy tiếp tục duy trì.")

            # HIỂN THỊ KHUYẾN NGHỊ
            st.write("---")
            if recommendations:
                st.subheader("🎯 Khuyến nghị Lối sống & Sàng lọc")
                st.info("Hãy ưu tiên các hành động sau:")
                for rec in recommendations:
                    st.markdown(f"* {rec}")
            else:
                st.success("Tuyệt vời! Các chỉ số cơ bản của bạn đều tốt.")
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi dự báo: {e}")

