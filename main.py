import os
import cv2
import numpy as np
import tensorflow as tf
import faiss
import gradio as gr
import tempfile
from PIL import Image

# Cấu hình
WIDTH, HEIGHT = 128, 128  # Kích thước ảnh đầu vào
VECTOR_DIMENSION = 8  # Kích thước vector đặc trưng
MODEL_PATH = "models/mobilenet_model_1286420.h5"
DATASET_PATH = "selected_images"
INDEX_PATH = "feature_database/faiss_index"

# Tải mô hình MobileNet đã được huấn luyện
mobilenet_model = tf.keras.models.load_model(MODEL_PATH)

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return None
    h, w = image.shape[:2]  # Lấy kích thước ảnh
    aspect_ratio = WIDTH / HEIGHT  # Tính tỷ lệ khung hình
    # Thay đổi kích thước ảnh theo tỷ lệ
    if w / h > aspect_ratio:
        new_w = WIDTH
        new_h = int(WIDTH * h / w)
    else:
        new_h = HEIGHT
        new_w = int(HEIGHT * w / h)
    # Thay đổi kích thước ảnh
    resized_image = cv2.resize(image, (new_w, new_h))
    delta_w = WIDTH - new_w
    delta_h = HEIGHT - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = image.astype("float32") / 255.0  # Chuyển đổi kiểu dữ liệu và chuẩn hóa
    return np.expand_dims(image, axis=0)   # Thêm chiều cho ảnh

# Hàm trích xuất vector đặc trưng
def extract_feature_vector(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # In ra số chiều trước khi trích đặc trưng
    print(f"Số chiều trước khi trích đặc trưng: {processed_image.shape}")
    feature_vector = mobilenet_model.predict(processed_image).flatten()
    # In ra độ dài vector và số chiều sau khi trích đặc trưng
    print(f"Độ dài vector sau khi trích đặc trưng: {feature_vector.shape}")
    
    return feature_vector
# Tạo cơ sở dữ liệu vector từ 'selected_images'
def create_feature_database():
    labels, index = [], faiss.IndexFlatL2(VECTOR_DIMENSION)
    for label in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(class_path):
            continue
        for image_path in os.listdir(class_path):
            full_image_path = os.path.join(class_path, image_path)
            feature_vector = extract_feature_vector(full_image_path) # Trích đặc trưn
            if feature_vector is not None:
                labels.append(label)  # Thêm nhãn vào danh sách
                index.add(np.array([feature_vector]))  # Thêm vector vào chỉ mục FAISS
    faiss.write_index(index, INDEX_PATH)
    with open("labels.txt", "w") as f:
        f.write("\n".join(labels))
    return labels

# Load danh sách labels từ file
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print("File labels.txt không tồn tại. Vui lòng chạy create_feature_database() trước.")
        return []  # Trả về danh sách rỗng nếu file không tồn tại

# Load chỉ mục FAISS
def load_faiss_index():
    try:
        return faiss.read_index(INDEX_PATH)
    except Exception as e:
        print(f"Lỗi khi tải FAISS index: {e}")
        return None

# Hàm ánh xạ nhãn tiếng Anh sang tên tiếng Việt
def map_label_to_vietnamese(label):
    label_mapping = {
        "CaiLuong": "Cải lương",
        "Cheo": "Chèo",
        "CaTru": "Ca trù",
        "HatChauVan": "Hát chầu văn",
        "HatXoan": "Hát xoan",
        "RoiNuoc": "Múa rối nước",
        "Tuong": "Tuồng",
        "Xam": "Xẩm"
    }
    return label_mapping.get(label, label)  # Nếu không tìm thấy nhãn, trả về chính nhãn đó

# Hàm query_image
def query_image(image_path, faiss_index, labels):
    query_vector = extract_feature_vector(image_path)  # Trích vector từ ảnh
    if query_vector is None:
        return "Không thể trích xuất vector từ hình ảnh."
    
    # Truy vấn FAISS với k=3 để lấy 3 loại hình có giá trị cao nhất
    distances, closest_indices = faiss_index.search(np.array([query_vector]), k=3)
    top_labels_with_scores = [(labels[i], 1 - distances[0][j]) for j, i in enumerate(closest_indices[0])]
    
    # Kiểm tra ngưỡng 0.7 cho loại hình có xác suất cao nhất
    if top_labels_with_scores[0][1] >= 0.7:
        # Nhãn xác suất cao nhất được khẳng định (chỉ trả về 1 nhãn), truy vấn đến LLM để mô tả
        label = top_labels_with_scores[0][0]
        vietnamese_label = map_label_to_vietnamese(label)  # Ánh xạ nhãn sang tiếng Việt
        description = generate_description(vietnamese_label)  # Truy vấn LLM với nhãn tiếng Việt
        result = f"Loại hình nghệ thuật: {vietnamese_label} ({top_labels_with_scores[0][1]:.2f})\n\nMô tả:\n{description}"
    else:
        # Nếu xác suất thấp hơn 0.7, không truy vấn LLM và hiển thị 3 nhãn có khả năng nhất
        result = "Dựa vào hình của bạn, chưa thể khẳng định được loại hình cụ thể nào. Dưới đây là 3 loại hình có khả năng phù hợp với ảnh bạn đưa ra:\n"
        result += "\n".join([f"{map_label_to_vietnamese(label)} ({score:.2f})" for label, score in top_labels_with_scores])
    
    return result

# Load Gemini-1.5-Flash
import google.generativeai as genai

# Cấu hình API cho mô hình LLM
genai.configure(api_key="AIzaSyCMeMuwnx05S5P59O59dUp5Sb9F49Al274")
model = genai.GenerativeModel("gemini-1.5-flash")

# LLM để tạo mô tả loại hình nghệ thuật
def generate_description(label):
    # Tạo prompt cho mô hình
    prompt = f"Viết một đoạn văn ngắn mô tả đặc trưng của loại hình nghệ thuật {label}."
    # Gọi mô hình để tạo mô tả
    response = model.generate_content(prompt)
    return response.text.strip()

# Giao diện Gradio
def predict_art_form(image):
    global faiss_index, labels
    if faiss_index is None or not labels:
        print("Lỗi: FAISS index hoặc labels chưa được tải.")
        return "Lỗi hệ thống: FAISS index hoặc labels không sẵn sàng."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Chuyển đổi ảnh nếu cần

    # Lưu ảnh tạm thời và kiểm tra lỗi
    try:
        temp_file_name = tempfile.mktemp(suffix=".jpg")
        image.save(temp_file_name)
    except Exception as e:
        print(f"Lỗi khi lưu ảnh tạm thời: {e}")
        return "Lỗi khi lưu ảnh tạm thời."

    # Truy vấn ảnh với các thay đổi mới
    result = query_image(temp_file_name, faiss_index, labels)
    
    os.remove(temp_file_name)  # Đảm bảo xóa file tạm thời
    return result


if __name__ == "__main__":
    # Tạo hoặc tải FAISS index và labels
    if os.path.exists(INDEX_PATH):
        print("Đang tải FAISS index và labels từ file...")
        # Tải chỉ mục và nhãn
        faiss_index, labels = load_faiss_index(), load_labels()
    else:
        print("Đang tạo cơ sở dữ liệu FAISS và labels...")
        # Tạo cơ sở dữ liệu
        labels = create_feature_database() 
        # Tải chỉ mục
        faiss_index = load_faiss_index()

    # image_path = "test_images/test_3.jpg"
    # # Gọi hàm trích xuất vector đặc trưng
    # feature_vector = extract_feature_vector(image_path)
    
    # if feature_vector is not None:
    #     print(f"Vector đặc trưng: {feature_vector}")
    # else:
    #     print("Không thể trích xuất vector đặc trưng.")

    # Khởi tạo Gradio
    iface = gr.Interface(fn=predict_art_form, inputs="image", outputs="text", title="Vietnamese Stage Art Recognition", allow_flagging="never")
    iface.launch()