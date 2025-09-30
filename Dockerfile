# 1. Chọn base image có Python
FROM python:3.13.7-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirement file và cài đặt package
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ project vào container
COPY . .

# 5. Expose port ứng dụng
EXPOSE 8000

# 6. Chạy FastAPI bằng uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]