GIAI ĐOẠN 3: Cài đặt môi trường trên Raspberry Pi
Sau khi code đã nằm trên Pi, bạn mở Terminal của Pi lên và gõ lệnh để cài các thư viện cần thiết (theo đúng file README của dự án):

Cài các thư viện hệ thống: sudo apt install libportaudio2
Cài trình quản lý gói uv: curl -LsSf https://astral.sh/uv/install.sh | sh
Tải các thư viện Python: uv sync
Tải các file Model AI nặng về Pi: uv run glados download