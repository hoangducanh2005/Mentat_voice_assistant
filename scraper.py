import requests
from bs4 import BeautifulSoup
import re
import time
import sys
import urllib.parse

# Cấu hình stdout sử dụng UTF-8 để tránh lỗi hiển thị ký tự Unicode trên Windows
sys.stdout.reconfigure(encoding='utf-8')

def scrape_fandom_page(url, output_filename):
    # Trích xuất tên trang từ URL
    parts = url.split("/wiki/")
    if len(parts) < 2:
        print(f"❌ URL không hợp lệ: {url}")
        return
    
    page_title = urllib.parse.unquote(parts[1]).strip()
    print(f"Đang tải dữ liệu từ: {url} (Page: {page_title}) ...")
    
    # Sử dụng MediaWiki API để lấy HTML nội dung bài viết, tránh bị Cloudflare chặn 403
    api_url = "https://dune.fandom.com/api.php"
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "redirects": "true",
        "format": "json"
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if "error" in data:
            print(f"❌ Lỗi API từ Fandom: {data['error'].get('info')}")
            return
            
        html_content = data.get("parse", {}).get("text", {}).get("*", "")
        if not html_content:
            print("❌ Không tìm thấy nội dung bài viết.")
            return
        
        # Đưa HTML vào BeautifulSoup để xử lý
        soup = BeautifulSoup(html_content, 'html.parser')
        
        text_blocks = []
        
        # Chỉ lấy các thẻ chứa văn bản có ý nghĩa: Tiêu đề (h2, h3), đoạn văn (p) và danh sách (li)
        for tag in soup.find_all(['p', 'h2', 'h3', 'li']):
            text = tag.get_text().strip()
            
            if text:
                # Dùng Regex để xóa các dấu trích dẫn kiểu [1], [2], [note 1] thường có trên wiki
                text = re.sub(r'\[.*?\]', '', text)
                text_blocks.append(text)

        # Gộp tất cả lại thành một đoạn văn bản hoàn chỉnh, cách nhau bởi dấu xuống dòng
        final_text = '\n'.join(text_blocks)
        
        # Lưu vào file text
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(final_text)
            
        print(f"✅ Đã lưu thành công vào file: {output_filename}\n")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

# Danh sách các trang quan trọng bạn muốn cho Soni học
dune_urls = [
    {"url": "https://dune.fandom.com/wiki/Paul_Atreides", "file": "Paul_Atreides.txt"},
    {"url": "https://dune.fandom.com/wiki/House_Atreides", "file": "House_Atreides.txt"},
    {"url": "https://dune.fandom.com/wiki/House_Harkonnen", "file": "House_Harkonnen.txt"},
    {"url": "https://dune.fandom.com/wiki/Melange", "file": "Melange.txt"},
    {"url": "https://dune.fandom.com/wiki/Bene_Gesserit", "file": "Bene_Gesserit.txt"},
    {"url": "https://dune.fandom.com/wiki/Dune_(novel)", "file": "Dune_(novel).txt"},
    {"url": "https://dune.fandom.com/wiki/Butlerian_Jihad", "file": "Butlerian_Jihad.txt"},
    {"url": "https://dune.fandom.com/wiki/Mentat", "file": "Mentat.txt"},
    {"url": "https://dune.fandom.com/wiki/Kwisatz_Haderach", "file": "Kwisatz_Haderach.txt"},
    {"url": "https://dune.fandom.com/wiki/Lisan_al-Gaib", "file": "Lisan_al-Gaib.txt"},
    {"url": "https://dune.fandom.com/wiki/Fremen", "file": "Fremen.txt"},
    {"url": "https://dune.fandom.com/wiki/Arrakis", "file": "Arrakis.txt"},
    {"url": "https://dune.fandom.com/wiki/Caladan", "file": "Caladan.txt"},
    {"url": "https://dune.fandom.com/wiki/Giedi_Prime", "file": "Giedi_Prime.txt"},
    {"url": "https://dune.fandom.com/wiki/Sandworm", "file": "Sandworm.txt"},
    {"url": "https://dune.fandom.com/wiki/Spice", "file": "Spice.txt"},
    {"url": "https://dune.fandom.com/wiki/Stillsuit", "file": "Stillsuit.txt"},
    {"url": "https://dune.fandom.com/wiki/Shield", "file": "Shield.txt"},
    {"url": "https://dune.fandom.com/wiki/Thumper", "file": "Thumper.txt"},
    {"url": "https://dune.fandom.com/wiki/Ornithopter", "file": "Ornithopter.txt"},
    {"url": "https://dune.fandom.com/wiki/Water_of_Life", "file": "Water_of_Life.txt"},
    {"url": "https://dune.fandom.com/wiki/Chani", "file": "Chani.txt"},
    {"url": "https://dune.fandom.com/wiki/Vladimir_Harkonnen", "file": "Vladimir_Harkonnen.txt"},
    {"url": "https://dune.fandom.com/wiki/Jessica_Atreides", "file": "Jessica_Atreides.txt"},
    {"url": "https://dune.fandom.com/wiki/Leto_Atreides_I", "file": "Leto_Atreides_I.txt"},
    {"url": "https://dune.fandom.com/wiki/Gurney_Halleck", "file": "Gurney_Halleck.txt"},
    {"url": "https://dune.fandom.com/wiki/Duncan_Idaho", "file": "Duncan_Idaho.txt"},
    {"url": "https://dune.fandom.com/wiki/Thufir_Hawat", "file": "Thufir_Hawat.txt"},
    {"url": "https://dune.fandom.com/wiki/Sardaukar", "file": "Sardaukar.txt"},
    {"url": "https://dune.fandom.com/wiki/Spacing_Guild", "file": "Spacing_Guild.txt"},
    {"url": "https://dune.fandom.com/wiki/Crysknife", "file": "Crysknife.txt"},
    {"url": "https://dune.fandom.com/wiki/Gom_Jabbar", "file": "Gom_Jabbar.txt"},
]

import os

# Cấu hình thư mục lưu trữ tri thức
output_dir = os.path.join("data", "knowledge")
os.makedirs(output_dir, exist_ok=True)

# Chạy vòng lặp để crawl tự động
for item in dune_urls:
    output_path = os.path.join(output_dir, item["file"])
    if os.path.exists(output_path):
        print(f"⏭️ File đã tồn tại, bỏ qua: {item['file']}")
        continue
    scrape_fandom_page(item["url"], output_path)
    # Nghỉ 2 giây giữa mỗi lần tải để tránh bị server khóa IP
    time.sleep(2) 

print("🎉Done collecting data!")