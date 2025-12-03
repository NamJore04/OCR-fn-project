"""
Script để tải và chuẩn bị dữ liệu cho dự án OCR Image-to-Text
Hỗ trợ tải các bộ dữ liệu như SynthText và ICDAR
"""

import os
import sys
import argparse
import wget
import zipfile
import gdown
import tarfile
from pathlib import Path

# Thêm thư mục gốc vào PATH để import các module từ dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Đảm bảo thư mục dữ liệu tồn tại
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Định nghĩa URL cho các bộ dữ liệu
DATASET_URLS = {
    'synthtext': {
        'url': 'https://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip',
        'filename': 'SynthText.zip'
    },
    'icdar2013': {
        'url': 'https://rrc.cvc.uab.es/?ch=2&com=downloads',
        'filename': 'ICDAR2013.zip',
        'note': 'Cần đăng nhập vào trang ICDAR để tải'
    },
    'icdar2015': {
        'url': 'https://rrc.cvc.uab.es/?ch=4&com=downloads',
        'filename': 'ICDAR2015.zip', 
        'note': 'Cần đăng nhập vào trang ICDAR để tải'
    }
}

def download_synthtext():
    """Tải bộ dữ liệu SynthText"""
    output_path = os.path.join(RAW_DATA_DIR, 'SynthText.zip')
    
    # Kiểm tra nếu file đã tồn tại
    if os.path.exists(output_path):
        print(f"File {output_path} đã tồn tại. Bỏ qua tải xuống.")
        return output_path
    
    print("Đang tải SynthText dataset...")
    try:
        wget.download(DATASET_URLS['synthtext']['url'], output_path)
        print(f"\nĐã tải xong và lưu tại {output_path}")
        return output_path
    except Exception as e:
        print(f"Lỗi khi tải SynthText: {str(e)}")
        return None

def extract_dataset(zip_path, dataset_name):
    """Giải nén file zip vào thư mục RAW_DATA_DIR"""
    if not os.path.exists(zip_path):
        print(f"File {zip_path} không tồn tại.")
        return False
    
    extract_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Đang giải nén {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Đã giải nén xong vào {extract_dir}")
        return True
    except Exception as e:
        print(f"Lỗi khi giải nén: {str(e)}")
        return False

def prepare_icdar_manual():
    """Hướng dẫn chuẩn bị dữ liệu ICDAR"""
    print("\nHướng dẫn chuẩn bị dữ liệu ICDAR:")
    print("1. Truy cập trang web ICDAR:")
    print("   - ICDAR 2013: https://rrc.cvc.uab.es/?ch=2&com=downloads")
    print("   - ICDAR 2015: https://rrc.cvc.uab.es/?ch=4&com=downloads")
    print("2. Đăng nhập và tải các file dữ liệu")
    print("3. Giải nén và đặt trong thư mục:", os.path.join(RAW_DATA_DIR, 'ICDAR2013') or os.path.join(RAW_DATA_DIR, 'ICDAR2015'))
    print(f"4. Chạy lại script này với tham số --process để xử lý dữ liệu")

def main():
    parser = argparse.ArgumentParser(description='Tải và chuẩn bị dữ liệu cho dự án OCR')
    parser.add_argument('--dataset', choices=['synthtext', 'icdar2013', 'icdar2015', 'all'], 
                        default='synthtext', help='Loại dataset cần tải')
    parser.add_argument('--extract', action='store_true', help='Giải nén file sau khi tải')
    parser.add_argument('--process', action='store_true', help='Xử lý dữ liệu sau khi giải nén')
    
    args = parser.parse_args()
    
    if args.dataset == 'synthtext' or args.dataset == 'all':
        zip_path = download_synthtext()
        if zip_path and args.extract:
            extract_dataset(zip_path, 'SynthText')
    
    if args.dataset == 'icdar2013' or args.dataset == 'icdar2015' or args.dataset == 'all':
        prepare_icdar_manual()
    
    if args.process:
        print("\nChức năng xử lý dữ liệu sẽ được triển khai trong các phiên làm việc tiếp theo.")

if __name__ == "__main__":
    main()
    print("\nHoàn tất công việc tải và chuẩn bị dữ liệu.")