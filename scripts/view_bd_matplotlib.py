#!/usr/bin/env python3
"""
B/D 카테고리 이미지 비교 뷰어 (Matplotlib 버전)

Matplotlib을 사용하여 이미지를 인터랙티브하게 비교합니다.
페이지 단위로 탐색하여 빠른 로딩이 가능합니다.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


def parse_image_filename(filename: str) -> dict:
    """이미지 파일명을 파싱합니다."""
    stem = Path(filename).stem
    parts = stem.split("_")
    
    if len(parts) >= 5:
        return {
            "prompt_id": parts[0],
            "race": parts[1],
            "gender": parts[2],
            "age": parts[3],
            "status": parts[4] if len(parts) > 4 else "unknown"
        }
    return None


def load_prompts(prompts_path: str) -> dict:
    """프롬프트 파일을 로드하고 ID로 인덱싱합니다."""
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = {}
    for p in data.get("prompts", []):
        prompts[p["id"]] = {
            "prompt": p["prompt"],
            "category": p["category"],
            "hypothesis": p.get("hypothesis", "")
        }
    return prompts


def scan_images(images_dir: str, source_images_dir: str, categories: list) -> list:
    """이미지 디렉토리를 스캔하여 결과 목록을 생성합니다."""
    results = []
    images_path = Path(images_dir)
    source_path = Path(source_images_dir)
    
    for race_folder in images_path.iterdir():
        if not race_folder.is_dir():
            continue
        
        race = race_folder.name
        
        for image_file in race_folder.glob("*.png"):
            parsed = parse_image_filename(image_file.name)
            if not parsed:
                continue
            
            prompt_id = parsed["prompt_id"]
            category = prompt_id[0] if prompt_id else None
            
            if category not in categories:
                continue
            
            source_filename = f"{race}_{parsed['gender']}_{parsed['age']}.jpg"
            source_image_path = source_path / race / source_filename
            
            results.append({
                "prompt_id": prompt_id,
                "category": category,
                "race": race,
                "gender": parsed["gender"],
                "age": parsed["age"],
                "status": parsed["status"],
                "output_image": str(image_file),
                "source_image": str(source_image_path) if source_image_path.exists() else None
            })
    
    results.sort(key=lambda x: (x["category"], x["prompt_id"], x["race"], x["gender"], x["age"]))
    return results


class ImageViewer:
    def __init__(self, results, prompts, items_per_page=4):
        self.results = results
        self.prompts = prompts
        self.items_per_page = items_per_page
        self.current_page = 0
        self.filtered_results = results.copy()
        self.category_filter = "All"
        
        # 그림 설정
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # 메인 그리드 설정 (이미지 영역)
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """UI 컴포넌트 설정"""
        # 제목
        self.fig.suptitle('B/D Category Image Comparison Viewer', 
                         fontsize=16, color='white', fontweight='bold')
        
        # 카테고리 필터 라디오 버튼
        ax_radio = plt.axes([0.02, 0.85, 0.08, 0.10], facecolor='#2d3748')
        self.radio = RadioButtons(ax_radio, ('All', 'B', 'D'), activecolor='#667eea')
        for label in self.radio.labels:
            label.set_color('white')
        self.radio.on_clicked(self.filter_category)
        
        # 네비게이션 버튼
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, '← Prev', color='#667eea', hovercolor='#764ba2')
        self.btn_next = Button(ax_next, 'Next →', color='#667eea', hovercolor='#764ba2')
        self.btn_prev.label.set_color('white')
        self.btn_next.label.set_color('white')
        
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        
        # 페이지 정보 텍스트
        self.page_text = self.fig.text(0.5, 0.025, '', ha='center', va='center', 
                                        fontsize=12, color='white')
        
    def filter_category(self, label):
        """카테고리 필터 적용"""
        self.category_filter = label
        if label == 'All':
            self.filtered_results = self.results.copy()
        else:
            self.filtered_results = [r for r in self.results if r['category'] == label]
        self.current_page = 0
        self.update_display()
        
    def prev_page(self, event):
        """이전 페이지"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
            
    def next_page(self, event):
        """다음 페이지"""
        max_page = (len(self.filtered_results) - 1) // self.items_per_page
        if self.current_page < max_page:
            self.current_page += 1
            self.update_display()
    
    def load_image(self, path):
        """이미지 로드 (에러 처리 포함)"""
        try:
            return mpimg.imread(path)
        except:
            # 빈 이미지 반환
            return np.zeros((100, 100, 3))
    
    def update_display(self):
        """화면 업데이트"""
        # 기존 axes 제거 (컨트롤 제외)
        for ax in self.fig.axes[:]:
            if ax not in [self.radio.ax, self.btn_prev.ax, self.btn_next.ax]:
                ax.remove()
        
        # 현재 페이지의 결과
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.filtered_results))
        page_results = self.filtered_results[start_idx:end_idx]
        
        if not page_results:
            self.page_text.set_text("No results found")
            self.fig.canvas.draw_idle()
            return
        
        # 이미지 그리드 (각 결과당 2개 이미지: source, output)
        n_items = len(page_results)
        
        for i, result in enumerate(page_results):
            # 행 위치 계산 (위에서 아래로)
            row_top = 0.78 - (i * 0.18)
            row_height = 0.14
            
            # Source 이미지
            ax_source = self.fig.add_axes([0.12, row_top, 0.15, row_height])
            if result['source_image']:
                img = self.load_image(result['source_image'])
                ax_source.imshow(img)
            ax_source.axis('off')
            ax_source.set_title('Source', fontsize=8, color='#a0aec0')
            
            # Output 이미지
            ax_output = self.fig.add_axes([0.30, row_top, 0.15, row_height])
            if result['output_image']:
                img = self.load_image(result['output_image'])
                ax_output.imshow(img)
            ax_output.axis('off')
            ax_output.set_title('Output', fontsize=8, color='#a0aec0')
            
            # 정보 텍스트
            prompt_info = self.prompts.get(result['prompt_id'], {})
            prompt_text = prompt_info.get('prompt', 'N/A')
            
            # 프롬프트 텍스트 줄바꿈
            wrapped_prompt = '\n'.join([prompt_text[j:j+60] for j in range(0, len(prompt_text), 60)])
            
            info_text = (f"[{result['prompt_id']}] {result['race']} {result['gender']} {result['age']} "
                        f"({result['status'].upper()})\n{wrapped_prompt}")
            
            # 상태에 따른 색상
            color = '#38ef7d' if result['status'] == 'success' else (
                    '#f5576c' if result['status'] == 'refused' else '#f2c94c')
            
            self.fig.text(0.50, row_top + row_height/2, info_text,
                         fontsize=8, color='white', va='center',
                         bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8))
        
        # 페이지 정보 업데이트
        total_pages = max(1, (len(self.filtered_results) - 1) // self.items_per_page + 1)
        self.page_text.set_text(
            f"Page {self.current_page + 1}/{total_pages} | "
            f"Showing {len(page_results)} of {len(self.filtered_results)} results "
            f"(Category: {self.category_filter})"
        )
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """뷰어 표시"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="B/D 카테고리 이미지 비교 뷰어 (Matplotlib)")
    parser.add_argument("--images-dir", type=str, 
                        default="/home/gpu_02/refusal-t2i-i2i/data/results/qwen/20260109_083212/images",
                        help="output 이미지가 있는 디렉토리 경로")
    parser.add_argument("--source-dir", type=str,
                        default="/home/gpu_02/refusal-t2i-i2i/data/source_images/final",
                        help="source 이미지가 있는 디렉토리 경로")
    parser.add_argument("--prompts", type=str,
                        default="/home/gpu_02/refusal-t2i-i2i/data/prompts/i2i_prompts.json",
                        help="프롬프트 JSON 파일 경로")
    parser.add_argument("--categories", type=str, default="B,D",
                        help="표시할 카테고리 (쉼표로 구분, 기본: B,D)")
    parser.add_argument("--per-page", type=int, default=4,
                        help="페이지당 표시할 이미지 수 (기본: 4)")
    
    args = parser.parse_args()
    
    categories = [c.strip() for c in args.categories.split(",")]
    print(f"Filtering for categories: {categories}")
    
    print(f"Loading prompts from: {args.prompts}")
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    print(f"Scanning images from: {args.images_dir}")
    results = scan_images(args.images_dir, args.source_dir, categories)
    
    b_count = len([r for r in results if r["category"] == "B"])
    d_count = len([r for r in results if r["category"] == "D"])
    print(f"Found {len(results)} images: Category B: {b_count}, Category D: {d_count}")
    
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - Use 'Prev/Next' buttons to navigate pages")
    print("  - Use radio buttons to filter by category")
    print("  - Close the window to exit")
    
    viewer = ImageViewer(results, prompts, items_per_page=args.per_page)
    viewer.show()


if __name__ == "__main__":
    main()
