#!/usr/bin/env python3
"""
B/D 카테고리 이미지 비교 뷰어 (페이지네이션 HTTP 서버)

한 페이지에 소수의 이미지만 로드하여 빠른 탐색이 가능합니다.
"""

import json
import argparse
from pathlib import Path
import http.server
import socketserver
import urllib.parse
import os
import base64


def parse_image_filename(filename: str) -> dict:
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


def image_to_base64(path):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lower()
            mime = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
            return f"data:{mime};base64,{data}"
    except:
        return ""


def generate_page_html(results, prompts, page, per_page, category_filter, prompt_filter, all_prompt_ids):
    # 필터링
    filtered = results
    if category_filter and category_filter != "all":
        filtered = [r for r in filtered if r["category"] == category_filter]
    if prompt_filter and prompt_filter != "all":
        filtered = [r for r in filtered if r["prompt_id"] == prompt_filter]
        # 프롬프트 필터링 시 race+gender 그룹별로 페이지네이션
        # 각 그룹을 하나의 페이지로 취급
        from collections import OrderedDict
        groups = OrderedDict()
        for r in filtered:
            key = (r["race"], r["gender"])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        group_list = list(groups.items())
        total_pages = len(group_list)
        page = max(0, min(page, total_pages - 1))
        
        if group_list:
            (race, gender), page_results = group_list[page]
            # 나이순 정렬
            page_results = sorted(page_results, key=lambda x: x["age"])
            total = len(filtered)
            b_count = len([r for r in filtered if r["category"] == "B"])
            d_count = len([r for r in filtered if r["category"] == "D"])
            group_info = f"{race} {gender}"
        else:
            page_results = []
            total = 0
            total_pages = 1
            b_count = 0
            d_count = 0
            group_info = "None"
    else:
        # 일반 페이지네이션
        total = len(filtered)
        total_pages = max(1, (total - 1) // per_page + 1)
        page = max(0, min(page, total_pages - 1))
        
        start = page * per_page
        end = min(start + per_page, total)
        page_results = filtered[start:end]
        
        b_count = len([r for r in filtered if r["category"] == "B"])
        d_count = len([r for r in filtered if r["category"] == "D"])
        group_info = None
    
    # 이미지 카드 생성
    cards_html = ""
    for r in page_results:
        prompt_info = prompts.get(r["prompt_id"], {})
        prompt_text = prompt_info.get("prompt", "N/A")
        hypothesis = prompt_info.get("hypothesis", "N/A")
        
        status_class = {"success": "badge-success", "refused": "badge-refused", "unchanged": "badge-unchanged"}.get(r["status"], "badge-success")
        
        source_b64 = image_to_base64(r["source_image"]) if r["source_image"] else ""
        output_b64 = image_to_base64(r["output_image"]) if r["output_image"] else ""
        
        cards_html += f'''
        <div class="card">
            <div class="card-header">
                <h3>{r["prompt_id"]} - {r["race"]} {r["gender"]} {r["age"]}</h3>
                <div>
                    <span class="badge badge-{r["category"]}">Category {r["category"]}</span>
                    <span class="badge {status_class}">{r["status"].capitalize()}</span>
                </div>
            </div>
            <div class="prompt-text"><strong>Prompt:</strong> {prompt_text}</div>
            <div class="meta-info">
                <span class="meta-tag">Hypothesis: {hypothesis}</span>
            </div>
            <div class="image-comparison">
                <div class="image-box">
                    <h4>Source</h4>
                    {"<img src='" + source_b64 + "'>" if source_b64 else "<div class='no-image'>Not found</div>"}
                </div>
                <div class="image-box">
                    <h4>Output</h4>
                    {"<img src='" + output_b64 + "'>" if output_b64 else "<div class='no-image'>Not found</div>"}
                </div>
            </div>
        </div>
        '''
    
    # 페이지네이션 링크
    cat_param = f"&cat={category_filter}" if category_filter and category_filter != "all" else ""
    prompt_param = f"&prompt={prompt_filter}" if prompt_filter and prompt_filter != "all" else ""
    params = cat_param + prompt_param
    prev_link = f"/?page={page-1}{params}" if page > 0 else "#"
    next_link = f"/?page={page+1}{params}" if page < total_pages - 1 else "#"
    
    prev_disabled = "disabled" if page == 0 else ""
    next_disabled = "disabled" if page >= total_pages - 1 else ""
    
    # 프롬프트 ID 드롭다운 옵션
    prompt_options = '<option value="all">All Prompts</option>'
    for pid in all_prompt_ids:
        selected = "selected" if pid == prompt_filter else ""
        prompt_options += f'<option value="{pid}" {selected}>{pid}</option>'
    
    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>B/D Image Viewer - Page {page+1}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1a1a2e, #0f3460); min-height: 100vh; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; color: #fff; margin-bottom: 20px; }}
.controls {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.controls a, .controls button {{ padding: 10px 20px; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; text-decoration: none; color: white; background: linear-gradient(135deg, #667eea, #764ba2); }}
.controls a:hover {{ opacity: 0.9; }}
.controls a.active {{ background: linear-gradient(135deg, #f093fb, #f5576c); }}
.controls a.disabled {{ opacity: 0.5; pointer-events: none; }}
.stats {{ text-align: center; margin-bottom: 15px; color: #a0aec0; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); flex-wrap: wrap; gap: 10px; }}
.card-header h3 {{ color: #fff; font-size: 1.1em; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 15px; font-size: 0.8em; font-weight: 600; margin-left: 5px; }}
.badge-B {{ background: linear-gradient(135deg, #667eea, #764ba2); }}
.badge-D {{ background: linear-gradient(135deg, #f093fb, #f5576c); }}
.badge-success {{ background: linear-gradient(135deg, #11998e, #38ef7d); }}
.badge-refused {{ background: linear-gradient(135deg, #eb3349, #f45c43); }}
.badge-unchanged {{ background: linear-gradient(135deg, #f2994a, #f2c94c); color: #333; }}
.prompt-text {{ background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; margin-bottom: 10px; font-style: italic; border-left: 3px solid #667eea; font-size: 0.9em; }}
.meta-info {{ margin-bottom: 10px; }}
.meta-tag {{ background: rgba(255,255,255,0.1); padding: 3px 8px; border-radius: 4px; font-size: 0.8em; }}
.image-comparison {{ display: flex; gap: 15px; justify-content: center; }}
.image-box {{ flex: 1; max-width: 400px; text-align: center; }}
.image-box h4 {{ margin-bottom: 8px; color: #a0aec0; font-size: 0.85em; text-transform: uppercase; }}
.image-box img {{ max-width: 100%; max-height: 350px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
.no-image {{ display: flex; align-items: center; justify-content: center; height: 200px; background: rgba(0,0,0,0.2); border-radius: 8px; color: #666; }}
.pagination {{ display: flex; justify-content: center; gap: 10px; margin-top: 20px; }}
</style>
</head>
<body>
<h1>🖼️ B/D Image Comparison Viewer</h1>

<div class="controls">
    <a href="/?page=0" class="{'active' if (not category_filter or category_filter == 'all') and (not prompt_filter or prompt_filter == 'all') else ''}">All ({len(results)})</a>
    <a href="/?page=0&cat=B" class="{'active' if category_filter == 'B' else ''}">Category B</a>
    <a href="/?page=0&cat=D" class="{'active' if category_filter == 'D' else ''}">Category D</a>
    <select onchange="location.href='/?page=0' + (document.getElementById('catSelect') ? '&cat=' + document.getElementById('catSelect').value : '') + '&prompt=' + this.value" style="padding: 10px; border-radius: 8px; background: #2d3748; color: #e0e0e0; border: 1px solid #4a5568;">
        {prompt_options}
    </select>
</div>

<div class="stats">Page {page+1} of {total_pages}{f" | 👤 {group_info}" if group_info else ""} | Showing {len(page_results)} of {total} results</div>

<div class="container">
{cards_html}
</div>

<div class="pagination">
    <a href="{prev_link}" class="{prev_disabled}">← Previous</a>
    <span style="color: #a0aec0; padding: 10px;">Page {page+1} / {total_pages}</span>
    <a href="{next_link}" class="{next_disabled}">Next →</a>
</div>

</body></html>'''
    return html


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    results = []
    prompts = {}
    per_page = 10
    all_prompt_ids = []
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        page = int(params.get("page", [0])[0])
        cat = params.get("cat", ["all"])[0]
        prompt = params.get("prompt", ["all"])[0]
        
        html = generate_page_html(
            self.results, self.prompts, page, self.per_page, cat,
            prompt, self.all_prompt_ids
        )
        
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="B/D 이미지 뷰어 (페이지네이션)")
    parser.add_argument("--images-dir", type=str, 
                        default="/home/gpu_02/refusal-t2i-i2i/data/results/qwen/20260109_083212/images")
    parser.add_argument("--source-dir", type=str,
                        default="/home/gpu_02/refusal-t2i-i2i/data/source_images/final")
    parser.add_argument("--prompts", type=str,
                        default="/home/gpu_02/refusal-t2i-i2i/data/prompts/i2i_prompts.json")
    parser.add_argument("--categories", type=str, default="B,D")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--per-page", type=int, default=10, help="이미지 수/페이지 (기본: 10)")
    
    args = parser.parse_args()
    
    categories = [c.strip() for c in args.categories.split(",")]
    print(f"Loading prompts...")
    prompts = load_prompts(args.prompts)
    
    print(f"Scanning images...")
    results = scan_images(args.images_dir, args.source_dir, categories)
    print(f"Found {len(results)} images")
    
    ViewerHandler.results = results
    ViewerHandler.prompts = prompts
    ViewerHandler.per_page = args.per_page
    ViewerHandler.all_prompt_ids = sorted(set(r["prompt_id"] for r in results))
    
    with socketserver.TCPServer(("", args.port), ViewerHandler) as httpd:
        print(f"\n✅ 서버 시작!")
        print(f"   👉 http://localhost:{args.port}")
        print(f"   (Ctrl+C로 종료)\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
