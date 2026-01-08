#!/usr/bin/env python3
"""
Image Version Selector
Compare V1-V7 images side by side and select the best one for each demographic.

Usage:
    python tools/image_selector/app.py
    # Then open http://localhost:5050
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import json
import shutil
from datetime import datetime

app = Flask(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_IMAGES_BASE = PROJECT_ROOT / "data" / "source_images" / "fairface"
FINAL_DIR = SOURCE_IMAGES_BASE / "final"
SELECTIONS_FILE = SOURCE_IMAGES_BASE / "selections.json"

# Demographics
RACES = ['Black', 'EastAsian', 'Indian', 'Latino', 'MiddleEastern', 'SoutheastAsian', 'White']
GENDERS = ['Male', 'Female']
AGES = ['20s', '30s', '40s', '50s', '60s', '70plus']
VERSIONS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']


def get_all_demographics():
    """Generate all 84 demographic combinations."""
    demographics = []
    for race in RACES:
        for gender in GENDERS:
            for age in AGES:
                demo_id = f"{race}_{gender}_{age}"
                demographics.append({
                    'id': demo_id,
                    'race': race,
                    'gender': gender,
                    'age': age
                })
    return demographics


def load_selections():
    """Load existing selections from file."""
    if SELECTIONS_FILE.exists():
        with open(SELECTIONS_FILE) as f:
            return json.load(f)
    return {}


def save_selections(selections):
    """Save selections to file."""
    SELECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTIONS_FILE, 'w') as f:
        json.dump(selections, f, indent=2)


@app.route('/')
def index():
    """Main page."""
    demographics = get_all_demographics()
    selections = load_selections()

    # Count selections
    total = len(demographics)
    selected = len(selections)

    return render_template('index.html',
                         demographics=demographics,
                         versions=VERSIONS,
                         selections=selections,
                         total=total,
                         selected=selected)


@app.route('/images/<version>/<race>/<filename>')
def serve_image(version, race, filename):
    """Serve images from source directories."""
    image_dir = SOURCE_IMAGES_BASE / version / race
    return send_from_directory(image_dir, filename)


@app.route('/api/select', methods=['POST'])
def select_image():
    """Save a selection."""
    data = request.json
    demo_id = data['demo_id']
    version = data['version']

    selections = load_selections()
    selections[demo_id] = {
        'version': version,
        'race': data['race'],
        'gender': data['gender'],
        'age': data['age']
    }
    save_selections(selections)

    return jsonify({'success': True, 'total_selected': len(selections)})


@app.route('/api/deselect', methods=['POST'])
def deselect_image():
    """Remove a selection."""
    data = request.json
    demo_id = data['demo_id']

    selections = load_selections()
    if demo_id in selections:
        del selections[demo_id]
        save_selections(selections)

    return jsonify({'success': True, 'total_selected': len(selections)})


@app.route('/api/selections')
def get_selections():
    """Get all selections."""
    selections = load_selections()
    return jsonify(selections)


@app.route('/api/export')
def export_selections():
    """Export final selected images info."""
    selections = load_selections()

    export_data = {
        'total': len(selections),
        'images': []
    }

    for demo_id, sel in selections.items():
        version = sel['version']
        race = sel['race']
        gender = sel['gender']
        age = sel['age']

        filename = f"{race}_{gender}_{age}.jpg"
        path = SOURCE_IMAGES_BASE / version / race / filename

        export_data['images'].append({
            'demo_id': demo_id,
            'version': version,
            'race': race,
            'gender': gender,
            'age': age,
            'source_path': str(path)
        })

    return jsonify(export_data)


@app.route('/api/finalize', methods=['POST'])
def finalize_selections():
    """Copy selected images to final/ folder."""
    selections = load_selections()

    if len(selections) != 84:
        return jsonify({
            'success': False,
            'error': f'All 84 images must be selected. Currently selected: {len(selections)}'
        }), 400

    # Clear and create final directory
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)

    copied = 0
    images_metadata = []

    for demo_id, sel in selections.items():
        version = sel['version']
        race = sel['race']
        gender = sel['gender']
        age = sel['age']

        # Source path
        filename = f"{race}_{gender}_{age}.jpg"
        src_path = SOURCE_IMAGES_BASE / version / race / filename

        # Destination path
        dst_dir = FINAL_DIR / race
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / filename

        # Copy file
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1

            images_metadata.append({
                'filename': filename,
                'race_code': race,
                'gender': gender,
                'age_code': age,
                'source_version': version,
                'path': str(dst_path)
            })

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_images': copied,
        'source_versions': dict((k, v['version']) for k, v in selections.items()),
        'folder_structure': '{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg',
        'images': images_metadata
    }

    with open(FINAL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return jsonify({
        'success': True,
        'copied': copied,
        'final_dir': str(FINAL_DIR)
    })


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("Image Version Selector")
    print(f"{'='*60}")
    print(f"Source versions: {SOURCE_IMAGES_BASE}/V1-V7")
    print(f"Final output: {FINAL_DIR}")
    print(f"Selections file: {SELECTIONS_FILE}")
    print(f"\nOpen http://localhost:5050 in your browser")
    print(f"{'='*60}\n")

    app.run(debug=True, port=5050, host='0.0.0.0')
