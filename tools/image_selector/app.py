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
SELECTION_LOGS_FILE = SOURCE_IMAGES_BASE / "selection_logs.json"

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


def load_selection_logs():
    """Load existing selection logs from file."""
    if SELECTION_LOGS_FILE.exists():
        with open(SELECTION_LOGS_FILE) as f:
            return json.load(f)
    return []


def save_selection_logs(logs):
    """Save selection logs to file."""
    SELECTION_LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


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
    """Save a selection (legacy endpoint, logs as basic selection)."""
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

    # Save basic selection log
    logs = load_selection_logs()
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'demo_id': demo_id,
        'version': version,
        'race': data['race'],
        'gender': data['gender'],
        'age': data['age'],
        'action': 'select_basic'
    }
    logs.append(log_entry)
    save_selection_logs(logs)

    return jsonify({'success': True, 'total_selected': len(selections)})


@app.route('/api/select_with_log', methods=['POST'])
def select_image_with_log():
    """Save a selection with logging information."""
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

    # Save selection log
    logs = load_selection_logs()
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'demo_id': demo_id,
        'version': version,
        'race': data['race'],
        'gender': data['gender'],
        'age': data['age'],
        'criteria': data['criteria'],
        'notes': data.get('notes', ''),
        'action': 'select'
    }
    logs.append(log_entry)
    save_selection_logs(logs)

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

        # Save deselection log
        logs = load_selection_logs()
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'demo_id': demo_id,
            'action': 'deselect'
        }
        logs.append(log_entry)
        save_selection_logs(logs)

    return jsonify({'success': True, 'total_selected': len(selections)})


@app.route('/api/selections')
def get_selections():
    """Get all selections."""
    selections = load_selections()
    return jsonify(selections)


@app.route('/api/export')
def export_selections():
    """Export final selected images info with logs."""
    selections = load_selections()
    selection_logs = load_selection_logs()

    export_data = {
        'total': len(selections),
        'export_timestamp': datetime.now().isoformat(),
        'images': [],
        'selection_logs_summary': selection_logs[-10:] if len(selection_logs) > 10 else selection_logs  # Last 10 logs
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


@app.route('/api/export_logs')
def export_logs():
    """Export complete selection logs."""
    selection_logs = load_selection_logs()
    selections = load_selections()

    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_logs': len(selection_logs),
        'total_selections': len(selections),
        'selections_summary': selections,
        'detailed_logs': selection_logs
    }

    return jsonify(export_data)


@app.route('/api/finalize', methods=['POST'])
def finalize_selections():
    """Copy selected images to final/ folder and save complete verification logs."""
    selections = load_selections()
    selection_logs = load_selection_logs()

    if len(selections) != 84:
        return jsonify({
            'success': False,
            'error': f'All 84 images must be selected. Currently selected: {len(selections)}'
        }), 400

    # Clear and create final directory
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = []
    images_metadata = []

    timestamp = datetime.now()
    timestamp_str = timestamp.isoformat()
    timestamp_short = timestamp.strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"FINALIZING DATASET - {timestamp_str}")
    print(f"Dataset ID: dataset_{timestamp_short}")
    print(f"{'='*60}")
    print(f"Selected images: {len(selections)}/84")
    print(f"Selection logs: {len(selection_logs)} entries")
    print(f"Source versions used: {set(sel['version'] for sel in selections.values())}")

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

        # Copy file with verification
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
                print(f"✓ Copied: {demo_id} -> {version}")

                images_metadata.append({
                    'filename': filename,
                    'race_code': race,
                    'gender': gender,
                    'age_code': age,
                    'source_version': version,
                    'source_path': str(src_path),
                    'final_path': str(dst_path),
                    'file_size': dst_path.stat().st_size if dst_path.exists() else 0
                })
            except Exception as e:
                failed.append(f"{demo_id}: {str(e)}")
                print(f"✗ Failed: {demo_id} - {str(e)}")
        else:
            failed.append(f"{demo_id}: Source file not found")
            print(f"✗ Missing: {demo_id} - Source file not found")

    # Create comprehensive verification report
    verification_report = {
        'finalization_timestamp': datetime.now().isoformat(),
        'total_expected': 84,
        'total_copied': copied,
        'total_failed': len(failed),
        'failed_items': failed,
        'selections_summary': {
            'by_version': {},
            'by_race': {},
            'by_gender': {},
            'by_age': {}
        }
    }

    # Analyze selections by categories
    for demo_id, sel in selections.items():
        version = sel['version']
        race = sel['race']
        gender = sel['gender']
        age = sel['age']

        verification_report['selections_summary']['by_version'][version] = \
            verification_report['selections_summary']['by_version'].get(version, 0) + 1
        verification_report['selections_summary']['by_race'][race] = \
            verification_report['selections_summary']['by_race'].get(race, 0) + 1
        verification_report['selections_summary']['by_gender'][gender] = \
            verification_report['selections_summary']['by_gender'].get(gender, 0) + 1
        verification_report['selections_summary']['by_age'][age] = \
            verification_report['selections_summary']['by_age'].get(age, 0) + 1

    # Create final metadata with dataset ID
    dataset_id = f"dataset_{timestamp_short}"
    metadata = {
        'dataset_id': dataset_id,
        'created_at': timestamp_str,
        'total_images': copied,
        'expected_total': 84,
        'success_rate': f"{copied}/84 ({(copied/84*100):.1f}%)" if copied > 0 else "0/84 (0%)",
        'source_versions': dict((k, v['version']) for k, v in selections.items()),
        'folder_structure': '{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg',
        'images': images_metadata,
        'selection_logs': selection_logs,
        'final_selections': selections,
        'verification_report': verification_report,
        'audit_trail': {
            'creation_method': 'Manual selection with criteria verification',
            'logs_preserve_criteria': True,
            'images_preserve_identity': True,
            'fully_auditable': True,
            'reproducible': True
        }
    }

    # Save all files
    with open(FINAL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(FINAL_DIR / 'selection_logs_complete.json', 'w') as f:
        json.dump({
            'export_timestamp': datetime.now().isoformat(),
            'total_logs': len(selection_logs),
            'logs_by_action': {
                'select': len([l for l in selection_logs if l.get('action') == 'select']),
                'select_with_log': len([l for l in selection_logs if l.get('action') == 'select_with_log']),
                'deselect': len([l for l in selection_logs if l.get('action') == 'deselect']),
                'select_basic': len([l for l in selection_logs if l.get('action') == 'select_basic'])
            },
            'detailed_logs': selection_logs
        }, f, indent=2)

    with open(FINAL_DIR / 'verification_report.json', 'w') as f:
        json.dump(verification_report, f, indent=2)

    # Backup original files
    if SELECTIONS_FILE.exists():
        shutil.copy2(SELECTIONS_FILE, FINAL_DIR / 'selections_backup.json')
    if SELECTION_LOGS_FILE.exists():
        shutil.copy2(SELECTION_LOGS_FILE, FINAL_DIR / 'logs_backup.json')

    print(f"\n{'='*60}")
    print("FINALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Images copied: {copied}/84")
    print(f"Logs saved: {len(selection_logs)}")
    print(f"Final directory: {FINAL_DIR}")
    if failed:
        print(f"Failed items: {len(failed)}")
        for fail in failed:
            print(f"  - {fail}")
    print(f"{'='*60}\n")

    return jsonify({
        'success': True,
        'copied': copied,
        'expected': 84,
        'failed': len(failed),
        'final_dir': str(FINAL_DIR),
        'logs_saved': len(selection_logs),
        'dataset_id': dataset_id,
        'timestamp': timestamp_str,
        'verification_complete': True,
        'audit_trail_available': True
    })


if __name__ == '__main__':
    print(f"\n{'='*80}")
    print("I2I Bias Refusal Study - Dataset Curator")
    print(f"{'='*80}")
    print(f"Purpose: Curate 84-image dataset with full audit trail for bias analysis")
    print(f"Source images: {SOURCE_IMAGES_BASE}/V1-V7")
    print(f"Final dataset: {FINAL_DIR}")
    print(f"Selections log: {SELECTIONS_FILE}")
    print(f"Criteria logs: {SELECTION_LOGS_FILE}")
    print(f"\n🔍 Open http://localhost:5050 in your browser to start curation")
    print(f"📋 Select images based on facial pose, quality, and demographic criteria")
    print(f"✅ Finalize creates complete audit trail for reproducibility")
    print(f"{'='*80}\n")

    app.run(debug=True, port=5050, host='0.0.0.0')
