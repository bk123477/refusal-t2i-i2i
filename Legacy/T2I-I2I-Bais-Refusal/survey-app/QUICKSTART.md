# ACRB Survey App - Quick Start Guide

## Overview

This is a fully-functional Firebase-integrated survey app for collecting human evaluations of T2I/I2I model outputs.

## What's Built

- Firebase integration (Firestore + Storage + Auth)
- 4-question evaluation form (refusal, attribute, faithfulness, confidence)
- Auto-save and session recovery
- Admin dashboard with analytics (Cohen's Kappa, per-attribute metrics)
- CSV/JSON export
- Mobile-optimized UI
- Dark mode support

## Immediate Next Steps

### 1. Start Development Server

```bash
cd /Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/survey-app
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 2. Upload Survey Images

You have two options:

#### Option A: Via Admin UI
1. Navigate to `http://localhost:3000/admin`
2. Click "Upload Images" tab
3. Fill in metadata and select image file
4. Click upload

#### Option B: Bulk Upload Script
```bash
# Create a results.json from your experiments
# Format: see scripts/upload_survey_items.ts

npx ts-node scripts/upload_survey_items.ts \
  --results ../experiments/results.json \
  --images ../experiments/outputs/
```

### 3. Test the Survey Flow

1. **Landing** - `http://localhost:3000`
   - Enter a test Prolific ID (e.g., "TEST001")

2. **Consent** - `/consent`
   - Click "I Consent"

3. **Demographics** - `/demographics`
   - Fill out demographic questions
   - Click "Continue"

4. **Survey** - `/survey`
   - You'll see the 4-question evaluation form
   - Test keyboard shortcuts (Ctrl+Enter to submit)
   - Check auto-save (wait 30 seconds, refresh page)
   - Every 10 items, you'll see a break prompt

5. **Complete** - `/complete`
   - Thank you page with completion code

### 4. View Admin Dashboard

```
http://localhost:3000/admin
```

You'll see:
- Total participants, evaluations, completion time
- Cohen's Kappa and percent agreement
- Per-attribute breakdown (refusal rate, faithfulness, attribute retention)
- Per-model statistics
- Export buttons for CSV/JSON

## Firebase Console

Visit https://console.firebase.google.com/project/acrb-e8cb4

### Collections to Monitor

1. **participants** - Evaluator info and demographics
2. **evaluations** - Individual image ratings
3. **survey_items** - Images available for evaluation
4. **sessions** - Auto-saved session data

### Storage

- `survey-images/` - All uploaded evaluation images

## Deploy to Production

### Vercel (Recommended)

```bash
npm install -g vercel
vercel login
vercel --prod
```

Your app will be live at `https://your-app.vercel.app`

### Alternative: Firebase Hosting

```bash
npm install -g firebase-tools
firebase login
firebase init hosting
npm run build
firebase deploy --only hosting
```

## Data Export for Paper

From the admin dashboard:

1. Click "Export Data" tab
2. Download CSV for statistical analysis
3. Download JSON for full dataset

Use exported data:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('acrb-evaluations-2026-01-05.csv')

# Calculate refusal disparity
refusal_by_attr = df.groupby(['attribute', 'attributeValue'])['isRefusal'].mean()
max_refusal = refusal_by_attr.max()
min_refusal = refusal_by_attr.min()
delta_refusal = max_refusal - min_refusal

print(f"Refusal disparity (Δ_refusal): {delta_refusal:.3f}")
```

## Firestore Security (Important!)

Before deploying to production, update Firestore rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Participants can only read/write their own data
    match /participants/{participantId} {
      allow read, write: if request.auth != null
                          && request.auth.uid == participantId;
    }

    // Evaluations are write-only during survey
    match /evaluations/{evaluationId} {
      allow create: if request.auth != null;
      allow read: if false; // Admin only via console
    }

    // Survey items are read-only
    match /survey_items/{itemId} {
      allow read: if request.auth != null;
      allow write: if false; // Admin upload only
    }

    // Sessions for auto-save
    match /sessions/{sessionId} {
      allow read, write: if request.auth != null
                          && request.auth.uid == sessionId;
    }
  }
}
```

## Prolific Integration

1. Create study on Prolific.co
2. Set completion URL:
   ```
   https://your-app.vercel.app/?PROLIFIC_PID={{%PROLIFIC_PID%}}
   ```

3. In your completion page, redirect back to Prolific:
   ```javascript
   window.location.href = `https://app.prolific.com/submissions/complete?cc=YOUR_COMPLETION_CODE`
   ```

## Troubleshooting

### "No items available" in survey

- You haven't uploaded any survey items yet
- Use the admin upload interface or bulk upload script

### Images not loading

- Check Firebase Storage rules (allow read for authenticated users)
- Verify image URLs in Firestore

### Build fails

```bash
rm -rf .next node_modules
npm install
npm run build
```

## Configuration

All configuration is in `/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/survey-app/lib/firebase.ts`

Firebase config (public, secured by rules):
```javascript
const firebaseConfig = {
  apiKey: "AIzaSyCYZNH-D5KEPc_2f1Gr9vcmFpyd29t97xU",
  authDomain: "acrb-e8cb4.firebaseapp.com",
  projectId: "acrb-e8cb4",
  storageBucket: "acrb-e8cb4.firebasestorage.app",
  messagingSenderId: "87810362498",
  appId: "1:87810362498:web:9a13c8f15886030ab6b89b"
}
```

## Paper Submission Checklist

- [ ] Deploy to production URL
- [ ] Upload all evaluation images (T2I + I2I)
- [ ] Test complete survey flow
- [ ] Recruit 50-100 participants via Prolific
- [ ] Monitor completion rate on admin dashboard
- [ ] Export data when target reached
- [ ] Calculate Cohen's Kappa (built-in)
- [ ] Compute per-attribute refusal disparities
- [ ] Include human evaluation results in Section 5

## Key Metrics Computed

The admin dashboard automatically calculates:

1. **Cohen's Kappa** - Inter-rater reliability
2. **Percent Agreement** - Overall agreement rate
3. **Per-attribute metrics**:
   - Refusal rate
   - Average faithfulness
   - Attribute retention rate
4. **Per-model metrics**:
   - Refusal rate
   - Average faithfulness

Export JSON includes full analysis object for paper.

## Support

Check README.md for detailed documentation.

For Firebase console access, project ID: `acrb-e8cb4`
