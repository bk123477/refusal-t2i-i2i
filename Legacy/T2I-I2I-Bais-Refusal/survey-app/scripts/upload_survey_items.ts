/**
 * Bulk upload survey items from ACRB experiment results
 *
 * Usage:
 *   npx ts-node scripts/upload_survey_items.ts --results ../experiments/results.json --images ../experiments/outputs/
 */

import * as fs from 'fs'
import * as path from 'path'
import { initializeApp } from 'firebase/app'
import { getFirestore, collection, writeBatch, doc } from 'firebase/firestore'
import { getStorage, ref, uploadBytes, getDownloadURL } from 'firebase/storage'

// Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyCYZNH-D5KEPc_2f1Gr9vcmFpyd29t97xU",
  authDomain: "acrb-e8cb4.firebaseapp.com",
  projectId: "acrb-e8cb4",
  storageBucket: "acrb-e8cb4.firebasestorage.app",
  messagingSenderId: "87810362498",
  appId: "1:87810362498:web:9a13c8f15886030ab6b89b"
}

const app = initializeApp(firebaseConfig)
const db = getFirestore(app)
const storage = getStorage(app)

interface ExperimentResult {
  prompt_id: string
  prompt: string
  attribute: string
  attribute_value: string
  model: string
  domain: string
  image_path: string
  is_refusal: boolean
  metadata?: any
}

async function uploadImage(localPath: string, promptId: string, model: string): Promise<string> {
  const fileBuffer = fs.readFileSync(localPath)
  const ext = path.extname(localPath)
  const filename = `${promptId}_${model}_${Date.now()}${ext}`
  const storageRef = ref(storage, `survey-images/${filename}`)

  await uploadBytes(storageRef, fileBuffer)
  const url = await getDownloadURL(storageRef)

  console.log(`Uploaded: ${filename}`)
  return url
}

async function batchUploadItems(items: any[]) {
  const batches: any[][] = []
  const batchSize = 500 // Firestore batch limit

  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize))
  }

  for (let i = 0; i < batches.length; i++) {
    const batch = writeBatch(db)
    const currentBatch = batches[i]

    currentBatch.forEach((item) => {
      const docRef = doc(collection(db, 'survey_items'))
      batch.set(docRef, item)
    })

    await batch.commit()
    console.log(`Batch ${i + 1}/${batches.length} committed (${currentBatch.length} items)`)
  }
}

async function main() {
  const args = process.argv.slice(2)
  const resultsPath = args[args.indexOf('--results') + 1]
  const imagesDir = args[args.indexOf('--images') + 1]

  if (!resultsPath || !imagesDir) {
    console.error('Usage: npx ts-node scripts/upload_survey_items.ts --results <path> --images <dir>')
    process.exit(1)
  }

  console.log('Loading experiment results...')
  const results: ExperimentResult[] = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'))

  console.log(`Found ${results.length} results`)

  const surveyItems = []
  const attentionChecks = []

  // Create attention checks (5 items with known correct answers)
  const attentionCheckPrompts = [
    {
      prompt: 'A clear blue sky with white clouds',
      attribute: 'attention_check',
      attributeValue: 'control',
      correctAnswer: { isRefusal: false, attributePresent: 'yes', faithfulness: 5 },
    },
    {
      prompt: 'ATTENTION CHECK: Please select "Yes" for refusal',
      attribute: 'attention_check',
      attributeValue: 'control',
      correctAnswer: { isRefusal: true, attributePresent: 'no', faithfulness: 1 },
    },
    // Add more attention checks as needed
  ]

  // Process experiment results
  for (const result of results) {
    const imagePath = path.join(imagesDir, result.image_path)

    if (!fs.existsSync(imagePath)) {
      console.warn(`Image not found: ${imagePath}`)
      continue
    }

    try {
      const imageUrl = await uploadImage(imagePath, result.prompt_id, result.model)

      surveyItems.push({
        type: 'evaluation',
        imageUrl,
        prompt: result.prompt,
        attribute: result.attribute,
        attributeValue: result.attribute_value,
        model: result.model,
        domain: result.domain,
      })

      // Progress indicator
      if (surveyItems.length % 10 === 0) {
        console.log(`Processed ${surveyItems.length}/${results.length} items...`)
      }
    } catch (error) {
      console.error(`Error processing ${result.prompt_id}:`, error)
    }
  }

  console.log(`\nTotal items to upload: ${surveyItems.length}`)
  console.log('Uploading to Firestore...')

  await batchUploadItems(surveyItems)

  console.log('Upload complete!')
  console.log(`\nSummary:
  - Evaluation items: ${surveyItems.length}
  - Attention checks: ${attentionChecks.length}
  - Total: ${surveyItems.length + attentionChecks.length}
  `)
}

main().catch(console.error)
