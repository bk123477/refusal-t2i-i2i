import { NextRequest, NextResponse } from 'next/server'
import { readFile, readdir, stat } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

interface EvalItem {
  id: string
  sourceImage: string
  outputImage: string
  promptId: string
  promptText: string
  category: string
  race: string
  gender: string
  age: string
  model: string
  isUnchanged: boolean
}

const RESULTS_BASE = '/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/data/results'
const SOURCE_IMAGES_BASE = '/Users/chan/IJCAI26/I2I-T2I-Bias-Refusal/data/source_images/final'

async function imageToBase64(imagePath: string): Promise<string> {
  try {
    const imageBuffer = await readFile(imagePath)
    const ext = imagePath.split('.').pop()?.toLowerCase() || 'jpg'
    const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg'
    return `data:${mimeType};base64,${imageBuffer.toString('base64')}`
  } catch (e) {
    console.error(`Failed to read image: ${imagePath}`)
    return ''
  }
}

async function loadResultsFromDir(model: string, expDir: string): Promise<EvalItem[]> {
  const resultsPath = join(RESULTS_BASE, model, expDir, 'results.json')

  if (!existsSync(resultsPath)) {
    console.log(`No results.json found at ${resultsPath}`)
    return []
  }

  try {
    const resultsData = await readFile(resultsPath, 'utf-8')
    const results = JSON.parse(resultsData)

    const items: EvalItem[] = []

    for (const result of results) {
      // Skip if no output image
      if (!result.output_image || result.status === 'error') continue

      const sourceImagePath = result.source_image || join(
        SOURCE_IMAGES_BASE,
        result.race_code || result.race,
        `${result.race_code || result.race}_${result.gender}_${result.age_code || result.age}.jpg`
      )

      // Load images as base64
      const sourceImage = await imageToBase64(sourceImagePath)
      const outputImage = await imageToBase64(result.output_image)

      if (!sourceImage || !outputImage) continue

      items.push({
        id: `${model}_${result.prompt_id}_${result.race_code || result.race}_${result.gender}_${result.age_code || result.age}`,
        sourceImage,
        outputImage,
        promptId: result.prompt_id,
        promptText: result.prompt_text || result.prompt || '',
        category: result.category || result.prompt_id?.charAt(0) || 'Unknown',
        race: result.race_code || result.race || 'Unknown',
        gender: result.gender || 'Unknown',
        age: result.age_code || result.age || 'Unknown',
        model: model === 'flux' ? 'FLUX' : model === 'step1x' ? 'Step1X' : model === 'qwen' ? 'Qwen' : model,
        isUnchanged: result.is_unchanged || result.status === 'unchanged'
      })
    }

    return items
  } catch (e) {
    console.error(`Error loading results from ${resultsPath}:`, e)
    return []
  }
}

async function findLatestExperimentDir(model: string): Promise<string | null> {
  const modelDir = join(RESULTS_BASE, model)

  if (!existsSync(modelDir)) return null

  try {
    const dirs = await readdir(modelDir)
    const expDirs = dirs.filter(d => /^\d{8}_\d{6}$/.test(d))

    if (expDirs.length === 0) return null

    // Sort by date (descending) and return latest
    expDirs.sort().reverse()
    return expDirs[0]
  } catch (e) {
    return null
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const model = searchParams.get('model') || 'all'
  const limit = parseInt(searchParams.get('limit') || '0')

  const allItems: EvalItem[] = []

  // Models to load
  const models = model === 'all' ? ['flux', 'step1x', 'qwen'] : [model.toLowerCase()]

  for (const m of models) {
    const expDir = await findLatestExperimentDir(m)
    if (expDir) {
      console.log(`Loading results for ${m} from ${expDir}`)
      const items = await loadResultsFromDir(m, expDir)
      allItems.push(...items)
    }
  }

  // Shuffle for fair evaluation order
  const shuffled = allItems.sort(() => Math.random() - 0.5)

  // Apply limit if specified
  const limited = limit > 0 ? shuffled.slice(0, limit) : shuffled

  return NextResponse.json({
    total: allItems.length,
    returned: limited.length,
    items: limited
  })
}
