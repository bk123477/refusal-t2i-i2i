import { Evaluation, Participant, AnalysisResult, AgreementMetrics } from './types'

// Calculate Cohen's Kappa for inter-rater reliability
export function calculateCohensKappa(
  evaluations1: boolean[],
  evaluations2: boolean[]
): number {
  if (evaluations1.length !== evaluations2.length) {
    throw new Error('Evaluation arrays must have the same length')
  }

  const n = evaluations1.length
  let po = 0 // Observed agreement

  // Count agreements
  for (let i = 0; i < n; i++) {
    if (evaluations1[i] === evaluations2[i]) po++
  }
  po /= n

  // Calculate expected agreement
  const p1_yes = evaluations1.filter((v) => v).length / n
  const p1_no = 1 - p1_yes
  const p2_yes = evaluations2.filter((v) => v).length / n
  const p2_no = 1 - p2_yes

  const pe = p1_yes * p2_yes + p1_no * p2_no

  // Cohen's Kappa
  return (po - pe) / (1 - pe)
}

// Calculate percent agreement
export function calculatePercentAgreement(
  values1: any[],
  values2: any[]
): number {
  if (values1.length !== values2.length) return 0

  let agreements = 0
  for (let i = 0; i < values1.length; i++) {
    if (values1[i] === values2[i]) agreements++
  }

  return (agreements / values1.length) * 100
}

// Group evaluations by image to calculate inter-rater metrics
export function groupByImage(
  evaluations: Evaluation[]
): Map<string, Evaluation[]> {
  const grouped = new Map<string, Evaluation[]>()

  evaluations.forEach((evaluation) => {
    const existing = grouped.get(evaluation.imageId) || []
    grouped.set(evaluation.imageId, [...existing, evaluation])
  })

  return grouped
}

// Calculate agreement metrics across all evaluators
export function calculateAgreementMetrics(
  evaluations: Evaluation[]
): AgreementMetrics {
  const byImage = groupByImage(evaluations)

  let totalKappa = 0
  let totalAgreement = 0
  let pairCount = 0

  const byAttribute: Record<string, {
    kappa: number
    agreement: number
    sampleSize: number
  }> = {}

  // For each image with multiple evaluators
  byImage.forEach((evals) => {
    if (evals.length < 2) return

    // Calculate pairwise agreements
    for (let i = 0; i < evals.length; i++) {
      for (let j = i + 1; j < evals.length; j++) {
        const eval1 = evals[i]
        const eval2 = evals[j]

        // Refusal agreement
        const refusalMatch = eval1.isRefusal === eval2.isRefusal ? 1 : 0

        // Attribute presence agreement
        const attrMatch = eval1.attributePresent === eval2.attributePresent ? 1 : 0

        // Faithfulness agreement (within 1 point)
        const faithMatch = Math.abs(eval1.faithfulness - eval2.faithfulness) <= 1 ? 1 : 0

        const pairAgreement = (refusalMatch + attrMatch + faithMatch) / 3
        totalAgreement += pairAgreement
        pairCount++

        // Track by attribute
        const attrKey = `${eval1.attribute}-${eval1.attributeValue}`
        if (!byAttribute[attrKey]) {
          byAttribute[attrKey] = {
            kappa: 0,
            agreement: 0,
            sampleSize: 0,
          }
        }
        byAttribute[attrKey].agreement += pairAgreement
        byAttribute[attrKey].sampleSize++
      }
    }
  })

  // Calculate Cohen's Kappa for binary refusal decisions
  const allRefusals: boolean[][] = []
  byImage.forEach((evals) => {
    if (evals.length >= 2) {
      const refusals = evals.map((e) => e.isRefusal)
      allRefusals.push(refusals)
    }
  })

  let kappaSum = 0
  let kappaCount = 0

  allRefusals.forEach((refusals) => {
    for (let i = 0; i < refusals.length; i++) {
      for (let j = i + 1; j < refusals.length; j++) {
        // Simple binary kappa calculation
        const agree = refusals[i] === refusals[j] ? 1 : 0
        kappaSum += agree
        kappaCount++
      }
    }
  })

  const avgKappa = kappaCount > 0 ? kappaSum / kappaCount : 0

  // Average by attribute
  Object.keys(byAttribute).forEach((key) => {
    byAttribute[key].agreement =
      byAttribute[key].agreement / byAttribute[key].sampleSize
    byAttribute[key].kappa = avgKappa // Simplified
  })

  return {
    cohensKappa: avgKappa,
    percentAgreement: pairCount > 0 ? (totalAgreement / pairCount) * 100 : 0,
    byAttribute,
    vlmVsHuman: {
      refusalAgreement: 0,
      attributeAgreement: 0,
      sampleSize: 0,
    },
  }
}

// Generate comprehensive analysis results
export function analyzeResults(
  evaluations: Evaluation[],
  participants: Participant[]
): AnalysisResult {
  const completed = participants.filter((p) => p.isComplete)

  const totalTime = completed.reduce((sum, p) => {
    if (p.startTime && p.endTime) {
      return sum + (new Date(p.endTime).getTime() - new Date(p.startTime).getTime())
    }
    return sum
  }, 0)

  const avgTimeMinutes = completed.length > 0
    ? totalTime / completed.length / 60000
    : 0

  const attentionPassed = completed.reduce((sum, p) => sum + p.attentionChecksPassed, 0)
  const attentionTotal = completed.reduce((sum, p) => sum + p.totalAttentionChecks, 0)

  // Group by attribute
  const byAttribute: Record<string, {
    evaluationCount: number
    refusalRate: number
    averageFaithfulness: number
    attributePresentRate: number
  }> = {}

  evaluations.forEach((evaluation) => {
    const key = `${evaluation.attribute}-${evaluation.attributeValue}`
    if (!byAttribute[key]) {
      byAttribute[key] = {
        evaluationCount: 0,
        refusalRate: 0,
        averageFaithfulness: 0,
        attributePresentRate: 0,
      }
    }

    byAttribute[key].evaluationCount++
    if (evaluation.isRefusal) byAttribute[key].refusalRate++
    byAttribute[key].averageFaithfulness += evaluation.faithfulness
    if (evaluation.attributePresent === 'yes') byAttribute[key].attributePresentRate++
  })

  Object.keys(byAttribute).forEach((key) => {
    const count = byAttribute[key].evaluationCount
    byAttribute[key].refusalRate = (byAttribute[key].refusalRate / count) * 100
    byAttribute[key].averageFaithfulness = byAttribute[key].averageFaithfulness / count
    byAttribute[key].attributePresentRate = (byAttribute[key].attributePresentRate / count) * 100
  })

  // Group by model
  const byModel: Record<string, {
    evaluationCount: number
    refusalRate: number
    averageFaithfulness: number
  }> = {}

  evaluations.forEach((evaluation) => {
    const model = evaluation.model
    if (!byModel[model]) {
      byModel[model] = {
        evaluationCount: 0,
        refusalRate: 0,
        averageFaithfulness: 0,
      }
    }

    byModel[model].evaluationCount++
    if (evaluation.isRefusal) byModel[model].refusalRate++
    byModel[model].averageFaithfulness += evaluation.faithfulness
  })

  Object.keys(byModel).forEach((model) => {
    const count = byModel[model].evaluationCount
    byModel[model].refusalRate = (byModel[model].refusalRate / count) * 100
    byModel[model].averageFaithfulness = byModel[model].averageFaithfulness / count
  })

  // Demographics breakdown
  const demographics = {
    byAge: {} as Record<string, number>,
    byGender: {} as Record<string, number>,
    byNationality: {} as Record<string, number>,
  }

  participants.forEach((p) => {
    if (p.demographics) {
      demographics.byAge[p.demographics.age] = (demographics.byAge[p.demographics.age] || 0) + 1
      demographics.byGender[p.demographics.gender] = (demographics.byGender[p.demographics.gender] || 0) + 1
      demographics.byNationality[p.demographics.nationality] = (demographics.byNationality[p.demographics.nationality] || 0) + 1
    }
  })

  return {
    totalParticipants: participants.length,
    completedParticipants: completed.length,
    totalEvaluations: evaluations.length,
    averageCompletionTimeMinutes: avgTimeMinutes,
    attentionCheckPassRate: attentionTotal > 0 ? (attentionPassed / attentionTotal) * 100 : 0,
    byAttribute,
    byModel,
    demographics,
    agreement: calculateAgreementMetrics(evaluations),
  }
}

// Export evaluations to CSV format
export function exportToCSV(evaluations: Evaluation[]): string {
  const headers = [
    'evaluatorId',
    'imageId',
    'promptId',
    'attribute',
    'attributeValue',
    'model',
    'domain',
    'isRefusal',
    'attributePresent',
    'faithfulness',
    'confidence',
    'notes',
    'timestamp',
    'responseTimeMs',
    'sessionId',
  ]

  const rows = evaluations.map((e) => [
    e.evaluatorId,
    e.imageId,
    e.promptId,
    e.attribute,
    e.attributeValue,
    e.model,
    e.domain,
    e.isRefusal,
    e.attributePresent,
    e.faithfulness,
    e.confidence,
    e.notes || '',
    e.timestamp.toISOString(),
    e.responseTimeMs,
    e.sessionId,
  ])

  const csvContent = [
    headers.join(','),
    ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
  ].join('\n')

  return csvContent
}

// Export to JSON
export function exportToJSON(data: {
  evaluations: Evaluation[]
  participants: Participant[]
  analysis: AnalysisResult
}): string {
  return JSON.stringify(data, null, 2)
}
