// ACRB-specific evaluation data structure
export interface Evaluation {
  evaluatorId: string
  imageId: string
  promptId: string
  attribute: string  // culture, disability, religion, etc.
  attributeValue: string  // Nigerian, wheelchair, hijab, etc.
  model: string
  domain: string

  // Responses
  isRefusal: boolean
  attributePresent: 'yes' | 'no' | 'partial'
  faithfulness: 1 | 2 | 3 | 4 | 5
  confidence: 'low' | 'medium' | 'high'

  // Optional notes
  notes?: string

  // Metadata
  timestamp: Date
  responseTimeMs: number
  sessionId: string
}

export interface SurveyItem {
  id: string
  type: 'evaluation' | 'attention'
  imageUrl: string
  prompt: string
  attribute?: string
  attributeValue?: string
  model?: string
  domain?: string

  // For attention checks
  correctAnswer?: {
    isRefusal?: boolean
    attributePresent?: 'yes' | 'no' | 'partial'
    faithfulness?: 1 | 2 | 3 | 4 | 5
  }
}

export interface Response {
  itemId: string
  type: string
  isRefusal?: boolean
  attributePresent?: 'yes' | 'no' | 'partial'
  faithfulness?: 1 | 2 | 3 | 4 | 5
  confidence?: 'low' | 'medium' | 'high'
  notes?: string
  responseTimeMs: number
  timestamp: string
}

export interface Demographics {
  age: string
  gender: string
  nationality: string
  ethnicity: string
  aiExperience: string
  imageGenerationFamiliarity: string
}

export interface Participant {
  id: string  // Firebase auth UID
  sessionId: string
  prolificId: string
  demographics?: Demographics
  consentTimestamp: string
  startTime: string
  endTime?: string
  isComplete: boolean
  attentionChecksPassed: number
  totalAttentionChecks: number
}

export interface SurveySession {
  participant: Participant
  responses: Response[]
  currentIndex: number
  lastSavedAt: string
}

// Admin dashboard types
export interface AgreementMetrics {
  cohensKappa: number
  percentAgreement: number
  byAttribute: Record<string, {
    kappa: number
    agreement: number
    sampleSize: number
  }>
  vlmVsHuman: {
    refusalAgreement: number
    attributeAgreement: number
    sampleSize: number
  }
}

export interface AnalysisResult {
  totalParticipants: number
  completedParticipants: number
  totalEvaluations: number
  averageCompletionTimeMinutes: number
  attentionCheckPassRate: number

  byAttribute: Record<string, {
    evaluationCount: number
    refusalRate: number
    averageFaithfulness: number
    attributePresentRate: number
  }>

  byModel: Record<string, {
    evaluationCount: number
    refusalRate: number
    averageFaithfulness: number
  }>

  demographics: {
    byAge: Record<string, number>
    byGender: Record<string, number>
    byNationality: Record<string, number>
  }

  agreement?: AgreementMetrics
}

// Image upload interface for admin
export interface ImageUpload {
  promptId: string
  prompt: string
  attribute: string
  attributeValue: string
  model: string
  domain: string
  file: File
  isRefusal?: boolean
}
