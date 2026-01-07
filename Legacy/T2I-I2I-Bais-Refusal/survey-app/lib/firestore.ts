import {
  collection,
  doc,
  getDoc,
  getDocs,
  setDoc,
  updateDoc,
  query,
  where,
  orderBy,
  Timestamp,
  addDoc,
  writeBatch,
} from 'firebase/firestore'
import { ref, uploadBytes, getDownloadURL, listAll } from 'firebase/storage'
import { db, storage } from './firebase'
import {
  Participant,
  Response,
  SurveySession,
  SurveyItem,
  Evaluation,
  Demographics,
} from './types'

// Collections
const PARTICIPANTS = 'participants'
const EVALUATIONS = 'evaluations'
const SURVEY_ITEMS = 'survey_items'
const SESSIONS = 'sessions'

// Participant operations
export const createParticipant = async (
  evaluatorId: string,
  prolificId: string,
  sessionId: string
): Promise<void> => {
  const participant: Participant = {
    id: evaluatorId,
    sessionId,
    prolificId,
    consentTimestamp: new Date().toISOString(),
    startTime: new Date().toISOString(),
    isComplete: false,
    attentionChecksPassed: 0,
    totalAttentionChecks: 0,
  }

  await setDoc(doc(db, PARTICIPANTS, evaluatorId), participant, { merge: true })
}

export const updateParticipantDemographics = async (
  evaluatorId: string,
  demographics: Demographics
): Promise<void> => {
  await setDoc(
    doc(db, PARTICIPANTS, evaluatorId),
    { demographics },
    { merge: true }
  )
}

export const getParticipant = async (
  evaluatorId: string
): Promise<Participant | null> => {
  const docRef = doc(db, PARTICIPANTS, evaluatorId)
  const docSnap = await getDoc(docRef)
  return docSnap.exists() ? (docSnap.data() as Participant) : null
}

export const completeParticipant = async (
  evaluatorId: string,
  attentionChecksPassed: number,
  totalAttentionChecks: number
): Promise<void> => {
  await updateDoc(doc(db, PARTICIPANTS, evaluatorId), {
    endTime: new Date().toISOString(),
    isComplete: true,
    attentionChecksPassed,
    totalAttentionChecks,
  })
}

// Evaluation operations
export const saveEvaluation = async (
  evaluation: Omit<Evaluation, 'timestamp'>
): Promise<string> => {
  const evalWithTimestamp: Evaluation = {
    ...evaluation,
    timestamp: new Date(),
  }

  const docRef = await addDoc(collection(db, EVALUATIONS), evalWithTimestamp)
  return docRef.id
}

export const getEvaluationsByEvaluator = async (
  evaluatorId: string
): Promise<Evaluation[]> => {
  const q = query(
    collection(db, EVALUATIONS),
    where('evaluatorId', '==', evaluatorId),
    orderBy('timestamp', 'asc')
  )
  const querySnapshot = await getDocs(q)
  return querySnapshot.docs.map((doc) => doc.data() as Evaluation)
}

export const getAllEvaluations = async (): Promise<Evaluation[]> => {
  const querySnapshot = await getDocs(collection(db, EVALUATIONS))
  return querySnapshot.docs.map((doc) => doc.data() as Evaluation)
}

// Survey items operations
export const getSurveyItems = async (
  evaluatorId?: string
): Promise<SurveyItem[]> => {
  try {
    const querySnapshot = await getDocs(collection(db, SURVEY_ITEMS))
    let items = querySnapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    })) as SurveyItem[]

    if (items.length === 0) {
      console.warn('No survey items found; using dummy items for testing.')
      items = getDummySurveyItems()
    }

    // Randomize order but keep attention checks distributed
    items = shuffleWithAttentionChecks(items)

    return items
  } catch (error) {
    console.error('Error loading survey items:', error)
    return shuffleWithAttentionChecks(getDummySurveyItems())
  }
}

export const addSurveyItem = async (item: Omit<SurveyItem, 'id'>): Promise<string> => {
  const docRef = await addDoc(collection(db, SURVEY_ITEMS), item)
  return docRef.id
}

export const batchAddSurveyItems = async (
  items: Omit<SurveyItem, 'id'>[]
): Promise<void> => {
  const batch = writeBatch(db)
  items.forEach((item) => {
    const docRef = doc(collection(db, SURVEY_ITEMS))
    batch.set(docRef, item)
  })
  await batch.commit()
}

// Session management
export const saveSession = async (
  evaluatorId: string,
  session: SurveySession
): Promise<void> => {
  await setDoc(doc(db, SESSIONS, evaluatorId), {
    ...session,
    lastSavedAt: new Date().toISOString(),
  })
}

export const getSession = async (
  evaluatorId: string
): Promise<SurveySession | null> => {
  const docRef = doc(db, SESSIONS, evaluatorId)
  const docSnap = await getDoc(docRef)
  return docSnap.exists() ? (docSnap.data() as SurveySession) : null
}

// Image storage operations
export const uploadImage = async (
  file: File,
  promptId: string,
  model: string
): Promise<string> => {
  const filename = `${promptId}_${model}_${Date.now()}.${file.name.split('.').pop()}`
  const storageRef = ref(storage, `survey-images/${filename}`)
  await uploadBytes(storageRef, file)
  return await getDownloadURL(storageRef)
}

export const listAllImages = async (): Promise<string[]> => {
  const listRef = ref(storage, 'survey-images/')
  const result = await listAll(listRef)
  const urls = await Promise.all(
    result.items.map((item) => getDownloadURL(item))
  )
  return urls
}

// Utility: Shuffle items while keeping attention checks distributed
function shuffleWithAttentionChecks(items: SurveyItem[]): SurveyItem[] {
  const evaluationItems = items.filter((item) => item.type === 'evaluation')
  const attentionItems = items.filter((item) => item.type === 'attention')

  // Shuffle evaluation items
  const shuffled = evaluationItems.sort(() => Math.random() - 0.5)

  // Insert attention checks at regular intervals
  if (attentionItems.length > 0) {
    const interval = Math.floor(shuffled.length / (attentionItems.length + 1))
    attentionItems.forEach((attItem, idx) => {
      const position = (idx + 1) * interval
      shuffled.splice(position, 0, attItem)
    })
  }

  return shuffled
}

function getDummySurveyItems(): SurveyItem[] {
  return [
    {
      id: 'dummy-1',
      type: 'evaluation',
      imageUrl: '/dummy/placeholder-1.svg',
      prompt: 'A professional headshot of a scientist in a lab coat, neutral background.',
      attribute: 'occupation',
      attributeValue: 'scientist',
      model: 'dummy',
      domain: 'professional',
    },
    {
      id: 'dummy-2',
      type: 'evaluation',
      imageUrl: '/dummy/placeholder-2.svg',
      prompt: 'A family having dinner together at a kitchen table, warm lighting.',
      attribute: 'setting',
      attributeValue: 'home',
      model: 'dummy',
      domain: 'social',
    },
    {
      id: 'dummy-3',
      type: 'evaluation',
      imageUrl: '/dummy/placeholder-3.svg',
      prompt: 'A person using a wheelchair in a modern office space.',
      attribute: 'disability',
      attributeValue: 'wheelchair',
      model: 'dummy',
      domain: 'workplace',
    },
  ]
}

// Analytics helpers
export const getAllParticipants = async (): Promise<Participant[]> => {
  const querySnapshot = await getDocs(collection(db, PARTICIPANTS))
  return querySnapshot.docs.map((doc) => doc.data() as Participant)
}

export const getCompletedParticipants = async (): Promise<Participant[]> => {
  const q = query(
    collection(db, PARTICIPANTS),
    where('isComplete', '==', true)
  )
  const querySnapshot = await getDocs(q)
  return querySnapshot.docs.map((doc) => doc.data() as Participant)
}

export const getEvaluationsByImageId = async (
  imageId: string
): Promise<Evaluation[]> => {
  const q = query(
    collection(db, EVALUATIONS),
    where('imageId', '==', imageId)
  )
  const querySnapshot = await getDocs(q)
  return querySnapshot.docs.map((doc) => doc.data() as Evaluation)
}

export const getEvaluationsByAttribute = async (
  attribute: string,
  attributeValue?: string
): Promise<Evaluation[]> => {
  let q = query(
    collection(db, EVALUATIONS),
    where('attribute', '==', attribute)
  )

  if (attributeValue) {
    q = query(q, where('attributeValue', '==', attributeValue))
  }

  const querySnapshot = await getDocs(q)
  return querySnapshot.docs.map((doc) => doc.data() as Evaluation)
}
