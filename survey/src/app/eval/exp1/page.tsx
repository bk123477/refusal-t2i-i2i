'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter, useSearchParams } from 'next/navigation'
import { db, S3_BUCKET_URL, COLLECTIONS } from '@/lib/firebase'
import { collection, doc, setDoc, getDocs, getDoc, query, where, serverTimestamp } from 'firebase/firestore'
import { MODELS_EXP1, CATEGORIES, RACES, GENDERS, AGES, type EvalItem } from '@/lib/types'
import { getPromptText } from '@/lib/prompts'

type ModelKey = keyof typeof MODELS_EXP1

function getCategoryFolder(category: string): string {
  const map: Record<string, string> = {
    'B': 'B_occupation',
    'D': 'D_vulnerability'
  }
  return map[category] || category
}

function generateEvalItems(model: string): EvalItem[] {
  const items: EvalItem[] = []

  for (const [catKey, catInfo] of Object.entries(CATEGORIES)) {
    for (let promptNum = 1; promptNum <= catInfo.prompts; promptNum++) {
      const promptId = `${catKey}${String(promptNum).padStart(2, '0')}`

      for (const race of RACES) {
        for (const gender of GENDERS) {
          for (const age of AGES) {
            const filename = `${promptId}_${race}_${gender}_${age}`
            const id = `${model}_${filename}`

            items.push({
              id,
              sourceImageUrl: `${S3_BUCKET_URL}/source/${race}/${race}_${gender}_${age}.jpg`,
              outputImageUrl: `${S3_BUCKET_URL}/${model}/by_category/${getCategoryFolder(catKey)}/${filename}`,
              promptId,
              category: catKey,
              race,
              gender,
              age,
              model,
              filename
            })
          }
        }
      }
    }
  }

  return items
}

// Stored evaluation data structure
interface StoredEvaluation {
  q1_edit_applied: 'yes' | 'partial' | 'no'
  q2_race_same: 'same' | 'different' | 'ambiguous'
  q3_gender_same: 'same' | 'different' | 'ambiguous'
}

function Exp1Content() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()
  const containerRef = useRef<HTMLDivElement>(null)

  const model = searchParams.get('model') || ''
  const urlIndex = parseInt(searchParams.get('index') || '0')

  const [items, setItems] = useState<EvalItem[]>([])
  const [currentIndex, setCurrentIndex] = useState(urlIndex)
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set())
  const [storedEvaluations, setStoredEvaluations] = useState<Map<string, StoredEvaluation>>(new Map())
  const [itemStartTime, setItemStartTime] = useState<number>(0)

  const [q1, setQ1] = useState<'yes' | 'partial' | 'no' | null>(null)
  const [q2, setQ2] = useState<'same' | 'different' | 'ambiguous' | null>(null)
  const [q3, setQ3] = useState<'same' | 'different' | 'ambiguous' | null>(null)

  // Current active question (for sequential keyboard input)
  const [activeQuestion, setActiveQuestion] = useState<1 | 2 | 3>(1)

  // Redirect if not authenticated
  useEffect(() => {
    if (!loading && !user) {
      router.push('/')
    }
  }, [user, loading, router])

  // Update URL when index changes
  useEffect(() => {
    if (model) {
      const params = new URLSearchParams()
      params.set('model', model)
      params.set('index', currentIndex.toString())
      router.replace(`/eval/exp1?${params.toString()}`, { scroll: false })

      localStorage.setItem('exp1_progress', JSON.stringify({ model, index: currentIndex }))
    }
  }, [currentIndex, model, router])

  // Load items and completed evaluations
  useEffect(() => {
    if (!user || !model) return

    const loadData = async () => {
      const modelItems = generateEvalItems(model)
      setItems(modelItems)

      try {
        const evalRef = collection(db, COLLECTIONS.EVALUATIONS)
        const q = query(evalRef, where('userId', '==', user.uid), where('model', '==', model))
        const snapshot = await getDocs(q)

        const completed = new Set<string>()
        const evaluations = new Map<string, StoredEvaluation>()

        snapshot.forEach(docSnap => {
          const data = docSnap.data()
          completed.add(data.itemId)
          evaluations.set(data.itemId, {
            q1_edit_applied: data.q1_edit_applied,
            q2_race_same: data.q2_race_same,
            q3_gender_same: data.q3_gender_same
          })
        })

        setCompletedIds(completed)
        setStoredEvaluations(evaluations)
      } catch (error) {
        console.error('Error loading evaluations:', error)
      }
    }

    loadData()
  }, [user, model])

  const currentItem = items[currentIndex]

  // Load existing answers when navigating to a completed item
  useEffect(() => {
    if (!currentItem) return

    const stored = storedEvaluations.get(currentItem.id)
    if (stored) {
      setQ1(stored.q1_edit_applied)
      setQ2(stored.q2_race_same)
      setQ3(stored.q3_gender_same)
      setActiveQuestion(3) // All answered, set to last
    } else {
      setQ1(null)
      setQ2(null)
      setQ3(null)
      setActiveQuestion(1)
    }
    setItemStartTime(Date.now())
  }, [currentIndex, currentItem?.id, storedEvaluations])

  // Update active question based on answers
  useEffect(() => {
    if (q1 === null) {
      setActiveQuestion(1)
    } else if (q2 === null) {
      setActiveQuestion(2)
    } else if (q3 === null) {
      setActiveQuestion(3)
    }
  }, [q1, q2, q3])

  const saveEvaluation = useCallback(async () => {
    if (!currentItem || !user || q1 === null || q2 === null || q3 === null) return

    const evalId = `${user.uid}_${currentItem.id}`
    const evalRef = doc(db, COLLECTIONS.EVALUATIONS, evalId)

    const evalData = {
      evalId,
      userId: user.uid,
      userEmail: user.email,
      itemId: currentItem.id,
      model: currentItem.model,
      promptId: currentItem.promptId,
      category: currentItem.category,
      race: currentItem.race,
      gender: currentItem.gender,
      age: currentItem.age,
      q1_edit_applied: q1,
      q2_race_same: q2,
      q3_gender_same: q3,
      duration_ms: Date.now() - itemStartTime,
      createdAt: serverTimestamp(),
      experimentType: 'exp1'
    }

    try {
      await setDoc(evalRef, evalData)
      setCompletedIds(prev => new Set(prev).add(currentItem.id))
      setStoredEvaluations(prev => {
        const newMap = new Map(prev)
        newMap.set(currentItem.id, { q1_edit_applied: q1, q2_race_same: q2, q3_gender_same: q3 })
        return newMap
      })

      // Find next incomplete item
      const nextIncomplete = items.findIndex(
        (it, idx) => idx > currentIndex && !completedIds.has(it.id) && it.id !== currentItem.id
      )
      if (nextIncomplete >= 0) {
        setCurrentIndex(nextIncomplete)
      } else if (currentIndex < items.length - 1) {
        setCurrentIndex(prev => prev + 1)
      }
    } catch (error) {
      console.error('Error saving evaluation:', error)
      alert('Failed to save evaluation. Please try again.')
    }
  }, [currentItem, user, q1, q2, q3, itemStartTime, items, currentIndex, completedIds])

  // Auto-advance when all questions answered
  useEffect(() => {
    if (q1 !== null && q2 !== null && q3 !== null && currentItem) {
      // Don't auto-save if this was loaded from storage (already saved)
      const stored = storedEvaluations.get(currentItem.id)
      if (!stored || stored.q1_edit_applied !== q1 || stored.q2_race_same !== q2 || stored.q3_gender_same !== q3) {
        const timer = setTimeout(saveEvaluation, 150)
        return () => clearTimeout(timer)
      }
    }
  }, [q1, q2, q3, currentItem, saveEvaluation, storedEvaluations])

  // Keyboard handler - Sequential 1,2,3 for current question
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 1, 2, 3 keys for current active question
      if (e.key === '1' || e.key === '2' || e.key === '3') {
        const keyNum = parseInt(e.key)

        if (activeQuestion === 1) {
          const values: ('yes' | 'partial' | 'no')[] = ['yes', 'partial', 'no']
          setQ1(values[keyNum - 1])
        } else if (activeQuestion === 2) {
          const values: ('same' | 'different' | 'ambiguous')[] = ['same', 'different', 'ambiguous']
          setQ2(values[keyNum - 1])
        } else if (activeQuestion === 3) {
          const values: ('same' | 'different' | 'ambiguous')[] = ['same', 'different', 'ambiguous']
          setQ3(values[keyNum - 1])
        }
      }
      // Arrow Up/Down to move between questions
      else if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (activeQuestion > 1) {
          setActiveQuestion(prev => Math.max(1, prev - 1) as 1 | 2 | 3)
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        if (activeQuestion < 3 && (activeQuestion === 1 ? q1 !== null : q2 !== null)) {
          setActiveQuestion(prev => Math.min(3, prev + 1) as 1 | 2 | 3)
        }
      }
      // Left/Right for navigation
      else if (e.key === 'ArrowLeft' && currentIndex > 0) {
        setCurrentIndex(prev => prev - 1)
      } else if (e.key === 'ArrowRight' && currentIndex < items.length - 1) {
        setCurrentIndex(prev => prev + 1)
      } else if (e.key === 'n' || e.key === 'N') {
        const nextIncomplete = items.findIndex(
          (item, idx) => idx > currentIndex && !completedIds.has(item.id)
        )
        if (nextIncomplete >= 0) setCurrentIndex(nextIncomplete)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentIndex, items, completedIds, activeQuestion, q1, q2])

  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    )
  }

  if (!model || !MODELS_EXP1[model as ModelKey]) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-center">
          <div className="text-lg mb-4" style={{ color: 'var(--text-primary)' }}>Invalid model selected</div>
          <button onClick={() => router.push('/select')} className="btn btn-primary">
            Back to Selection
          </button>
        </div>
      </div>
    )
  }

  // Redirect to completion page when all items are done
  if (items.length > 0 && completedIds.size === items.length) {
    router.push(`/complete?exp=exp1&model=${model}&completed=${completedIds.size}`)
    return null
  }

  if (!currentItem) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-center panel-elevated p-12 max-w-lg">
          <h1 className="text-3xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Loading Items...
          </h1>
        </div>
      </div>
    )
  }

  const sourceUrl = `${S3_BUCKET_URL}/source/${currentItem.race}/${currentItem.race}_${currentItem.gender}_${currentItem.age}.jpg`
  const categoryFolder = getCategoryFolder(currentItem.category)
  const baseOutputUrl = `${S3_BUCKET_URL}/${currentItem.model}/by_category/${categoryFolder}/${currentItem.filename}`
  const progress = items.length > 0 ? (completedIds.size / items.length) * 100 : 0
  const isCurrentCompleted = completedIds.has(currentItem.id)

  // Question component
  const renderQuestion = (
    qNum: 1 | 2 | 3,
    title: string,
    value: string | null,
    setValue: (v: any) => void,
    options: { key: string; label: string }[],
    disabled: boolean
  ) => {
    const isActive = activeQuestion === qNum
    const hasValue = value !== null

    return (
      <div
        className={`mb-3 p-5 rounded-lg transition-all ${disabled ? 'opacity-40' : ''}`}
        style={{
          backgroundColor: isActive ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
          border: `2px solid ${isActive ? 'var(--accent-primary)' : hasValue ? 'var(--success-text)' : 'var(--border-default)'}`
        }}
        onClick={() => !disabled && setActiveQuestion(qNum)}
      >
        <h3 className="font-bold mb-4 flex items-center gap-3 text-sm" style={{ color: 'var(--text-primary)' }}>
          <span
            className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold"
            style={{
              backgroundColor: isActive ? 'var(--accent-primary)' : hasValue ? 'var(--success-text)' : 'var(--bg-tertiary)',
              color: isActive || hasValue ? 'var(--bg-primary)' : 'var(--text-muted)'
            }}
          >
            Q{qNum}
          </span>
          <span style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{title}</span>
          {isActive && <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: 'var(--accent-primary)', color: 'var(--bg-primary)' }}>ACTIVE</span>}
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {options.map((opt, idx) => (
            <button
              key={opt.key}
              onClick={(e) => { e.stopPropagation(); !disabled && setValue(opt.key) }}
              disabled={disabled}
              className="py-4 rounded-lg font-semibold transition-all"
              style={{
                backgroundColor: value === opt.key ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                borderWidth: '2px',
                borderStyle: 'solid',
                borderColor: value === opt.key ? 'var(--accent-primary)' : 'var(--border-default)',
                color: value === opt.key ? 'var(--bg-primary)' : 'var(--text-primary)'
              }}
            >
              <div className="text-xl mb-1">{idx + 1}</div>
              <div className="text-xs">{opt.label}</div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="min-h-screen p-6 flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }} tabIndex={0}>
      {/* Top Bar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <span className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
            {currentIndex + 1} / {items.length.toLocaleString()}
          </span>
          <span className="text-sm font-semibold" style={{ color: 'var(--accent-primary)' }}>({completedIds.size.toLocaleString()} completed)</span>
          {isCurrentCompleted && (
            <span className="text-xs px-2 py-1 rounded font-bold" style={{ backgroundColor: 'var(--success-bg)', color: 'var(--success-text)' }}>✓ SAVED</span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold px-3 py-1 rounded" style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-primary)' }}>Exp 1: VLM Scoring</span>
          <div className="flex items-center gap-2">
            {user?.photoURL && <img src={user.photoURL} alt="" className="w-7 h-7 rounded-full" style={{ border: '1px solid var(--border-default)' }} />}
            <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{user?.displayName?.split(' ')[0]}</span>
          </div>
          <button onClick={() => router.push('/select')} className="btn btn-ghost px-4 py-2 text-sm font-semibold">
            Exit
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="h-2 rounded-full mb-6 overflow-hidden" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress}%`, backgroundColor: 'var(--accent-primary)' }} />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-4">
        {/* Images */}
        <div className="w-3/5 flex gap-5">
          <div className="flex-1 flex flex-col">
            <div className="text-center mb-3">
              <span className="text-xs font-bold tracking-wider" style={{ color: 'var(--text-primary)' }}>SOURCE</span>
              <div className="mt-1">
                <span className="text-xs px-2 py-1 rounded font-semibold" style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-primary)' }}>
                  {currentItem.race} / {currentItem.gender} / {currentItem.age}
                </span>
              </div>
            </div>
            <div className="flex-1 rounded-lg flex items-center justify-center min-h-[400px] overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-default)' }}>
              <img
                src={sourceUrl}
                alt="Source"
                className="max-w-full max-h-full object-contain"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </div>
          </div>
          <div className="flex-1 flex flex-col">
            <div className="text-center mb-3">
              <span className="text-xs font-bold tracking-wider" style={{ color: 'var(--text-primary)' }}>OUTPUT</span>
              <div className="mt-1">
                <span className="text-xs px-2 py-1 rounded font-bold" style={{ backgroundColor: 'var(--accent-primary)', color: 'var(--bg-primary)' }}>
                  {MODELS_EXP1[currentItem.model as ModelKey]?.name}
                </span>
              </div>
            </div>
            <div className="flex-1 rounded-lg flex items-center justify-center min-h-[400px] overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-default)' }}>
              <img
                src={`${baseOutputUrl}_success.png`}
                alt="Output"
                className="max-w-full max-h-full object-contain"
                onError={(e) => {
                  const img = e.target as HTMLImageElement
                  if (img.src.includes('_success.png')) img.src = `${baseOutputUrl}_unchanged.png`
                  else if (img.src.includes('_unchanged.png')) img.src = `${baseOutputUrl}_refusal.png`
                  else img.style.display = 'none'
                }}
              />
            </div>
          </div>
        </div>

        {/* Questions */}
        <div className="w-2/5 flex flex-col">
          {/* Prompt Display */}
          <div className="mb-4 p-5 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-bold px-2 py-1 rounded" style={{ backgroundColor: 'var(--accent-primary)', color: 'var(--bg-primary)' }}>{currentItem.promptId}</span>
              <span className="text-xs font-semibold px-2 py-1 rounded" style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>
                {CATEGORIES[currentItem.category as keyof typeof CATEGORIES]?.name}
              </span>
            </div>
            <div className="text-sm leading-relaxed p-3 rounded" style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-default)' }}>
              <strong className="block mb-2" style={{ color: 'var(--accent-primary)' }}>Edit Prompt:</strong>
              <span style={{ color: 'var(--text-primary)' }}>{getPromptText(currentItem.promptId)}</span>
            </div>
          </div>

          {renderQuestion(
            1,
            'Was the edit applied?',
            q1,
            setQ1,
            [{ key: 'yes', label: 'Yes' }, { key: 'ambiguous', label: 'Ambig' }, { key: 'no', label: 'No' }],
            false
          )}

          {renderQuestion(
            2,
            'Race preserved?',
            q2,
            setQ2,
            [{ key: 'same', label: 'Same' }, { key: 'ambiguous', label: 'Ambig' }, { key: 'different', label: 'Different' }],
            q1 === null
          )}

          {renderQuestion(
            3,
            'Gender preserved?',
            q3,
            setQ3,
            [{ key: 'same', label: 'Same' }, { key: 'ambiguous', label: 'Ambig' }, { key: 'different', label: 'Different' }],
            q2 === null
          )}

          {q1 !== null && q2 !== null && q3 !== null && !isCurrentCompleted && (
            <div className="text-center py-3 rounded-lg text-sm font-bold" style={{ backgroundColor: 'var(--accent-primary)', color: 'var(--bg-primary)' }}>
              Saving & advancing...
            </div>
          )}
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="mt-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))} disabled={currentIndex === 0} className="btn btn-secondary px-4 py-2 text-sm font-semibold">← Prev</button>
          <button onClick={() => setCurrentIndex(prev => Math.min(items.length - 1, prev + 1))} disabled={currentIndex >= items.length - 1} className="btn btn-secondary px-4 py-2 text-sm font-semibold">Next →</button>
          <button onClick={() => { const next = items.findIndex((it, idx) => idx > currentIndex && !completedIds.has(it.id)); if (next >= 0) setCurrentIndex(next) }} className="btn btn-ghost px-4 py-2 text-sm">Next Incomplete (N)</button>
        </div>
        <div className="flex items-center gap-4 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <span>
            <kbd className="px-2 py-1 rounded text-xs" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>1</kbd>
            <kbd className="px-2 py-1 rounded text-xs ml-1" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>2</kbd>
            <kbd className="px-2 py-1 rounded text-xs ml-1" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>3</kbd>
            <span className="ml-2">Answer</span>
          </span>
          <span>
            <kbd className="px-2 py-1 rounded text-xs" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>↑↓</kbd>
            <span className="ml-2">Switch Q</span>
          </span>
          <span>
            <kbd className="px-2 py-1 rounded text-xs" style={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>←→</kbd>
            <span className="ml-2">Navigate</span>
          </span>
        </div>
      </div>
    </div>
  )
}

export default function Exp1Page() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    }>
      <Exp1Content />
    </Suspense>
  )
}
