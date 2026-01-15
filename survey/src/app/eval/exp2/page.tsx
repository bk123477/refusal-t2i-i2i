'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter, useSearchParams } from 'next/navigation'
import { db } from '@/lib/firebase'
import { collection, doc, setDoc, getDocs, query, where, serverTimestamp } from 'firebase/firestore'
import type { PairwiseItem } from '@/lib/types'
import { MODELS_EXP2 } from '@/lib/types'
import { getPromptText } from '@/lib/prompts'

// Load pairwise items from JSON file
async function loadPairwiseItems(model: string): Promise<PairwiseItem[]> {
  try {
    console.log('Loading exp2_items.json for model:', model)
    const response = await fetch('/data/exp2_items.json')
    if (!response.ok) {
      console.error('Failed to load exp2_items.json:', response.status, response.statusText)
      throw new Error(`Failed to load exp2_items.json: ${response.status} ${response.statusText}`)
    }
    const data = await response.json()
    console.log('Loaded JSON data:', { totalItems: data.items?.length, model })
    
    // Filter items for the specified model
    const modelItems = data.items.filter((item: any) => item.model === model)
    console.log(`Filtered items for ${model}:`, modelItems.length)
    
    if (modelItems.length === 0) {
      console.warn(`No items found for model: ${model}`)
    }
    
    // Map to PairwiseItem format
    const mappedItems = modelItems.map((item: any) => ({
      id: item.id,
      model: item.model,
      promptId: item.promptId,
      category: item.category,
      categoryName: item.categoryName,
      race: item.race,
      gender: item.gender,
      age: item.age,
      sourceImageUrl: item.sourceImageUrl,
      preservedImageUrl: item.preservedImageUrl,
      editedImageUrl: item.editedImageUrl || null,
      hasEditedPair: item.hasEditedPair || false
    }))
    
    // Log first item for debugging
    if (mappedItems.length > 0) {
      console.log('First item URLs:', {
        source: mappedItems[0].sourceImageUrl,
        preserved: mappedItems[0].preservedImageUrl,
        edited: mappedItems[0].editedImageUrl
      })
    }
    
    return mappedItems
  } catch (error) {
    console.error('Error loading pairwise items:', error)
    return []
  }
}

function Exp2Content() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()
  const containerRef = useRef<HTMLDivElement>(null)

  const model = searchParams.get('model') || 'step1x'
  const urlIndex = parseInt(searchParams.get('index') || '0')

  const [items, setItems] = useState<PairwiseItem[]>([])
  const [loadingItems, setLoadingItems] = useState(true)
  const [itemsError, setItemsError] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(urlIndex)
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set())
  const [itemStartTime, setItemStartTime] = useState<number>(0)

  // Simple A/B choice: 'A' = preserved, 'B' = edited, 'tie' = equal
  const [choice, setChoice] = useState<'A' | 'B' | 'tie' | null>(null)

  // Randomize which side shows which (to avoid position bias)
  const [leftIsPreserved, setLeftIsPreserved] = useState(true)

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
      router.replace(`/eval/exp2?${params.toString()}`, { scroll: false })
      localStorage.setItem('exp2_progress', JSON.stringify({ model, index: currentIndex }))
    }
  }, [currentIndex, model, router])

  // Load items and completed evaluations
  useEffect(() => {
    if (!user || !model) return

    async function loadItems() {
      setLoadingItems(true)
      setItemsError(null)
      try {
        const loadedItems = await loadPairwiseItems(model)
        if (loadedItems.length === 0) {
          setItemsError(`No items found for model: ${model}. Please check exp2_items.json`)
        } else {
          setItems(loadedItems)
        }
      } catch (error: any) {
        console.error('Failed to load items:', error)
        setItemsError(`Failed to load items: ${error.message}`)
      } finally {
        setLoadingItems(false)
      }
    }
    loadItems()

    // Load completed
    async function loadCompleted() {
      if (!user) return
      
      try {
        const q = query(
          collection(db, 'pairwise_evaluations'),
          where('userId', '==', user.uid),
          where('model', '==', model)
        )
        const snapshot = await getDocs(q)
        const completed = new Set<string>()
        snapshot.forEach(doc => completed.add(doc.data().itemId))
        setCompletedIds(completed)
      } catch (err) {
        console.error('Error loading completed:', err)
      }
    }
    loadCompleted()
  }, [user, model])

  // Reset choice and randomize sides when item changes
  useEffect(() => {
    setChoice(null)
    setItemStartTime(Date.now())
    setLeftIsPreserved(Math.random() > 0.5)
  }, [currentIndex])

  const currentItem = items[currentIndex]

  // Save evaluation
  const saveEvaluation = useCallback(async () => {
    if (!user || !currentItem || choice === null) return

    const evalId = `${user.uid}_${currentItem.id}`
    const evalRef = doc(db, 'pairwise_evaluations', evalId)

    // Convert choice back to preserved/edited
    let preference: 'preserved' | 'edited' | 'tie'
    if (choice === 'tie') {
      preference = 'tie'
    } else if ((choice === 'A' && leftIsPreserved) || (choice === 'B' && !leftIsPreserved)) {
      preference = 'preserved'
    } else {
      preference = 'edited'
    }

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
      preference,
      leftWasPreserved: leftIsPreserved,
      rawChoice: choice,
      duration_ms: Date.now() - itemStartTime,
      createdAt: serverTimestamp(),
      experimentType: 'exp2_pairwise'
    }

    try {
      await setDoc(evalRef, evalData)
      setCompletedIds(prev => new Set(prev).add(currentItem.id))

      // Auto advance to next incomplete
      const nextIncomplete = items.findIndex(
        (it, idx) => idx > currentIndex && !completedIds.has(it.id) && it.id !== currentItem.id
      )
      if (nextIncomplete >= 0) {
        setCurrentIndex(nextIncomplete)
      } else if (currentIndex < items.length - 1) {
        setCurrentIndex(prev => prev + 1)
      }
    } catch (error) {
      console.error('Error saving:', error)
      alert('Failed to save. Please try again.')
    }
  }, [currentItem, user, choice, leftIsPreserved, itemStartTime, items, currentIndex, completedIds])

  // Auto-save when choice made
  useEffect(() => {
    if (choice !== null && currentItem) {
      const timer = setTimeout(saveEvaluation, 200)
      return () => clearTimeout(timer)
    }
  }, [choice, currentItem, saveEvaluation])

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (key === 'a' || key === '1') setChoice('A')
      else if (key === 'b' || key === '2') setChoice('B')
      else if (key === 't' || key === '3' || key === '=') setChoice('tie')
      else if (e.key === 'ArrowLeft' && currentIndex > 0) {
        setCurrentIndex(prev => prev - 1)
      } else if (e.key === 'ArrowRight' && currentIndex < items.length - 1) {
        setCurrentIndex(prev => prev + 1)
      } else if (key === 'n') {
        const nextIncomplete = items.findIndex(
          (item, idx) => idx > currentIndex && !completedIds.has(item.id)
        )
        if (nextIncomplete >= 0) setCurrentIndex(nextIncomplete)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentIndex, items, completedIds])

  // Redirect to completion page when all items are done
  useEffect(() => {
    if (items.length > 0 && completedIds.size === items.length) {
      router.push(`/complete?exp=exp2&model=${model}&completed=${completedIds.size}`)
    }
  }, [items.length, completedIds.size, model, router])

  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    )
  }

  // Show loading state while items are being loaded
  if (loadingItems) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-center panel-elevated p-12 max-w-lg">
          <div className="text-base mb-4" style={{ color: 'var(--text-primary)' }}>Loading evaluation items...</div>
          <div className="text-sm" style={{ color: 'var(--text-muted)' }}>Fetching from exp2_items.json</div>
        </div>
      </div>
    )
  }

  // Show error if items failed to load
  if (itemsError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-center panel-elevated p-12 max-w-lg">
          <h1 className="text-2xl font-semibold mb-4" style={{ color: 'var(--error-text)' }}>Error Loading Items</h1>
          <p className="text-base mb-4" style={{ color: 'var(--text-secondary)' }}>{itemsError}</p>
          <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
            Model: {model}<br />
            Please check the browser console for details.
          </p>
          <button onClick={() => window.location.reload()} className="btn btn-primary px-8 py-3 text-base font-semibold">
            Reload Page
          </button>
        </div>
      </div>
    )
  }

  if (!currentItem || items.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-center panel-elevated p-12 max-w-lg">
          <h1 className="text-3xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            {completedIds.size === items.length ? 'Redirecting...' : 'Evaluation Complete'}
          </h1>
          <p className="text-base mb-8" style={{ color: 'var(--text-secondary)' }}>
            Evaluated: <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{completedIds.size}</span> / {items.length}
          </p>
          <button onClick={() => router.push(`/complete?exp=exp2&model=${model}&completed=${completedIds.size}`)} className="btn btn-primary px-8 py-3 text-base font-semibold">
            Get Completion Code
          </button>
        </div>
      </div>
    )
  }

  const progress = items.length > 0 ? (completedIds.size / items.length) * 100 : 0
  const isCurrentCompleted = completedIds.has(currentItem.id)
  const promptText = getPromptText(currentItem.promptId)

  // Determine which image goes where based on randomization
  const leftImage = leftIsPreserved ? currentItem.preservedImageUrl : currentItem.editedImageUrl
  const rightImage = leftIsPreserved ? currentItem.editedImageUrl : currentItem.preservedImageUrl
  const leftLabel = leftIsPreserved ? 'Result A' : 'Result A'
  const rightLabel = leftIsPreserved ? 'Result B' : 'Result B'

  return (
    <div ref={containerRef} className="min-h-screen p-6 flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }} tabIndex={0}>
      {/* Top Bar */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-4">
          <span className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>{currentIndex + 1} / {items.length}</span>
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>({completedIds.size} done)</span>
          {isCurrentCompleted && <span className="badge badge-strong text-xs">DONE</span>}
        </div>
        <div className="flex items-center gap-4">
          <span className="badge badge-default text-xs">{MODELS_EXP2[model as keyof typeof MODELS_EXP2]?.name || model}</span>
          <div className="flex items-center gap-2">
            {user?.photoURL && <img src={user.photoURL} alt="" className="w-7 h-7 rounded-full" style={{ border: '1px solid var(--border-default)' }} />}
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{user?.displayName?.split(' ')[0]}</span>
          </div>
          <button onClick={() => router.push('/select')} className="btn btn-ghost px-4 py-2 text-sm">Exit</button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-bar h-1.5 mb-4">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      {/* Source Image + Prompt */}
      <div className="mb-4 panel p-4">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <div className="text-xs mb-1 text-center" style={{ color: 'var(--text-muted)' }}>ORIGINAL</div>
            <div className="w-24 h-24 image-container flex items-center justify-center overflow-hidden">
              <img src={currentItem.sourceImageUrl} alt="Source" className="w-full h-full object-cover" onError={(e) => { (e.target as HTMLImageElement).src = '/placeholder.svg' }} />
            </div>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="badge badge-strong font-semibold">{currentItem.promptId}</span>
              <span className="badge badge-default text-xs">{currentItem.race} / {currentItem.gender} / {currentItem.age}</span>
            </div>
            <div className="text-sm p-3 rounded" style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--accent-primary)' }}>Edit Prompt: </strong>
              &ldquo;{promptText}&rdquo;
            </div>
          </div>
        </div>
      </div>

      {/* Main Comparison - Side by Side */}
      <div className="flex-1 flex gap-4 mb-4">
        {/* Left Option (A) */}
        <div
          className={`flex-1 panel p-4 cursor-pointer transition-all ${choice === 'A' ? 'ring-2' : 'hover:ring-1'}`}
          style={{
            borderColor: choice === 'A' ? 'var(--accent-primary)' : 'var(--border-default)',
            backgroundColor: choice === 'A' ? 'var(--bg-elevated)' : 'var(--bg-secondary)'
          }}
          onClick={() => setChoice('A')}
        >
          <div className="text-center mb-2">
            <span className="text-lg font-bold" style={{ color: choice === 'A' ? 'var(--accent-primary)' : 'var(--text-primary)' }}>A</span>
            <span className="ml-2 keyboard-hint">1</span>
          </div>
          <div className="image-container flex items-center justify-center" style={{ minHeight: '350px' }}>
            <img
              src={leftImage || ''}
              alt="Option A"
              className="max-w-full max-h-full object-contain"
              onError={(e) => { (e.target as HTMLImageElement).src = '/placeholder.svg' }}
            />
          </div>
        </div>

        {/* VS Divider */}
        <div className="flex flex-col items-center justify-center px-2">
          <div className="text-2xl font-bold" style={{ color: 'var(--text-muted)' }}>VS</div>
        </div>

        {/* Right Option (B) */}
        <div
          className={`flex-1 panel p-4 cursor-pointer transition-all ${choice === 'B' ? 'ring-2' : 'hover:ring-1'}`}
          style={{
            borderColor: choice === 'B' ? 'var(--accent-primary)' : 'var(--border-default)',
            backgroundColor: choice === 'B' ? 'var(--bg-elevated)' : 'var(--bg-secondary)'
          }}
          onClick={() => setChoice('B')}
        >
          <div className="text-center mb-2">
            <span className="text-lg font-bold" style={{ color: choice === 'B' ? 'var(--accent-primary)' : 'var(--text-primary)' }}>B</span>
            <span className="ml-2 keyboard-hint">2</span>
          </div>
          <div className="image-container flex items-center justify-center" style={{ minHeight: '350px' }}>
            <img
              src={rightImage || ''}
              alt="Option B"
              className="max-w-full max-h-full object-contain"
              onError={(e) => { (e.target as HTMLImageElement).src = '/placeholder.svg' }}
            />
          </div>
        </div>
      </div>

      {/* Question + Tie Option */}
      <div className="panel p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-base mb-1" style={{ color: 'var(--text-primary)' }}>
              Which result better applies the edit while preserving the person&apos;s identity?
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Consider: Does the edit match the prompt? Is the original person still recognizable?
            </p>
          </div>
          <button
            onClick={() => setChoice('tie')}
            className={`px-6 py-3 panel font-semibold transition-all ${choice === 'tie' ? 'selected' : ''}`}
            style={{
              backgroundColor: choice === 'tie' ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
              borderColor: choice === 'tie' ? 'var(--accent-primary)' : 'var(--border-default)',
              color: choice === 'tie' ? 'var(--bg-primary)' : 'var(--text-primary)'
            }}
          >
            <span className="keyboard-hint mr-2">T</span>
            Tie / Can&apos;t Decide
          </button>
        </div>
      </div>

      {/* Keyboard Hints */}
      <div className="mt-3 text-center text-xs" style={{ color: 'var(--text-disabled)' }}>
        <span className="keyboard-hint mx-1">A/1</span> Left
        <span className="keyboard-hint mx-1 ml-4">B/2</span> Right
        <span className="keyboard-hint mx-1 ml-4">T/3</span> Tie
        <span className="keyboard-hint mx-1 ml-4">←→</span> Navigate
        <span className="keyboard-hint mx-1 ml-4">N</span> Next incomplete
      </div>
    </div>
  )
}

export default function Exp2Page() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    }>
      <Exp2Content />
    </Suspense>
  )
}
