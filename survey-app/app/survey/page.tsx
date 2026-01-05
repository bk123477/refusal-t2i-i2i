'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { User } from 'firebase/auth'
import ProgressBar from '@/components/ProgressBar'
import EvaluationForm from '@/components/EvaluationForm'
import { generateSessionId, onAuthChange, auth } from '@/lib/firebase'
import {
  getSurveyItems,
  saveEvaluation,
  saveSession,
  getSession,
  createParticipant,
  completeParticipant,
} from '@/lib/firestore'
import { SurveyItem, Response } from '@/lib/types'

export default function SurveyPage() {
  const router = useRouter()
  const [items, setItems] = useState<SurveyItem[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [responses, setResponses] = useState<Response[]>([])
  const [startTime, setStartTime] = useState<number>(Date.now())
  const [sessionId] = useState<string>(() => generateSessionId())
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [attentionChecksPassed, setAttentionChecksPassed] = useState(0)
  const [showBreakPrompt, setShowBreakPrompt] = useState(false)

  const currentItem = items[currentIndex]
  const isLastItem = currentIndex === items.length - 1
  const totalItems = items.length

  // Auth listener
  useEffect(() => {
    const unsubscribe = onAuthChange(async (authUser) => {
      if (!authUser) {
        router.push('/')
        return
      }

      setUser(authUser)

      // Check consent
      const consentGiven = localStorage.getItem('consentGiven')
      if (!consentGiven) {
        router.push('/consent')
        return
      }

      try {
        // Check for existing session
        const existingSession = await getSession(authUser.uid)

        if (existingSession) {
          // Resume previous session
          setResponses(existingSession.responses)
          setCurrentIndex(existingSession.currentIndex)
        } else {
          // Create new participant
          const prolificId = localStorage.getItem('prolificId') || authUser.email || 'google_user'
          await createParticipant(authUser.uid, prolificId, sessionId)
        }

        // Load survey items
        const surveyItems = await getSurveyItems(authUser.uid)
        setItems(surveyItems)
      } catch (error) {
        console.error('Error initializing survey:', error)
      }

      setLoading(false)
    })

    return () => unsubscribe()
  }, [sessionId, router])

  // Auto-save session every 30 seconds
  useEffect(() => {
    if (!user || items.length === 0) return

    const saveInterval = setInterval(async () => {
      try {
        await saveSession(user.uid, {
          participant: {
            id: user.uid,
            sessionId,
            prolificId: localStorage.getItem('prolificId') || user.email || 'google_user',
            consentTimestamp: new Date().toISOString(),
            startTime: new Date().toISOString(),
            isComplete: false,
            attentionChecksPassed,
            totalAttentionChecks: items.filter((i) => i.type === 'attention').length,
          },
          responses,
          currentIndex,
          lastSavedAt: new Date().toISOString(),
        })
      } catch (error) {
        console.error('Error auto-saving:', error)
      }
    }, 30000)

    return () => clearInterval(saveInterval)
  }, [user, responses, currentIndex, sessionId, items, attentionChecksPassed])

  // Reset start time when moving to next question
  useEffect(() => {
    setStartTime(Date.now())
  }, [currentIndex])

  // Show break prompt every 10 questions
  useEffect(() => {
    if (currentIndex > 0 && currentIndex % 10 === 0) {
      setShowBreakPrompt(true)
    }
  }, [currentIndex])

  const handleSubmit = async (response: {
    isRefusal: boolean
    attributePresent: 'yes' | 'no' | 'partial'
    faithfulness: 1 | 2 | 3 | 4 | 5
    confidence: 'low' | 'medium' | 'high'
    notes?: string
  }) => {
    if (!user) return

    const responseTimeMs = Date.now() - startTime

    try {
      await saveEvaluation({
        evaluatorId: user.uid,
        imageId: currentItem.id,
        promptId: currentItem.id,
        attribute: currentItem.attribute || 'unknown',
        attributeValue: currentItem.attributeValue || 'unknown',
        model: currentItem.model || 'unknown',
        domain: currentItem.domain || 'unknown',
        isRefusal: response.isRefusal,
        attributePresent: response.attributePresent,
        faithfulness: response.faithfulness,
        confidence: response.confidence,
        notes: response.notes,
        responseTimeMs,
        sessionId,
      })

      // Check attention check
      if (currentItem.type === 'attention') {
        const correctAnswer = currentItem.correctAnswer
        let isPassed = false

        if (correctAnswer) {
          if (
            correctAnswer.isRefusal !== undefined &&
            response.isRefusal === correctAnswer.isRefusal
          ) {
            isPassed = true
          } else if (
            correctAnswer.attributePresent !== undefined &&
            response.attributePresent === correctAnswer.attributePresent
          ) {
            isPassed = true
          } else if (
            correctAnswer.faithfulness !== undefined &&
            Math.abs(response.faithfulness - correctAnswer.faithfulness) <= 1
          ) {
            isPassed = true
          }
        }

        if (isPassed) {
          setAttentionChecksPassed((prev) => prev + 1)
        }
      }

      // Update responses state
      const newResponse: Response = {
        itemId: currentItem.id,
        type: currentItem.type,
        ...response,
        responseTimeMs,
        timestamp: new Date().toISOString(),
      }

      const updatedResponses = [...responses, newResponse]
      setResponses(updatedResponses)

      // Move to next or complete
      if (isLastItem) {
        await completeParticipant(
          user.uid,
          attentionChecksPassed,
          items.filter((i) => i.type === 'attention').length
        )
        router.push('/complete')
      } else {
        setCurrentIndex((prev) => prev + 1)
      }
    } catch (error) {
      console.error('Error saving evaluation:', error)
      alert('Failed to save response. Please try again.')
    }
  }

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1)
    }
  }

  const handleContinueAfterBreak = () => {
    setShowBreakPrompt(false)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-neutral-900 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-neutral-600">Loading survey...</p>
        </div>
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-neutral-900 font-medium mb-2">No items available</p>
          <p className="text-sm text-neutral-500">
            Please contact the administrator
          </p>
        </div>
      </div>
    )
  }

  // Break prompt overlay
  if (showBreakPrompt) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="bg-white border border-neutral-200 rounded-lg shadow-sm p-8 max-w-md w-full">
          <div className="text-center">
            <p className="text-xs tracking-widest text-neutral-500 uppercase mb-6">
              Break Time
            </p>
            <h2 className="text-2xl font-light text-neutral-900 mb-4">
              Take a moment to rest
            </h2>
            <p className="text-neutral-600 mb-8 leading-relaxed">
              You have completed {currentIndex} out of {totalItems} evaluations.
              Feel free to take a short break before continuing.
            </p>
            <button
              onClick={handleContinueAfterBreak}
              className="w-full py-3 bg-neutral-900 text-white rounded-lg font-medium hover:bg-neutral-800"
            >
              Continue Survey
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-8 px-4 bg-neutral-50">
      <div className="max-w-2xl mx-auto">
        {/* Header with user info */}
        <div className="flex items-center justify-between mb-6">
          <div className="text-sm text-neutral-500">
            {user?.isAnonymous ? 'Anonymous' : user?.displayName}
          </div>
          <div className="text-sm text-neutral-500">
            {currentIndex + 1} / {totalItems}
          </div>
        </div>

        <ProgressBar current={currentIndex + 1} total={totalItems} />

        <div className="bg-white border border-neutral-200 rounded-lg shadow-sm p-6 mt-6">
          <div className="mb-8">
            <p className="text-xs tracking-widest text-neutral-500 uppercase mb-3">
              Evaluation {currentIndex + 1} of {totalItems}
            </p>
            <div className="w-12 h-px bg-neutral-300" />
          </div>

          <EvaluationForm
            item={currentItem}
            onSubmit={handleSubmit}
            onPrevious={handlePrevious}
            showPrevious={currentIndex > 0}
          />
        </div>

        {/* Help text */}
        <div className="mt-6 text-center">
          <p className="text-xs text-neutral-400">
            Your progress is automatically saved every 30 seconds
          </p>
        </div>
      </div>
    </div>
  )
}
