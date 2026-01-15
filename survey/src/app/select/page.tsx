'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'
import { db } from '@/lib/firebase'
import { doc, getDoc } from 'firebase/firestore'
import { MODELS_EXP1, MODELS_EXP2, MODELS_EXP3 } from '@/lib/types'

const EXPERIMENT_TYPES = {
  exp1: {
    name: 'Experiment 1: VLM Scoring',
    desc: 'Rate edit quality, identity preservation (Categories B, D)',
    route: '/eval/exp1',
    models: MODELS_EXP1
  },
  exp2: {
    name: 'Experiment 2: Pairwise A/B',
    desc: 'Compare edited vs identity-preserved images (Soft Erasure Detection)',
    route: '/eval/exp2',
    models: MODELS_EXP2
  },
  exp3: {
    name: 'Experiment 3: WinoBias',
    desc: 'Binary stereotype detection for gender-occupation prompts',
    route: '/eval/exp3',
    models: MODELS_EXP3
  }
}

/**
 * Experiment type and model selection page
 */
export default function SelectPage() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()
  const [experimentType, setExperimentType] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [amtWorkerId, setAmtWorkerId] = useState<string | null>(null)
  const [checkingAmt, setCheckingAmt] = useState(true)

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!loading && !user) {
      router.push('/')
    }
  }, [user, loading, router])

  // Check consent
  useEffect(() => {
    const consent = localStorage.getItem('irb_consent_i2i_bias')
    if (consent !== 'agreed') {
      router.push('/consent')
    }
  }, [router])

  // Check AMT info (optional - user may have skipped)
  useEffect(() => {
    async function checkAMT() {
      if (!user) {
        setCheckingAmt(false)
        return
      }

      try {
        const userDoc = await getDoc(doc(db, 'users', user.uid))
        if (userDoc.exists()) {
          const data = userDoc.data()
          // Check if user has gone through AMT page (either with or without Worker ID)
          if (!data.createdAt) {
            // User hasn't been through AMT page yet
            router.push('/amt')
            return
          }
          // Set Worker ID if available (may be null if user skipped)
          setAmtWorkerId(data.amtWorkerId || null)
        } else {
          // No user doc, redirect to AMT page
          router.push('/amt')
          return
        }
      } catch (err) {
        console.error('Error checking AMT:', err)
        // Fallback to localStorage
        const localWorkerId = localStorage.getItem('amt_worker_id')
        setAmtWorkerId(localWorkerId || null)
      }
      setCheckingAmt(false)
    }

    if (!loading && user) {
      checkAMT()
    }
  }, [user, loading, router])

  // Reset model when experiment changes
  useEffect(() => {
    setSelectedModel('')
  }, [experimentType])

  const handleStart = () => {
    if (!experimentType || !selectedModel) {
      alert('Please select both experiment type and model')
      return
    }

    const expInfo = EXPERIMENT_TYPES[experimentType as keyof typeof EXPERIMENT_TYPES]
    router.push(`${expInfo.route}?model=${selectedModel}&index=0`)
  }

  if (loading || !user || checkingAmt) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    )
  }

  const currentExp = experimentType ? EXPERIMENT_TYPES[experimentType as keyof typeof EXPERIMENT_TYPES] : null

  return (
    <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="max-w-2xl w-full panel-elevated p-10">
        <div className="flex items-center justify-between mb-10">
          <h1 className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>
            {experimentType ? 'Select Model' : 'Select Experiment Type'}
          </h1>
          <div className="flex items-center gap-3">
            {user.photoURL && (
              <img src={user.photoURL} alt="" className="w-8 h-8 rounded-full" style={{ border: '1px solid var(--border-default)' }} />
            )}
            <div className="text-right">
              <div className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{user.displayName}</div>
              {amtWorkerId && (
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Worker: {amtWorkerId}</div>
              )}
              <button onClick={logout} className="text-xs transition-colors" style={{ color: 'var(--text-muted)' }}>
                Sign out
              </button>
            </div>
          </div>
        </div>

        {!experimentType ? (
          // Step 1: Select Experiment Type
          <div className="space-y-3">
            {Object.entries(EXPERIMENT_TYPES).map(([key, info]) => (
              <button
                key={key}
                onClick={() => setExperimentType(key)}
                className="w-full p-6 panel text-left transition-all hover-lift group"
                style={{ borderColor: 'var(--border-default)' }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="font-semibold text-base mb-1" style={{ color: 'var(--text-primary)' }}>
                      {info.name}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-muted)' }}>{info.desc}</div>
                    <div className="text-xs mt-2" style={{ color: 'var(--text-disabled)' }}>
                      Models: {Object.values(info.models).map(m => m.name).join(', ')}
                    </div>
                  </div>
                  <svg className="w-5 h-5 flex-shrink-0 ml-4" style={{ color: 'var(--text-muted)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </button>
            ))}
          </div>
        ) : (
          // Step 2: Select Model
          <div className="space-y-8">
            <button
              onClick={() => setExperimentType('')}
              className="text-sm transition-colors btn-ghost px-3 py-1.5"
              style={{ color: 'var(--text-muted)' }}
            >
              ← Back to experiment selection
            </button>

            <div className="panel p-4 mb-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{currentExp?.name}</div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{currentExp?.desc}</div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                Select Model to Evaluate
              </label>
              <div className="grid grid-cols-1 gap-3">
                {currentExp && Object.entries(currentExp.models).map(([key, info]) => {
                  const isAvailable = 'available' in info ? info.available : true
                  const isSelected = selectedModel === key

                  return (
                    <button
                      key={key}
                      onClick={() => isAvailable && setSelectedModel(key)}
                      disabled={!isAvailable}
                      className={`p-5 panel text-left transition-all ${isAvailable ? 'hover-lift' : 'cursor-not-allowed'} ${isSelected ? 'selected' : ''}`}
                      style={{
                        borderColor: isSelected ? 'var(--accent-primary)' : 'var(--border-default)',
                        backgroundColor: isSelected ? 'var(--accent-primary)' : isAvailable ? 'var(--bg-secondary)' : 'var(--bg-tertiary)',
                        opacity: isAvailable ? 1 : 0.6
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold" style={{ color: isSelected ? 'var(--bg-primary)' : 'var(--text-primary)' }}>
                            {info.name}
                          </div>
                          <div className="text-sm mt-1" style={{ color: isSelected ? 'var(--bg-primary)' : 'var(--text-muted)' }}>
                            {info.total.toLocaleString()} items
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {!isAvailable && (
                            <span className="text-xs px-2 py-1 rounded font-medium" style={{ backgroundColor: 'var(--warning-bg)', color: 'var(--warning-text)' }}>
                              Coming Soon
                            </span>
                          )}
                          {key === 'gemini' && isAvailable && (
                            <span className="text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-muted)' }}>
                              Example
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            <button
              onClick={handleStart}
              disabled={!selectedModel}
              className="btn btn-primary w-full py-4 text-base font-semibold"
            >
              START EVALUATION
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
