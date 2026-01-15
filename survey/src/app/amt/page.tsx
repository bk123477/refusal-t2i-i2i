'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'
import { db } from '@/lib/firebase'
import { doc, setDoc, getDoc, serverTimestamp } from 'firebase/firestore'

/**
 * AMT (Amazon Mechanical Turk) Code Entry Page
 * Workers enter their Worker ID and Assignment ID here
 */
export default function AMTPage() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()

  const [workerId, setWorkerId] = useState('')
  const [assignmentId, setAssignmentId] = useState('')
  const [hitId, setHitId] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

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

  // Check if user has already been through this page
  useEffect(() => {
    async function checkExistingUser() {
      if (!user) return

      try {
        const userDoc = await getDoc(doc(db, 'users', user.uid))
        if (userDoc.exists()) {
          const data = userDoc.data()
          // If user doc exists with createdAt, they've already been through AMT page
          if (data.createdAt) {
            router.push('/select')
          }
        }
      } catch (err) {
        console.error('Error checking user info:', err)
      }
    }

    checkExistingUser()
  }, [user, router])

  // Auto-fill from URL params (MTurk often passes these)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const urlWorkerId = params.get('workerId') || params.get('worker_id')
    const urlAssignmentId = params.get('assignmentId') || params.get('assignment_id')
    const urlHitId = params.get('hitId') || params.get('hit_id')

    if (urlWorkerId) setWorkerId(urlWorkerId)
    if (urlAssignmentId) setAssignmentId(urlAssignmentId)
    if (urlHitId) setHitId(urlHitId)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!user) return

    setSaving(true)
    setError('')

    try {
      // Save to users collection (with or without AMT info)
      await setDoc(doc(db, 'users', user.uid), {
        email: user.email,
        displayName: user.displayName,
        photoURL: user.photoURL,
        amtWorkerId: workerId.trim() || null,
        amtAssignmentId: assignmentId.trim() || null,
        amtHitId: hitId.trim() || null,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      }, { merge: true })

      // Also save to localStorage as backup
      if (workerId.trim()) {
        localStorage.setItem('amt_worker_id', workerId.trim())
        if (assignmentId) localStorage.setItem('amt_assignment_id', assignmentId.trim())
      }

      router.push('/select')
    } catch (err) {
      console.error('Error saving AMT info:', err)
      setError('Failed to save. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const handleSkip = async () => {
    if (!user) return

    setSaving(true)
    setError('')

    try {
      // Save user without AMT info
      await setDoc(doc(db, 'users', user.uid), {
        email: user.email,
        displayName: user.displayName,
        photoURL: user.photoURL,
        amtWorkerId: null,
        amtAssignmentId: null,
        amtHitId: null,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      }, { merge: true })

      router.push('/select')
    } catch (err) {
      console.error('Error saving user info:', err)
      setError('Failed to proceed. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <div className="text-base" style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="max-w-md w-full panel-elevated p-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
            Worker Information
          </h1>
          <div className="flex items-center gap-2">
            {user.photoURL && (
              <img src={user.photoURL} alt="" className="w-7 h-7 rounded-full" style={{ border: '1px solid var(--border-default)' }} />
            )}
            <button onClick={logout} className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Sign out
            </button>
          </div>
        </div>

        <div className="panel p-4 mb-6" style={{ backgroundColor: 'var(--info-bg)', borderColor: 'var(--info-text)' }}>
          <p className="text-sm" style={{ color: 'var(--info-text)' }}>
            If you're an Amazon Mechanical Turk worker, please enter your Worker ID to track your progress and ensure proper credit. Otherwise, you can skip this step.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              Worker ID <span style={{ color: 'var(--text-muted)' }}>(optional)</span>
            </label>
            <input
              type="text"
              value={workerId}
              onChange={(e) => setWorkerId(e.target.value)}
              placeholder="e.g., A1B2C3D4E5F6G7"
              className="w-full px-4 py-3 rounded text-base focus:outline-none focus:ring-2"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)'
              }}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              Assignment ID <span style={{ color: 'var(--text-muted)' }}>(optional)</span>
            </label>
            <input
              type="text"
              value={assignmentId}
              onChange={(e) => setAssignmentId(e.target.value)}
              placeholder="From your HIT assignment"
              className="w-full px-4 py-3 rounded text-base focus:outline-none focus:ring-2"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)'
              }}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              HIT ID <span style={{ color: 'var(--text-muted)' }}>(optional)</span>
            </label>
            <input
              type="text"
              value={hitId}
              onChange={(e) => setHitId(e.target.value)}
              placeholder="From your HIT"
              className="w-full px-4 py-3 rounded text-base focus:outline-none focus:ring-2"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)'
              }}
            />
          </div>

          {error && (
            <div className="p-3 rounded text-sm" style={{ backgroundColor: '#450a0a', color: '#f87171' }}>
              {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              type="button"
              onClick={handleSkip}
              disabled={saving}
              className="btn w-full py-3 text-base font-semibold"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-secondary)'
              }}
            >
              Skip
            </button>
            <button
              type="submit"
              disabled={saving}
              className="btn btn-primary w-full py-3 text-base font-semibold"
            >
              {saving ? 'Saving...' : 'Continue'}
            </button>
          </div>
        </form>

        <p className="text-xs mt-6 text-center" style={{ color: 'var(--text-muted)' }}>
          {workerId.trim() 
            ? 'Your Worker ID will be used to track your evaluations and generate a completion code.'
            : 'You can skip entering a Worker ID and proceed directly to the evaluation.'}
        </p>
      </div>
    </div>
  )
}
