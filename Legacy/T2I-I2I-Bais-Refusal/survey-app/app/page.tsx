'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { User } from 'firebase/auth'
import {
  signInWithGoogle,
  signInAnonymouslyUser,
  signOutUser,
  onAuthChange
} from '@/lib/firebase'
import { getParticipant, getEvaluationsByEvaluator } from '@/lib/firestore'

export default function LandingPage() {
  const router = useRouter()
  const [prolificId, setProlificId] = useState('')
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [authLoading, setAuthLoading] = useState(false)
  const [userStats, setUserStats] = useState<{
    totalEvaluations: number
    isComplete: boolean
  } | null>(null)

  // Listen for auth state changes
  useEffect(() => {
    const unsubscribe = onAuthChange(async (authUser) => {
      setUser(authUser)
      setLoading(false)

      if (authUser) {
        // Load user's previous progress
        try {
          const participant = await getParticipant(authUser.uid)
          const evaluations = await getEvaluationsByEvaluator(authUser.uid)
          setUserStats({
            totalEvaluations: evaluations.length,
            isComplete: participant?.isComplete || false
          })
        } catch (error) {
          console.error('Error loading user stats:', error)
        }
      } else {
        setUserStats(null)
      }
    })

    return () => unsubscribe()
  }, [])

  const handleGoogleSignIn = async () => {
    setAuthLoading(true)
    try {
      await signInWithGoogle()
      // User will be set by onAuthChange
    } catch (error) {
      console.error('Google sign-in failed:', error)
      alert('Google sign-in failed. Please try again.')
    } finally {
      setAuthLoading(false)
    }
  }

  const handleAnonymousStart = async () => {
    if (!prolificId.trim()) return

    setAuthLoading(true)
    try {
      await signInAnonymouslyUser()
      localStorage.setItem('prolificId', prolificId.trim())
      router.push('/consent')
    } catch (error) {
      console.error('Anonymous sign-in failed:', error)
      alert('Sign-in failed. Please try again.')
    } finally {
      setAuthLoading(false)
    }
  }

  const handleSignOut = async () => {
    try {
      await signOutUser()
      localStorage.removeItem('prolificId')
      setUserStats(null)
    } catch (error) {
      console.error('Sign-out failed:', error)
    }
  }

  const handleContinueSurvey = () => {
    router.push('/consent')
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-neutral-900 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-neutral-50">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs tracking-widest text-neutral-400 uppercase mb-3">
            Research Study
          </p>
          <h1 className="text-2xl font-light text-neutral-900 mb-2">
            ACRB Human Evaluation
          </h1>
          <p className="text-sm text-neutral-500">
            AI Image Generation Bias Assessment
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white border border-neutral-200 rounded-lg shadow-sm p-6">
          {user ? (
            // Logged in state
            <div className="space-y-6">
              {/* User Info */}
              <div className="flex items-center gap-4 pb-4 border-b border-neutral-100">
                {user.photoURL ? (
                  <img
                    src={user.photoURL}
                    alt="Profile"
                    className="w-12 h-12 rounded-full"
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-neutral-200 flex items-center justify-center">
                    <span className="text-neutral-500 text-lg">
                      {user.isAnonymous ? '?' : (user.displayName?.[0] || 'U')}
                    </span>
                  </div>
                )}
                <div className="flex-1">
                  <p className="font-medium text-neutral-900">
                    {user.isAnonymous ? 'Anonymous User' : user.displayName}
                  </p>
                  <p className="text-xs text-neutral-500">
                    {user.isAnonymous ? 'Prolific Participant' : user.email}
                  </p>
                </div>
                <button
                  onClick={handleSignOut}
                  className="text-xs text-neutral-400 hover:text-neutral-600"
                >
                  Sign out
                </button>
              </div>

              {/* Progress Stats */}
              {userStats && (
                <div className="bg-neutral-50 rounded-lg p-4">
                  <p className="text-xs tracking-widest text-neutral-400 uppercase mb-3">
                    Your Progress
                  </p>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-2xl font-light text-neutral-900">
                        {userStats.totalEvaluations}
                      </p>
                      <p className="text-xs text-neutral-500">Evaluations</p>
                    </div>
                    <div className="text-right">
                      {userStats.isComplete ? (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-700">
                          ✓ Completed
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-700">
                          In Progress
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Continue Button */}
              <button
                onClick={handleContinueSurvey}
                className="w-full py-3 bg-neutral-900 text-white rounded-lg font-medium
                         hover:bg-neutral-800 transition-colors"
              >
                {userStats?.totalEvaluations ? 'Continue Survey' : 'Start Survey'}
              </button>

              {/* View History Link */}
              {userStats && userStats.totalEvaluations > 0 && (
                <button
                  onClick={() => router.push('/history')}
                  className="w-full py-2 text-sm text-neutral-600 hover:text-neutral-900"
                >
                  View My Evaluation History →
                </button>
              )}
            </div>
          ) : (
            // Not logged in state
            <div className="space-y-6">
              {/* Study Info */}
              <div className="space-y-3 text-sm text-neutral-600">
                <div className="flex items-start gap-3">
                  <span className="w-5 h-5 rounded-full bg-neutral-100 flex items-center justify-center flex-shrink-0 text-xs">1</span>
                  <span>15-20 minutes, about 50 images to evaluate</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-5 h-5 rounded-full bg-neutral-100 flex items-center justify-center flex-shrink-0 text-xs">2</span>
                  <span>Evaluate refusals and bias in AI image generation models</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-5 h-5 rounded-full bg-neutral-100 flex items-center justify-center flex-shrink-0 text-xs">3</span>
                  <span>Your progress is saved automatically</span>
                </div>
              </div>

              <div className="border-t border-neutral-100 pt-6">
                {/* Google Login */}
                <button
                  onClick={handleGoogleSignIn}
                  disabled={authLoading}
                  className="w-full py-3 bg-white border border-neutral-300 rounded-lg font-medium
                           text-neutral-700 hover:bg-neutral-50 transition-colors
                           flex items-center justify-center gap-3 mb-4"
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  {authLoading ? 'Loading...' : 'Continue with Google'}
                </button>

                <div className="relative my-4">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-neutral-200" />
                  </div>
                  <div className="relative flex justify-center text-xs">
                    <span className="px-2 bg-white text-neutral-400">Or</span>
                  </div>
                </div>

                {/* Prolific ID Input */}
                <div className="space-y-3">
                  <label className="block text-xs tracking-widest text-neutral-500 uppercase">
                    Participate with Prolific ID
                  </label>
                  <input
                    type="text"
                    value={prolificId}
                    onChange={(e) => setProlificId(e.target.value)}
                    placeholder="Enter your Prolific ID"
                    className="w-full px-4 py-3 border border-neutral-300 rounded-lg
                             focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent
                             text-sm"
                  />
                  <button
                    onClick={handleAnonymousStart}
                    disabled={!prolificId.trim() || authLoading}
                    className={`w-full py-3 rounded-lg font-medium transition-colors
                      ${prolificId.trim() && !authLoading
                        ? 'bg-neutral-900 text-white hover:bg-neutral-800'
                        : 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
                      }`}
                  >
                    {authLoading ? 'Loading...' : 'Begin Study'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-xs text-neutral-400 text-center mt-6">
          ACRB Research Study · IRB Approved
        </p>
      </div>
    </div>
  )
}
