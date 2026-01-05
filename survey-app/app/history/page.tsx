'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { User } from 'firebase/auth'
import { onAuthChange } from '@/lib/firebase'
import { getEvaluationsByEvaluator, getParticipant } from '@/lib/firestore'
import { Evaluation, Participant } from '@/lib/types'

export default function HistoryPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [evaluations, setEvaluations] = useState<Evaluation[]>([])
  const [participant, setParticipant] = useState<Participant | null>(null)
  const [filter, setFilter] = useState<'all' | 'refusal' | 'non-refusal'>('all')

  useEffect(() => {
    const unsubscribe = onAuthChange(async (authUser) => {
      if (!authUser) {
        router.push('/')
        return
      }

      setUser(authUser)

      try {
        const [evals, part] = await Promise.all([
          getEvaluationsByEvaluator(authUser.uid),
          getParticipant(authUser.uid)
        ])
        setEvaluations(evals)
        setParticipant(part)
      } catch (error) {
        console.error('Error loading history:', error)
      }

      setLoading(false)
    })

    return () => unsubscribe()
  }, [router])

  const filteredEvaluations = evaluations.filter(e => {
    if (filter === 'refusal') return e.isRefusal
    if (filter === 'non-refusal') return !e.isRefusal
    return true
  })

  const stats = {
    total: evaluations.length,
    refusals: evaluations.filter(e => e.isRefusal).length,
    avgFaithfulness: evaluations.length > 0
      ? (evaluations.reduce((sum, e) => sum + e.faithfulness, 0) / evaluations.length).toFixed(1)
      : 0,
    avgResponseTime: evaluations.length > 0
      ? Math.round(evaluations.reduce((sum, e) => sum + e.responseTimeMs, 0) / evaluations.length / 1000)
      : 0
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-neutral-900 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-neutral-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-light text-neutral-900">Evaluation History</h1>
            <p className="text-sm text-neutral-500 mt-1">
              {user?.isAnonymous ? 'Anonymous User' : user?.displayName}
            </p>
          </div>
          <button
            onClick={() => router.push('/')}
            className="text-sm text-neutral-600 hover:text-neutral-900"
          >
            ← Back to Home
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white border border-neutral-200 rounded-lg p-4">
            <p className="text-2xl font-light text-neutral-900">{stats.total}</p>
            <p className="text-xs text-neutral-500 uppercase tracking-wider">Total Evaluations</p>
          </div>
          <div className="bg-white border border-neutral-200 rounded-lg p-4">
            <p className="text-2xl font-light text-red-600">{stats.refusals}</p>
            <p className="text-xs text-neutral-500 uppercase tracking-wider">Refusals Detected</p>
          </div>
          <div className="bg-white border border-neutral-200 rounded-lg p-4">
            <p className="text-2xl font-light text-neutral-900">{stats.avgFaithfulness}</p>
            <p className="text-xs text-neutral-500 uppercase tracking-wider">Avg Faithfulness</p>
          </div>
          <div className="bg-white border border-neutral-200 rounded-lg p-4">
            <p className="text-2xl font-light text-neutral-900">{stats.avgResponseTime}s</p>
            <p className="text-xs text-neutral-500 uppercase tracking-wider">Avg Response Time</p>
          </div>
        </div>

        {/* Completion Status */}
        {participant && (
          <div className={`mb-8 p-4 rounded-lg ${participant.isComplete ? 'bg-green-50 border border-green-200' : 'bg-yellow-50 border border-yellow-200'}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className={`font-medium ${participant.isComplete ? 'text-green-800' : 'text-yellow-800'}`}>
                  {participant.isComplete ? '✓ Survey Completed' : 'Survey In Progress'}
                </p>
                <p className={`text-sm ${participant.isComplete ? 'text-green-600' : 'text-yellow-600'}`}>
                  Attention checks passed: {participant.attentionChecksPassed} / {participant.totalAttentionChecks}
                </p>
              </div>
              {!participant.isComplete && (
                <button
                  onClick={() => router.push('/survey')}
                  className="px-4 py-2 bg-neutral-900 text-white text-sm rounded-lg hover:bg-neutral-800"
                >
                  Continue Survey
                </button>
              )}
            </div>
          </div>
        )}

        {/* Filter */}
        <div className="flex gap-2 mb-6">
          {(['all', 'refusal', 'non-refusal'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                filter === f
                  ? 'bg-neutral-900 text-white'
                  : 'bg-white border border-neutral-200 text-neutral-600 hover:bg-neutral-50'
              }`}
            >
              {f === 'all' ? 'All' : f === 'refusal' ? 'Refusals Only' : 'Non-Refusals'}
            </button>
          ))}
        </div>

        {/* Evaluations List */}
        <div className="bg-white border border-neutral-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-neutral-50 border-b border-neutral-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">#</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Attribute</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Model</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Refusal</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Attr. Present</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Faithfulness</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Confidence</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100">
                {filteredEvaluations.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-neutral-500">
                      No evaluations found
                    </td>
                  </tr>
                ) : (
                  filteredEvaluations.map((eval_, idx) => (
                    <tr key={idx} className="hover:bg-neutral-50">
                      <td className="px-4 py-3 text-neutral-900">{idx + 1}</td>
                      <td className="px-4 py-3">
                        <span className="text-neutral-900">{eval_.attribute}</span>
                        <span className="text-neutral-400 text-xs block">{eval_.attributeValue}</span>
                      </td>
                      <td className="px-4 py-3 text-neutral-600">{eval_.model}</td>
                      <td className="px-4 py-3">
                        {eval_.isRefusal ? (
                          <span className="inline-flex px-2 py-0.5 text-xs rounded-full bg-red-100 text-red-700">Yes</span>
                        ) : (
                          <span className="inline-flex px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-700">No</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex px-2 py-0.5 text-xs rounded-full ${
                          eval_.attributePresent === 'yes' ? 'bg-green-100 text-green-700' :
                          eval_.attributePresent === 'partial' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {eval_.attributePresent}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1">
                          {[1, 2, 3, 4, 5].map(n => (
                            <div
                              key={n}
                              className={`w-2 h-2 rounded-full ${
                                n <= eval_.faithfulness ? 'bg-neutral-900' : 'bg-neutral-200'
                              }`}
                            />
                          ))}
                          <span className="ml-1 text-neutral-500">{eval_.faithfulness}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`text-xs ${
                          eval_.confidence === 'high' ? 'text-green-600' :
                          eval_.confidence === 'medium' ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {eval_.confidence}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-neutral-500">
                        {(eval_.responseTimeMs / 1000).toFixed(1)}s
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Export Button */}
        {evaluations.length > 0 && (
          <div className="mt-6 text-center">
            <button
              onClick={() => {
                const csv = [
                  ['#', 'Attribute', 'Value', 'Model', 'Domain', 'Refusal', 'AttrPresent', 'Faithfulness', 'Confidence', 'ResponseTime'],
                  ...evaluations.map((e, i) => [
                    i + 1,
                    e.attribute,
                    e.attributeValue,
                    e.model,
                    e.domain,
                    e.isRefusal ? 'Yes' : 'No',
                    e.attributePresent,
                    e.faithfulness,
                    e.confidence,
                    e.responseTimeMs
                  ])
                ].map(row => row.join(',')).join('\n')

                const blob = new Blob([csv], { type: 'text/csv' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `acrb-evaluations-${user?.uid?.slice(0, 8)}.csv`
                a.click()
              }}
              className="text-sm text-neutral-600 hover:text-neutral-900 underline"
            >
              Download as CSV
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
