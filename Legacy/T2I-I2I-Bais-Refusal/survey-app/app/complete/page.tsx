'use client'

import { useEffect, useState } from 'react'

export default function CompletePage() {
  const [completionCode, setCompletionCode] = useState<string>('')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const code = `ACRB-${Math.random().toString(36).substring(2, 8).toUpperCase()}`
    setCompletionCode(code)
    localStorage.setItem('completionCode', code)
  }, [])

  const handleCopy = () => {
    navigator.clipboard.writeText(completionCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleReturnToProlific = () => {
    window.location.href = `https://app.prolific.com/submissions/complete?cc=${completionCode}`
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="survey-card text-center">
        <div className="mb-10">
          <p className="text-xs tracking-widest text-neutral-500 uppercase mb-3">
            Step 4 of 4
          </p>
          <h1 className="text-2xl font-light text-neutral-900">
            Complete
          </h1>
          <div className="w-12 h-px bg-neutral-300 mt-4 mx-auto" />
        </div>

        <p className="text-sm text-neutral-600 mb-10">
          Your responses have been recorded. Thank you for contributing to
          our research on AI image generation fairness.
        </p>

        <div className="border border-neutral-200 p-6 mb-10">
          <p className="text-xs tracking-widest text-neutral-500 uppercase mb-4">
            Completion Code
          </p>
          <div className="flex items-center justify-center gap-4">
            <code className="text-xl font-mono text-neutral-900 tracking-wider">
              {completionCode}
            </code>
            <button
              onClick={handleCopy}
              className="text-xs text-neutral-500 hover:text-neutral-900 transition-colors uppercase tracking-widest"
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <p className="text-xs text-neutral-500 mt-4">
            Copy this code to Prolific for compensation.
          </p>
        </div>

        <button
          onClick={handleReturnToProlific}
          className="btn-primary w-full"
        >
          Return to Prolific
        </button>

        <p className="text-xs text-neutral-400 mt-8">
          Questions? Contact research@institution.edu
        </p>
      </div>
    </div>
  )
}
