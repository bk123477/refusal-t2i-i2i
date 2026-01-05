'use client'

import { useState } from 'react'
import { SurveyItem } from '@/lib/types'
import Image from 'next/image'

interface EvaluationFormProps {
  item: SurveyItem
  onSubmit: (response: {
    isRefusal: boolean
    attributePresent: 'yes' | 'no' | 'partial'
    faithfulness: 1 | 2 | 3 | 4 | 5
    confidence: 'low' | 'medium' | 'high'
    notes?: string
  }) => void
  onPrevious?: () => void
  showPrevious?: boolean
}

export default function EvaluationForm({
  item,
  onSubmit,
  onPrevious,
  showPrevious = false,
}: EvaluationFormProps) {
  const [isRefusal, setIsRefusal] = useState<boolean | null>(null)
  const [attributePresent, setAttributePresent] = useState<'yes' | 'no' | 'partial' | null>(null)
  const [faithfulness, setFaithfulness] = useState<1 | 2 | 3 | 4 | 5 | null>(null)
  const [confidence, setConfidence] = useState<'low' | 'medium' | 'high' | null>(null)
  const [notes, setNotes] = useState('')
  const [showImage, setShowImage] = useState(false)
  const [imageError, setImageError] = useState(false)

  const handleSubmit = () => {
    if (
      isRefusal !== null &&
      attributePresent !== null &&
      faithfulness !== null &&
      confidence !== null
    ) {
      onSubmit({
        isRefusal,
        attributePresent,
        faithfulness,
        confidence,
        notes: notes.trim() || undefined,
      })

      // Reset form
      setIsRefusal(null)
      setAttributePresent(null)
      setFaithfulness(null)
      setConfidence(null)
      setNotes('')
      setShowImage(false)
    }
  }

  const isComplete =
    isRefusal !== null &&
    attributePresent !== null &&
    faithfulness !== null &&
    confidence !== null

  // Keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey && isComplete) {
      handleSubmit()
    }
  }

  return (
    <div className="space-y-8" onKeyDown={handleKeyDown}>
      {/* Prompt Display */}
      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <p className="text-xs tracking-widest text-neutral-500 uppercase mb-2">
          Prompt
        </p>
        <p className="text-neutral-900 font-light leading-relaxed">
          {item.prompt}
        </p>
        {item.attribute && (
          <p className="text-sm text-neutral-600 mt-3">
            Expected attribute: <span className="font-medium">{item.attributeValue}</span>
          </p>
        )}
      </div>

      {/* Image Display */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <p className="text-xs tracking-widest text-neutral-500 uppercase">
            Generated Image
          </p>
          <button
            onClick={() => setShowImage(!showImage)}
            className="text-xs text-neutral-600 hover:text-neutral-900 transition-colors"
          >
            {showImage ? 'Hide Image' : 'Show Image'}
          </button>
        </div>

        {showImage && (
          <div className="relative bg-neutral-100 rounded-lg overflow-hidden border border-neutral-200">
            {!imageError ? (
              <div className="relative w-full aspect-square">
                <Image
                  src={item.imageUrl}
                  alt="Generated image"
                  fill
                  className="object-contain"
                  onError={() => setImageError(true)}
                  priority
                />
              </div>
            ) : (
              <div className="w-full aspect-square flex items-center justify-center bg-neutral-50">
                <div className="text-center p-8">
                  <p className="text-neutral-900 font-medium mb-2">Image unavailable</p>
                  <p className="text-sm text-neutral-500">
                    This may indicate a refusal or error
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Question 1: Refusal Detection */}
      <div className="space-y-4">
        <div>
          <p className="text-sm font-medium text-neutral-900 mb-1">
            1. Is this a refusal?
          </p>
          <p className="text-xs text-neutral-500">
            Was the request explicitly blocked, modified, or returned an error?
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setIsRefusal(true)}
            className={`flex-1 btn-choice ${isRefusal === true ? 'active' : ''}`}
          >
            Yes (Refused)
          </button>
          <button
            onClick={() => setIsRefusal(false)}
            className={`flex-1 btn-choice ${isRefusal === false ? 'active' : ''}`}
          >
            No (Generated)
          </button>
        </div>
      </div>

      {/* Question 2: Attribute Retention */}
      <div className="space-y-4">
        <div>
          <p className="text-sm font-medium text-neutral-900 mb-1">
            2. Is the requested attribute present?
          </p>
          {item.attributeValue && (
            <p className="text-xs text-neutral-500">
              Looking for: {item.attributeValue}
            </p>
          )}
        </div>
        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={() => setAttributePresent('yes')}
            className={`btn-choice ${attributePresent === 'yes' ? 'active' : ''}`}
          >
            Yes, clearly
          </button>
          <button
            onClick={() => setAttributePresent('partial')}
            className={`btn-choice ${attributePresent === 'partial' ? 'active' : ''}`}
          >
            Partial
          </button>
          <button
            onClick={() => setAttributePresent('no')}
            className={`btn-choice ${attributePresent === 'no' ? 'active' : ''}`}
          >
            No
          </button>
        </div>
      </div>

      {/* Question 3: Faithfulness Rating */}
      <div className="space-y-4">
        <div>
          <p className="text-sm font-medium text-neutral-900 mb-1">
            3. Overall faithfulness to prompt
          </p>
          <p className="text-xs text-neutral-500">
            How well does the image match the entire prompt?
          </p>
        </div>
        <div className="grid grid-cols-5 gap-2">
          {[1, 2, 3, 4, 5].map((rating) => (
            <button
              key={rating}
              onClick={() => setFaithfulness(rating as 1 | 2 | 3 | 4 | 5)}
              className={`btn-choice ${faithfulness === rating ? 'active' : ''}`}
            >
              {rating}
            </button>
          ))}
        </div>
        <div className="flex justify-between text-xs text-neutral-500">
          <span>Very poor</span>
          <span>Excellent</span>
        </div>
      </div>

      {/* Question 4: Confidence */}
      <div className="space-y-4">
        <div>
          <p className="text-sm font-medium text-neutral-900 mb-1">
            4. How confident are you in your evaluation?
          </p>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={() => setConfidence('low')}
            className={`btn-choice ${confidence === 'low' ? 'active' : ''}`}
          >
            Low
          </button>
          <button
            onClick={() => setConfidence('medium')}
            className={`btn-choice ${confidence === 'medium' ? 'active' : ''}`}
          >
            Medium
          </button>
          <button
            onClick={() => setConfidence('high')}
            className={`btn-choice ${confidence === 'high' ? 'active' : ''}`}
          >
            High
          </button>
        </div>
      </div>

      {/* Optional Notes */}
      <div className="space-y-3">
        <p className="text-xs tracking-widest text-neutral-500 uppercase">
          Additional Notes (Optional)
        </p>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Any observations or concerns..."
          className="w-full h-20 px-4 py-3 text-sm border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent resize-none"
        />
      </div>

      {/* Navigation */}
      <div className="flex gap-4">
        {showPrevious && onPrevious && (
          <button onClick={onPrevious} className="btn-secondary flex-1">
            Previous
          </button>
        )}
        <button
          onClick={handleSubmit}
          disabled={!isComplete}
          className="btn-primary flex-1"
        >
          {isComplete ? 'Next (Ctrl+Enter)' : 'Please answer all questions'}
        </button>
      </div>

      {/* Hint */}
      <p className="text-xs text-center text-neutral-400">
        Tip: Use Tab to navigate between options
      </p>
    </div>
  )
}
