'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { onAuthChange } from '@/lib/firebase'

export default function ConsentPage() {
  const router = useRouter()
  const [ageConfirmed, setAgeConfirmed] = useState(false)
  const [consentAgreed, setConsentAgreed] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Block access unless signed in; require Prolific ID only for anonymous users.
  useEffect(() => {
    const unsubscribe = onAuthChange((authUser) => {
      if (!authUser) {
        router.push('/')
        return
      }

      const prolificId = localStorage.getItem('prolificId')
      if (authUser.isAnonymous && !prolificId) {
        router.push('/')
      }
    })

    return () => unsubscribe()
  }, [router])

  const canContinue = ageConfirmed && consentAgreed

  const handleContinue = async () => {
    if (!canContinue) return

    setIsSubmitting(true)
    try {
      localStorage.setItem('consentGiven', 'true')
      localStorage.setItem('consentTimestamp', new Date().toISOString())
      router.push('/demographics')
    } catch (error) {
      console.error('Error saving consent:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDecline = () => {
    localStorage.removeItem('prolificId')
    router.push('/')
  }

  return (
    <div className="min-h-screen bg-neutral-50 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <p className="text-xs tracking-widest text-neutral-400 uppercase mb-2">
            Research Study
          </p>
          <h1 className="text-2xl font-light text-neutral-900">
            Informed Consent Form
          </h1>
          <p className="text-sm text-neutral-500 mt-2">
            ACRB: Attribute-Conditioned Refusal Bias Evaluation
          </p>
        </div>

        {/* Progress */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <div className="w-8 h-1 bg-neutral-900 rounded" />
          <div className="w-8 h-1 bg-neutral-200 rounded" />
          <div className="w-8 h-1 bg-neutral-200 rounded" />
          <div className="w-8 h-1 bg-neutral-200 rounded" />
        </div>

        {/* Main Card */}
        <div className="bg-white border border-neutral-200 rounded-lg shadow-sm">
          {/* Info Banner */}
          <div className="bg-blue-50 border-b border-blue-100 px-6 py-4">
            <div className="flex items-start gap-3">
              <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-sm text-blue-800">
                You are being asked to participate in a research study on evaluating refusal bias in AI image generation systems.
                Please read this form carefully before deciding whether to participate.
              </p>
            </div>
          </div>

          {/* Scrollable Content */}
          <div className="px-6 py-6 max-h-[50vh] overflow-y-auto">
            <div className="space-y-6 text-sm text-neutral-700 leading-relaxed">

              {/* Purpose */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Purpose of the Study
                </h3>
                <p>
                  This study investigates how AI image generation systems exhibit differential refusal behavior
                  based on demographic and cultural attributes. We aim to understand whether safety filters
                  disproportionately block or modify content related to specific cultural groups, genders,
                  disabilities, or religions. Your evaluation helps identify and quantify these potential biases.
                </p>
              </section>

              {/* What You Will Do */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  What You Will Do
                </h3>
                <p className="mb-2">You will evaluate approximately 50 AI-generated images by answering four questions for each:</p>
                <ul className="space-y-1 ml-4">
                  <li className="flex items-start gap-2">
                    <span className="text-neutral-400">1.</span>
                    <span>Whether the image represents a &quot;refusal&quot; (blocked content)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-neutral-400">2.</span>
                    <span>Whether the requested attribute is present in the image</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-neutral-400">3.</span>
                    <span>Overall faithfulness of the image to the original prompt</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-neutral-400">4.</span>
                    <span>Your confidence level in your assessment</span>
                  </li>
                </ul>
                <p className="mt-2 text-neutral-500">
                  Estimated time: 15-25 minutes total
                </p>
              </section>

              {/* Risks */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Potential Risks
                </h3>
                <p>
                  This study involves minimal risk, similar to everyday web browsing. Some images may
                  depict cultural content, religious symbols, or accessibility-related themes. All explicit
                  or harmful content has been pre-screened and excluded. You may skip any image that
                  makes you uncomfortable.
                </p>
              </section>

              {/* Benefits */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Benefits
                </h3>
                <p>
                  There is no direct benefit to you. However, your participation contributes to research
                  that may improve fairness in AI systems and inform policy decisions regarding
                  AI governance (EU AI Act, Executive Order 14110).
                </p>
              </section>

              {/* Compensation */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Compensation
                </h3>
                <p>
                  You will receive compensation at a rate of $12-15/hour through Prolific upon
                  successful completion of the survey. Partial compensation may be provided if
                  you complete at least 50% of the evaluations.
                </p>
              </section>

              {/* Voluntary Participation */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Voluntary Participation
                </h3>
                <p>
                  Your participation is entirely voluntary. You may stop at any time by closing the
                  browser window. If you wish to withdraw your responses after submission, contact
                  the research team within 7 days with your Prolific ID.
                </p>
              </section>

              {/* Confidentiality */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Confidentiality &amp; Data Protection
                </h3>
                <p>
                  We collect only your Prolific ID and basic demographic information (age range,
                  cultural background). Your Prolific ID is used solely for payment and quality control,
                  and will be separated from your responses before analysis. De-identified results
                  may be shared in academic publications and public datasets. Data will be stored
                  securely using Firebase with encryption.
                </p>
              </section>

              {/* Eligibility */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Eligibility
                </h3>
                <p>
                  You must be 18 years of age or older to participate in this study. You should be
                  fluent in English and have normal or corrected-to-normal vision.
                </p>
              </section>

              {/* Contact */}
              <section>
                <h3 className="font-semibold text-neutral-900 mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                  Contact Information
                </h3>
                <p className="mb-2">
                  If you have questions about the research, please contact the study team at:
                </p>
                <p className="text-neutral-500 ml-4">
                  acrb-research@example.com
                </p>
                <p className="mt-2">
                  For questions about your rights as a research participant, contact the
                  Institutional Review Board (IRB).
                </p>
              </section>

            </div>
          </div>

          {/* Consent Checkboxes */}
          <div className="border-t border-neutral-200 px-6 py-6 bg-neutral-50">
            <div className="space-y-4">
              <label className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={ageConfirmed}
                  onChange={(e) => setAgeConfirmed(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-neutral-300 text-neutral-900
                           focus:ring-neutral-500 focus:ring-offset-0"
                />
                <span className="text-sm text-neutral-700 group-hover:text-neutral-900">
                  <strong>I confirm that I am 18 years of age or older.</strong>
                </span>
              </label>

              <label className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={consentAgreed}
                  onChange={(e) => setConsentAgreed(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-neutral-300 text-neutral-900
                           focus:ring-neutral-500 focus:ring-offset-0"
                />
                <span className="text-sm text-neutral-700 group-hover:text-neutral-900">
                  <strong>I have read and understood the consent information above.</strong> I voluntarily
                  agree to participate and understand that I can withdraw at any time without penalty.
                </span>
              </label>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="border-t border-neutral-200 px-6 py-4 flex gap-3">
            <button
              onClick={handleDecline}
              className="flex-1 px-4 py-3 text-sm font-medium text-neutral-600
                       bg-white border border-neutral-300 rounded-lg
                       hover:bg-neutral-50 transition-colors"
            >
              Decline &amp; Exit
            </button>
            <button
              onClick={handleContinue}
              disabled={!canContinue || isSubmitting}
              className={`flex-1 px-4 py-3 text-sm font-medium rounded-lg transition-colors
                ${canContinue && !isSubmitting
                  ? 'bg-neutral-900 text-white hover:bg-neutral-800'
                  : 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
                }`}
            >
              {isSubmitting ? 'Processing...' : 'I Agree — Continue'}
            </button>
          </div>
        </div>

        {/* Footer */}
        <p className="text-xs text-neutral-400 text-center mt-6">
          Your participation is valuable to this research. Thank you for your time.
        </p>
      </div>
    </div>
  )
}
