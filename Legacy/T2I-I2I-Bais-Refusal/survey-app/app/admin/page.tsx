'use client'

import { useState, useEffect } from 'react'
import {
  getAllEvaluations,
  getAllParticipants,
  getCompletedParticipants,
  batchAddSurveyItems,
  uploadImage,
} from '@/lib/firestore'
import {
  analyzeResults,
  exportToCSV,
  exportToJSON,
} from '@/lib/analytics'
import { Evaluation, Participant, AnalysisResult, SurveyItem } from '@/lib/types'

export default function AdminDashboard() {
  const [evaluations, setEvaluations] = useState<Evaluation[]>([])
  const [participants, setParticipants] = useState<Participant[]>([])
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'upload' | 'export'>('overview')

  // Image upload state
  const [uploadForm, setUploadForm] = useState({
    promptId: '',
    prompt: '',
    attribute: '',
    attributeValue: '',
    model: '',
    domain: '',
    isRefusal: false,
  })
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [evals, parts] = await Promise.all([
        getAllEvaluations(),
        getAllParticipants(),
      ])

      setEvaluations(evals)
      setParticipants(parts)

      const analysisResult = analyzeResults(evals, parts)
      setAnalysis(analysisResult)

      setLoading(false)
    } catch (error) {
      console.error('Error loading data:', error)
      setLoading(false)
    }
  }

  const handleExportCSV = () => {
    const csv = exportToCSV(evaluations)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `acrb-evaluations-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
  }

  const handleExportJSON = () => {
    const json = exportToJSON({
      evaluations,
      participants,
      analysis: analysis!,
    })
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `acrb-full-data-${new Date().toISOString().split('T')[0]}.json`
    a.click()
  }

  const handleImageUpload = async () => {
    if (!uploadFile || !uploadForm.promptId) {
      alert('Please fill all required fields')
      return
    }

    setUploading(true)
    try {
      // Upload image to Firebase Storage
      const imageUrl = await uploadImage(uploadFile, uploadForm.promptId, uploadForm.model)

      // Create survey item
      const surveyItem: Omit<SurveyItem, 'id'> = {
        type: 'evaluation',
        imageUrl,
        prompt: uploadForm.prompt,
        attribute: uploadForm.attribute,
        attributeValue: uploadForm.attributeValue,
        model: uploadForm.model,
        domain: uploadForm.domain,
      }

      await batchAddSurveyItems([surveyItem])

      alert('Image uploaded successfully!')

      // Reset form
      setUploadForm({
        promptId: '',
        prompt: '',
        attribute: '',
        attributeValue: '',
        model: '',
        domain: '',
        isRefusal: false,
      })
      setUploadFile(null)
      setUploading(false)
    } catch (error) {
      console.error('Error uploading:', error)
      alert('Failed to upload image')
      setUploading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-neutral-900 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-neutral-600">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <h1 className="text-3xl font-light text-neutral-900 mb-2">
            ACRB Admin Dashboard
          </h1>
          <p className="text-neutral-600">Human evaluation data management</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 border-b border-neutral-200">
          <button
            onClick={() => setActiveTab('overview')}
            className={`pb-3 px-4 text-sm font-medium transition-colors ${
              activeTab === 'overview'
                ? 'text-neutral-900 border-b-2 border-neutral-900'
                : 'text-neutral-500 hover:text-neutral-700'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`pb-3 px-4 text-sm font-medium transition-colors ${
              activeTab === 'upload'
                ? 'text-neutral-900 border-b-2 border-neutral-900'
                : 'text-neutral-500 hover:text-neutral-700'
            }`}
          >
            Upload Images
          </button>
          <button
            onClick={() => setActiveTab('export')}
            className={`pb-3 px-4 text-sm font-medium transition-colors ${
              activeTab === 'export'
                ? 'text-neutral-900 border-b-2 border-neutral-900'
                : 'text-neutral-500 hover:text-neutral-700'
            }`}
          >
            Export Data
          </button>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && analysis && (
          <div className="space-y-8">
            {/* Summary Cards */}
            <div className="grid grid-cols-4 gap-6">
              <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
                <p className="text-xs tracking-widest text-neutral-500 uppercase mb-2">
                  Total Participants
                </p>
                <p className="text-3xl font-light text-neutral-900">
                  {analysis.totalParticipants}
                </p>
                <p className="text-sm text-neutral-500 mt-1">
                  {analysis.completedParticipants} completed
                </p>
              </div>

              <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
                <p className="text-xs tracking-widest text-neutral-500 uppercase mb-2">
                  Total Evaluations
                </p>
                <p className="text-3xl font-light text-neutral-900">
                  {analysis.totalEvaluations}
                </p>
              </div>

              <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
                <p className="text-xs tracking-widest text-neutral-500 uppercase mb-2">
                  Avg Completion Time
                </p>
                <p className="text-3xl font-light text-neutral-900">
                  {analysis.averageCompletionTimeMinutes.toFixed(1)}
                </p>
                <p className="text-sm text-neutral-500 mt-1">minutes</p>
              </div>

              <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
                <p className="text-xs tracking-widest text-neutral-500 uppercase mb-2">
                  Attention Pass Rate
                </p>
                <p className="text-3xl font-light text-neutral-900">
                  {analysis.attentionCheckPassRate.toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Agreement Metrics */}
            {analysis.agreement && (
              <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
                <h2 className="text-lg font-medium text-neutral-900 mb-4">
                  Inter-Rater Agreement
                </h2>
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-neutral-600 mb-2">Cohen's Kappa</p>
                    <p className="text-2xl font-light text-neutral-900">
                      {analysis.agreement.cohensKappa.toFixed(3)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-neutral-600 mb-2">Percent Agreement</p>
                    <p className="text-2xl font-light text-neutral-900">
                      {analysis.agreement.percentAgreement.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* By Attribute */}
            <div className="bg-white p-6 rounded-lg border border-neutral-200">
              <h2 className="text-lg font-medium text-neutral-900 mb-4">
                Results by Attribute
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-neutral-200">
                    <tr>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900">
                        Attribute
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Evaluations
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Refusal Rate
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Avg Faithfulness
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Attr Present Rate
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.byAttribute).map(([attr, data]) => (
                      <tr key={attr} className="border-b border-neutral-100">
                        <td className="py-3 px-4 text-neutral-900">{attr}</td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.evaluationCount}
                        </td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.refusalRate.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.averageFaithfulness.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.attributePresentRate.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* By Model */}
            <div className="bg-white p-6 rounded-lg border border-neutral-200">
              <h2 className="text-lg font-medium text-neutral-900 mb-4">
                Results by Model
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-neutral-200">
                    <tr>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900">
                        Model
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Evaluations
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Refusal Rate
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-neutral-900">
                        Avg Faithfulness
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.byModel).map(([model, data]) => (
                      <tr key={model} className="border-b border-neutral-100">
                        <td className="py-3 px-4 text-neutral-900">{model}</td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.evaluationCount}
                        </td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.refusalRate.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-right text-neutral-600">
                          {data.averageFaithfulness.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-2xl">
            <div className="bg-white p-8 rounded-lg border border-neutral-200 space-y-6">
              <h2 className="text-xl font-light text-neutral-900 mb-6">
                Upload Survey Image
              </h2>

              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Prompt ID *
                </label>
                <input
                  type="text"
                  value={uploadForm.promptId}
                  onChange={(e) => setUploadForm({ ...uploadForm, promptId: e.target.value })}
                  className="w-full"
                  placeholder="e.g., prompt_001"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Prompt Text *
                </label>
                <textarea
                  value={uploadForm.prompt}
                  onChange={(e) => setUploadForm({ ...uploadForm, prompt: e.target.value })}
                  className="w-full h-20"
                  placeholder="A Korean person at a wedding"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Attribute *
                  </label>
                  <input
                    type="text"
                    value={uploadForm.attribute}
                    onChange={(e) => setUploadForm({ ...uploadForm, attribute: e.target.value })}
                    className="w-full"
                    placeholder="culture"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Attribute Value *
                  </label>
                  <input
                    type="text"
                    value={uploadForm.attributeValue}
                    onChange={(e) => setUploadForm({ ...uploadForm, attributeValue: e.target.value })}
                    className="w-full"
                    placeholder="Korean"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Model *
                  </label>
                  <input
                    type="text"
                    value={uploadForm.model}
                    onChange={(e) => setUploadForm({ ...uploadForm, model: e.target.value })}
                    className="w-full"
                    placeholder="flux-2-dev"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Domain *
                  </label>
                  <input
                    type="text"
                    value={uploadForm.domain}
                    onChange={(e) => setUploadForm({ ...uploadForm, domain: e.target.value })}
                    className="w-full"
                    placeholder="social"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Image File *
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full"
                />
              </div>

              <button
                onClick={handleImageUpload}
                disabled={uploading}
                className="btn-primary w-full"
              >
                {uploading ? 'Uploading...' : 'Upload Image'}
              </button>
            </div>
          </div>
        )}

        {/* Export Tab */}
        {activeTab === 'export' && (
          <div className="max-w-2xl space-y-6">
            <div className="bg-white p-8 rounded-lg border border-neutral-200">
              <h2 className="text-xl font-light text-neutral-900 mb-4">
                Export Evaluations (CSV)
              </h2>
              <p className="text-neutral-600 mb-6 text-sm">
                Download all evaluation data in CSV format for statistical analysis
              </p>
              <button onClick={handleExportCSV} className="btn-primary w-full">
                Download CSV
              </button>
            </div>

            <div className="bg-white p-8 rounded-lg border border-neutral-200">
              <h2 className="text-xl font-light text-neutral-900 mb-4">
                Export Full Dataset (JSON)
              </h2>
              <p className="text-neutral-600 mb-6 text-sm">
                Download complete dataset including participants, evaluations, and analysis
              </p>
              <button onClick={handleExportJSON} className="btn-primary w-full">
                Download JSON
              </button>
            </div>

            <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
              <p className="text-sm text-neutral-600">
                <strong>Note:</strong> Exported data is anonymized. Prolific IDs are hashed
                and no personally identifiable information is included.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
