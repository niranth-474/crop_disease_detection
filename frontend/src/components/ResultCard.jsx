import React from 'react'

function ResultCard({ result, loading, error }) {
  if (loading) {
    return (
      <div className="mt-8 flex items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 p-10">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-200 border-t-emerald-600" />
      </div>
    )
  }

  if (error) {
    return <p className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>
  }

  if (!result) {
    return (
      <div className="mt-8 rounded-2xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
        Upload an image to begin inference.
      </div>
    )
  }

  return (
    <div className="mt-6 space-y-4">
      <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-emerald-700">Predicted class</p>
        <p className="mt-2 text-2xl font-semibold text-slate-900">{result.predicted_class}</p>
        <p className="mt-2 text-sm text-slate-600">Confidence: {result.confidence}%</p>
      </div>
    </div>
  )
}

export default ResultCard
