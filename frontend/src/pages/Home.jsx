import { useMemo, useState } from 'react'
import { predictImage } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import ResultCard from '../components/ResultCard'

function Home() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const onFileSelect = (selectedFile) => {
    if (!selectedFile) return
    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setError('')
    setResult(null)
  }

  const uploadImage = async () => {
    if (!file) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await predictImage(file)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try another image.')
    } finally {
      setLoading(false)
    }
  }

  const previewLabel = useMemo(() => {
    if (!file) return 'No file selected yet'
    return file.name
  }, [file])

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-10 text-slate-800">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-xl shadow-slate-200/75 backdrop-blur sm:p-10">
        <header className="space-y-3">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-emerald-600">Crop Leaf Disease Detection</p>
          <h1 className="text-3xl font-semibold sm:text-4xl">Upload a leaf image to identify potential disease</h1>
          <p className="max-w-2xl text-base text-slate-600">
            This interface sends your image to the FastAPI backend for inference using your trained PyTorch model.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <ImageUploader onFileSelect={onFileSelect} previewUrl={previewUrl} previewLabel={previewLabel} loading={loading} />
            <button
              onClick={uploadImage}
              className="mt-6 w-full rounded-2xl bg-emerald-600 px-4 py-3 font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400"
              disabled={loading || !file}
            >
              {loading ? 'Analyzing...' : 'Predict disease'}
            </button>
            {error ? <p className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
          </div>

          <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <h2 className="text-xl font-semibold">Prediction result</h2>
            <p className="mt-2 text-sm text-slate-600">The top prediction and confidence will appear here.</p>
            <ResultCard result={result} loading={loading} error={error} />
          </section>
        </div>
      </div>
    </div>
  )
}

export default Home
