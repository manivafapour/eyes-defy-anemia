import { useEffect, useState } from 'react'
import UploadPanel from './components/UploadPanel'
import ResultCard from './components/ResultCard'
import Disclaimer from './components/Disclaimer'
import { predict, getVersion } from './api'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [version, setVersion] = useState(null)

  useEffect(() => { getVersion().then(setVersion).catch(() => setVersion(null)) }, [])
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  function handleSelect(selected) {
    setError(null)
    setResult(null)
    setFile(selected)
    setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(selected) })
  }

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await predict(file))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setResult(null)
    setError(null)
    setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null })
  }

  const mockSegmenter = version?.segmenter?.is_mock

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto max-w-2xl px-5 py-10">
        <header className="mb-6 text-center">
          <h1 className="text-3xl font-bold tracking-tight">Eyes-Defy-Anemia</h1>
          <p className="mt-1 text-slate-500 dark:text-slate-400">
            Non-invasive anemia screening from a conjunctiva photo
          </p>
        </header>

        {mockSegmenter && (
          <div className="mb-5 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
            <span className="font-semibold">Demo mode.</span> Stage&nbsp;1 (segmentation) is
            simulated — the classifier is real, but it runs on a placeholder crop, so results are
            not clinically meaningful yet.
          </div>
        )}

        <UploadPanel
          previewUrl={previewUrl}
          onSelect={handleSelect}
          onAnalyze={handleAnalyze}
          onReset={handleReset}
          hasFile={!!file}
          loading={loading}
        />

        {error && (
          <div className="mt-5 rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300">
            {error}
          </div>
        )}

        {loading && <LoadingCard />}
        {!loading && result && <ResultCard result={result} />}

        <Disclaimer />
      </div>
    </div>
  )
}

function LoadingCard() {
  return (
    <div className="mt-6 flex items-center gap-3 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600 dark:border-slate-700 dark:border-t-blue-400" />
      <span className="text-slate-600 dark:text-slate-300">Analyzing image…</span>
    </div>
  )
}
