import { useRef, useState } from 'react'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']

// Client-side validation mirrors the backend limits so the user gets instant
// feedback instead of a round-trip 413/415.
export default function UploadPanel({ previewUrl, onSelect, onAnalyze, onReset, hasFile, loading }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [localError, setLocalError] = useState(null)

  function validateAndSelect(file) {
    setLocalError(null)
    if (!file) return
    if (!ACCEPT.includes(file.type)) {
      setLocalError('Please upload a JPEG, PNG, or WebP image.')
      return
    }
    if (file.size > MAX_BYTES) {
      setLocalError('Image is larger than 10 MB.')
      return
    }
    onSelect(file)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    validateAndSelect(e.dataTransfer.files?.[0])
  }

  const zoneClasses = [
    'flex min-h-[240px] cursor-pointer items-center justify-center rounded-xl border-2 border-dashed p-4 text-center transition-colors',
    dragOver ? 'border-blue-500 bg-blue-50 dark:bg-blue-500/10' : 'border-slate-300 dark:border-slate-700',
    previewUrl ? 'border-solid' : '',
  ].join(' ')

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div
        role="button"
        tabIndex={0}
        className={zoneClasses}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="selected eye" className="max-h-80 max-w-full rounded-lg object-contain" />
        ) : (
          <div className="text-slate-500 dark:text-slate-400">
            <div className="mb-2 text-4xl">📷</div>
            <p className="font-medium text-slate-700 dark:text-slate-200">Drop an eye photo here</p>
            <p className="text-sm">or click to browse · JPEG / PNG / WebP · up to 10 MB</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT.join(',')}
          hidden
          onChange={(e) => validateAndSelect(e.target.files?.[0])}
        />
      </div>

      {localError && <p className="mt-2 text-sm text-red-600 dark:text-red-400">{localError}</p>}

      <div className="mt-4 flex gap-3">
        <button
          onClick={onAnalyze}
          disabled={!hasFile || loading}
          className="flex-1 rounded-lg bg-blue-600 px-5 py-2.5 font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
        <button
          onClick={onReset}
          disabled={loading || !hasFile}
          className="rounded-lg border border-slate-300 px-5 py-2.5 font-semibold text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          Reset
        </button>
      </div>
    </section>
  )
}
