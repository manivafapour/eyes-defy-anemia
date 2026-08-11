// Renders one /api/predict response: label, P(anemic) with a threshold marker,
// the exact crop the classifier saw, warnings, and model provenance.
export default function ResultCard({ result }) {
  const {
    probability,
    threshold,
    is_anemic,
    is_mock,
    warnings = [],
    crop_preview,
    segmenter_backend,
    classifier_backend,
  } = result

  const pct = (probability * 100).toFixed(1)
  const thresholdPct = (threshold * 100).toFixed(0)

  return (
    <section
      className={`mt-6 rounded-2xl border border-t-4 border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 ${
        is_anemic ? 'border-t-amber-500' : 'border-t-emerald-500'
      }`}
    >
      <div className="mb-5 flex items-center gap-3">
        <span
          className={`text-2xl font-bold ${
            is_anemic ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'
          }`}
        >
          {is_anemic ? 'Anemic' : 'Non-Anemic'}
        </span>
        {is_mock && (
          <span className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-xs uppercase tracking-wide text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-300">
            demo
          </span>
        )}
      </div>

      <div className="mb-5">
        <div className="mb-1.5 flex justify-between text-sm text-slate-500 dark:text-slate-400">
          <span>P(anemic)</span>
          <span className="font-semibold text-slate-900 dark:text-slate-100">{pct}%</span>
        </div>
        <div className="relative h-3 rounded-full bg-slate-200 dark:bg-slate-700">
          <div
            className={`absolute inset-y-0 left-0 rounded-full ${is_anemic ? 'bg-amber-500' : 'bg-emerald-500'}`}
            style={{ width: `${pct}%` }}
          />
          <div
            className="absolute -inset-y-1 w-0.5 bg-slate-500/60"
            style={{ left: `${thresholdPct}%` }}
            title={`threshold ${thresholdPct}%`}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-slate-400">
          <span>0%</span>
          <span>threshold {thresholdPct}%</span>
          <span>100%</span>
        </div>
      </div>

      {crop_preview && (
        <figure className="mb-5 rounded-xl bg-black p-3 text-center">
          <img src={crop_preview} alt="analyzed conjunctiva region" className="mx-auto max-h-56 rounded-md" />
          <figcaption className="mt-2 text-xs text-slate-300">
            Region analyzed — Stage&nbsp;1 output → classifier input
          </figcaption>
        </figure>
      )}

      {warnings.length > 0 && (
        <ul className="mb-5 space-y-1 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
          {warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}

      <footer className="flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-200 pt-3 text-xs text-slate-400 dark:border-slate-800">
        <span>segmenter: {segmenter_backend}</span>
        <span>classifier: {classifier_backend}</span>
      </footer>
    </section>
  )
}
