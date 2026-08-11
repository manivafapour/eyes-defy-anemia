const TEXT =
  'Research screening tool only. Not a medical device and not a substitute for clinical ' +
  'diagnosis. Confirm any result with a laboratory hemoglobin test.'

export default function Disclaimer() {
  return (
    <footer className="mt-8 border-t border-slate-200 pt-4 text-center text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
      <span className="font-semibold">⚕ Disclaimer.</span> {TEXT}
    </footer>
  )
}
