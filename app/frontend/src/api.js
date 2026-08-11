// Thin API client. Relative paths so it works in dev (Vite proxy) and prod
// (FastAPI serves this SPA same-origin). Errors surface the API's `detail`.
const BASE = '/api'

export async function predict(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: form })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`)
  }
  return data
}

export async function getVersion() {
  const res = await fetch(`${BASE}/version`)
  if (!res.ok) throw new Error(`Version check failed (${res.status})`)
  return res.json()
}
