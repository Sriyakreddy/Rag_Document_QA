import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Sidebar: Document Manager ───────────────────────────────────────
function DocumentPanel({ documents, onRefresh, onDelete }) {
  return (
    <div className="doc-panel">
      <h3>Indexed Documents</h3>
      {documents.length === 0 ? (
        <p className="doc-empty">No documents yet. Upload one to get started.</p>
      ) : (
        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.name} className="doc-item">
              <div className="doc-info">
                <span className="doc-name">{doc.name}</span>
                <span className="doc-chunks">{doc.chunks} chunks</span>
              </div>
              <button className="doc-delete" onClick={() => onDelete(doc.name)} title="Remove">
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
      <button className="doc-refresh" onClick={onRefresh}>Refresh</button>
    </div>
  )
}

// ── Upload Section ──────────────────────────────────────────────────
function UploadBox({ onUploadComplete }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  async function handleUpload() {
    if (!file) return
    setUploading(true)
    setStatus(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
      const data = await res.json()
      setStatus({ type: 'success', text: `Indexed ${data.chunks_added} chunks from "${data.filename}"` })
      setFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      onUploadComplete?.()
    } catch (err) {
      setStatus({ type: 'error', text: err.message })
    } finally {
      setUploading(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragActive(false)
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) setFile(dropped)
  }

  return (
    <div
      className={`upload-box ${dragActive ? 'drag-active' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <div className="upload-inner">
        <p className="upload-label">
          {file ? file.name : 'Drop a file here or click to browse'}
        </p>
        <p className="upload-hint">Supports PDF, DOCX, TXT, CSV</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="upload-input"
        />
        <button onClick={handleUpload} disabled={!file || uploading} className="btn btn-primary">
          {uploading ? 'Indexing...' : 'Upload & Index'}
        </button>
      </div>
      {status && (
        <p className={`upload-status ${status.type}`}>{status.text}</p>
      )}
    </div>
  )
}

// ── Chat Message ────────────────────────────────────────────────────
function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`msg ${message.role}`}>
      <div className="msg-avatar">{isUser ? 'You' : 'AI'}</div>
      <div className="msg-content">
        {isUser ? (
          <p>{message.text}</p>
        ) : (
          <ReactMarkdown>{message.text}</ReactMarkdown>
        )}
        {message.sources && message.sources.length > 0 && (
          <details className="msg-sources">
            <summary>Sources ({message.sources.length})</summary>
            <ul>
              {message.sources.map((s, i) => (
                <li key={i}>
                  <strong>{s.source}</strong>
                  {s.page != null && <span className="source-page"> p.{s.page}</span>}
                  <p className="source-snippet">{s.snippet}</p>
                </li>
              ))}
            </ul>
          </details>
        )}
        {message.streaming && <span className="cursor-blink">|</span>}
      </div>
    </div>
  )
}

// ── Main App ────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [documents, setDocuments] = useState([])
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load documents on mount
  const loadDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`)
      if (res.ok) {
        const data = await res.json()
        setDocuments(data.documents || [])
      }
    } catch {
      // silently fail — documents panel just shows empty
    }
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  async function handleDeleteDoc(filename) {
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      if (res.ok) loadDocuments()
    } catch { /* ignore */ }
  }

  async function handleAsk(e) {
    e.preventDefault()
    if (!question.trim() || streaming) return

    const userMsg = { role: 'user', text: question }
    setMessages((prev) => [...prev, userMsg])
    setQuestion('')
    setStreaming(true)

    // Add a placeholder assistant message
    const assistantIdx = messages.length + 1
    setMessages((prev) => [...prev, { role: 'assistant', text: '', sources: [], streaming: true }])

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullText = ''
      let sources = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))

            if (event.type === 'sources') {
              sources = event.sources
              setMessages((prev) => {
                const updated = [...prev]
                updated[assistantIdx] = { ...updated[assistantIdx], sources }
                return updated
              })
            } else if (event.type === 'token') {
              fullText += event.token
              setMessages((prev) => {
                const updated = [...prev]
                updated[assistantIdx] = { ...updated[assistantIdx], text: fullText }
                return updated
              })
            } else if (event.type === 'done') {
              setMessages((prev) => {
                const updated = [...prev]
                updated[assistantIdx] = { ...updated[assistantIdx], streaming: false }
                return updated
              })
            } else if (event.type === 'error') {
              throw new Error(event.message)
            }
          } catch (parseErr) {
            // skip malformed SSE lines
          }
        }
      }

      // Ensure streaming flag is cleared
      setMessages((prev) => {
        const updated = [...prev]
        if (updated[assistantIdx]) {
          updated[assistantIdx] = { ...updated[assistantIdx], streaming: false }
        }
        return updated
      })

    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev]
        updated[assistantIdx] = {
          role: 'assistant',
          text: `Error: ${err.message}`,
          sources: [],
          streaming: false,
        }
        return updated
      })
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2 className="logo">RAG Doc Q&A</h2>
        <p className="logo-sub">v2.0 — Production</p>
        <UploadBox onUploadComplete={loadDocuments} />
        <DocumentPanel
          documents={documents}
          onRefresh={loadDocuments}
          onDelete={handleDeleteDoc}
        />
      </aside>

      <main className="chat-area">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Ask anything about your documents</h2>
              <p>Upload a PDF, DOCX, TXT, or CSV in the sidebar, then ask a question.</p>
              <div className="suggestions">
                <button onClick={() => setQuestion('Summarize the key points of this document')}>
                  Summarize key points
                </button>
                <button onClick={() => setQuestion('What are the main topics covered?')}>
                  Main topics covered
                </button>
                <button onClick={() => setQuestion('List any specific dates, numbers, or figures mentioned')}>
                  Dates and figures
                </button>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleAsk} className="ask-form">
          <input
            type="text"
            placeholder="Ask something about your documents..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={streaming}
          />
          <button type="submit" disabled={streaming || !question.trim()} className="btn btn-send">
            {streaming ? 'Streaming...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}
