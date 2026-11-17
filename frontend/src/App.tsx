import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

type Health = {
  status: string
  models_loaded: { detector: boolean; embedder: boolean; metric_adapter: boolean }
  gallery: { objects_count: number; total_embeddings: number; use_adapted_metric: boolean }
}
type ObjectItem = {
  id: number; name: string; embedding_count: number; last_seen_phrase: string | null
  location: string | null; confuser_group: number | null
}
type DetectionItem = {
  name: string | null; score: number; decision: string; direction: { horizontal: string; vertical: string }
  distance_phrase: string; spoken_phrase: string
}
type Recognition = {
  detections: DetectionItem[]; spoken_announcements: string[]; guidance_hint: string | null
  find_status: { found: boolean; spoken_phrase: string | null; beep: { enabled: boolean; interval_ms: number; freq_hz: number; pan: number } } | null
  fps: number | null
}
type QueueItem = { id: string; name: string; file: File }
type Tab = 'live' | 'teach' | 'memory'

const api = (path: string) => `/api${path}`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(api(path), init)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json() as { detail?: string }
      if (body.detail) message = body.detail
    } catch { /* the response may not be JSON */ }
    throw new Error(message)
  }
  return await response.json() as T
}

function speak(text: string) {
  if (!text.trim() || !('speechSynthesis' in window)) return
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = 'en-IN'
  utterance.rate = 0.94
  window.speechSynthesis.speak(utterance)
}

function App() {
  const [tab, setTab] = useState<Tab>('live')
  const [health, setHealth] = useState<Health | null>(null)
  const [objects, setObjects] = useState<ObjectItem[]>([])
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [cameraError, setCameraError] = useState('')
  const [cameraBusy, setCameraBusy] = useState(false)
  const [live, setLive] = useState(false)
  const [mode, setMode] = useState<'IDENTIFY' | 'FIND'>('IDENTIFY')
  const [target, setTarget] = useState('')
  const [recognition, setRecognition] = useState<Recognition | null>(null)
  const [speechOn, setSpeechOn] = useState(true)
  const [notice, setNotice] = useState('Connect to the local vision service to begin.')
  const [objectName, setObjectName] = useState('')
  const [recording, setRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)
  const [queue, setQueue] = useState<QueueItem[]>([])
  const [batchBusy, setBatchBusy] = useState(false)
  const [renameId, setRenameId] = useState<number | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [lastError, setLastError] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recordTimerRef = useRef<number | undefined>(undefined)
  const busyRef = useRef(false)
  const lastSpeechRef = useRef('')
  const lastBeepRef = useRef(0)
  const audioRef = useRef<AudioContext | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [newHealth, newObjects] = await Promise.all([
        request<Health>('/health'), request<ObjectItem[]>('/objects'),
      ])
      setHealth(newHealth)
      setObjects(newObjects)
      setLastError('')
    } catch (error) {
      setLastError(error instanceof Error ? error.message : String(error))
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, 5000)
    return () => window.clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    streamRef.current = cameraStream
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream
      void videoRef.current.play().catch(() => undefined)
    }
  }, [cameraStream, tab])

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
    if (audioRef.current) void audioRef.current.close()
  }, [])

  const startCamera = async () => {
    setCameraBusy(true)
    setCameraError('')
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not provide camera access. Open the app on localhost or HTTPS.')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 960 }, height: { ideal: 720 } },
        audio: false,
      })
      setCameraStream(stream)
      streamRef.current = stream
      audioRef.current ??= new AudioContext()
      if (audioRef.current.state === 'suspended') await audioRef.current.resume()
      setNotice('Camera is ready. Start live recognition or record an object video.')
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : String(error))
    } finally {
      setCameraBusy(false)
    }
  }

  const stopCamera = () => {
    setLive(false)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraStream(null)
    setRecognition(null)
    setNotice('Camera stopped.')
  }

  const sendCurrentFrame = useCallback(async () => {
    const video = videoRef.current
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || busyRef.current) return
    busyRef.current = true
    const canvas = document.createElement('canvas')
    const scale = Math.min(1, 960 / video.videoWidth)
    canvas.width = Math.round(video.videoWidth * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    const drawing = canvas.getContext('2d')
    if (!drawing) { busyRef.current = false; return }
    drawing.drawImage(video, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.78))
    if (!blob) { busyRef.current = false; return }
    const form = new FormData()
    form.append('frame', blob, 'camera.jpg')
    form.append('mode', mode)
    if (mode === 'FIND') form.append('target', target.trim())
    try {
      const result = await request<Recognition>('/recognize', { method: 'POST', body: form })
      setRecognition(result)
      const phrase = mode === 'FIND'
        ? result.find_status?.spoken_phrase
        : result.spoken_announcements[0] || result.guidance_hint
      if (speechOn && phrase && phrase !== lastSpeechRef.current) {
        lastSpeechRef.current = phrase
        speak(phrase)
      }
      if (!phrase) lastSpeechRef.current = ''
      const beep = result.find_status?.beep
      if (mode === 'FIND' && beep?.enabled && audioRef.current) {
        const now = performance.now()
        if (now - lastBeepRef.current >= beep.interval_ms) {
          lastBeepRef.current = now
          const audio = audioRef.current
          const oscillator = audio.createOscillator()
          const gain = audio.createGain()
          const pan = audio.createStereoPanner()
          oscillator.frequency.value = beep.freq_hz
          gain.gain.setValueAtTime(0.0001, audio.currentTime)
          gain.gain.exponentialRampToValueAtTime(0.12, audio.currentTime + 0.015)
          gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.12)
          pan.pan.value = Math.max(-1, Math.min(1, beep.pan))
          oscillator.connect(gain).connect(pan).connect(audio.destination)
          oscillator.start()
          oscillator.stop(audio.currentTime + 0.13)
        }
      }
    } catch (error) {
      setLastError(error instanceof Error ? error.message : String(error))
      if (lastSpeechRef.current !== 'Recognition service error.') {
        lastSpeechRef.current = 'Recognition service error.'
        if (speechOn) speak('The vision service is not ready. Please check the screen.')
      }
    } finally {
      busyRef.current = false
    }
  }, [mode, target, speechOn])

  useEffect(() => {
    if (!live || !cameraStream) return
    let cancelled = false
    let timeout = 0
    const tick = async () => {
      if (cancelled) return
      await sendCurrentFrame()
      if (!cancelled) timeout = window.setTimeout(() => { void tick() }, 120)
    }
    void tick()
    return () => { cancelled = true; window.clearTimeout(timeout) }
  }, [live, cameraStream, sendCurrentFrame])

  const startRecording = () => {
    const stream = streamRef.current
    const name = objectName.trim()
    if (!stream) { setNotice('Start the camera before recording.'); return }
    if (!name) { setNotice('Enter the personal object name first.'); return }
    if (queue.length >= 10) { setNotice('A registration batch can contain at most 10 objects.'); return }
    if (queue.some((item) => item.name.toLocaleLowerCase() === name.toLocaleLowerCase())) {
      setNotice('Each queued video needs a different object name.'); return
    }
    if (!('MediaRecorder' in window)) { setNotice('This browser cannot record video. Upload a 5–10 second video file instead.'); return }

    const mimeType = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4']
      .find((type) => MediaRecorder.isTypeSupported(type))
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    const chunks: BlobPart[] = []
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data) }
    recorder.onstop = () => {
      const type = recorder.mimeType || 'video/webm'
      const ext = type.includes('mp4') ? 'mp4' : 'webm'
      const file = new File(chunks, `${name.replace(/[^a-z0-9_-]+/gi, '_')}.${ext}`, { type })
      setQueue((current) => [...current, { id: crypto.randomUUID(), name, file }])
      setObjectName('')
      setRecording(false)
      setRecordSeconds(0)
      setNotice(`Recorded a six-second video for ${name}. Add another item or register this batch.`)
      if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
    }
    recorderRef.current = recorder
    recorder.start(250)
    setRecording(true)
    setRecordSeconds(0)
    setNotice('Recording. Hold the object centered and slowly turn it so different sides are visible.')
    recordTimerRef.current = window.setInterval(() => setRecordSeconds((seconds) => {
      if (seconds >= 5) recorder.stop()
      return seconds + 1
    }), 1000)
  }

  const cancelRecording = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
    setRecording(false)
    setRecordSeconds(0)
    setNotice('Recording stopped and kept in the registration queue.')
  }

  const addVideoFiles = (files: FileList | null) => {
    if (!files) return
    const incoming = Array.from(files).slice(0, Math.max(0, 10 - queue.length)).map((file) => ({
      id: crypto.randomUUID(),
      name: file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim(),
      file,
    }))
    setQueue((current) => [...current, ...incoming])
    setNotice(`${incoming.length} video file(s) added. Check each object name before registration.`)
  }

  const registerBatch = async () => {
    if (!queue.length || batchBusy) return
    if (queue.some((item) => !item.name.trim())) { setNotice('Every video needs an object name.'); return }
    if (new Set(queue.map((item) => item.name.toLocaleLowerCase())).size !== queue.length) {
      setNotice('Object names must be unique within a batch.'); return
    }
    setBatchBusy(true)
    setNotice('Processing video views, creating object embeddings, and updating the metric head…')
    const form = new FormData()
    queue.forEach((item) => { form.append('videos', item.file, item.file.name); form.append('names', item.name) })
    try {
      const result = await request<{ success: boolean; registered_count: number; failures: { name: string; detail: string }[]; joint_adaptation: string; adaptation_error: string | null }>(
        '/register/batch', { method: 'POST', body: form },
      )
      const failedNames = new Set(result.failures.map((failure) => failure.name.toLocaleLowerCase()))
      setQueue((current) => current.filter((item) => failedNames.has(item.name.toLocaleLowerCase())))
      setNotice(`${result.registered_count} video(s) registered. ${result.joint_adaptation}${result.adaptation_error ? ` Metric adaptation/checkpoint warning: ${result.adaptation_error}` : ''}${result.failures.length ? ` ${result.failures.length} item(s) need correction.` : ''}`)
      await refresh()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error))
    } finally {
      setBatchBusy(false)
    }
  }

  const renameObject = async (item: ObjectItem) => {
    if (!renameValue.trim()) return
    try {
      await request(`/objects/${item.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: renameValue.trim() }) })
      setRenameId(null)
      await refresh()
      setNotice(`Renamed object to ${renameValue.trim()}.`)
    } catch (error) { setNotice(error instanceof Error ? error.message : String(error)) }
  }

  const deleteObject = async (item: ObjectItem) => {
    if (!window.confirm(`Delete “${item.name}” and its stored views?`)) return
    try {
      await request(`/objects/${item.id}`, { method: 'DELETE' })
      await refresh()
      setNotice(`${item.name} removed from your local gallery.`)
    } catch (error) { setNotice(error instanceof Error ? error.message : String(error)) }
  }

  const clearGallery = async () => {
    if (!window.confirm('Permanently clear all enrolled objects, embeddings, and sightings from this computer?')) return
    try {
      await request('/objects', { method: 'DELETE' })
      await refresh()
      setNotice('The personal gallery and object memory were cleared.')
    } catch (error) { setNotice(error instanceof Error ? error.message : String(error)) }
  }

  const loaded = Boolean(health?.models_loaded.detector && health?.models_loaded.embedder)

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">V</div>
        <div className="brand-copy"><p className="eyebrow">LOCAL COMPUTER VISION</p><h1>See & Remember</h1></div>
        <div className={`service-status ${loaded ? 'ready' : ''}`} role="status">
          <span className="status-dot" />{loaded ? 'Vision system ready' : 'Connecting to vision system'}
        </div>
      </header>

      <nav className="tabs" aria-label="Project sections">
        <button className={tab === 'live' ? 'active' : ''} onClick={() => setTab('live')}>Live assistant</button>
        <button className={tab === 'teach' ? 'active' : ''} onClick={() => setTab('teach')}>Teach personal objects</button>
        <button className={tab === 'memory' ? 'active' : ''} onClick={() => setTab('memory')}>Object memory</button>
      </nav>

      {lastError && <div className="banner error" role="alert">{lastError}. Start the backend with <code>python run.py --mode web</code>.</div>}
      <div className="sr-only" aria-live="polite">{notice}</div>

      {tab === 'live' && <section className="page-grid">
        <div className="main-column">
          <section className="panel camera-panel">
            <div className="panel-heading">
              <div><p className="eyebrow">REAL-TIME PERSONAL OBJECT RECOGNITION</p><h2>What is around you?</h2></div>
              <span className={`tag ${loaded ? 'tag-green' : ''}`}>{loaded ? 'Models loaded' : 'Waiting for models'}</span>
            </div>
            <div className="camera-frame">
              {cameraStream
                ? <video ref={videoRef} autoPlay muted playsInline aria-label="Live camera preview" />
                : <div className="camera-placeholder"><span className="camera-icon" aria-hidden="true">◉</span><strong>Your camera preview appears here</strong><span>Video stays on this computer and is sent only to the local service for recognition.</span></div>}
              {live && <div className="live-indicator"><span /> LIVE</div>}
            </div>
            {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}
            <div className="control-row">
              {!cameraStream
                ? <button className="button primary" onClick={() => void startCamera()} disabled={cameraBusy}>{cameraBusy ? 'Requesting camera…' : 'Allow camera access'}</button>
                : <button className="button secondary" onClick={stopCamera}>Stop camera</button>}
              <button className="button primary" onClick={() => { if (!cameraStream) setNotice('Allow camera access first.'); else if (mode === 'FIND' && !target.trim()) setNotice('Enter the object you want to find first.'); else { setLastError(''); setLive((value) => !value); lastSpeechRef.current = ''; } }} disabled={!cameraStream || !loaded}>
                {live ? 'Pause recognition' : 'Start live recognition'}
              </button>
              <button className={`button ${speechOn ? 'secondary' : 'muted-button'}`} onClick={() => setSpeechOn((on) => !on)} aria-pressed={speechOn}>
                {speechOn ? 'Speech on' : 'Speech off'}
              </button>
            </div>
            <p className="help-text">On a phone, open this page on the same Wi-Fi network only if the server was deliberately configured for network access. The default server is private to this computer.</p>
          </section>

          <section className="panel results-panel">
            <div className="panel-heading compact"><div><p className="eyebrow">AUDIO GUIDANCE</p><h2>Recognition results</h2></div><span className="fps">{recognition?.fps ? `${recognition.fps} req/s` : '—'}</span></div>
            <div className="mode-switch" role="group" aria-label="Assistant mode">
              <button className={mode === 'IDENTIFY' ? 'active' : ''} onClick={() => setMode('IDENTIFY')}>Identify objects</button>
              <button className={mode === 'FIND' ? 'active' : ''} onClick={() => setMode('FIND')}>Find one of mine</button>
            </div>
            {mode === 'FIND' && <label className="field-label">Object to find
              <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder="For example, my keys" list="object-names" />
              <datalist id="object-names">{objects.map((item) => <option key={item.id} value={item.name} />)}</datalist>
            </label>}
            {recognition?.guidance_hint && <div className="guidance-hint" role="status"><span aria-hidden="true">↗</span>{recognition.guidance_hint}</div>}
            {recognition?.find_status?.spoken_phrase && <div className="guidance-hint" role="status">{recognition.find_status.spoken_phrase}</div>}
            <ul className="result-list" aria-label="Detected objects">
              {recognition?.detections.map((detection, index) => <li key={`${detection.name}-${index}`}>
                <span className={`decision-dot ${detection.name ? 'known' : ''}`} />
                <div><strong>{detection.name || (detection.decision === 'AMBIGUOUS' ? 'Not sure yet' : 'Unfamiliar object')}</strong>
                  <span>{detection.name ? `${detection.direction.horizontal}, ${detection.distance_phrase}` : 'Move closer or turn the item for a clearer view.'}</span></div>
                {detection.name && <span className="score">{Math.round(detection.score * 100)}%</span>}
              </li>)}
              {!recognition?.detections.length && <li className="empty-result"><span>{live ? 'Looking for objects…' : 'Start the camera to hear and see personal object results.'}</span></li>}
            </ul>
          </section>
        </div>

        <aside className="side-column">
          <section className="panel summary-panel"><p className="eyebrow">YOUR PERSONAL GALLERY</p><div className="summary-number">{objects.length}<span> / 10</span></div><p>personal items taught to the recognizer</p>
            <div className="progress-track"><span style={{ width: `${Math.min(100, objects.length * 10)}%` }} /></div>
            <div className="summary-foot"><span>{health?.gallery.total_embeddings ?? 0} stored views</span><span>{health?.gallery.use_adapted_metric ? 'Metric adapted' : 'Frozen DINO features'}</span></div>
            <button className="button secondary full-width" onClick={() => setTab('teach')}>Teach an object</button>
          </section>
          <section className="panel steps-panel"><p className="eyebrow">HOW IT WORKS</p><ol>
            <li><span>01</span><div><strong>Teach</strong><p>Record a short video while turning the item.</p></div></li>
            <li><span>02</span><div><strong>Learn</strong><p>DINOv2 features are adapted across your enrolled objects.</p></div></li>
            <li><span>03</span><div><strong>Listen</strong><p>Get spoken identity, direction and approximate distance.</p></div></li>
          </ol></section>
          <p className="privacy-note"><span aria-hidden="true">⌑</span> Local processing · No cloud account · Gallery stays on this computer</p>
        </aside>
      </section>}

      {tab === 'teach' && <section className="teach-layout">
        <div className="panel teach-panel">
          <div className="panel-heading"><div><p className="eyebrow">FEW-SHOT REGISTRATION</p><h2>Teach the system your items</h2><p className="subheading">Register 5–10 personal things you want it to recognise. Each short video contributes multiple views; the shared metric head learns after examples from at least two different objects.</p></div><span className="tag">{objects.length} enrolled</span></div>
          <div className="record-card">
            <div className="record-top"><div><h3>Record with the live camera</h3><p>Hold the object centered; rotate it slowly to show several sides.</p></div><span className="duration-chip">6 seconds</span></div>
            <label className="field-label">Personal object name
              <input value={objectName} onChange={(event) => setObjectName(event.target.value)} placeholder="For example, my house keys" maxLength={60} />
            </label>
            <div className="control-row">
              {!cameraStream ? <button className="button secondary" onClick={() => void startCamera()} disabled={cameraBusy}>{cameraBusy ? 'Requesting camera…' : 'Start camera'}</button> : null}
              {!recording
                ? <button className="button primary" onClick={startRecording} disabled={!cameraStream || batchBusy}>Record six-second video</button>
                : <button className="button danger" onClick={cancelRecording}>Stop and queue ({recordSeconds}s)</button>}
            </div>
            <label className="upload-label">Or add prepared 5–10 second videos
              <input type="file" accept="video/*" multiple onChange={(event) => { addVideoFiles(event.target.files); event.target.value = '' }} disabled={batchBusy || queue.length >= 10} />
            </label>
            <p className="help-text">Only sharp frames with a usable object view are stored. Videos and learned embeddings remain in the local gallery database.</p>
          </div>
          {cameraStream && <div className="registration-preview"><video ref={videoRef} autoPlay muted playsInline aria-label="Camera view while recording an object" />{recording && <span className="recording-badge">● Recording · {recordSeconds}s</span>}</div>}

          <div className="queue-heading"><div><h3>Registration batch</h3><p>{queue.length} of 10 videos queued</p></div>
            <button className="button secondary" onClick={() => setQueue([])} disabled={!queue.length || batchBusy}>Clear queue</button></div>
          {!queue.length ? <div className="queue-empty">Your video queue is empty. Add one video or register several objects together.</div> : <div className="queue-list">
            {queue.map((item, index) => <div className="queue-item" key={item.id}>
              <span className="queue-index">{String(index + 1).padStart(2, '0')}</span>
              <label className="field-label">Object name<input value={item.name} onChange={(event) => setQueue((current) => current.map((row) => row.id === item.id ? { ...row, name: event.target.value } : row))} /></label>
              <span className="file-name">{item.file.name}</span>
              <button className="icon-button" onClick={() => setQueue((current) => current.filter((row) => row.id !== item.id))} aria-label={`Remove ${item.name} from queue`}>×</button>
            </div>)}
          </div>}
          <button className="button primary register-button" onClick={() => void registerBatch()} disabled={!queue.length || batchBusy || !loaded}>
            {batchBusy ? 'Processing videos and adapting…' : `Register ${queue.length || ''} queued object${queue.length === 1 ? '' : 's'}`}
          </button>
          <p className="notice" role="status">{notice}</p>
        </div>

        <aside className="panel instructions-panel"><p className="eyebrow">A GOOD ENROLLMENT VIDEO</p><h3>Help it learn the right details</h3><ul>
          <li><span>1</span><p>Keep one item large and near the center of the frame.</p></li>
          <li><span>2</span><p>Slowly turn it; show front, back and side surfaces.</p></li>
          <li><span>3</span><p>Use steady light and avoid fast movement or glare.</p></li>
          <li><span>4</span><p>Give each personal item a distinct name, such as “my keys”.</p></li>
        </ul><div className="metric-note"><strong>What “metric learning” means here</strong><p>The DINOv2 vision backbone remains frozen. A small projection head learns to bring views of your enrolled items together and separate them from other enrolled items. It needs at least two real object classes; it never invents a fake negative object.</p></div>
          <p className="privacy-note">Registration videos are processed locally and discarded after embeddings are created. The gallery stores embeddings and object names, not video files.</p>
        </aside>
      </section>}

      {tab === 'memory' && <section className="panel memory-panel">
        <div className="panel-heading"><div><p className="eyebrow">LOCAL OBJECT MEMORY</p><h2>Your enrolled items</h2><p className="subheading">Recent sightings help answer “Where did I leave it?” A location is shown only when the system has real scene data.</p></div>
          <button className="button danger" onClick={() => void clearGallery()} disabled={!objects.length}>Clear all personal data</button></div>
        {!objects.length ? <div className="queue-empty">No personal items are enrolled yet. Teach a few items to start your gallery.</div> : <div className="object-table">
          {objects.map((item) => <article className="object-row" key={item.id}>
            <div className="object-avatar" aria-hidden="true">{item.name.charAt(0).toLocaleUpperCase()}</div>
            <div className="object-info">{renameId === item.id
              ? <div className="rename-controls"><input value={renameValue} onChange={(event) => setRenameValue(event.target.value)} aria-label="New object name" /><button className="button small primary" onClick={() => void renameObject(item)}>Save</button><button className="button small secondary" onClick={() => setRenameId(null)}>Cancel</button></div>
              : <><h3>{item.name}</h3><p>{item.last_seen_phrase || 'Not seen yet'}{item.location ? ` · ${item.location}` : ''}</p></>}
              <span className="object-meta">{item.embedding_count} stored views{item.confuser_group ? ' · similar-item group' : ''}</span>
            </div>
            <div className="object-actions">
              <button className="button small secondary" onClick={() => { setTab('live'); setMode('FIND'); setTarget(item.name) }}>Find</button>
              <button className="button small secondary" onClick={() => { setRenameId(item.id); setRenameValue(item.name) }}>Rename</button>
              <button className="icon-button" onClick={() => void deleteObject(item)} aria-label={`Delete ${item.name}`}>×</button>
            </div>
          </article>)}
        </div>}
        <button className="button secondary" onClick={() => void refresh()}>Refresh memory</button>
      </section>}

      <footer className="footer"><span>See & Remember · Few-shot personal object recognition</span><span>For assistance only; verify safety-critical actions independently.</span></footer>
    </main>
  )
}

export default App
