import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { createSession, getEvents, getHealth, getMessages, getRuns } from './api';
import MemoryPanel from './MemoryPanel';
import type { AISHAEvent, ChatMessage, ModelRun, RuntimeHealth, TurnLatency } from './types';

const SESSION_KEY = 'aisha.studio.session.v1';

function numeric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function millis(value: number | null | undefined): string {
  return value == null ? '—' : Math.round(value).toLocaleString() + ' ms';
}

function shortId(value: string): string {
  return value.length > 19 ? value.slice(0, 13) + '…' + value.slice(-5) : value;
}

function toMessage(row: {
  message_id: string;
  role: string;
  text: string;
}): ChatMessage {
  return {
    id: row.message_id,
    role: row.role === 'assistant' ? 'assistant' : 'user',
    text: row.text,
    status: 'committed',
  };
}

function upsertAssistant(
  previous: ChatMessage[],
  turnId: string,
  transform: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const existing = previous.find((message) => message.turnId === turnId);
  if (existing) {
    return previous.map((message) => (message.turnId === turnId ? transform(message) : message));
  }
  return [...previous, transform({
    id: turnId,
    turnId,
    role: 'assistant',
    text: '',
    status: 'streaming',
  })];
}

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => window.localStorage.getItem(SESSION_KEY) ?? '',
  );
  const [health, setHealth] = useState<RuntimeHealth | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [connection, setConnection] = useState<'connecting' | 'online' | 'offline'>('connecting');
  const [busy, setBusy] = useState(false);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [latency, setLatency] = useState<TurnLatency>({ firstTokenMs: null, totalMs: null });
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [events, setEvents] = useState<AISHAEvent[]>([]);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<'runs' | 'events'>('runs');
  const [selectedRunId, setSelectedRunId] = useState('');
  const [notice, setNotice] = useState('');
  const [creatingSession, setCreatingSession] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const creatingRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await getHealth());
    } catch {
      setHealth(null);
      setNotice('AISHA Core is unavailable. Start the Python server on port 8000.');
    }
  }, []);

  const refreshInspector = useCallback(async (id: string) => {
    const results = await Promise.allSettled([getRuns(id), getEvents(id)]);
    if (results[0].status === 'fulfilled') setRuns(results[0].value);
    if (results[1].status === 'fulfilled') setEvents(results[1].value);
    if (results.some((result) => result.status === 'rejected')) {
      setNotice('Unable to load some developer records from AISHA Core.');
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  useEffect(() => {
    if (sessionId || creatingRef.current) return;
    creatingRef.current = true;
    void createSession()
      .then((id) => {
        window.localStorage.setItem(SESSION_KEY, id);
        setSessionId(id);
        setNotice('');
      })
      .catch(() => {
        setConnection('offline');
        setNotice('Cannot create a session. Check that AISHA Core is running on port 8000.');
      })
      .finally(() => {
        creatingRef.current = false;
      });
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    let closed = false;
    setConnection('connecting');
    setBusy(false);
    setActiveTurnId(null);

    void getMessages(sessionId)
      .then((history) => {
        if (!closed) setMessages(history.map(toMessage));
      })
      .catch(() => {
        if (!closed) setNotice('Could not restore prior messages. The session may be new.');
      });

    void refreshInspector(sessionId);

    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(
      scheme + '//' + window.location.host + '/v1/ws/' + encodeURIComponent(sessionId),
    );
    wsRef.current = socket;

    socket.onopen = () => {
      if (!closed) {
        setConnection('online');
        setNotice('');
      }
    };
    socket.onerror = () => {
      if (!closed) setNotice('Connection to AISHA Core failed. Check port 8000.');
    };
    socket.onclose = () => {
      if (!closed) {
        setConnection('offline');
        setBusy(false);
        setActiveTurnId(null);
        setNotice('Connection lost. Refresh the page after restarting AISHA Core.');
      }
    };

    socket.onmessage = (raw: MessageEvent<string>) => {
      if (closed) return;
      let event: AISHAEvent;
      try {
        event = JSON.parse(raw.data) as AISHAEvent;
      } catch {
        setNotice('Received an invalid event from AISHA Core.');
        return;
      }
      const turnId = event.turn_id ?? '';
      const payload = event.payload ?? {};

      switch (event.type) {
        case 'aisha.turn.started':
          setActiveTurnId(turnId);
          setBusy(true);
          setMessages((previous) => upsertAssistant(previous, turnId, (message) => message));
          break;
        case 'aisha.assistant.text_delta':
          setMessages((previous) => upsertAssistant(previous, turnId, (message) => ({
            ...message,
            text: message.text + (typeof payload.text === 'string' ? payload.text : ''),
          })));
          break;
        case 'aisha.turn.finished':
          setMessages((previous) => upsertAssistant(previous, turnId, (message) => ({
            ...message,
            status: 'committed',
          })));
          setLatency({
            firstTokenMs: numeric(payload.first_token_ms),
            totalMs: numeric(payload.total_ms),
          });
          setBusy(false);
          setActiveTurnId(null);
          void refreshInspector(sessionId);
          break;
        case 'aisha.turn.cancelled':
          setMessages((previous) => upsertAssistant(previous, turnId, (message) => ({
            ...message,
            text: message.text || 'Stopped before AISHA responded.',
            status: 'cancelled',
          })));
          setBusy(false);
          setActiveTurnId(null);
          void refreshInspector(sessionId);
          break;
        case 'aisha.turn.failed':
          setMessages((previous) => upsertAssistant(previous, turnId, (message) => ({
            ...message,
            text: String(payload.error ?? 'The provider failed.'),
            status: 'failed',
          })));
          setBusy(false);
          setActiveTurnId(null);
          void refreshInspector(sessionId);
          break;
        case 'aisha.control.cancel_ack':
          if (!payload.cancelled_turn_id) {
            setNotice('There was no active turn to cancel.');
          }
          break;
        case 'aisha.error':
          setNotice(String(payload.error ?? 'AISHA Core reported an error.'));
          break;
        default:
          break;
      }
    };

    return () => {
      closed = true;
      socket.close();
      if (wsRef.current === socket) wsRef.current = null;
    };
  }, [sessionId, refreshInspector]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  function sendMessage() {
    const text = draft.trim();
    if (!text || busy || connection !== 'online' || wsRef.current?.readyState !== WebSocket.OPEN) {
      return;
    }
    wsRef.current.send(JSON.stringify({ type: 'aisha.user.text', payload: { text } }));
    setMessages((previous) => [
      ...previous,
      { id: 'local_' + crypto.randomUUID(), role: 'user', text, status: 'committed' },
    ]);
    setDraft('');
    setBusy(true);
    setNotice('');
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    sendMessage();
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  }

  function cancelTurn() {
    if (wsRef.current?.readyState !== WebSocket.OPEN || !busy) return;
    wsRef.current.send(JSON.stringify({
      type: 'aisha.turn.cancel',
      payload: activeTurnId ? { turn_id: activeTurnId } : {},
    }));
  }

  async function newSession() {
    if (creatingSession || busy) return;
    setCreatingSession(true);
    try {
      const id = await createSession();
      setMessages([]);
      setRuns([]);
      setEvents([]);
      setSelectedRunId('');
      setLatency({ firstTokenMs: null, totalMs: null });
      setDraft('');
      window.localStorage.setItem(SESSION_KEY, id);
      setSessionId(id);
      setNotice('');
    } catch {
      setNotice('Could not create a new session. Check AISHA Core.');
    } finally {
      setCreatingSession(false);
    }
  }

  const currentRun = runs.find((run) => run.run_id === selectedRunId) ?? runs.at(-1);
  const statusText = connection === 'online' ? 'Connected' : connection === 'connecting'
    ? 'Connecting' : 'Disconnected';

  return (
    <div className="studio">
      <aside className="rail">
        <div className="identity">
          <div className="identity-mark">A<span>✦</span></div>
          <div>
            <div className="identity-name">AISHA</div>
            <div className="identity-subtitle">STUDIO · LOCAL</div>
          </div>
        </div>

        <div className="rail-group">
          <div className="rail-caption">YOUR WORKSPACE</div>
          <button className="session-button" onClick={() => void newSession()}
            disabled={busy || creatingSession}>
            <span aria-hidden="true">＋</span> New conversation
          </button>
          <div className="session-card">
            <div className="session-dot" aria-hidden="true" />
            <div className="session-copy">
              <strong>Current session</strong>
              <span title={sessionId}>{sessionId ? shortId(sessionId) : 'Creating…'}</span>
            </div>
          </div>
        </div>

        <div className="rail-bottom">
          <div className="small-label">LOCAL RUNTIME</div>
          <div className="runtime-pill">
            <span className={'signal ' + connection} />
            <span>{statusText}</span>
          </div>
          <div className="runtime-model">{health?.model ?? 'AISHA Core'}</div>
          <div className="runtime-profile">{health?.profile ?? 'Check server connection'}</div>
          <button className="text-action" onClick={() => void refreshHealth()}>
            Refresh runtime ↗
          </button>
          <div className="rail-footnote">Your model and SQLite history stay on this computer.</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-title">
            <span className="eyebrow">CONVERSATION</span>
            <h1>Talk to AISHA<span className="title-star">✦</span></h1>
          </div>
          <div className="topbar-actions">
            <button className="inspector-toggle" onClick={() => setMemoryOpen(true)}>Memory ✦</button>
            <span className={'connection-badge ' + connection}>
              <span className="status-dot" /> {statusText}
            </span>
            <button className={'inspector-toggle ' + (inspectorOpen ? 'active' : '')}
              onClick={() => setInspectorOpen((open) => !open)}
              aria-expanded={inspectorOpen}>
              {inspectorOpen ? 'Close inspector' : 'Developer panel'}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
        </header>

        {notice && <div className="notice" role="status">
          <span>{notice}</span>
          <button onClick={() => setNotice('')} aria-label="Dismiss notice">×</button>
        </div>}

        <section className="chat-scroll" aria-label="Conversation">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-orb">✦</div>
              <div className="eyebrow">A PLACE TO BEGIN</div>
              <h2>She's listening.</h2>
              <p>Start a conversation. AISHA Core will handle the model, persona, and memory of this session.</p>
              <div className="welcome-meta">LOCAL MODEL · NO HOSTED API REQUIRED</div>
            </div>
          ) : (
            <div className="chat-thread">
              {messages.map((message) => (
                <article className={'message message-' + message.role} key={message.id}>
                  <div className={'avatar avatar-' + message.role}>
                    {message.role === 'assistant' ? 'A' : 'YOU'}
                  </div>
                  <div className="message-body">
                    <div className="message-heading">
                      <strong>{message.role === 'assistant' ? 'AISHA' : 'You'}</strong>
                      {message.status === 'streaming' && <span className="typing-dot">● LIVE</span>}
                      {message.status === 'cancelled' && <span className="message-note">Interrupted · not saved</span>}
                      {message.status === 'failed' && <span className="message-note error">Provider error</span>}
                    </div>
                    <div className="message-text">{message.text || (message.status === 'streaming'
                      ? <span className="thinking-dots">● ● ●</span> : '')}</div>
                  </div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-wrap">
          <div className="latency-strip">
            <span>TTFT <strong>{millis(latency.firstTokenMs)}</strong></span>
            <span>Total <strong>{millis(latency.totalMs)}</strong></span>
            <span className="latency-right">{health?.provider ?? 'offline'} · {health?.persona_version ?? '—'}</span>
          </div>
          <form className="composer" onSubmit={onSubmit}>
            <textarea aria-label="Message AISHA" rows={2} value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={connection === 'online' ? 'Say something to AISHA…' : 'Connect AISHA Core to begin…'}
              disabled={connection !== 'online'} />
            <div className="composer-footer">
              <span>Enter to send · Shift+Enter for a new line</span>
              {busy ? (
                <button type="button" className="stop-button" onClick={cancelTurn}
                  disabled={connection !== 'online'}>■ Stop turn</button>
              ) : (
                <button type="submit" className="send-button"
                  disabled={!draft.trim() || connection !== 'online'}>Send <span>↗</span></button>
              )}
            </div>
          </form>
          <div className="composer-disclaimer">Development build · AISHA has no camera or microphone access yet.</div>
        </div>
      </main>

      {memoryOpen && <MemoryPanel sessionId={sessionId} onClose={() => setMemoryOpen(false)} />}
      {inspectorOpen && (
        <aside className="inspector" aria-label="Developer inspector">
          <div className="inspector-heading">
            <span className="eyebrow">OBSERVABILITY</span>
            <h2>Under the hood</h2>
            <p>Inspect the actual local model run and persisted events.</p>
          </div>
          <div className="inspector-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={inspectorTab === 'runs'}
              className={inspectorTab === 'runs' ? 'selected' : ''}
              onClick={() => setInspectorTab('runs')}>Model runs</button>
            <button type="button" role="tab" aria-selected={inspectorTab === 'events'}
              className={inspectorTab === 'events' ? 'selected' : ''}
              onClick={() => setInspectorTab('events')}>Event log</button>
          </div>
          <div className="inspector-scroll">
            {inspectorTab === 'runs' ? (
              <div>
                {runs.length === 0 ? (
                  <div className="empty-records">No model runs in this session yet.</div>
                ) : (
                  <>
                    <label className="small-label" htmlFor="run-select">SELECT TURN</label>
                    <select id="run-select" className="run-select"
                      value={currentRun?.run_id ?? ''}
                      onChange={(event) => setSelectedRunId(event.target.value)}>
                      {runs.slice().reverse().map((run) => (
                        <option key={run.run_id} value={run.run_id}>
                          {shortId(run.turn_id)} · {run.status}
                        </option>
                      ))}
                    </select>
                    {currentRun && (
                      <div className="run-detail">
                        <div className="detail-header">
                          <span className="small-label">LATEST RUN</span>
                          <span className={'run-status status-' + currentRun.status}>
                            {currentRun.status}
                          </span>
                        </div>
                        <div className="model-name">{currentRun.model}</div>
                        <div className="metrics-grid">
                          <div><span>First token</span><strong>{millis(currentRun.first_token_ms)}</strong></div>
                          <div><span>Total</span><strong>{millis(currentRun.total_ms)}</strong></div>
                          <div><span>Characters</span><strong>{currentRun.output_chars}</strong></div>
                          <div><span>Output tokens</span><strong>{String(currentRun.backend_metrics.output_tokens ?? '—')}</strong></div>
                        </div>
                        <div className="small-label metrics-label">OLLAMA BREAKDOWN</div>
                        <dl className="timing-rows">
                          <div><dt>Model load</dt><dd>{millis(numeric(currentRun.backend_metrics.load_ms))}</dd></div>
                          <div><dt>Prompt evaluation</dt><dd>{millis(numeric(currentRun.backend_metrics.prompt_eval_ms))}</dd></div>
                          <div><dt>Token generation</dt><dd>{millis(numeric(currentRun.backend_metrics.eval_ms))}</dd></div>
                          <div><dt>Prompt tokens</dt><dd>{String(currentRun.backend_metrics.prompt_tokens ?? '—')}</dd></div>
                          <div><dt>Cached prompt tokens</dt><dd>{String(currentRun.backend_metrics.cached_prompt_tokens ?? '—')}</dd></div>
                        </dl>
                        <div className="small-label metrics-label">TURN ID</div>
                        <code className="run-id">{currentRun.turn_id}</code>
                      </div>
                    )}
                  </>
                )}
              </div>
            ) : (
              <div className="event-list">
                {events.length === 0 ? (
                  <div className="empty-records">No persisted events in this session yet.</div>
                ) : (
                  events.slice(-100).reverse().map((event, index) => (
                    <details className="event-entry" key={event.turn_id + ':' + index}>
                      <summary>
                        <span className="event-dot" />
                        <span>{event.type.replace('aisha.', '')}</span>
                        <small>{event.timestamp?.slice(11, 19) ?? ''}</small>
                      </summary>
                      <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                    </details>
                  ))
                )}
              </div>
            )}
          </div>
          <button className="refresh-inspector" onClick={() => void refreshInspector(sessionId)}
            disabled={!sessionId}>↻ Refresh records</button>
        </aside>
      )}
    </div>
  );
}
