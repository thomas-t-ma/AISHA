import { useEffect, useState } from 'react';
import { addMemory, editMemory, getMemories, removeMemory } from './api';
import type { MemoryRecord } from './types';

interface Props {
  sessionId: string;
  onClose: () => void;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function MemoryPanel({ sessionId, onClose }: Props) {
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [newText, setNewText] = useState('');
  const [editingId, setEditingId] = useState('');
  const [editingText, setEditingText] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  async function reload() {
    try {
      setMemories(await getMemories());
      setError('');
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void reload();
  }, []);

  async function create() {
    const text = newText.trim();
    if (!text || saving) return;
    setSaving(true);
    try {
      await addMemory(text, sessionId);
      setNewText('');
      await reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSaving(false);
    }
  }

  async function save() {
    const text = editingText.trim();
    if (!text || !editingId || saving) return;
    setSaving(true);
    try {
      await editMemory(editingId, text);
      setEditingId('');
      setEditingText('');
      await reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSaving(false);
    }
  }

  async function forget(memoryId: string) {
    if (saving || !window.confirm('Remove this memory from future AISHA turns?')) return;
    setSaving(true);
    try {
      await removeMemory(memoryId);
      await reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="memory-backdrop" role="presentation"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="memory-panel" role="dialog" aria-modal="true"
        aria-labelledby="memory-title">
        <div className="memory-head">
          <div>
            <span className="eyebrow">EXPLICIT MEMORY · LOCAL ONLY</span>
            <h2 id="memory-title">What AISHA remembers</h2>
          </div>
          <button className="memory-close" onClick={onClose} aria-label="Close memory">×</button>
        </div>
        <p className="memory-help">
          Write facts you want AISHA to know across conversations. She does not
          automatically save or infer memories. You can edit or remove any entry.
          Updates affect future turns, not messages AISHA has already written.
        </p>

        <div className="memory-form">
          <label className="small-label" htmlFor="memory-input">NEW USER-APPROVED MEMORY</label>
          <textarea id="memory-input" rows={3} maxLength={500} value={newText}
            onChange={(event) => setNewText(event.target.value)}
            placeholder="Example: I prefer concise answers and a little dry humor." />
          <div className="memory-form-actions">
            <span className="memory-count">{newText.length} / 500</span>
            <button className="memory-button primary" onClick={() => void create()}
              disabled={saving || !newText.trim()}>Remember this</button>
          </div>
        </div>

        {error && <p className="memory-error" role="alert">{error}</p>}
        <div className="memory-list">
          {loading && <p className="memory-help">Loading memories…</p>}
          {!loading && memories.length === 0 && (
            <p className="memory-help">No saved memories yet. Add one above to test cross-session recall.</p>
          )}
          {memories.map((memory) => (
            <article className="memory-item" key={memory.memory_id}>
              {editingId === memory.memory_id ? (
                <div className="memory-edit">
                  <textarea aria-label="Edit memory" rows={3} maxLength={500}
                    value={editingText}
                    onChange={(event) => setEditingText(event.target.value)} />
                  <div className="memory-actions">
                    <button className="memory-button primary"
                      disabled={saving || !editingText.trim()} onClick={() => void save()}>
                      Save correction
                    </button>
                    <button className="memory-button" onClick={() => setEditingId('')}>
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="memory-text">{memory.text}</p>
                  <div className="memory-provenance">
                    User-approved · {memory.source_session_id
                      ? 'saved from session ' + memory.source_session_id.slice(0, 18) + '…'
                      : 'added locally'}
                    {' · '}{new Date(memory.updated_at).toLocaleString()}
                  </div>
                  <div className="memory-actions">
                    <button className="memory-button" disabled={saving}
                      onClick={() => {
                        setEditingId(memory.memory_id);
                        setEditingText(memory.text);
                      }}>Edit</button>
                    <button className="memory-button danger" disabled={saving}
                      onClick={() => void forget(memory.memory_id)}>Forget</button>
                  </div>
                </>
              )}
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
