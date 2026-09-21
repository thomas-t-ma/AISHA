import { useEffect, useState } from 'react';
import {
  forgetLearnedBelief,
  getAutomaticMemoryStatus,
  getBeliefVersions,
  getLearnedBeliefs,
} from './api';
import type { AutomaticMemoryStatus, BeliefVersion, LearnedBelief } from './types';

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function LearnedMemory() {
  const [beliefs, setBeliefs] = useState<LearnedBelief[]>([]);
  const [status, setStatus] = useState<AutomaticMemoryStatus | null>(null);
  const [versions, setVersions] = useState<Record<string, BeliefVersion[]>>({});
  const [expanded, setExpanded] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function refresh() {
    try {
      const [nextBeliefs, nextStatus] = await Promise.all([
        getLearnedBeliefs(), getAutomaticMemoryStatus(),
      ]);
      setBeliefs(nextBeliefs);
      setStatus(nextStatus);
      setError('');
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void refresh(); }, []);

  async function toggleVersions(id: string) {
    if (expanded === id) {
      setExpanded('');
      return;
    }
    setExpanded(id);
    try {
      setVersions((old) => ({ ...old, [id]: [] }));
      const history = await getBeliefVersions(id);
      setVersions((old) => ({ ...old, [id]: history }));
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function forget(id: string) {
    if (busy || !window.confirm('Forget this learned belief? The original chat remains.')) {
      return;
    }
    setBusy(true);
    try {
      await forgetLearnedBelief(id);
      setExpanded('');
      await refresh();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="learned-section" aria-label="AISHA learned memories">
      <div className="learned-header">
        <h3>What she's learned</h3>
        <button type="button" className="memory-button" onClick={() => void refresh()}>
          ↻ Refresh
        </button>
      </div>
      <p className="learned-help">
        AISHA forms these after conversations without requesting approval. This
        view separates the original quote from her interpretation and shows revisions.
      </p>
      <div className="learned-status" role="status">
        {loading ? 'Loading…' : !status?.enabled
          ? 'Automatic reflection is disabled in this profile.'
          : status.active_reflections
            ? 'Thinking about the last conversation…'
            : 'Automatic reflection is ready.'}
      </div>
      {status?.last_result && (
        <p className="learned-help" role="status">
          Last reflection: {status.last_result.outcome.replace('_', ' ')} ·
          {' '}{status.last_result.proposed} proposed ·
          {' '}{status.last_result.saved} saved ·
          {' '}{status.last_result.rejected} rejected
          {status.last_result.outcome === 'no_candidate'
            ? ' (the model proposed no memory for that message)'
            : status.last_result.outcome === 'rejected'
              ? ' (proposals failed grounding or revision checks)'
              : ''}
        </p>
      )}
      {status?.last_result?.rejections?.map((failure, index) => (
        <div className="memory-error" key={status.last_result?.episode_id + ':' + index}>
          Rejected {failure.action || 'proposal'} for {failure.topic_key || 'unknown topic'}:
          {' '}{failure.reason.replaceAll('_', ' ')}
          {failure.source_quote && (
            <div className="learned-quote">
              Proposed source: “{failure.source_quote}”
            </div>
          )}
        </div>
      ))}
      {status?.last_error && (
        <p className="memory-error">Last reflection error: {status.last_error}</p>
      )}
      {error && <p className="memory-error" role="alert">{error}</p>}
      {!loading && !beliefs.length && (
        <p className="learned-help">
          No learned beliefs yet. Tell AISHA something meaningful, let her reply,
          then refresh this panel after the reflection finishes.
        </p>
      )}
      {beliefs.map((belief) => (
        <article key={belief.belief_id} className="learned-card">
          <div className="learned-topic">
            {belief.topic_key} · revision {belief.revision}
          </div>
          <p className="memory-text">{belief.text}</p>
          <div className="memory-actions">
            <span className="learned-badge">{belief.epistemic_status}</span>
            <span className="learned-badge">
              {belief.evidence_status === 'verified'
                ? 'Evidence checked (model-assisted)'
                : 'Legacy: evidence not checked'}
            </span>
          </div>
          <div className="learned-quote">
            {belief.revision > 1 ? "New evidence for this revision" : "Source quote"}: “{belief.source_quote}”
          </div>
          {belief.open_question && (
            <div className="learned-question">Open question: {belief.open_question}</div>
          )}
          {belief.evidence_status !== 'verified' && (
            <p className="learned-help">
              Created before the evidence checker; the quote may not support the
              whole memory. This entry has not been rewritten or deleted.
            </p>
          )}
          {belief.revision > 1 && <p className="learned-help">
            This quote supports the latest change. Earlier claims may rely on the
            preceding source quotes under Revision history.
          </p>}
          <div className="memory-provenance" title={belief.source_turn_id}>
            Source: {belief.source_session_id.slice(0, 19)}… ·
            {' '}{new Date(belief.updated_at).toLocaleString()}
          </div>
          <div className="memory-actions">
            <button type="button" className="memory-button"
              onClick={() => void toggleVersions(belief.belief_id)}>
              {expanded === belief.belief_id ? 'Hide history' : 'Revision history'}
            </button>
            <button type="button" className="memory-button danger" disabled={busy}
              onClick={() => void forget(belief.belief_id)}>
              Forget
            </button>
          </div>
          {expanded === belief.belief_id && (
            <div className="learned-history">
              {(versions[belief.belief_id] ?? []).map((version) => (
                <div key={version.version_id}>
                  <strong>Version {version.revision}: </strong>{version.text}
                  <p>Evidence: “{version.source_quote}” ·
                    {' '}{version.evidence_status === 'verified'
                      ? 'checked' : 'legacy unchecked'}</p>
                </div>
              ))}
            </div>
          )}
        </article>
      ))}
    </section>
  );
}
