import { useState } from 'react';
import { GameEngine } from '../game/GameEngine';
import { ApiJevController } from '../game/ApiJevController';
import { BALL_SPEEDS } from '../game/types';
import { GameCanvas } from './GameCanvas';

export function Game() {
  const [controller] = useState(() => new ApiJevController({
    url: import.meta.env.VITE_JEV_API_URL || undefined,
    intervalMs: Number(import.meta.env.VITE_JEV_INFERENCE_INTERVAL_MS ?? 100),
    timeoutMs: Number(import.meta.env.VITE_JEV_REQUEST_TIMEOUT_MS ?? 30000),
  }));
  const [engine] = useState(() => new GameEngine(controller));
  const [snapshot, setSnapshot] = useState(() => engine.getSnapshot());
  const [inference, setInference] = useState(() => controller.getStatus());
  const { state, humanScore, jevScore, speed, winner, lastScorer } = snapshot;
  const decision = inference.prediction;
  const start = () => { controller.resetPrediction(); engine.start(); setSnapshot(engine.getSnapshot()); };
  const restart = () => { controller.resetPrediction(); engine.restart(); setSnapshot(engine.getSnapshot()); };
  const decisionIcon = decision === 'LEFT' ? '←' : decision === 'RIGHT' ? '→' : '—';

  return (
    <main className="shell">
      <header className="masthead">
        <a className="wordmark" href="./" aria-label="JEV Keep Up home"><span className="brand-icon">j.</span> JEV <span>PLAYGROUND</span></a>
        <span className={`version ${inference.connected ? '' : 'unavailable'}`} role="status"><span className="status-dot" /> JEV: {inference.connected ? 'Connected' : 'Unavailable'}</span>
      </header>

      <section className="intro">
        <div><p className="eyebrow">01 / HUMAN VS MACHINE</p><h1>Keep <span>up.</span></h1></div>
        <p className="intro-copy">One ball. Two players. No excuses.<br /> Keep it in play. First to 10 takes it.</p>
      </section>

      <div className="game-layout">
        <section className="arena" aria-label="JEV Keep Up game">
          <div className="scoreboard">
            <div className="score-side human"><span className="player-mark">YOU</span><span>Human</span><strong>{humanScore.toString().padStart(2, '0')}</strong></div>
            <div className="score-middle"><span>FIRST TO 10</span><span className="match-state">{state === 'READY' ? 'READY TO PLAY' : state === 'GAME_OVER' ? 'MATCH COMPLETE' : 'MATCH IN PROGRESS'}</span></div>
            <div className="score-side jev"><strong>{jevScore.toString().padStart(2, '0')}</strong><span>JEV</span><span className="player-mark">BOT</span></div>
          </div>
          <div className="court">
            <GameCanvas engine={engine} controller={controller} onSnapshot={setSnapshot} onInference={setInference} />
            {state === 'READY' && <div className="court-overlay"><div className="overlay-card"><span className="eyebrow">THE COURT IS YOURS</span><h2>Can you keep up?</h2><p>Meet JEV on the other side.</p><button className="primary-button" onClick={start}>Start game <span>↗</span></button></div></div>}
            {state === 'POINT_SCORED' && <div className="point-overlay" role="status"><span>{lastScorer === 'human' ? 'Your point' : 'JEV scores'}</span><small>Next serve incoming</small></div>}
            {state === 'GAME_OVER' && <div className="court-overlay"><div className="overlay-card" role="status"><span className="eyebrow">MATCH COMPLETE</span><h2>{winner === 'human' ? 'You kept up.' : 'JEV takes it.'}</h2><p>{humanScore} – {jevScore} · {winner === 'human' ? 'A well-earned win.' : 'Another round?'}</p><button className="primary-button" onClick={restart}>Play again <span>↗</span></button></div></div>}
          </div>
          <div className="court-footer"><span><span className="status-dot" /> {state === 'PLAYING' ? 'LIVE RALLY' : state.replaceAll('_', ' ')}</span><span>YOU’RE THE GREEN PADDLE <span className="green-dot" /></span></div>
        </section>

        <aside className="sidebar">
          <section className="panel">
            <p className="eyebrow">SET THE PACE</p>
            <div className="panel-heading"><h2>Ball speed</h2><span className="speed-readout">{speed}×</span></div>
            <div className="speed-options" role="group" aria-label="Ball speed">
              {BALL_SPEEDS.map(value => <button key={value} aria-pressed={speed === value} onClick={() => { engine.setSpeed(value); setSnapshot(engine.getSnapshot()); }}>{value}×</button>)}
            </div>
            <p className="helper">Change it anytime. Feel it instantly.</p>
          </section>
          <section className="panel jev-panel">
            <div className="panel-heading"><p className="eyebrow">INSIDE JEV’S HEAD</p><span className="mock-badge">API</span></div>
            <div className="decision"><span className="decision-icon">{decisionIcon}</span><div><small>CURRENT DECISION</small><strong>{decision}</strong></div><span className={`thinking-dot ${inference.connected ? '' : 'unavailable'}`} /></div>
            <dl className="inference-metrics">
              <div><dt>Model</dt><dd className="model-name">{inference.provider === 'mock' ? 'Mock' : 'OpenJEV-style 0.8B'}</dd></div>
              <div><dt>Status</dt><dd>{inference.connected ? 'Connected' : 'Unavailable'}</dd></div>
              <div><dt>Confidence</dt><dd>{inference.confidence === null ? '—' : `${Math.round(inference.confidence * 100)}%`}</dd></div>
              <div><dt>Inference</dt><dd>{inference.latencyMs === null ? '—' : `${inference.latencyMs.toFixed(1)} ms`}</dd></div>
            </dl>
            <p className="helper">{inference.connected ? 'A rival tracking the ball, one move at a time.' : 'Inference unavailable. JEV stays still; you can keep playing.'}</p>
          </section>
          <section className="panel controls-panel">
            <p className="eyebrow">YOUR MOVE</p><h2>Stay in the rally.</h2>
            <div className="keyboard"><kbd>←</kbd><kbd>→</kbd><span>or</span><kbd>A</kbd><kbd>D</kbd></div>
            <p className="helper">Move left and right. Catch the ball.<br />A miss gives your rival a point.</p>
            <div className="touch-controls" aria-label="Paddle controls">
              {(['a', 'd'] as const).map((key) => <button key={key} aria-label={key === 'a' ? 'Move left' : 'Move right'} onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); engine.setKey(key, true); }} onPointerUp={() => engine.setKey(key, false)} onPointerCancel={() => engine.setKey(key, false)} onLostPointerCapture={() => engine.setKey(key, false)}>{key === 'a' ? '←' : '→'}</button>)}
            </div>
          </section>
          <button className={state === 'READY' ? 'primary-button session-button' : 'restart-button session-button'} onClick={state === 'READY' ? start : restart}>{state === 'READY' ? 'Start game' : 'Restart game'}<span>{state === 'READY' ? '↗' : '↻'}</span></button>
        </aside>
      </div>
      <footer className="page-footer"><span>JEV KEEP UP</span><span>A little competition. A lot of reflexes.</span><span>BUILT TO PLAY.</span></footer>
    </main>
  );
}
