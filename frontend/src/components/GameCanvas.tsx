import { useEffect, useRef } from 'react';
import type { GameEngine } from '../game/GameEngine';
import type { ApiJevController, JevInferenceStatus } from '../game/ApiJevController';
import { renderGame } from '../game/renderGame';
import { COURT } from '../game/types';
import type { GameSnapshot } from '../game/types';

interface Props {
  engine: GameEngine;
  controller: ApiJevController;
  onSnapshot: (snapshot: GameSnapshot) => void;
  onInference: (status: JevInferenceStatus) => void;
}

export function GameCanvas({ engine, controller, onSnapshot, onInference }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    let frame = 0;
    let previousTime: number | null = null;
    let previousSnapshot = '';
    let previousInference = '';
    let previousState: GameSnapshot['state'] | null = null;
    controller.start();

    const loop = (time: number) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.round(canvas.clientWidth * dpr);
      const pixelHeight = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      ctx.setTransform(canvas.width / COURT.width, 0, 0, canvas.height / COURT.height, 0, 0);
      if (!document.hidden && previousTime !== null) engine.update((time - previousTime) / 1000);
      previousTime = document.hidden ? null : time;
      renderGame(ctx, engine);
      const snapshot = engine.getSnapshot();
      if (snapshot.state !== previousState) {
        controller.resetPrediction();
        previousState = snapshot.state;
      }
      if (!document.hidden) {
        const velocityScale = snapshot.state === 'PLAYING' ? snapshot.speed : 0;
        void controller.requestPrediction({
          ball_x: engine.ball.x,
          ball_y: engine.ball.y,
          ball_vx: engine.ball.vx * velocityScale,
          ball_vy: engine.ball.vy * velocityScale,
          jev_x: engine.jev.x,
          paddle_width: engine.jev.width,
          game_width: COURT.width,
          game_height: COURT.height,
        }, time);
      }
      const serialized = JSON.stringify(snapshot);
      if (serialized !== previousSnapshot) {
        onSnapshot(snapshot);
        previousSnapshot = serialized;
      }
      const inference = controller.getStatus();
      const serializedInference = JSON.stringify(inference);
      if (serializedInference !== previousInference) {
        onInference(inference);
        previousInference = serializedInference;
      }
      frame = requestAnimationFrame(loop);
    };

    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName))) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (['arrowleft', 'arrowright', 'a', 'd'].includes(event.key.toLowerCase())) {
        event.preventDefault();
        engine.setKey(event.key, event.type === 'keydown');
      }
    };
    const clearInput = () => { engine.clearInput(); controller.resetPrediction(); previousTime = null; };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', clearInput);
    document.addEventListener('visibilitychange', clearInput);
    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      controller.stop();
      engine.clearInput();
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
      window.removeEventListener('blur', clearInput);
      document.removeEventListener('visibilitychange', clearInput);
    };
  }, [engine, controller, onSnapshot, onInference]);

  return <canvas ref={canvasRef} width={COURT.width} height={COURT.height}
    aria-label="Game court. You control the bottom paddle; JEV controls the top paddle. Use Left and Right arrows or A and D."
    role="img" />;
}
