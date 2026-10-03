"""Python mirror of Ball/Player/GameEngine. Constants and order match frontend.

No wall-clock dependency: policies run at a configurable simulated interval.
"""
import math
import random
from backend.app.schemas import GameState
from training.oracle import OraclePolicy

WIDTH, HEIGHT, RADIUS, PADDLE_WIDTH, BASE_SPEED = 960, 640, 9, 128, 330
DT = 1 / 240


class Simulator:
    def __init__(self, seed=0, speed=1):
        self.rng = random.Random(seed)
        self.speed = speed
        self.top = self.bottom = WIDTH / 2
        self.top_score = self.bottom_score = 0
        self.top_hits = self.top_misses = 0
        self.timer = 0
        self.done = False
        self.serve("human")

    def serve(self, toward):
        self.x, self.y = WIDTH / 2, HEIGHT / 2
        angle = (self.rng.random() - 0.5) * 0.8
        self.vx = math.sin(angle) * BASE_SPEED
        self.vy = math.cos(angle) * BASE_SPEED * (1 if toward == "human" else -1)

    def state(self):
        return GameState(ball_x=self.x, ball_y=self.y, ball_vx=self.vx * self.speed,
                         ball_vy=self.vy * self.speed, jev_x=self.top, paddle_width=PADDLE_WIDTH,
                         game_width=WIDTH, game_height=HEIGHT)

    def randomize(self):
        self.timer = 0
        self.done = False
        self.top_score = self.bottom_score = 0
        self.x = self.rng.uniform(RADIUS, WIDTH - RADIUS)
        self.y = self.rng.uniform(64, HEIGHT - 64)
        self.top = self.rng.uniform(PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2)
        self.bottom = self.rng.uniform(PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2)
        angle = self.rng.uniform(-math.pi, math.pi)
        velocity = BASE_SPEED * self.rng.choice([0.5, 1, 1.5, 2, 3])
        self.vx, self.vy = math.sin(angle) * velocity, math.cos(angle) * velocity

    @staticmethod
    def move(x, action, speed):
        direction = -1 if action == "LEFT" else 1 if action == "RIGHT" else 0
        return max(PADDLE_WIDTH / 2, min(WIDTH - PADDLE_WIDTH / 2, x + direction * speed * DT))

    def bottom_action(self):
        # Symmetric oracle opponent with the existing human paddle speed (560).
        s = self.state()
        mirrored = s.model_copy(update={"ball_y": HEIGHT - s.ball_y, "ball_vy": -s.ball_vy, "jev_x": self.bottom})
        return OraclePolicy().predict(mirrored).prediction

    def step(self, action="STAY", bottom_action=None):
        if self.done:
            return
        if self.timer > 0:
            self.timer = max(0, self.timer - DT)
            return
        self.bottom = self.move(self.bottom, bottom_action or self.bottom_action(), 560)
        self.top = self.move(self.top, action, 240)
        self.x += self.vx * DT * self.speed
        self.y += self.vy * DT * self.speed
        if self.x < RADIUS:
            self.x, self.vx = 2 * RADIUS - self.x, abs(self.vx)
        elif self.x > WIDTH - RADIUS:
            self.x, self.vx = 2 * (WIDTH - RADIUS) - self.x, -abs(self.vx)
        bottom = self.vy > 0
        face = HEIGHT - 55 if bottom else 55
        paddle = self.bottom if bottom else self.top
        if abs(self.x - paddle) <= PADDLE_WIDTH / 2 + RADIUS and abs(self.y - face) <= RADIUS:
            offset = max(-1, min(1, (self.x - paddle) / (PADDLE_WIDTH / 2)))
            angle = offset * math.pi / 3
            self.vx, self.vy = math.sin(angle) * BASE_SPEED, math.cos(angle) * BASE_SPEED * (-1 if bottom else 1)
            self.y = face + (-RADIUS if bottom else RADIUS)
            if not bottom:
                self.top_hits += 1
        if self.y - RADIUS > HEIGHT or self.y + RADIUS < 0:
            top_scored = self.y - RADIUS > HEIGHT
            self.top_score += int(top_scored)
            self.bottom_score += int(not top_scored)
            self.top_misses += int(not top_scored)
            self.top = self.bottom = WIDTH / 2
            self.serve("human" if top_scored else "jev")
            self.done = max(self.top_score, self.bottom_score) >= 10
            self.timer = 0.9


def play(policy, seed, max_seconds=120, interval_ms=100, latency_ms=0, speed=1, random_start=False):
    sim = Simulator(seed, speed)
    if random_start:
        sim.randomize()
    action, next_decision = "STAY", 0.0
    pending, ready_at = None, 0.0
    for step in range(int(max_seconds / DT)):
        now = step * DT
        if sim.done:
            break
        if sim.timer > 0:
            action, pending = "STAY", None
        else:
            if pending is not None and now >= ready_at:
                action, pending = pending, None
            if pending is None and now >= next_decision:
                decision = policy.predict(sim.state()).prediction
                next_decision = now + interval_ms / 1000
                if latency_ms:
                    pending, ready_at = decision, now + latency_ms / 1000
                else:
                    action = decision
        sim.step(action)
    return {"score": sim.top_score, "opponent_score": sim.bottom_score, "hits": sim.top_hits,
            "misses": sim.top_misses, "completed": sim.done}
