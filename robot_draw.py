# PAI最終課題：自分の腕を学ぶAIロボットアームが図形を描く
#
# 流れ
#  1) 体験（モーターバブリング）: ロボットが腕をランダムな姿勢にし、
#     「関節の角度」と「そのとき手先がどこにあったか」を記録する
#  2) 学習: ニューラルネットワーク(AI)が「手先をここに置くには関節を何度にすればよいか」を学ぶ
#  3) 推論: 描きたい図形の各点をAIに入力 → 関節の目標角度を出力 → モーターで追従
#  4) 動画: 左にシミュレーション、右に「お手本(灰色)」と「AIが描いた線(赤)」を並べて保存
import math, time
import numpy as np
import gymnasium as gym
import imageio
from sklearn.neural_network import MLPRegressor

OUT = "draw_robot.mp4"
FPS = 30
STEPS_PER_POINT = 4      # 1点あたりのシミュレーション回数（大きいほどゆっくり描く）
N_BABBLING = 20000       # 体験する姿勢の数

CAMERA = {"distance": 0.75, "elevation": -90.0, "azimuth": 90.0,
          "lookat": np.array([0.0, 0.0, 0.0])}   # 真上から見下ろすカメラ
env = gym.make("Reacher-v5", render_mode="rgb_array", width=480, height=480,
               default_camera_config=CAMERA)
env.reset(seed=0)
u = env.unwrapped
rng = np.random.default_rng(0)

def tip():
    return u.get_body_com("fingertip")[:2].copy()

def set_arm(q0, q1):
    qpos = u.data.qpos.copy(); qvel = u.data.qvel.copy() * 0
    qpos[0], qpos[1] = q0, q1
    u.set_state(qpos, qvel)

# ---------- 1. 体験（モーターバブリング） ----------
t0 = time.time()
Q, P = [], []
for _ in range(N_BABBLING):
    q0 = rng.uniform(-math.pi, math.pi)
    q1 = rng.uniform(0.05, 2.9)          # ひじは片側だけに曲げる（答えを1つに決めるため）
    set_arm(q0, q1)
    Q.append([math.cos(q0), math.sin(q0), q1])
    P.append(tip())
Q, P = np.array(Q), np.array(P)
print(f"体験データ {len(P)} 件を収集（{time.time()-t0:.1f}秒）")

# ---------- 2. 学習 ----------
t0 = time.time()
ai = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, random_state=0)
ai.fit(P, Q)                              # 入力: 手先の位置 → 出力: 関節角度
print(f"AIの学習完了（{time.time()-t0:.1f}秒）")

def ai_joint_target(xy):
    c, s, q1 = ai.predict(np.asarray(xy).reshape(1, 2))[0]   # AIの推論
    return math.atan2(s, c), q1

# ---------- 3. 推論 + モーター制御 ----------
KP, KD = 4.0, 0.25   # 関節を目標角度へ動かすPD制御のゲイン

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def move_toward(goal_q, target_xy):
    q, dq = u.data.qpos[:2], u.data.qvel[:2]
    err = np.array([wrap(goal_q[0] - q[0]), goal_q[1] - q[1]])
    action = np.clip(KP * err - KD * dq, -1, 1)
    qpos = u.data.qpos.copy(); qvel = u.data.qvel.copy()
    qpos[2:4] = target_xy; qvel[2:4] = 0             # 赤い目標点を表示用に移動
    u.set_state(qpos, qvel)
    u.step(action)

# ---------- 描く図形 ----------
def heart(n=240, scale=0.14):
    t = np.linspace(0, 2 * math.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    p = np.stack([x, y], 1) / 17.0 * scale
    p[:, 1] += 0.01
    return p

def star(n=240, r_out=0.16, r_in=0.07):
    c = [[(r_out if k % 2 == 0 else r_in) * math.cos(math.pi / 2 + k * math.pi / 5),
          (r_out if k % 2 == 0 else r_in) * math.sin(math.pi / 2 + k * math.pi / 5)]
         for k in range(11)]
    c = np.array(c); seg = n // 10
    return np.concatenate([np.linspace(c[i], c[i + 1], seg, endpoint=False) for i in range(10)])

# ---------- 自分で決めた図形：イニシャル ----------
# 文字は「線（ストローク）」の集まり。線と線の間はペンを上げて移動する
def line(a, b, step=0.003):
    a, b = np.array(a), np.array(b)
    n = max(int(np.linalg.norm(b - a) / step), 2)
    return np.linspace(a, b, n)

def ellipse(cx, cy, rx, ry, n=200):
    t = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)

def initials():
    T = [line((-0.165, 0.09), (-0.035, 0.09)),   # Tの横棒
         line((-0.10, 0.09), (-0.10, -0.09))]    # Tの縦棒
    O = [ellipse(0.10, 0.0, 0.06, 0.09)]         # O
    return T + O

# 図形は「ストロークのリスト」で表す
SHAPES = [("initials", initials()), ("heart", [heart()]), ("star", [star()])]

def panel(path, trail, size=480, lim=0.22):
    img = np.full((size, size, 3), 255, np.uint8)
    def dot(p, color, w):
        cc = int((p[0] + lim) / (2 * lim) * (size - 1))
        rr = int((lim - p[1]) / (2 * lim) * (size - 1))
        img[max(rr - w, 0):rr + w + 1, max(cc - w, 0):cc + w + 1] = color
    for p in path:
        dot(p, (180, 180, 180), 1)
    for p in trail:
        dot(p, (220, 30, 30), 2)
    return img

# ---------- 4. 動画 ----------
frames = []
set_arm(0.0, 1.0)
for name, strokes in SHAPES:
    guide = np.concatenate(strokes)          # お手本（灰色）
    trail, errs = [], []
    for stroke in strokes:
        goal = ai_joint_target(stroke[0])
        for _ in range(45):                  # ペンを上げて書き始めの点へ移動（線は描かない）
            move_toward(goal, stroke[0])
            frames.append(np.concatenate([env.render(), panel(guide, trail)], 1))
        for p in stroke:                     # ペンを下ろして線を描く
            goal = ai_joint_target(p)
            for _ in range(STEPS_PER_POINT):
                move_toward(goal, p)
                trail.append(tip()); errs.append(np.linalg.norm(tip() - p))
                frames.append(np.concatenate([env.render(), panel(guide, trail)], 1))
    frames += [frames[-1]] * FPS * 2
    print(f"{name}: 平均誤差 {np.mean(errs)*100:.2f} cm / 最大誤差 {np.max(errs)*100:.2f} cm")

imageio.mimsave(OUT, frames, fps=FPS)
print(f"動画を保存しました: {OUT}（{len(frames)/FPS:.0f}秒）")
