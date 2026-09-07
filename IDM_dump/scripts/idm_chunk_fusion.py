"""IDM 라벨에서 프레임 단위 잡음을 걷어내는 두 가지 후처리.

왜 필요한지는 ../docs/IDM_ACTION_LABELING.md 에 전부 적어 두었다. 한 줄 요약:

    IDM은 매 프레임 H-step 청크를 예측하는데 지금까지 step 0 하나만 쓰고 H-1개를 버렸다.
    프레임 t는 t-H+1 … t 에서 시작한 청크 H개에 모두 들어 있으므로, 그것들을 평균하면
    추가 forward pass 없이 프레임 단위 잡음이 크게 줄어든다.

이 모듈은 그 평균(ChunkAccumulator)과, 그 위에 얹는 약한 Savitzky-Golay 스무딩
(smooth_actions), 그리고 효과를 바로 확인할 수 있는 품질 지표(label_quality)를 제공한다.

두 dump 스크립트(dump_idm_actions_h2r.py, dump_idm_traj_openarm.py)가 공유한다.
"""
from __future__ import annotations

import numpy as np

# 28-dim 관절 벡터의 구간. 팔/목은 부드러운 궤적, 손(그리퍼)은 가장자리가 살아 있어야 한다.
ARMLIKE_SLICES = ((0, 2), (2, 9), (9, 16))     # neck, left_arm, right_arm
HANDLIKE_SLICES = ((16, 22), (22, 28))         # left_hand, right_hand
RIGHT_ARM = slice(9, 16)


class ChunkAccumulator:
    """프레임 t를 덮는 모든 청크의 예측을 모아 평균한다.

    IDM은 프레임 t에서 (t, t+H) 두 프레임을 보고 a[t], a[t+1], …, a[t+H-1] 을 예측한다.
    청크 시작점 t' 가 stride 간격으로 훑고 지나가면 프레임 u 는 t' ∈ [u-H+1, u] 인 모든
    청크의 step (u - t') 위치에 나타난다. 그 값들을 전부 더해 개수로 나눈다.

    stride=1이면 프레임당 최대 H개가 모인다. 에피소드 앞뒤 H 프레임은 그보다 적게 모이므로
    (1개→H개로 램프업) 평활 효과가 약하다 — 의도된 동작이고, 없는 정보를 지어내지 않는다.

    잡음이 청크끼리 완전히 독립이면 표준편차가 √H배 줄지만, 이웃 청크는 입력 프레임을
    상당 부분 공유하므로 실제 이득은 그보다 작다. 측정값은 dump 로그의 QC 줄로 확인할 것.
    """

    def __init__(self, length: int, action_dim: int, horizon: int):
        self.L = int(length)
        self.H = int(horizon)
        self._sum = np.zeros((self.L, action_dim), dtype=np.float64)
        self._cnt = np.zeros((self.L, 1), dtype=np.float64)

    def add(self, start: int, chunk: np.ndarray) -> None:
        """start 프레임에서 시작한 (H, A) 청크를 더한다."""
        n = min(self.H, len(chunk), self.L - start)
        if n <= 0:
            return
        self._sum[start:start + n] += chunk[:n]
        self._cnt[start:start + n] += 1.0

    def result(self) -> np.ndarray:
        if (self._cnt == 0).any():
            missing = int((self._cnt == 0).sum())
            raise RuntimeError(
                f"{missing} frames were covered by no chunk — stride must be <= horizon")
        return (self._sum / self._cnt).astype(np.float32)

    @property
    def coverage(self) -> np.ndarray:
        """프레임별로 몇 개의 청크가 평균됐는지. QC 로그용."""
        return self._cnt[:, 0].copy()


def smooth_actions(actions: np.ndarray, window: int = 9, hand_window: int = 5,
                   polyorder: int = 2) -> np.ndarray:
    """청크 평균 뒤에 남은 고주파를 Savitzky-Golay로 약하게 걷어낸다.

    이동평균이 아니라 SG를 쓰는 이유: 이동평균은 램프의 시작/끝과 그리퍼가 닫히는
    가장자리를 뭉갠다. SG(polyorder=2)는 국소 2차 곡률을 보존하므로 궤적의 모양과
    파지 순간의 계단이 살아남는다.

    팔/목은 window(기본 9프레임 = 20Hz에서 0.45초), 손은 hand_window(기본 5 = 0.25초).
    손을 짧게 두는 것은 파지 타이밍이 라벨에서 제일 중요한 가장자리이기 때문이다.
    window<=1 이면 해당 구간은 건드리지 않는다.
    """
    from scipy.signal import savgol_filter

    out = np.array(actions, dtype=np.float32, copy=True)
    L = len(out)
    for slices, w in ((ARMLIKE_SLICES, window), (HANDLIKE_SLICES, hand_window)):
        if w is None or w <= 1:
            continue
        w = int(w) | 1                       # savgol은 홀수 창만 받는다
        if w > L:
            continue
        if w <= polyorder:
            continue
        for a, b in slices:
            out[:, a:b] = savgol_filter(out[:, a:b], w, polyorder, axis=0, mode="nearest")
    return out


def label_quality(actions: np.ndarray, fps: int = 20, cols: slice = RIGHT_ARM) -> dict:
    """라벨이 '궤적'인지 '잡음'인지 한 줄로 보여 주는 지표.

    lag1        속도(Δq)의 lag-1 자기상관. 실제 teleop ≈ +0.44, 잡음 ≈ 0, 톱니 < 0.
                이 값이 음수면 후처리가 아직 부족한 것이다.
    flip        Δq의 부호가 다음 프레임에 뒤집히는 비율. 동전 던지기(50%)에 가까우면 잡음.
                실제 teleop ≈ 2.6%.
    step        프레임당 |Δq| 평균 (rad). 잡음이 실려 있으면 부풀어 오른다.
    """
    q = np.asarray(actions, dtype=np.float64)[:, cols]
    if len(q) < 4:
        return {"lag1": float("nan"), "flip": float("nan"), "step": float("nan")}
    v = np.diff(q, axis=0)
    rho = []
    for j in range(v.shape[1]):
        a, b = v[:-1, j], v[1:, j]
        if a.std() > 1e-9 and b.std() > 1e-9:
            rho.append(float(np.corrcoef(a, b)[0, 1]))
    s = np.sign(v)
    return {
        "lag1": float(np.mean(rho)) if rho else float("nan"),
        "flip": float((s[1:] * s[:-1] < 0).mean()),
        "step": float(np.abs(v).mean()),
    }


def quality_line(before: np.ndarray, after: np.ndarray, fps: int = 20) -> str:
    b, a = label_quality(before, fps), label_quality(after, fps)
    return (f"QC r_arm | lag1 {b['lag1']:+.3f} -> {a['lag1']:+.3f} "
            f"| signflip {b['flip']*100:4.1f}% -> {a['flip']*100:4.1f}% "
            f"| |dq| {b['step']:.5f} -> {a['step']:.5f} rad "
            f"(teleop reference: lag1 +0.44, signflip 2.6%, |dq| 0.0030)")


# ---------------------------------------------------------------------------
# observation.state 를 action으로부터 만드는 모델
# ---------------------------------------------------------------------------
# 왜 action[t-1]이 아닌가
# ----------------------
# IDM은 action(관절 명령)만 내놓으므로 state(측정된 관절각)는 만들어 줘야 한다.
# 처음에는 state[t] := action[t-1] 을 썼는데, 실제 teleop에서 재보니 서보 추종 지연이
# 1프레임이 아니라 **4프레임(200 ms)**이었다. 게다가 그리퍼는 물체에 막혀 명령을 아예
# 다 못 따라간다 (명령 > 측정인 프레임이 63%, 평균 차이 0.11 rad).
#
# openarm-teleop-v4-rldx-v2 (635 ep / 117,769 frame) 전체에 대해 재고 피팅한 결과:
#
#     |모델 - 실제 state|          neck   l_arm  r_arm  l_hand  r_hand   ALL28
#     state = action[t-1]  (구)   0.0043 0.0108 0.0173 0.0603  0.1124  0.0443
#     저역통과만                  0.0043 0.0104 0.0123 0.0601  0.1039  0.0411
#     저역통과 + gain/offset      0.0001 0.0024 0.0057 0.0157  0.0466  0.0154   <- 채택
#
# 모델:  s[t] = (1-ALPHA) * s[t-1] + ALPHA * a[t-1]        (1차 저역통과)
#        state[t] = GAIN * s[t] + OFFSET                    (관절별 상수)
#
# ALPHA=0.25 는 시정수 200 ms. 오른팔 최적은 0.28, 전체 28-dim 최적은 0.23이고
# 0.22~0.30 구간이 평평하므로 0.25로 고정했다.
#
# GAIN은 팔에서는 거의 1.0 (0.96~1.02) 이라 사실상 무해하고, 실효는 관절별 상수
# OFFSET(중력 처짐/영점 보정, 0.01~0.04 rad)에서 나온다. 손은 다르다 — GAIN이
# 0.30~0.98 로 크게 떨어지는데, 이는 **손가락이 물체에 막혀 명령한 만큼 못 닫히는
# 접촉 현상**이다. 액션만 보고 접촉 시점을 알 수 없으므로 전 구간 평균 이득으로
# 근사했고, 그래도 오른손 잔차가 0.104 -> 0.047 로 절반이 된다.
#
# 한계 (알고 쓸 것)
# ----------------
# 이건 h2r의 state를 teleop의 state와 **통계적으로 닮게** 만드는 프록시다. 측정값이
# 아니다. 남는 잔차 0.0154 rad은 부하/백래시/접촉 타이밍처럼 액션만으로는 재현할 수
# 없는 실제 동역학이다. VLA는 state를 입력으로 받고(state_dropout_prob=0.3) action을
# 예측하므로, 중요한 것은 "state와 action의 관계"가 두 데이터셋에서 같아지는 것이다.
STATE_ALPHA = 0.25          # 하위호환: model="lag" 및 외부 호출자가 참조하는 단일값
# 2026-08-27 재적합. openarm-idm-v3v4-teleop20-stereo/train 269,983 프레임(20Hz)에서
# alpha를 그룹별로, gain/offset을 관절별로 최소제곱 적합했다. held-out test split 잔차
# 0.0366 -> 0.0328 (10.4 퍼센트 감소), 왼손 0.0750 -> 0.0651 (13.2 퍼센트 감소).
#
# 왜 다시 적합했나: 이전 상수는 openarm-teleop-v4-rldx-v2 에서 나왔는데 그 코퍼스는
# 635 에피소드 전체에서 물리적 왼손이 0.2 rad 이상 접힌 적이 없다. 움직이지 않는 신호에서
# 적합한 gain은 식별되지 않으며, 실제로 이전 왼손 gain은 전부 1 근처(0.79~1.03)였다.
# 신규 teleop 0.86 h 가 들어와 좌우가 대등해지면서 처음으로 식별됐다 —
# 왼손 thumb_2 gain 0.96 -> 0.40 으로, 오른손 0.17 과 같은 접촉 포화 패턴이 나타났다.
#
# alpha 는 그룹별로 갈린다 (교차상관 실측: 팔 167 ms, 손 267 ms):
#   팔 0.300 (tau 167 ms) / 손 0.175 (tau 286 ms) / neck 0.150
# ★ alpha 만 바꾸면 오히려 나빠진다 (0.0366 -> 0.0507). 저역통과가 약해져 진폭이 커지는
#   것을 gain 이 다시 눌러주는 구조이므로 둘은 반드시 함께 갱신한다.
STATE_ALPHA_PER_JOINT = np.array([0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175], dtype=np.float32)
STATE_GAIN = np.array([0.96307, 1, 0.99126, 1.0038, 1.0015, 1.0012, 1.0068, 0.99574, 0.95935, 0.9944, 0.98978, 1.0002, 0.99819, 1.0047, 0.9856, 0.97043, 0.95311, 0.40123, 0.73347, 0.78679, 0.90972, 0.95852, 0.84201, 0.17091, 0.64135, 0.69039, 0.86337, 0.9748], dtype=np.float32)
STATE_OFFSET = np.array([0.041054, 0.00052919, 0.0071754, 0.0017314, 0.00070954, -0.01195, -0.0044035, -0.00028781, 0.027063, -0.0081584, 0.0021651, -0.0020525, -0.0073352, -0.00055948, 0.0077511, -0.028317, 0.11047, 0.028555, -0.062008, 0.02309, -0.043666, -0.068508, 0.050516, 0.072369, -0.005585, 0.037243, -0.05149, 0.011545], dtype=np.float32)


def state_from_action(actions: np.ndarray, model: str = "fitted",
                      alpha: float = STATE_ALPHA) -> np.ndarray:
    """(L, 28) action -> (L, 28) observation.state.

    model="fitted"  저역통과 + teleop에서 피팅한 관절별 gain/offset (권장, 기본값)
    model="lag"     저역통과만 (gain/offset 없이)
    model="prev"    state[t] = action[t-1]  (2026-08-10 이전 동작, A/B용)
    """
    A = np.asarray(actions, dtype=np.float32)
    if model == "prev":
        return np.concatenate([A[:1], A[:-1]], axis=0)
    # "fitted" uses the per-group time constants; "lag" keeps the single legacy alpha so
    # A/B comparisons against the old behaviour stay meaningful.
    al = (STATE_ALPHA_PER_JOINT if model == "fitted"
          else np.full(A.shape[1], alpha, np.float32))
    s = np.empty_like(A)
    s[0] = A[0]
    one_minus = 1.0 - al
    for t in range(1, len(A)):
        s[t] = one_minus * s[t - 1] + al * A[t - 1]
    if model == "lag":
        return s
    if model != "fitted":
        raise ValueError(f"unknown state model {model!r}")
    return (s * STATE_GAIN + STATE_OFFSET).astype(np.float32)
