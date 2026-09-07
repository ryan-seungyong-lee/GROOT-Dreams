"""Check the Depth Anything tower honours every contract the IDM action head relies on.

Run this before spending GPU hours. Each assertion targets one failure that is otherwise
silent: wrong token count, CLS left in, SigLIP normalisation reaching the depth tower, the
DDP-only mask_token crash, and the DPT decoder surviving into the parameter count.
"""
import sys, torch

sys.path.insert(0, "/sjw_alinlab/home/seungyong/project/anyh2r/GR00T-Dreams")
from gr00t.model.action_head.depth_anything_tower import DepthAnythingTower
from gr00t.model.action_head.multimodal_projector import (
    MultimodalProjector, MultimodalProjectorConfig)

MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
SIGLIP_VISION_TOWER_PARAMS_M = 316.0   # measured from the baseline checkpoint
ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {label:52s} got={got} want={want}")


print(f"loading {MODEL} ...", flush=True)
tower = DepthAnythingTower.from_pretrained(MODEL).eval()

print("\n=== 1. action head 가 접근하는 속성 ===")
for path in ("text_model", "logit_scale", "logit_bias",
             "vision_model.encoder.layers", "vision_model.head"):
    obj = tower
    try:
        for part in path.split("."):
            obj = getattr(obj, part)
        print(f"  PASS  {path:52s} {type(obj).__name__}")
    except AttributeError as exc:
        ok = False
        print(f"  FAIL  {path:52s} {exc}")
check("encoder.layers 개수 (인덱스 11 존재)", len(tower.vision_model.encoder.layers), 24)

print("\n=== 2. 토큰/차원 계약 (224 입력) ===")
x = torch.randn(4, 3, 224, 224) * 2.5      # ImageNet 범위를 흉내내 tripwire 통과
with torch.no_grad():
    raw = tower.vision_model.backbone(x).feature_maps[-1]
    out = tower.vision_model(x)["last_hidden_state"]
check("CLS 포함 원본 토큰 (DINOv2 는 CLS 를 붙인다)", raw.shape[1], 257)
check("CLS 제거 후 토큰", out.shape[1], 256)
check("hidden dim (= siglip_hidden_size)", out.shape[2], 1024)
check("sqrt 가 정수 (mm_projector 요구)", int(out.shape[1] ** 0.5) ** 2, 256)

print("\n=== 3. 해상도 드리프트가 조용히 통과하지 않는가 ===")
with torch.no_grad():
    o256 = tower.vision_model(torch.randn(1, 3, 256, 256) * 2.5)["last_hidden_state"]
check("256 입력은 324 토큰 (16 토큰이 안 나옴 = assert 가 필요한 이유)", o256.shape[1], 324)

print("\n=== 4. mm_projector 통과 후 16 토큰 ===")
proj = MultimodalProjector(MultimodalProjectorConfig(
    hidden_size=1024, mm_hidden_size=1024, mm_projector_type="mlp_doubledownsample")).eval()
with torch.no_grad():
    p = proj(out)
check("mm_projector 출력 토큰 (= num_visual_tokens_per_frame)", p.shape[1], 16)
check("mm_projector 출력 차원", p.shape[2], 1024)

print("\n=== 5. 정규화 tripwire (SigLIP 입력이 새어들어오면 잡는가) ===")
t2 = DepthAnythingTower.from_pretrained(MODEL).eval()   # 새 인스턴스: 플래그 미사용
try:
    with torch.no_grad():
        t2.vision_model(torch.rand(1, 3, 224, 224) * 2 - 1)   # SigLIP 범위 [-1,1]
    ok = False
    print("  FAIL  SigLIP 범위 입력이 통과해버렸다 (tripwire 미작동)")
except ValueError as exc:
    print(f"  PASS  tripwire 작동: {str(exc)[:74]}...")

print("\n=== 6. DDP 함정: gradient 를 못 받는 파라미터 ===")
mt = tower.vision_model.backbone.embeddings.mask_token
check("mask_token.requires_grad (True 면 DDP 첫 스텝에서 크래시)", mt.requires_grad, False)
never_grad = [n for n, q in tower.named_parameters()
              if q.requires_grad and ("mask_token" in n)]
check("학습 대상에 남은 mask_token 수", len(never_grad), 0)

print("\n=== 7. DPT 디코더가 딸려오지 않았는가 ===")
bad = [n for n, _ in tower.named_parameters() if ".neck." in n or "head.projection" in n]
check("neck/depth-head 파라미터 수", len(bad), 0)
n_tot = sum(q.numel() for q in tower.parameters())
print(f"  총 파라미터 {n_tot/1e6:.1f} M  (기준선 SigLIP 비전타워 {SIGLIP_VISION_TOWER_PARAMS_M} M, "
      f"차이 {abs(n_tot/1e6 - SIGLIP_VISION_TOWER_PARAMS_M)/SIGLIP_VISION_TOWER_PARAMS_M:.1%})")
check("규모가 기준선과 20% 이내 (공정 비교 조건)",
      abs(n_tot/1e6 - SIGLIP_VISION_TOWER_PARAMS_M) / SIGLIP_VISION_TOWER_PARAMS_M < 0.20, True)

print("\n=== 8. action head 의 동결 로직이 그대로 돌아가는가 ===")
try:
    for q in tower.parameters():
        q.requires_grad = True
    tower.freeze_unused_parameters()                       # mask_token 다시 동결
    tower.logit_scale.requires_grad = False
    tower.logit_bias.requires_grad = False
    for q in tower.vision_model.encoder.layers[11].parameters():
        q.requires_grad = False
    for q in tower.vision_model.head.parameters():
        q.requires_grad = False
    tr = sum(q.numel() for q in tower.parameters() if q.requires_grad)
    print(f"  PASS  실행됨. trainable {tr/1e6:.1f} M / frozen {(n_tot-tr)/1e6:.1f} M")
    check("백본 대부분이 학습 대상 (파인튜닝)", tr > 0.9 * n_tot, True)
    check("layers[11] 이 실제로 동결됨 (기준선과 대칭)",
          any(not q.requires_grad for q in tower.vision_model.encoder.layers[11].parameters()),
          True)
except Exception as exc:
    ok = False
    print(f"  FAIL  {type(exc).__name__}: {exc}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
