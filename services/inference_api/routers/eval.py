"""Değerlendirme router'ı — /eval/run, /eval/results, /eval/results/export.

M7: endpoint iskeleti + son sonuçları sunma. M9: gerçek QoD A/B harness bağlanır
(roadguard.eval.harness). Dashboard QoD A/B paneli bu endpoint'leri tüketir.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import PlainTextResponse

from services.inference_api.models import EvalRunRequest
from services.inference_api.security import resolve_ground_truth, validate_source, verify_token

router = APIRouter(tags=["eval"])
log = logging.getLogger("roadguard.api.eval")

# Varsayılan ground-truth yolu (istek başına literal kurmak yerine modül-sabiti).
_DEFAULT_GT = "data/samples/ornek_gt.json"


@router.get("/eval/results")
def eval_results(request: Request):
    res = getattr(request.app.state, "eval_results", None)
    if not res:
        return {
            "status": "no_results",
            "message": "Henüz eval çalıştırılmadı. POST /eval/run ile başlatın.",
        }
    return res


@router.post("/eval/run")
def eval_run(
    req: EvalRunRequest,
    request: Request,
    background: BackgroundTasks,
    _=Depends(verify_token),
):
    sm = request.app.state.stream
    # SEC-002/003: kullanici girdisini run_eval'e gecmeden once dogrula (istek
    # thread'inde → traversal/SSRF 400 ile caller'a doner, background'a sizmaz).
    validated_source = validate_source(req.source)
    source = validated_source or sm.cfg.get("runtime.source")
    gt = resolve_ground_truth(req.ground_truth) or _DEFAULT_GT

    def _job():
        try:
            from roadguard.eval.harness import run_eval  # M9

            request.app.state.eval_results = run_eval(
                sm.cfg, source, gt, qod_comparison=req.qod_comparison
            )
            log.info("Eval tamamlandı.")
        except Exception as e:  # noqa: BLE001
            # SEC: ham istisna metni (mutlak dosya yolları / iç yapı detayı) auth'suz
            # GET /eval/results ile sızabilirdi → jenerik mesaj sakla; kök-neden yalnız
            # sunucu log'unda kalır.
            log.warning("Eval harness henüz yok/başarısız: %s", e)
            request.app.state.eval_results = {
                "status": "error",
                "error": "eval çalıştırılamadı (ayrıntı sunucu log'unda)",
            }

    background.add_task(_job)
    return {
        "status": "queued",
        "source": source,
        "ground_truth": gt,
        "qod_comparison": req.qod_comparison,
    }


@router.get("/eval/results/export")
def eval_export(request: Request):
    res = getattr(request.app.state, "eval_results", None)
    if not res or res.get("status") in ("no_results", "error"):
        return PlainTextResponse("# Eval sonucu yok\n", media_type="text/markdown")
    md = (
        res.get("report_md")
        or "```json\n" + json.dumps(res, indent=2, ensure_ascii=False) + "\n```"
    )
    return PlainTextResponse(md, media_type="text/markdown")
