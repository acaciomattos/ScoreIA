import html
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.ml.artifacts import nome_artefato


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        if isinstance(value, (int,)):
            return str(value)
        if isinstance(value, (float,)):
            return f"{float(value):.4f}"
        if hasattr(value, "item"):
            v = value.item()
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                return f"{float(v):.4f}"
    except Exception:
        pass
    return str(value)

def _simplify_month_label(value: Any) -> str:
    s = str(value)
    if len(s) >= 6 and s[:6].isdigit():
        return f"{s[:4]}-{s[4:6]}"
    return s

def _table(headers: List[str], rows: List[List[Any]]) -> str:
    head = "".join([f"<th>{html.escape(str(h))}</th>" for h in headers])
    body_rows = []
    for row in rows:
        cols = "".join([f"<td>{html.escape(_fmt(c))}</td>" for c in row])
        body_rows.append(f"<tr>{cols}</tr>")
    body = "".join(body_rows) if body_rows else f"<tr><td colspan='{len(headers)}'>-</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _dict_rows(payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for k, v in payload.items():
        if isinstance(v, dict):
            rows.append([k, str(v)])
        else:
            rows.append([k, v])
    return rows


def _as_float_list(values: Sequence[Any]) -> List[float]:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _min_max(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        pad = abs(vmin) * 0.1 if vmin != 0 else 1.0
        return vmin - pad, vmax + pad
    return vmin, vmax


def _svg_multi_line_chart(
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    x_label: str = "x",
    y_label: str = "y",
    width: int = 760,
    height: int = 320,
    x_tick_labels: Optional[Sequence[str]] = None,
) -> str:
    x_vals = _as_float_list(x_values)
    clean_series: List[Tuple[str, List[float], str]] = []

    for name, vals, color in series:
        arr = _as_float_list(vals)
        if arr:
            clean_series.append((name, arr, color))

    if not x_vals or not clean_series:
        return f"<p>{html.escape(title)}: sem dados para grafico.</p>"

    n = min(len(x_vals), min(len(vals) for _, vals, _ in clean_series))
    x_vals = x_vals[:n]
    clean_series = [(name, vals[:n], color) for name, vals, color in clean_series]

    tick_labels = None
    if x_tick_labels is not None and len(x_tick_labels) >= n:
        tick_labels = [str(v) for v in x_tick_labels[:n]]

    margin_left, margin_right, margin_top, margin_bottom = 56, 16, 46, 38
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min, x_max = _min_max(x_vals)
    all_y = [v for _, vals, _ in clean_series for v in vals]
    y_min, y_max = _min_max(all_y)

    def sx(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return margin_top + ((y_max - y) / (y_max - y_min)) * plot_h

    y_grid = []
    for i in range(5):
        frac = i / 4
        yv = y_min + (y_max - y_min) * frac
        py = sy(yv)
        y_grid.append(
            f"<line x1='{margin_left}' y1='{py:.2f}' x2='{width - margin_right}' y2='{py:.2f}' stroke='#ececec' />"
            f"<text x='{margin_left - 8}' y='{py + 4:.2f}' text-anchor='end' font-size='11' fill='#555'>{_fmt(yv)}</text>"
        )

    x_grid = []
    if tick_labels is not None:
        step = max(1, int(len(x_vals) / 10) or 1)
        for i in range(0, len(x_vals), step):
            xv = x_vals[i]
            px = sx(xv)
            lbl = tick_labels[i]
            x_grid.append(
                f"<line x1='{px:.2f}' y1='{margin_top}' x2='{px:.2f}' y2='{height - margin_bottom}' stroke='#f2f2f2' />"
                f"<text x='{px:.2f}' y='{height - margin_bottom + 16}' text-anchor='middle' font-size='10' fill='#555'>{html.escape(lbl)}</text>"
            )
    else:
        for i in range(5):
            frac = i / 4
            xv = x_min + (x_max - x_min) * frac
            px = sx(xv)
            x_grid.append(
                f"<line x1='{px:.2f}' y1='{margin_top}' x2='{px:.2f}' y2='{height - margin_bottom}' stroke='#f2f2f2' />"
                f"<text x='{px:.2f}' y='{height - margin_bottom + 16}' text-anchor='middle' font-size='11' fill='#555'>{_fmt(xv)}</text>"
            )

    lines = []
    legend = []
    lx = margin_left
    for name, vals, color in clean_series:
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_vals, vals))
        lines.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2' />")
        legend.append(
            f"<rect x='{lx}' y='26' width='10' height='10' fill='{color}' />"
            f"<text x='{lx + 14}' y='35' font-size='11' fill='#444'>{html.escape(name)}</text>"
        )
        lx += 130

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='{width / 2:.2f}' y='18' text-anchor='middle' font-size='14' fill='#222'>{html.escape(title)}</text>"
        + "".join(y_grid)
        + "".join(x_grid)
        + f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#222' />"
        + f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#222' />"
        + "".join(lines)
        + "".join(legend)
        + f"<text x='{width / 2:.2f}' y='{height - 5}' text-anchor='middle' font-size='12' fill='#555'>{html.escape(x_label)}</text>"
        + f"<text x='14' y='{height / 2:.2f}' text-anchor='middle' transform='rotate(-90 14 {height / 2:.2f})' font-size='12' fill='#555'>{html.escape(y_label)}</text>"
        + "</svg>"
    )

def _svg_hbar_chart(items: Sequence[Tuple[str, float]], title: str, width: int = 760) -> str:
    if not items:
        return f"<p>{html.escape(title)}: sem dados para grafico.</p>"

    valid = [(name, float(val)) for name, val in items if val is not None]
    if not valid:
        return f"<p>{html.escape(title)}: sem dados para grafico.</p>"

    n = len(valid)
    height = max(240, 32 * n + 40)
    margin_left, margin_right, margin_top, margin_bottom = 220, 24, 26, 20
    plot_w = width - margin_left - margin_right

    max_val = max(abs(v) for _, v in valid) or 1.0

    bars = []
    for i, (name, val) in enumerate(valid):
        y = margin_top + i * 30
        bar_w = (abs(val) / max_val) * plot_w
        bars.append(
            f"<text x='{margin_left - 8}' y='{y + 14}' text-anchor='end' font-size='11' fill='#333'>{html.escape(name)}</text>"
            f"<rect x='{margin_left}' y='{y + 4}' width='{bar_w:.2f}' height='14' fill='#2a9d8f' />"
            f"<text x='{margin_left + bar_w + 6:.2f}' y='{y + 15}' font-size='11' fill='#333'>{_fmt(val)}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='{width / 2:.2f}' y='18' text-anchor='middle' font-size='14' fill='#222'>{html.escape(title)}</text>"
        + "".join(bars)
        + "</svg>"
    )


def _svg_scatter_chart(
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 520,
    height: int = 260,
) -> str:
    x_vals = _as_float_list(x_values)
    y_vals = _as_float_list(y_values)
    n = min(len(x_vals), len(y_vals))

    if n == 0:
        return f"<p>{html.escape(title)}: sem dados para grafico.</p>"

    x_vals = x_vals[:n]
    y_vals = y_vals[:n]

    margin_left, margin_right, margin_top, margin_bottom = 52, 16, 24, 36
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min, x_max = _min_max(x_vals)
    y_min, y_max = _min_max(y_vals)

    def sx(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return margin_top + ((y_max - y) / (y_max - y_min)) * plot_h

    dots = [
        f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='1.8' fill='#457b9d' opacity='0.45' />"
        for x, y in zip(x_vals, y_vals)
    ]

    zero_line = ""
    if y_min <= 0 <= y_max:
        zero_line = (
            f"<line x1='{margin_left}' y1='{sy(0):.2f}' x2='{width - margin_right}' y2='{sy(0):.2f}' "
            "stroke='#e63946' stroke-dasharray='4 3' />"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='{width / 2:.2f}' y='18' text-anchor='middle' font-size='13' fill='#222'>{html.escape(title)}</text>"
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#222' />"
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#222' />"
        + zero_line
        + "".join(dots)
        + f"<text x='{width / 2:.2f}' y='{height - 6}' text-anchor='middle' font-size='11' fill='#555'>{html.escape(x_label)}</text>"
        + f"<text x='13' y='{height / 2:.2f}' text-anchor='middle' transform='rotate(-90 13 {height / 2:.2f})' font-size='11' fill='#555'>{html.escape(y_label)}</text>"
        + "</svg>"
    )


def _build_optuna_trials_section(resultado_otimizacao: Optional[Dict[str, Any]]) -> str:
    if not resultado_otimizacao:
        return "<p>Otimizacao nao executada.</p>"

    studies: List[Tuple[str, Any]] = []

    if "melhor_modelo" in resultado_otimizacao:
        for modelo, dados in (resultado_otimizacao.get("todos", {}) or {}).items():
            study = dados.get("study")
            if study is not None:
                studies.append((modelo, study))
    else:
        study = resultado_otimizacao.get("study")
        if study is not None:
            studies.append(("modelo", study))

    if not studies:
        return "<p>Sem estudos Optuna disponiveis para visualizacao.</p>"

    charts = []

    for modelo, study in studies:
        values = []
        x_values = []

        for trial in getattr(study, "trials", []):
            if getattr(trial, "value", None) is None:
                continue
            state = str(getattr(trial, "state", ""))
            if "COMPLETE" not in state.upper():
                continue
            x_values.append(float(getattr(trial, "number", len(x_values))))
            values.append(float(trial.value))

        if not values:
            charts.append(f"<p>{html.escape(modelo)}: sem trials completos para grafico.</p>")
            continue

        direction = str(getattr(study, "direction", "")).lower()
        maximize = "maximize" in direction

        best_so_far: List[float] = []
        current = values[0]
        for v in values:
            if maximize:
                current = max(current, v)
            else:
                current = min(current, v)
            best_so_far.append(current)

        charts.append(
            _svg_multi_line_chart(
                x_values=x_values,
                series=[
                    ("trial_value", values, "#9aa0a6"),
                    ("best_so_far", best_so_far, "#1d3557"),
                ],
                title=f"Optuna - {modelo}",
                x_label="trial",
                y_label="score",
            )
        )

    return "".join([f"<div class='chart-card'>{c}</div>" for c in charts])


def _build_learning_curve_section(learning_curve: Optional[Dict[str, Any]]) -> str:
    if not learning_curve:
        return "<p>Curva de aprendizado indisponivel.</p>"

    values = _as_float_list(learning_curve.get("values", []))
    if not values:
        return "<p>Curva de aprendizado indisponivel.</p>"

    metric_name = str(learning_curve.get("metric_name") or "metric")
    model = str(learning_curve.get("model") or "model")
    x_values = list(range(1, len(values) + 1))

    return _svg_multi_line_chart(
        x_values=x_values,
        series=[(metric_name, values, "#264653")],
        title=f"Curva de aprendizado - {model}",
        x_label="epoca/iteracao",
        y_label=metric_name,
    )


def _build_feature_importance_chart(interpretabilidade: Dict[str, Any]) -> str:
    top = interpretabilidade.get("importancia_global", {}).get("top_features", [])
    items = [(str(it.get("feature")), float(it.get("importance", 0.0))) for it in top[:20]]
    return _svg_hbar_chart(items, "Top 20 features por importancia")


def _build_shap_section(interpretabilidade: Dict[str, Any]) -> Tuple[str, str]:
    shap_data = interpretabilidade.get("shap", {})
    if not shap_data.get("available"):
        reason = html.escape(str(shap_data.get("reason", "not_available")))
        return f"<p>SHAP indisponivel: {reason}</p>", "<p>Sem interpretacao de inversao SHAP.</p>"

    method = shap_data.get("method")
    method_info = f"<p>Metodo SHAP: <code>{html.escape(str(method))}</code>.</p>" if method else ""
    fallback_reason = shap_data.get("fallback_reason")
    if fallback_reason:
        method_info += f"<p>Fallback acionado: <code>{html.escape(str(fallback_reason))}</code>.</p>"

    top_shap = shap_data.get("top_shap", []) or []
    scatter = shap_data.get("scatter", []) or []

    rows = []
    inverted = []
    for item in top_shap:
        rows.append(
            [
                item.get("feature"),
                item.get("mean_abs_shap"),
                item.get("effect_direction"),
                item.get("effect_inversion_vs_target"),
            ]
        )
        if bool(item.get("effect_inversion_vs_target")):
            inverted.append(str(item.get("feature")))

    table_html = _table(["Feature", "mean_abs_shap", "direction", "inversion"], rows)

    cards = []
    for item in scatter[:20]:
        feature = str(item.get("feature"))
        title = f"SHAP scatter - {feature}"
        cards.append(
            "<div class='chart-card'>"
            + _svg_scatter_chart(
                x_values=item.get("x", []),
                y_values=item.get("shap", []),
                title=title,
                x_label=f"{feature} (dominio)",
                y_label="SHAP value",
            )
            + "</div>"
        )

    charts_html = "".join(cards) if cards else "<p>Sem pontos SHAP para plotagem.</p>"

    if inverted:
        inversion_text = "<p>Inversao de efeito estimada para: <code>" + html.escape(", ".join(inverted)) + "</code>.</p>"
    else:
        inversion_text = "<p>Nao foi identificada inversao de efeito SHAP vs correlacao feature-target nas top features.</p>"

    return method_info + table_html + "<div class='chart-grid'>" + charts_html + "</div>", inversion_text


def _build_ale_section(interpretabilidade: Dict[str, Any]) -> str:
    ale_data = interpretabilidade.get("ale", {})
    if not ale_data.get("available"):
        return "<p>ALE nao aplicavel para este modelo.</p>"

    rows = []
    cards = []

    for item in ale_data.get("top_features_ale", [])[:20]:
        feature = str(item.get("feature"))
        direction = item.get("direction")
        rows.append([feature, direction])

        grid = _as_float_list(item.get("grid", []))
        ale = _as_float_list(item.get("ale", []))

        cards.append(
            "<div class='chart-card'>"
            + _svg_multi_line_chart(
                x_values=grid,
                series=[(f"ALE ({feature})", ale, "#e76f51")],
                title=f"ALE - {feature}",
                x_label=f"{feature} (dominio)",
                y_label="ALE",
                width=520,
                height=260,
            )
            + "</div>"
        )

    table_html = _table(["Feature", "Direcao ALE"], rows)
    charts_html = "".join(cards) if cards else "<p>Sem curvas ALE para plotagem.</p>"

    return table_html + "<div class='chart-grid'>" + charts_html + "</div>"


def _table_from_dict_records(records: Optional[List[Dict[str, Any]]]) -> str:
    records = records or []
    if not records:
        return "<p>Sem dados.</p>"

    headers: List[str] = []
    for rec in records:
        for key in rec.keys():
            if key not in headers:
                headers.append(key)

    rows = [[rec.get(h) for h in headers] for rec in records]
    return _table(headers, rows)


def _svg_dual_axis_temporal_chart(distribuicao_tempo: List[Dict[str, Any]]) -> str:
    if not distribuicao_tempo:
        return "<p>Sem dados de distribuicao temporal.</p>"

    meses = sorted({str(r.get("temporal_month")) for r in distribuicao_tempo if r.get("temporal_month") is not None})
    grupos = sorted({str(r.get("split_group")) for r in distribuicao_tempo if r.get("split_group") is not None})

    if not meses or not grupos:
        return "<p>Sem dados de distribuicao temporal.</p>"

    x_labels = [_simplify_month_label(m) for m in meses]
    palette = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#6a4c93", "#264653"]

    width = 700
    height = 250
    margin_left, margin_right, margin_top, margin_bottom = 52, 52, 38, 42
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    month_step = plot_w / max(len(meses), 1)

    map_values = {}
    volume_values = []
    target_values = []
    for grupo in grupos:
        for mes in meses:
            row = next(
                (
                    r
                    for r in distribuicao_tempo
                    if str(r.get("split_group")) == grupo and str(r.get("temporal_month")) == mes
                ),
                {},
            )
            volume = float(row.get("volume", 0.0) or 0.0)
            target = float(row.get("target_mean", 0.0) or 0.0)
            map_values[(grupo, mes)] = {"volume": volume, "target": target}
            volume_values.append(volume)
            target_values.append(target)

    vol_max = max(volume_values) if volume_values else 1.0
    if vol_max <= 0:
        vol_max = 1.0
    target_min, target_max = _min_max(target_values)
    if target_min >= 0:
        target_min = 0.0

    def sx(month_idx: int) -> float:
        return margin_left + ((month_idx + 0.5) * month_step)

    def sy1(value: float) -> float:
        return margin_top + ((vol_max - value) / vol_max) * plot_h

    def sy2(value: float) -> float:
        return margin_top + ((target_max - value) / (target_max - target_min)) * plot_h

    y_grid = []
    for i in range(5):
        frac = i / 4
        vol_tick = vol_max * frac
        y = sy1(vol_tick)
        target_tick = target_max - ((target_max - target_min) * ((y - margin_top) / plot_h))
        y_grid.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='#eef2f7' />"
            f"<text x='{margin_left - 6}' y='{y + 3:.2f}' text-anchor='end' font-size='9' fill='#475569'>{_fmt(vol_tick)}</text>"
            f"<text x='{width - margin_right + 6}' y='{y + 3:.2f}' text-anchor='start' font-size='9' fill='#475569'>{_fmt(target_tick)}</text>"
        )

    month_labels = []
    for idx, label in enumerate(x_labels):
        x = sx(idx)
        month_labels.append(
            f"<text x='{x:.2f}' y='{height - margin_bottom + 14}' text-anchor='middle' font-size='9' fill='#475569'>{html.escape(label)}</text>"
        )

    group_count = max(len(grupos), 1)
    bar_width = min(16.0, (month_step * 0.75) / group_count)
    bars = []
    lines = []
    legend = []
    legend_x = margin_left

    for g_idx, grupo in enumerate(grupos):
        color = palette[g_idx % len(palette)]
        line_points = []
        month_center_offset = ((g_idx - ((group_count - 1) / 2.0)) * bar_width)

        for m_idx, mes in enumerate(meses):
            x_center = sx(m_idx) + month_center_offset
            volume = map_values[(grupo, mes)]["volume"]
            target = map_values[(grupo, mes)]["target"]
            y_bar = sy1(volume)
            bars.append(
                f"<rect x='{x_center - (bar_width / 2):.2f}' y='{y_bar:.2f}' width='{bar_width:.2f}' height='{(height - margin_bottom - y_bar):.2f}' fill='{color}' opacity='0.45' />"
            )
            line_points.append(f"{sx(m_idx):.2f},{sy2(target):.2f}")

        lines.append(
            f"<polyline points='{' '.join(line_points)}' fill='none' stroke='{color}' stroke-width='1.8' />"
        )
        lines.extend(
            [
                f"<circle cx='{sx(m_idx):.2f}' cy='{sy2(map_values[(grupo, mes)]['target']):.2f}' r='2' fill='{color}' />"
                for m_idx, mes in enumerate(meses)
            ]
        )
        legend.append(
            f"<rect x='{legend_x}' y='18' width='8' height='8' fill='{color}' opacity='0.45' />"
            f"<line x1='{legend_x + 12}' y1='22' x2='{legend_x + 24}' y2='22' stroke='{color}' stroke-width='1.8' />"
            f"<text x='{legend_x + 28}' y='25' font-size='9' fill='#334155'>{html.escape(str(grupo))}</text>"
        )
        legend_x += 110

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Distribuicao temporal de volumetria e media da variavel resposta'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='{width / 2:.2f}' y='12' text-anchor='middle' font-size='12' fill='#0f172a'>Distribuicao temporal de volumetria e media da variavel resposta</text>"
        + "".join(legend)
        + "".join(y_grid)
        + f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#334155' />"
        + f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#334155' />"
        + f"<line x1='{width - margin_right}' y1='{margin_top}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#334155' />"
        + "".join(bars)
        + "".join(lines)
        + "".join(month_labels)
        + f"<text x='{width / 2:.2f}' y='{height - 4}' text-anchor='middle' font-size='10' fill='#475569'>mes</text>"
        + f"<text x='14' y='{height / 2:.2f}' text-anchor='middle' transform='rotate(-90 14 {height / 2:.2f})' font-size='9' fill='#475569'>volume</text>"
        + f"<text x='{width - 12}' y='{height / 2:.2f}' text-anchor='middle' transform='rotate(90 {width - 12} {height / 2:.2f})' font-size='9' fill='#475569'>media target</text>"
        + "</svg>"
    )


def _build_metricas_tempo_chart(metricas_por_tempo: List[Dict[str, Any]]) -> str:
    if not metricas_por_tempo:
        return "<p>Sem dados de metricas por tempo.</p>"

    meses = sorted({str(r.get("temporal_month")) for r in metricas_por_tempo if r.get("temporal_month") is not None})
    grupos = sorted({str(r.get("split_group")) for r in metricas_por_tempo if r.get("split_group") is not None})
    if not meses or not grupos:
        return "<p>Sem dados de metricas por tempo.</p>"

    metric_keys = []
    for k in ["auc", "f1", "rmse", "r2"]:
        if any(r.get(k) is not None for r in metricas_por_tempo):
            metric_keys.append(k)

    if not metric_keys:
        return "<p>Sem metricas por tempo para plotagem.</p>"

    x_values = list(range(len(meses)))
    x_labels = [_simplify_month_label(m) for m in meses]
    palette = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#6a4c93", "#264653"]

    cards = []
    for metrica in metric_keys:
        series = []
        for idx, grupo in enumerate(grupos):
            cor = palette[idx % len(palette)]
            map_grupo = {
                str(r.get("temporal_month")): r
                for r in metricas_por_tempo
                if str(r.get("split_group")) == grupo
            }
            vals = []
            for m in meses:
                v = (map_grupo.get(m) or {}).get(metrica)
                vals.append(float(v) if v is not None else 0.0)
            series.append((f"{metrica}_{grupo}", vals, cor))

        cards.append(
            "<div class='chart-card'>"
            + _svg_multi_line_chart(
                x_values=x_values,
                x_tick_labels=x_labels,
                series=series,
                title=f"Metricas por tempo - {metrica}",
                x_label="mes",
                y_label=metrica,
            )
            + "</div>"
        )

    return "".join(cards)

def _build_temporal_segment_section(analise_tempo_segmento: Optional[Dict[str, Any]]) -> str:
    if not analise_tempo_segmento:
        return "<p>Analise temporal/segmento nao configurada nesta execucao.</p>"

    distribuicao_tempo = analise_tempo_segmento.get("distribuicao_tempo", []) or []
    distribuicao_segmento = analise_tempo_segmento.get("distribuicao_segmento", []) or []
    distribuicao_tempo_segmento = analise_tempo_segmento.get("distribuicao_tempo_segmento", []) or []

    metricas_por_tempo = analise_tempo_segmento.get("metricas_por_tempo", []) or []
    metricas_por_segmento = analise_tempo_segmento.get("metricas_por_segmento", []) or []
    metricas_por_tempo_segmento = analise_tempo_segmento.get("metricas_por_tempo_segmento", []) or []

    estabilidade_resumo = analise_tempo_segmento.get("estabilidade_resumo", {}) or {}
    estabilidade_rows = []
    for eixo, metricas in estabilidade_resumo.items():
        for nome_metrica, stats in (metricas or {}).items():
            estabilidade_rows.append(
                {
                    "eixo": eixo,
                    "metrica": nome_metrica,
                    "n_buckets": stats.get("n_buckets"),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "range": stats.get("range"),
                }
            )

    dist_ts_items = []
    for r in distribuicao_tempo_segmento[:40]:
        label = f"{r.get('split_group')} | {_simplify_month_label(r.get('temporal_month'))} | {r.get('segment_key')}"
        dist_ts_items.append((label, float(r.get('volume', 0.0))))

    met_tempo_segmento_items = []
    primary_metric = "auc" if any(r.get("auc") is not None for r in metricas_por_tempo_segmento) else "rmse"
    for r in metricas_por_tempo_segmento[:40]:
        v = r.get(primary_metric)
        if v is None:
            continue
        label = f"{r.get('split_group')} | {_simplify_month_label(r.get('temporal_month'))} | {r.get('segment_key')}"
        met_tempo_segmento_items.append((label, float(v)))

    met_segmento_items = []
    metric_segmento = "auc" if any(r.get("auc") is not None for r in metricas_por_segmento) else "rmse"
    for r in metricas_por_segmento[:40]:
        v = r.get(metric_segmento)
        if v is None:
            continue
        label = f"{r.get('split_group')} | {r.get('segment_key')}"
        met_segmento_items.append((label, float(v)))

    return (
        "<h3>Distribuicao por Tempo</h3>"
        + "<div class='chart-card'>"
        + _svg_dual_axis_temporal_chart(distribuicao_tempo)
        + "</div>"
        + "<h3>Distribuicao por Segmento</h3>"
        + ("<div class='chart-card'>" + _svg_hbar_chart([(f"{r.get('split_group')} | {r.get('segment_key')}", float(r.get('volume', 0.0))) for r in distribuicao_segmento[:40]], "Volume por segmento") + "</div>" if distribuicao_segmento else "<p>Sem dados.</p>")
        + "<h3>Distribuicao por Tempo/Segmento</h3>"
        + ("<div class='chart-card'>" + _svg_hbar_chart(dist_ts_items, "Volume por tempo/segmento") + "</div>" if dist_ts_items else "<p>Sem dados.</p>")
        + "<h3>Metricas por Tempo</h3>"
        + _build_metricas_tempo_chart(metricas_por_tempo)
        + "<h3>Metricas por Segmento</h3>"
        + ("<div class='chart-card'>" + _svg_hbar_chart(met_segmento_items, f"{metric_segmento} por segmento") + "</div>" if met_segmento_items else "<p>Sem dados.</p>")
        + "<h3>Metricas por Tempo/Segmento</h3>"
        + ("<div class='chart-card'>" + _svg_hbar_chart(met_tempo_segmento_items, f"{primary_metric} por tempo/segmento") + "</div>" if met_tempo_segmento_items else "<p>Sem dados.</p>")
        + "<h3>Resumo de Estabilidade de Metricas</h3>"
        + _table_from_dict_records(estabilidade_rows)
    )


def _build_feature_selection_stability_section(feature_selection_stability: Optional[Dict[str, Any]]) -> str:
    if not feature_selection_stability:
        return "<p>Selecao de features por estabilidade temporal nao executada.</p>"

    summary = {
        "enabled": feature_selection_stability.get("enabled"),
        "applied": feature_selection_stability.get("applied"),
        "reason": feature_selection_stability.get("reason"),
        "selector_model": feature_selection_stability.get("selector_model"),
        "metric_key": feature_selection_stability.get("metric_key"),
        "direction": feature_selection_stability.get("direction"),
        "n_features_original": feature_selection_stability.get("n_features_original"),
        "n_features_selected": feature_selection_stability.get("n_features_selected"),
        "best_mean_score": feature_selection_stability.get("best_mean_score"),
    }

    rows = []
    for item in feature_selection_stability.get("candidate_results", []) or []:
        rows.append(
            [
                item.get("n_features"),
                item.get("mean"),
                item.get("std"),
                item.get("range"),
                item.get("cv"),
                item.get("n_folds_valid"),
                item.get("performance_close_to_best"),
                item.get("selected"),
            ]
        )

    temporal_cv = feature_selection_stability.get("temporal_cv", {}) or {}
    fold_rows = []
    for fd in temporal_cv.get("folds", []) or []:
        fold_rows.append(
            [
                fd.get("fold_id"),
                ", ".join([_simplify_month_label(m) for m in fd.get("train_months", [])]),
                _simplify_month_label(fd.get("val_month")),
                fd.get("n_train"),
                fd.get("n_val"),
            ]
        )

    return (
        _table(["Campo", "Valor"], _dict_rows(summary))
        + "<h3>Candidatos avaliados (estabilidade)</h3>"
        + _table(
            [
                "n_features",
                "mean",
                "std",
                "range",
                "cv",
                "n_folds_valid",
                "close_to_best",
                "selected",
            ],
            rows,
        )
        + "<h3>Folds temporais utilizados</h3>"
        + _table(
            ["fold", "train_months", "val_month", "n_train", "n_val"],
            fold_rows,
        )
    )
def gerar_relatorio_tecnico_html(
    run_context: Dict[str, str],
    config: Dict[str, Any],
    resumo_dados: Dict[str, Any],
    metricas_treino: Dict[str, Any],
    metricas_teste: Dict[str, Any],
    modelo_final: str,
    best_params: Optional[Dict[str, Any]],
    best_cv_score: Optional[float],
    threshold_classificacao: Optional[float],
    metadata: Dict[str, Any],
    interpretabilidade: Dict[str, Any],
    resultado_otimizacao: Optional[Dict[str, Any]] = None,
    learning_curve: Optional[Dict[str, Any]] = None,
    metricas_oot: Optional[Dict[str, Any]] = None,
    analise_tempo_segmento: Optional[Dict[str, Any]] = None,
    feature_selection_stability: Optional[Dict[str, Any]] = None,
) -> str:
    params_table = _table(["Parametro", "Valor"], _dict_rows(config))
    run_table = _table(["Campo", "Valor"], _dict_rows(run_context))
    dados_table = _table(["Campo", "Valor"], _dict_rows(resumo_dados))
    metadata_table = _table(["Campo", "Valor"], _dict_rows(metadata))

    metricas_rows = [["treino", k, v] for k, v in metricas_treino.items()]
    metricas_rows += [["teste", k, v] for k, v in metricas_teste.items()]
    if metricas_oot:
        metricas_rows += [["oot", k, v] for k, v in metricas_oot.items()]

    metricas_table = _table(["Conjunto", "Metrica", "Valor"], metricas_rows)

    model_summary = {
        "modelo_final": modelo_final,
        "best_cv_score": best_cv_score,
        "threshold_classificacao": threshold_classificacao,
        "best_params": best_params or {},
    }
    model_table = _table(["Campo", "Valor"], _dict_rows(model_summary))

    if resultado_otimizacao and "melhor_modelo" in resultado_otimizacao:
        rows = []
        for m, data in resultado_otimizacao.get("todos", {}).items():
            rows.append([m, data.get("best_score"), data.get("best_params")])
        opt_table = _table(["Modelo", "Melhor score", "Melhores parametros"], rows)
    elif resultado_otimizacao:
        opt_table = _table(
            ["Campo", "Valor"],
            _dict_rows(
                {
                    "best_score": resultado_otimizacao.get("best_score"),
                    "best_params": resultado_otimizacao.get("best_params"),
                    "n_trials": len(resultado_otimizacao.get("study").trials) if resultado_otimizacao.get("study") is not None else 0,
                }
            ),
        )
    else:
        opt_table = "<p>Otimizacao nao executada.</p>"

    imp_top = interpretabilidade.get("importancia_global", {}).get("top_features", [])
    imp_table = _table(
        ["Feature", "Importancia"],
        [[it.get("feature"), it.get("importance")] for it in imp_top],
    )

    shap_summary_html, shap_inversion_html = _build_shap_section(interpretabilidade)
    ale_summary_html = _build_ale_section(interpretabilidade)

    ts = run_context.get("timestamp", "")
    title = f"ScoreIA Technical Report - {ts}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; background: #fff; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
    h3 {{ margin-top: 22px; margin-bottom: 6px; }}
    p.meta {{ color: #444; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f6f6; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
    .chart-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 6px; background: #fff; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>ScoreIA Technical Report</h1>
  <p class="meta">Timestamp: <code>{html.escape(ts)}</code></p>

  <h2>Run Context</h2>
  {run_table}

  <h2>Input Parameters</h2>
  {params_table}

  <h2>Data Volumetry and Split</h2>
  {dados_table}

  <h2>Pipeline Metadata</h2>
  {metadata_table}

  <h2>Optimization Summary (Optuna)</h2>
  {opt_table}

  <h2>Optimization Process Charts (Optuna)</h2>
  <div class="chart-grid">{_build_optuna_trials_section(resultado_otimizacao)}</div>

  <h2>Feature Selection Stability</h2>
  {_build_feature_selection_stability_section(feature_selection_stability)}

  <h2>Model and Decision Summary</h2>
  {model_table}

  <h2>Learning Curve</h2>
  <div class="chart-card">{_build_learning_curve_section(learning_curve)}</div>

  <h2>Metrics (Train/Test/OOT)</h2>
  {metricas_table}

  <h2>Global Feature Importance</h2>
  {imp_table}
  <div class="chart-card">{_build_feature_importance_chart(interpretabilidade)}</div>

  <h2>SHAP Summary</h2>
  {shap_inversion_html}
  {shap_summary_html}

  <h2>ALE Summary</h2>
  {ale_summary_html}

  <h2>Temporal/Segment Distribution and Stability</h2>
  {_build_temporal_segment_section(analise_tempo_segmento)}
</body>
</html>"""


def salvar_relatorio_html(
    html_content: str,
    output_dir: str,
    run_context: Dict[str, str],
    file_name: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report_name = file_name or nome_artefato("report", run_context, "html")
    path = os.path.join(output_dir, report_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return path




