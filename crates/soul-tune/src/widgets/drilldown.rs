use ratatui::layout::Constraint;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Frame;

use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::data::RetrieveCaseData;

fn render_metric_row(frame: &mut Frame, y: u16, k: &usize, recall: f64, precision: f64, ndcg: f64) {
    let col_widths: [Constraint; 6] = [
        Constraint::Length(2),
        Constraint::Length(6),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Fill(1),
        Constraint::Length(0),
    ];
    // layout passed from caller, just use y directly
    // This is called from build_drilldown_lines which returns Vec<Line>,
    // so we can't use Layout here. Leave as Vec<Line> for callers.
}

/// Build text lines for drill-down view. The comparison table
/// uses proper Layout constraints.
pub fn build_drilldown_lines(
    data: &RetrieveCaseData,
    case_index: usize,
    case_total: usize,
) -> Vec<Line> {
    let mut l = Vec::new();
    let hdr = Style::new().yellow().bold();
    let green = Style::new().green();
    let red = Style::new().red();
    let gray = Style::new().dark_gray();

    l.push(Line::from(Span::raw(format!(
        " 用例: #{} {}",
        case_index + 1,
        data.case_name
    ))));
    let passed =
        data.combined_ranking_metrics.hit_rate > 0.0 || data.combined_ranking_metrics.mrr > 0.0;
    l.push(Line::from(vec![
        Span::raw(" 状态: "),
        Span::styled(
            if passed { "✓ 通过" } else { "✗ 失败" },
            if passed { green } else { red },
        ),
    ]));
    l.push(Line::from(""));

    l.push(Line::from(Span::styled(" ── 综合排序指标 ──", hdr)));
    for (k, r) in &data.combined_ranking_metrics.recall_at {
        let p = data
            .combined_ranking_metrics
            .precision_at
            .iter()
            .find(|(pk, _)| pk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        let n = data
            .combined_ranking_metrics
            .ndcg_at
            .iter()
            .find(|(nk, _)| nk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        l.push(Line::from(Span::raw(format!(
            "  @{:<2}   {:.4}    {:.4}    {:.4}",
            k, r, p, n
        ))));
    }
    l.push(Line::from(Span::styled(
        format!(
            "  MRR: {:.4}     Hit: {:.2}",
            data.combined_ranking_metrics.mrr, data.combined_ranking_metrics.hit_rate
        ),
        gray,
    )));

    l.push(Line::from(""));
    l.push(Line::from(Span::styled(" ── 子查询详情 ──", hdr)));
    for m in &data.per_query_metrics {
        l.push(Line::from(Span::styled(
            format!(
                "  Q{}  MRR={:.4}  Hit={:.2}",
                m.query_index, m.ranking_metrics.mrr, m.ranking_metrics.hit_rate,
            ),
            gray,
        )));
    }

    if !data.expected_combined_ranking.is_empty() || !data.combined_retrieved_ids.is_empty() {
        l.push(Line::from(""));
        l.push(Line::from(Span::styled(" ── 检索结果 vs 预期 ──", hdr)));
        let retrieved_set: std::collections::HashSet<&MemoryId> =
            data.combined_retrieved_ids.iter().take(10).collect();
        let n_max = data
            .combined_retrieved_ids
            .len()
            .min(10)
            .max(data.expected_combined_ranking.len().min(5));
        for pos in 0..n_max {
            let col0 = format!("  #{}", pos + 1);
            let (retrieved_str, retrieved_style) =
                if let Some(id) = data.combined_retrieved_ids.get(pos) {
                    let name = data
                        .graph_names
                        .as_ref()
                        .and_then(|m| m.get(id))
                        .cloned()
                        .unwrap_or_default();
                    let is_hit = data.expected_combined_ranking.iter().any(|eid| eid == id);
                    (format!(" {}", name), if is_hit { green } else { gray })
                } else {
                    (format!(" {}", "\u{2014}"), gray)
                };
            l.push(Line::from(vec![
                Span::raw(col0),
                Span::styled(retrieved_str, retrieved_style),
            ]));
        }
    }

    l
}
