//! Post-processing rule stack for the smart router.
//!
//! Ported from rwkv-router `router.rs`: given the four tier probabilities
//! produced by the classifier, apply (1) under-routing safety upgrade and
//! (2) sticky tier, then produce the final routing decision.

use super::tier::RouteClass;
use crate::config::RouterConfig;
use serde::{Deserialize, Serialize};

/// Where a routing decision came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionSource {
    /// Rule short-circuit (trivial ack).
    TrivialAck,
    /// Model classification probabilities.
    Model,
    /// Fallback (classification unavailable / failed).
    Fallback,
}

/// Final routing decision produced by the smart router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub route: RouteClass,
    pub confidence: f32,
    pub probabilities: [f32; 4],
    /// Difference between the top-1 and top-2 probabilities.
    pub margin: f32,
    pub sticky_applied: bool,
    pub safety_applied: bool,
    pub source: DecisionSource,
}

/// Applies the post-processing rule stack to raw class probabilities.
///
/// Rules (in order):
/// 1. **Under-routing safety**: when argmax is R0/R1 but `P(R2)+P(R3)` exceeds
///    the safety threshold, upgrade to R2 — prefer over-routing to under-routing.
/// 2. **Sticky tier**: for short messages without code blocks, never route
///    below the previous turn's tier (prevents mid-conversation downgrades).
pub fn postprocess(
    probs: &[f32; 4],
    prev_tier: Option<RouteClass>,
    is_short: bool,
    config: &RouterConfig,
) -> RoutingDecision {
    // argmax
    let (top1_idx, &top1_prob) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let mut route = RouteClass::from_index(top1_idx).unwrap_or(RouteClass::R1);
    let mut confidence = top1_prob;

    // margin = top1 - top2
    let mut sorted = *probs;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = sorted[0] - sorted[1];

    let mut sticky_applied = false;
    let mut safety_applied = false;

    // Under-routing safety: R0/R1 with heavy tail probability -> upgrade to R2.
    let heavy_prob = probs[2] + probs[3];
    if matches!(route, RouteClass::R0 | RouteClass::R1) && heavy_prob > config.safety_threshold {
        route = RouteClass::R2;
        safety_applied = true;
        confidence = heavy_prob;
    }

    // Sticky tier: short messages never route below the previous tier.
    if config.sticky_enabled && is_short {
        if let Some(last) = prev_tier {
            if route < last {
                route = last;
                sticky_applied = true;
            }
        }
    }

    RoutingDecision {
        route,
        confidence,
        probabilities: *probs,
        margin,
        sticky_applied,
        safety_applied,
        source: DecisionSource::Model,
    }
}

/// Builds a trivial-ack decision (rule short-circuit, R0 with full confidence).
pub fn trivial_ack_decision() -> RoutingDecision {
    RoutingDecision {
        route: RouteClass::R0,
        confidence: 1.0,
        probabilities: [1.0, 0.0, 0.0, 0.0],
        margin: 1.0,
        sticky_applied: false,
        safety_applied: false,
        source: DecisionSource::TrivialAck,
    }
}

/// Builds a fallback decision (classification unavailable).
pub fn fallback_decision() -> RoutingDecision {
    RoutingDecision {
        route: RouteClass::R1,
        confidence: 0.0,
        probabilities: [0.0, 0.0, 0.0, 0.0],
        margin: 0.0,
        sticky_applied: false,
        safety_applied: false,
        source: DecisionSource::Fallback,
    }
}

/// Converts raw candidate-token logits into normalized probabilities via softmax.
pub fn softmax(logits: &[f32]) -> [f32; 4] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut out = [0.0f32; 4];
    if sum <= 0.0 || !sum.is_finite() {
        // Degenerate logits: uniform distribution.
        return [0.25; 4];
    }
    for (i, e) in exps.iter().enumerate().take(4) {
        out[i] = e / sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RouterConfig {
        RouterConfig::default()
    }

    #[test]
    fn argmax_wins_when_clear() {
        let probs = [0.05, 0.85, 0.08, 0.02];
        let d = postprocess(&probs, None, false, &config());
        assert_eq!(d.route, RouteClass::R1);
        assert!((d.confidence - 0.85).abs() < 1e-6);
        assert!(!d.safety_applied);
        assert!(!d.sticky_applied);
        assert!((d.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn safety_upgrades_r1_with_heavy_tail() {
        // argmax is R1 (0.48) but P(R2)+P(R3) = 0.47 > 0.45 threshold.
        let probs = [0.05, 0.48, 0.32, 0.15];
        let d = postprocess(&probs, None, false, &config());
        assert_eq!(d.route, RouteClass::R2);
        assert!(d.safety_applied);
        assert!((d.confidence - 0.47).abs() < 1e-6);
    }

    #[test]
    fn safety_not_applied_below_threshold() {
        // P(R2)+P(R3) = 0.30 < 0.45.
        let probs = [0.05, 0.65, 0.20, 0.10];
        let d = postprocess(&probs, None, false, &config());
        assert_eq!(d.route, RouteClass::R1);
        assert!(!d.safety_applied);
    }

    #[test]
    fn safety_not_applied_when_r2_or_r3() {
        let probs = [0.05, 0.10, 0.15, 0.70];
        let d = postprocess(&probs, None, false, &config());
        assert_eq!(d.route, RouteClass::R3);
        assert!(!d.safety_applied);
    }

    #[test]
    fn sticky_keeps_previous_tier_for_short_messages() {
        // Model says R1, previous turn was R2, message is short -> stay at R2.
        let probs = [0.10, 0.60, 0.20, 0.10];
        let d = postprocess(&probs, Some(RouteClass::R2), true, &config());
        assert_eq!(d.route, RouteClass::R2);
        assert!(d.sticky_applied);
    }

    #[test]
    fn sticky_ignored_for_long_messages() {
        let probs = [0.10, 0.60, 0.20, 0.10];
        let d = postprocess(&probs, Some(RouteClass::R2), false, &config());
        assert_eq!(d.route, RouteClass::R1);
        assert!(!d.sticky_applied);
    }

    #[test]
    fn sticky_disabled_by_config() {
        let mut cfg = config();
        cfg.sticky_enabled = false;
        let probs = [0.10, 0.60, 0.20, 0.10];
        let d = postprocess(&probs, Some(RouteClass::R2), true, &cfg);
        assert_eq!(d.route, RouteClass::R1);
    }

    #[test]
    fn sticky_never_downgrades_below_model_tier() {
        // Model says R3, previous was R1 -> keep R3 (sticky only lifts, never lowers).
        let probs = [0.05, 0.10, 0.15, 0.70];
        let d = postprocess(&probs, Some(RouteClass::R1), true, &config());
        assert_eq!(d.route, RouteClass::R3);
        assert!(!d.sticky_applied);
    }

    #[test]
    fn trivial_and_fallback_decisions() {
        let t = trivial_ack_decision();
        assert_eq!(t.route, RouteClass::R0);
        assert_eq!(t.source, DecisionSource::TrivialAck);
        assert!((t.confidence - 1.0).abs() < 1e-6);

        let f = fallback_decision();
        assert_eq!(f.source, DecisionSource::Fallback);
    }

    #[test]
    fn softmax_normalizes() {
        let p = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[3] > p[0]);
        // Large values must not overflow.
        let p2 = softmax(&[1000.0, 1001.0, 1002.0, 1003.0]);
        assert!((p2.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // Degenerate input -> uniform.
        let p3 = softmax(&[f32::NAN, f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY]);
        assert!((p3[0] - 0.25).abs() < 1e-6);
    }
}
