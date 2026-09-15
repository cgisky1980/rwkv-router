//! Route tier definition for the smart router.
//!
//! Ported from the rwkv-router project (OpenSquilla R0-R3 route classes):
//! requests are classified into one of four complexity tiers, each mapped
//! to a configurable model/API.

use serde::{Deserialize, Serialize};

/// Route classification tier (complexity of the request).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RouteClass {
    /// Trivial chat ("thanks" / "ok" / "好的") — lightest model.
    R0,
    /// Simple task — single-step operations, simple Q&A (fast model).
    R1,
    /// Complex task — code generation, multi-step reasoning (mid-tier model).
    R2,
    /// High-stakes task — debugging, error diagnosis, long context (flagship model).
    R3,
}

impl RouteClass {
    pub const ALL: [RouteClass; 4] = [
        RouteClass::R0,
        RouteClass::R1,
        RouteClass::R2,
        RouteClass::R3,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            RouteClass::R0 => "R0",
            RouteClass::R1 => "R1",
            RouteClass::R2 => "R2",
            RouteClass::R3 => "R3",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            RouteClass::R0 => 0,
            RouteClass::R1 => 1,
            RouteClass::R2 => 2,
            RouteClass::R3 => 3,
        }
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(RouteClass::R0),
            1 => Some(RouteClass::R1),
            2 => Some(RouteClass::R2),
            3 => Some(RouteClass::R3),
            _ => None,
        }
    }

    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s {
            "R0" => Some(RouteClass::R0),
            "R1" => Some(RouteClass::R1),
            "R2" => Some(RouteClass::R2),
            "R3" => Some(RouteClass::R3),
            _ => None,
        }
    }

    /// Upgrade to the next-higher tier (more capable model).
    pub fn upgrade(&self) -> RouteClass {
        match self {
            RouteClass::R0 => RouteClass::R1,
            RouteClass::R1 => RouteClass::R2,
            RouteClass::R2 => RouteClass::R3,
            RouteClass::R3 => RouteClass::R3,
        }
    }
}

impl std::fmt::Display for RouteClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_class_roundtrip() {
        for rc in RouteClass::ALL {
            let idx = rc.index();
            assert_eq!(RouteClass::from_index(idx), Some(rc));
            assert_eq!(RouteClass::parse_from_str(rc.as_str()), Some(rc));
        }
    }

    #[test]
    fn order_and_upgrade() {
        assert!(RouteClass::R0 < RouteClass::R3);
        assert_eq!(RouteClass::R1.upgrade(), RouteClass::R2);
        assert_eq!(RouteClass::R3.upgrade(), RouteClass::R3);
        assert_eq!(RouteClass::parse_from_str("R4"), None);
        assert_eq!(RouteClass::from_index(9), None);
    }
}
