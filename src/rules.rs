//! Rule-based shortcuts for the smart router.
//!
//! Ported from rwkv-router's trivial-ack short circuit and rule labeler:
//! obvious chit-chat/acknowledgment messages skip model inference entirely
//! and route straight to R0 (lightest tier).

/// Maximum trimmed character length for a trivial-ack message.
const TRIVIAL_MAX_CHARS: usize = 10;

/// English trivial-ack phrases (exact match, case-insensitive).
const EN_TRIVIAL: &[&str] = &[
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "ok",
    "okay",
    "yes",
    "no",
    "yep",
    "nope",
    "sure",
    "bye",
    "goodbye",
    "good",
    "great",
    "nice",
    "cool",
    "done",
    "got it",
    "understood",
    "fine",
    "please",
    "sorry",
    "wow",
    "ok then",
];

/// Chinese trivial-ack phrases (substring match).
const ZH_TRIVIAL: &[&str] = &[
    "你好",
    "您好",
    "好的",
    "好滴",
    "嗯",
    "哦",
    "哈",
    "哈哈",
    "呵呵",
    "谢谢",
    "感谢",
    "再见",
    "拜拜",
    "收到",
    "明白",
    "知道了",
    "懂了",
    "对",
    "是的",
    "行",
    "可以",
    "没问题",
];

/// True when the message is an obvious trivial acknowledgment and can skip
/// classification inference entirely (route directly to R0).
///
/// Criteria (all must hold): trimmed length <= 10 chars, no code block,
/// no question mark, and the message matches a chit-chat phrase.
pub fn is_trivial_ack(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.chars().count() > TRIVIAL_MAX_CHARS {
        return false;
    }
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains("```") {
        return false;
    }
    if trimmed.contains('?') || trimmed.contains('？') {
        return false;
    }
    is_chitchat(trimmed)
}

fn is_chitchat(trimmed: &str) -> bool {
    let lower = trimmed.to_lowercase();
    if EN_TRIVIAL.iter().any(|kw| *kw == lower) {
        return true;
    }
    ZH_TRIVIAL.iter().any(|kw| trimmed.contains(kw))
}

/// True when the message is "short" for the sticky-tier rule:
/// at most 10 characters and no code block (mirrors rwkv-router
/// `sticky_max_length` semantics).
pub fn is_short_message(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.chars().count() <= TRIVIAL_MAX_CHARS && !trimmed.contains("```")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_trivial_ack() {
        assert!(is_trivial_ack("thanks"));
        assert!(is_trivial_ack("  OK  "));
        assert!(is_trivial_ack("好的"));
        assert!(is_trivial_ack("嗯嗯"));
        assert!(is_trivial_ack("收到"));
        assert!(is_trivial_ack("Hello"));
        assert!(is_trivial_ack("got it"));
    }

    #[test]
    fn rejects_non_trivial() {
        assert!(!is_trivial_ack("")); // empty
        assert!(!is_trivial_ack("fix this bug now")); // too long + not chit-chat
        assert!(!is_trivial_ack("what?")); // question mark
        assert!(!is_trivial_ack("怎么办？")); // question mark (zh)
        assert!(!is_trivial_ack("```rust")); // code block
        assert!(!is_trivial_ack("token")); // not an exact chit-chat word
        assert!(!is_trivial_ack("帮我写个函数")); // real request
                                                  // "继续" is a directive (continue the task), never a trivial ack —
                                                  // misrouting it to R0 mid-task degrades quality.
        assert!(!is_trivial_ack("继续"));
    }

    #[test]
    fn short_message_check() {
        assert!(is_short_message("继续"));
        assert!(is_short_message("ok go"));
        assert!(!is_short_message("```code"));
        assert!(!is_short_message(
            "this message is way too long for the sticky rule"
        ));
    }
}
