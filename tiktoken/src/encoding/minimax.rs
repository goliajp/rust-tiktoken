//! MiniMax encodings: minimax_m2 (the M2 and M3 families).

use super::*;

/// Construct the minimax_m2 encoding (MiniMax M2 / M2.1 / M2.5 / M2.7).
/// Vocabulary size: 200,000 regular tokens + 54 special tokens
/// (byte-identical tokenizer across the whole M2 family).
#[cfg(feature = "vocab-minimax_m2")]
pub fn minimax_m2() -> CoreBpe {
    let encoder = parse_tkv(MINIMAX_M2_CHAIN);
    let special = special_tokens(&[
        ("]!p~[", 200000),
        ("<fim_prefix>", 200001),
        ("<fim_middle>", 200002),
        ("<fim_suffix>", 200003),
        ("<fim_pad>", 200004),
        ("<reponame>", 200005),
        ("<filename>", 200006),
        ("<gh_stars>", 200007),
        ("<issue_start>", 200008),
        ("<issue_comment>", 200009),
        ("<issue_closed>", 200010),
        ("<jupyter_start>", 200011),
        ("<jupyter_text>", 200012),
        ("<jupyter_code>", 200013),
        ("<jupyter_output>", 200014),
        ("<empty_output>", 200015),
        ("<commit_before>", 200016),
        ("<commit_msg>", 200017),
        ("<commit_after>", 200018),
        ("]~b]", 200019),
        ("[e~[", 200020),
        ("]!d~[", 200021),
        ("<function_call>", 200022),
        ("<code_interpreter>", 200023),
        ("]<]speech[>[", 200024),
        ("]<]image[>[", 200025),
        ("]<]video[>[", 200026),
        ("]<]start of speech[>[", 200027),
        ("]<]end of speech[>[", 200028),
        ("]<]start of image[>[", 200029),
        ("]<]end of image[>[", 200030),
        ("]<]start of video[>[", 200031),
        ("]<]end of video[>[", 200032),
        ("]<]vision pad[>[", 200033),
        ("]~!b[", 200034),
        ("<jupyter_error>", 200035),
        ("<add_file>", 200036),
        ("<delete_file>", 200037),
        ("<rename_file>", 200038),
        ("<edit_file>", 200039),
        ("<commit_message>", 200040),
        ("<empty_source_file>", 200041),
        ("<repo_struct>", 200042),
        ("<code_context>", 200043),
        ("<file_content>", 200044),
        ("<source_files>", 200045),
        ("<pr_start>", 200046),
        ("<review_comment>", 200047),
        ("<filepath>", 200048),
        ("<file_sep>", 200049),
        ("<think>", 200050),
        ("</think>", 200051),
        ("<minimax:tool_call>", 200052),
        ("</minimax:tool_call>", 200053),
    ]);
    CoreBpe::new(
        encoder,
        special,
        MINIMAX_M2_PATTERN,
        FastPath::MiniMax,
        WhitespaceRules::NewlineFirst,
    )
}
