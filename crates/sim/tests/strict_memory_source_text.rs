/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_case::Manifest;
use ptxv_sim::strict_memory_frontend_launch;
use ptxv_sim::strict_memory_source;
use serde_json::json;

fn manifest() -> Manifest {
    serde_json::from_value(json!({
        "name": "source-text", "description": "admit only the declared PTX text byte subset",
        "entry": "k", "grid": [1,1,1], "block": [1,1,1],
        "outputs": [], "buffers": [], "params": [], "variants": []
    }))
    .unwrap()
}

fn source(comment: &[u8], trailing: bool) -> Vec<u8> {
    let mut source = b".version 9.0\n.target sm_103a\n.address_size 64\n.global .align 8 .u64 x[1]={0};\n.visible .entry k(){\n.reg .pred %p<1>;\nsetp.eq.u32 %p0,0,1;\n".to_vec();
    if !trailing {
        source.extend_from_slice(comment);
    }
    source.extend_from_slice(b"@%p0 st.global.u64[x],1; ret; }\n");
    if trailing {
        source.extend_from_slice(comment);
    }
    source
}

#[test]
fn refuses_invalid_comment_bytes_before_producing_a_candidate() {
    let manifest = manifest();
    for byte in [0, 1, 8, 14, 31, 127, 128, 255] {
        for trailing in [false, true] {
            for comment in [
                vec![b'/', b'/', byte, b'\n'],
                vec![b'/', b'*', b' ', byte, b' ', b'*', b'/'],
            ] {
                let source = source(&comment, trailing);
                assert!(
                    strict_memory_frontend_launch::build_program(&source, b"k", &manifest)
                        .is_none()
                );
                let report = strict_memory_source::check(&source, b"k", &manifest, 64, |_, _| {
                    panic!("candidate producer called for refused source byte {byte}")
                });
                let report = serde_json::to_value(report).unwrap();
                assert_eq!(
                    report["status"], "inconclusive",
                    "byte={byte}, trailing={trailing}: {report}"
                );
                assert_eq!(report["reason"]["kind"], "source_refused", "{report}");
            }
        }
    }
}

#[test]
fn accepts_printable_ascii_and_ordinary_whitespace_in_comments() {
    let manifest = manifest();
    for byte in [9, 10, 11, 12, 13, 32, 33, 65, 92, 126] {
        for trailing in [false, true] {
            for comment in [
                vec![b'/', b'/', byte, b'\n'],
                vec![b'/', b'*', b' ', byte, b' ', b'*', b'/'],
            ] {
                let source = source(&comment, trailing);
                let report = strict_memory_source::check(
                    &source,
                    b"k",
                    &manifest,
                    64,
                    strict_memory_source::native_candidate,
                );
                assert!(
                    report.is_safe(),
                    "byte={byte}, trailing={trailing}: {report:?}"
                );
            }
        }
    }
}

#[test]
fn refuses_valid_utf8_non_ascii_comments_as_a_whole_source_policy() {
    let manifest = manifest();
    for text in ["é", "\u{85}", "\u{2028}", "\u{2029}"] {
        let comment = format!("/* {text} */");
        for trailing in [false, true] {
            let source = source(comment.as_bytes(), trailing);
            assert!(
                strict_memory_frontend_launch::build_program(&source, b"k", &manifest).is_none()
            );
        }
    }
}
