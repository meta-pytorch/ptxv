// Uniform-register windows (urwin.rs) over a list of (cubin, kernel) lines on stdin.
use std::io::BufRead;
fn main() {
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let Some((path, name)) = line.split_once('\t') else {
            continue;
        };
        let Ok(k) = ptxv_sass::load_cubin(std::path::Path::new(path), name) else {
            println!("{path}\t{name}\tLIFT-ERR");
            continue;
        };
        let w = ptxv_sim::urwin::windows(&k);
        if w.is_empty() {
            println!("{path}\t{name}\t0");
            continue;
        }
        // Each window: writer pc, end pc, writer text, the instructions inside.
        let mut pc = 0;
        let mut out = Vec::new();
        while pc < w.len() {
            if w[pc] {
                let s = pc - 1;
                let mut e = pc;
                while e + 1 < w.len() && w[e + 1] {
                    e += 1;
                }
                let inner: Vec<String> = (pc..e)
                    .map(|i| {
                        k.code[i]
                            .text
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect();
                out.push(format!(
                    "{}|{}|{}",
                    k.code[s].text.split_whitespace().next().unwrap_or(""),
                    k.code[e]
                        .text
                        .split_whitespace()
                        .take(2)
                        .collect::<Vec<_>>()
                        .join(" "),
                    inner.join(",")
                ));
                pc = e + 1;
            } else {
                pc += 1;
            }
        }
        println!("{path}\t{name}\t{}\t{}", out.len(), out.join(";"));
    }
}
