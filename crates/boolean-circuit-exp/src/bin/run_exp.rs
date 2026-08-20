//! CLI for boolean circuit streaming experiments.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use boolean_circuit_exp::circuit::TopologyMode;
use boolean_circuit_exp::metrics::{run_single, RunMetrics};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TopologyArg {
    Independent,
    Matched,
}

impl From<TopologyArg> for TopologyMode {
    fn from(v: TopologyArg) -> Self {
        match v {
            TopologyArg::Independent => TopologyMode::Independent,
            TopologyArg::Matched => TopologyMode::Matched,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "run_exp", about = "Boolean circuit streaming experiments")]
struct Args {
    /// Experiment: quick, a (stream length), b (depth/complexity scaling)
    #[arg(long, default_value = "quick")]
    experiment: String,

    #[arg(long, default_value_t = 32)]
    d: usize,

    #[arg(long, default_value_t = 4)]
    depth: usize,

    #[arg(long, default_value_t = 32)]
    width: usize,

    #[arg(long, default_value_t = 65536)]
    stream_len: u64,

    #[arg(long, default_value_t = 50)]
    n_seeds: usize,

    #[arg(long, value_enum, default_value_t = TopologyArg::Matched)]
    topology: TopologyArg,

    #[arg(long, default_value = "target/exp")]
    out_dir: PathBuf,
}

fn main() {
    let args = Args::parse();
    if !valid_config(args.d, args.depth, args.width) {
        eprintln!(
            "invalid config: N = d + depth*width = {} > 256",
            args.d + args.depth * args.width
        );
        std::process::exit(1);
    }
    fs::create_dir_all(&args.out_dir).expect("create out dir");

    match args.experiment.as_str() {
        "quick" => run_quick(&args),
        "a" => run_experiment_a(&args),
        "b" => run_experiment_b(&args),
        other => {
            eprintln!("Unknown experiment {other}; use quick, a, or b");
            std::process::exit(1);
        }
    }
}

fn valid_config(d: usize, depth: usize, width: usize) -> bool {
    d + depth * width <= 256
}

fn run_quick(args: &Args) {
    let gates = args.depth * args.width;
    println!("=== Quick run: topology comparison ===\n");
    println!(
        "d={} depth={} width={} gates={} T={} seeds={}\n",
        args.d, args.depth, args.width, gates, args.stream_len, args.n_seeds
    );

    let configs = [
        (TopologyMode::Matched, "matched"),
        (TopologyMode::Independent, "independent"),
    ];

    println!(
        "{:<14} {:>8} {:>8} {:>10} {:>12} {:>12}",
        "topology", "acc_mean", "acc_std", "steps95", "predict_ns", "observe_ns"
    );
    println!("{}", "-".repeat(70));

    let mut rows = Vec::new();
    for (topo, label) in configs {
        let results: Vec<RunMetrics> = (0..args.n_seeds)
            .map(|i| {
                run_single(
                    i as u64 + 1,
                    args.d,
                    args.depth,
                    args.width,
                    args.stream_len,
                    topo,
                )
            })
            .collect();
        let summary = summarize(&results);
        println!(
            "{:<14} {:>8.4} {:>8.4} {:>10.1} {:>12.0} {:>12.0}",
            label,
            summary.mean_acc,
            summary.std_acc,
            summary.mean_steps95.unwrap_or(f64::NAN),
            summary.mean_predict_ns,
            summary.mean_observe_ns,
        );
        rows.push((label.to_string(), results));
    }

    let path = args.out_dir.join("quick.csv");
    write_csv(&path, &rows);
    println!("\nWrote {}", path.display());
}

fn run_experiment_a(args: &Args) {
    let ds = [16usize, 32];
    let depths = [4usize, 8];
    let widths = [32usize, 64];
    let stream_lens = [16_384u64, 65_536, 262_144];
    let topologies = [TopologyMode::Independent, TopologyMode::Matched];

    let path = args.out_dir.join("exp_a.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "seed,d,depth,width,gates,T,topology,final_accuracy,steps_to_95,mean_predict_ns,mean_observe_ns"
    )
    .unwrap();

    for &d in &ds {
        for &depth in &depths {
            for &width in &widths {
                if !valid_config(d, depth, width) {
                    continue;
                }
                let gates = depth * width;
                for &t in &stream_lens {
                    for &topo in &topologies {
                        let results: Vec<RunMetrics> = (0..args.n_seeds)
                            .into_par_iter()
                            .map(|i| run_single(i as u64 + 1, d, depth, width, t, topo))
                            .collect();
                        for r in &results {
                            writeln!(
                                f,
                                "{},{},{},{},{},{},{:?},{:.6},{},{:.1},{:.1}",
                                r.seed,
                                r.d,
                                r.depth,
                                r.width,
                                r.num_gates,
                                r.stream_len,
                                r.topology,
                                r.final_accuracy,
                                r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                                r.mean_predict_ns,
                                r.mean_observe_ns,
                            )
                            .unwrap();
                        }
                        eprintln!("done d={d} depth={depth} width={width} gates={gates} T={t} topo={topo:?}");
                    }
                }
            }
        }
    }
    println!("Wrote {}", path.display());
}

fn run_experiment_b(args: &Args) {
    let ds = [16usize, 32];
    let depths = [2usize, 4, 8, 12];
    let widths = [16usize, 32, 64];
    let topologies = [TopologyMode::Independent, TopologyMode::Matched];
    let t = args.stream_len;

    let path = args.out_dir.join("exp_b.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "seed,d,depth,width,gates,T,topology,final_accuracy,steps_to_95,mean_predict_ns,mean_observe_ns"
    )
    .unwrap();

    for &d in &ds {
        for &depth in &depths {
            for &width in &widths {
                if !valid_config(d, depth, width) {
                    continue;
                }
                let gates = depth * width;
                for &topo in &topologies {
                    let results: Vec<RunMetrics> = (0..args.n_seeds)
                        .into_par_iter()
                        .map(|i| run_single(i as u64 + 1, d, depth, width, t, topo))
                        .collect();
                    for r in &results {
                        writeln!(
                            f,
                            "{},{},{},{},{},{},{:?},{:.6},{},{:.1},{:.1}",
                            r.seed,
                            r.d,
                            r.depth,
                            r.width,
                            r.num_gates,
                            r.stream_len,
                            r.topology,
                            r.final_accuracy,
                            r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                            r.mean_predict_ns,
                            r.mean_observe_ns,
                        )
                        .unwrap();
                    }
                    eprintln!("done d={d} depth={depth} width={width} gates={gates} topo={topo:?}");
                }
            }
        }
    }
    println!("Wrote {}", path.display());
}

struct Summary {
    mean_acc: f64,
    std_acc: f64,
    mean_steps95: Option<f64>,
    mean_predict_ns: f64,
    mean_observe_ns: f64,
}

fn summarize(results: &[RunMetrics]) -> Summary {
    let n = results.len() as f64;
    let mean_acc = results.iter().map(|r| r.final_accuracy).sum::<f64>() / n;
    let var = results
        .iter()
        .map(|r| (r.final_accuracy - mean_acc).powi(2))
        .sum::<f64>()
        / n;
    let steps: Vec<f64> = results
        .iter()
        .filter_map(|r| r.steps_to_95pct.map(|s| s as f64))
        .collect();
    let mean_steps95 = if steps.is_empty() {
        None
    } else {
        Some(steps.iter().sum::<f64>() / steps.len() as f64)
    };
    Summary {
        mean_acc,
        std_acc: var.sqrt(),
        mean_steps95,
        mean_predict_ns: results.iter().map(|r| r.mean_predict_ns).sum::<f64>() / n,
        mean_observe_ns: results.iter().map(|r| r.mean_observe_ns).sum::<f64>() / n,
    }
}

fn write_csv(path: &PathBuf, rows: &[(String, Vec<RunMetrics>)]) {
    let mut f = File::create(path).expect("create csv");
    writeln!(
        f,
        "topology,seed,d,depth,width,gates,T,final_accuracy,steps_to_95,mean_predict_ns,mean_observe_ns"
    )
    .unwrap();
    for (label, results) in rows {
        for r in results {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.6},{},{:.1},{:.1}",
                label,
                r.seed,
                r.d,
                r.depth,
                r.width,
                r.num_gates,
                r.stream_len,
                r.final_accuracy,
                r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                r.mean_predict_ns,
                r.mean_observe_ns,
            )
            .unwrap();
        }
    }
}
