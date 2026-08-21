//! CLI for boolean circuit streaming experiments.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use boolean_circuit_exp::circuit::TopologyMode;
use boolean_circuit_exp::diag_wire::{print_wire_diag_summary, run_wire_diagnostic};
use boolean_circuit_exp::metrics::{run_single, LearnerKind, RunMetrics};
use boolean_circuit_exp::u8_linear::{
    pool_size_for, run_seed, run_seed_with_split, summarize as summarize_u8, TrainSplit,
    U8RunMetrics, U8Variant, WeightMode,
};

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LearnerArg {
    Pair,
    Wire,
    Both,
}

impl LearnerArg {
    fn kinds(self) -> Vec<LearnerKind> {
        match self {
            LearnerArg::Pair => vec![LearnerKind::Pair],
            LearnerArg::Wire => vec![LearnerKind::Wire],
            LearnerArg::Both => vec![LearnerKind::Pair, LearnerKind::Wire],
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "run_exp", about = "Boolean circuit streaming experiments")]
struct Args {
    /// Experiment: quick, a, b, c, diag, u8-linear, u8-linear-mem, u8-linear-sym, u8-linear-compare, u8-linear-benchmark
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

    /// Steps excluded from accuracy (cold-start warmup); must be < stream_len.
    #[arg(long, default_value_t = 16384)]
    warmup_len: u64,

    #[arg(long, default_value_t = 50)]
    n_seeds: usize,

    #[arg(long, value_enum, default_value_t = TopologyArg::Matched)]
    topology: TopologyArg,

    #[arg(long, value_enum, default_value_t = LearnerArg::Both)]
    learner: LearnerArg,

    /// Eval samples per gate for post-train node error rate (diag only).
    #[arg(long, default_value_t = 4096)]
    diag_eval_samples: u64,

    #[arg(long, default_value = "target/exp")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = 100)]
    epochs: usize,

    #[arg(long, default_value_t = 0.8)]
    train_frac: f64,

    /// Comma-separated train fractions for u8-linear-benchmark.
    #[arg(long, default_value = "0.001,0.01,0.1,0.3,0.5,0.7,0.8,0.9")]
    train_fracs: String,

    /// Comma-separated fixed train sample counts for u8-linear-benchmark.
    #[arg(long, default_value = "1,2,3,5,10")]
    train_counts: String,

    /// Limit u8 experiments to this many inputs (default: 1..=5 for compare/benchmark).
    #[arg(long)]
    u8_n_inputs: Option<usize>,

    /// Comma-separated u8 input counts (overrides --u8-n-inputs).
    #[arg(long)]
    u8_n_inputs_list: Option<String>,

    /// Comma-separated DGP weight modes for u8-linear-benchmark: i8, binary.
    #[arg(long, default_value = "i8,binary")]
    u8_weight_modes: String,
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
    if args.warmup_len >= args.stream_len {
        eprintln!("warmup_len ({}) must be < stream_len ({})", args.warmup_len, args.stream_len);
        std::process::exit(1);
    }
    fs::create_dir_all(&args.out_dir).expect("create out dir");

    match args.experiment.as_str() {
        "quick" => run_quick(&args),
        "a" => run_experiment_a(&args),
        "b" => run_experiment_b(&args),
        "c" => run_experiment_c(&args),
        "diag" => run_diag(&args),
        "u8-linear" => run_u8_linear(&args, 1, U8Variant::Perceptron, "u8_linear.csv"),
        "u8-linear-mem" => run_u8_linear(&args, 1, U8Variant::Mem, "u8_linear_mem.csv"),
        "u8-linear-sym" => run_u8_linear(&args, 1, U8Variant::Sym, "u8_linear_sym.csv"),
        "u8-linear-compare" => run_u8_compare(&args),
        "u8-linear-benchmark" => run_u8_benchmark(&args),
        other => {
            eprintln!(
                "Unknown experiment {other}; use quick, a, b, c, diag, u8-linear, u8-linear-mem, u8-linear-sym, u8-linear-compare, or u8-linear-benchmark"
            );
            std::process::exit(1);
        }
    }
}

fn valid_config(d: usize, depth: usize, width: usize) -> bool {
    d + depth * width <= 256
}

fn run_for_learner(
    args: &Args,
    topo: TopologyMode,
    learner: LearnerKind,
) -> Vec<RunMetrics> {
    (0..args.n_seeds)
        .map(|i| {
            run_single(
                i as u64 + 1,
                args.d,
                args.depth,
                args.width,
                args.stream_len,
                args.warmup_len,
                topo,
                learner,
            )
        })
        .collect()
}

fn u8_input_range(args: &Args) -> Vec<usize> {
    if let Some(ref list) = args.u8_n_inputs_list {
        return list
            .split(',')
            .map(|s| {
                s.trim().parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("invalid u8_n_inputs_list entry: {s:?}");
                    std::process::exit(1);
                })
            })
            .inspect(|&n| {
                if !(1..=5).contains(&n) {
                    eprintln!("each u8 n_inputs must be 1..=5, got {n}");
                    std::process::exit(1);
                }
            })
            .collect();
    }
    match args.u8_n_inputs {
        Some(n) if (1..=5).contains(&n) => vec![n],
        Some(n) => {
            eprintln!("u8_n_inputs must be 1..=5, got {n}");
            std::process::exit(1);
        }
        None => (1..=5).collect(),
    }
}

fn run_u8_linear(args: &Args, n_inputs: usize, variant: U8Variant, csv_name: &str) {
    if args.train_frac <= 0.0 || args.train_frac >= 1.0 {
        eprintln!("train_frac must be in (0, 1), got {}", args.train_frac);
        std::process::exit(1);
    }

    println!(
        "=== u8 linear ({}, n_inputs={}) ===\n",
        variant.as_str(),
        n_inputs
    );
    println!(
        "samples={} epochs={} train_frac={} seeds={}\n",
        pool_size_for(n_inputs),
        args.epochs,
        args.train_frac,
        args.n_seeds
    );

    let results: Vec<U8RunMetrics> = (0..args.n_seeds)
        .into_par_iter()
        .map(|i| {
            run_seed(
                i as u64 + 1,
                n_inputs,
                args.train_frac,
                args.epochs,
                WeightMode::I8,
                variant,
            )
        })
        .collect();

    print_u8_summary(&results);

    let path = args.out_dir.join(csv_name);
    write_u8_csv(&path, &results);
    println!("\nWrote {}", path.display());
}

fn run_u8_compare(args: &Args) {
    if args.train_frac <= 0.0 || args.train_frac >= 1.0 {
        eprintln!("train_frac must be in (0, 1), got {}", args.train_frac);
        std::process::exit(1);
    }

    println!("=== u8 linear compare (perceptron vs mem vs sym) ===\n");
    println!(
        "epochs={} train_frac={} seeds={}\n",
        args.epochs, args.train_frac, args.n_seeds
    );

    let path = args.out_dir.join("u8_linear_compare.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "n_inputs,n_samples,variant,train_acc,train_std,test_acc,test_std,mean_epochs,mean_fit_ns,mean_predict_ns"
    )
    .unwrap();

    println!(
        "{:>8} {:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n_in", "samples", "variant", "train", "train_std", "test", "test_std", "fit_ns", "pred_ns"
    );
    println!("{}", "-".repeat(92));

    for n_inputs in u8_input_range(args) {
        let n_samples = pool_size_for(n_inputs);
        for variant in [U8Variant::Perceptron, U8Variant::Mem, U8Variant::Sym] {
            let results: Vec<U8RunMetrics> = (0..args.n_seeds)
                .into_par_iter()
                .map(|i| {
                    run_seed(
                        i as u64 + 1,
                        n_inputs,
                        args.train_frac,
                        args.epochs,
                        WeightMode::I8,
                        variant,
                    )
                })
                .collect();
            let summary = summarize_u8(&results);
            println!(
                "{:>8} {:>8} {:>12} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.0} {:>10.0}",
                n_inputs,
                n_samples,
                variant.as_str(),
                summary.mean_train_acc,
                summary.std_train_acc,
                summary.mean_test_acc,
                summary.std_test_acc,
                summary.mean_fit_ns,
                summary.mean_predict_ns,
            );
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.1},{:.1},{:.1}",
                n_inputs,
                n_samples,
                variant.as_str(),
                summary.mean_train_acc,
                summary.std_train_acc,
                summary.mean_test_acc,
                summary.std_test_acc,
                summary.mean_epochs,
                summary.mean_fit_ns,
                summary.mean_predict_ns,
            )
            .unwrap();
        }
    }
    println!("\nWrote {}", path.display());
}

fn u8_benchmark_train_fracs(args: &Args) -> Vec<f64> {
    args.train_fracs
        .split(',')
        .map(|s| {
            s.trim().parse::<f64>().unwrap_or_else(|_| {
                eprintln!("invalid train_fracs entry: {s:?}");
                std::process::exit(1);
            })
        })
        .inspect(|&f| {
            if f <= 0.0 || f >= 1.0 {
                eprintln!("each train fraction must be in (0, 1), got {f}");
                std::process::exit(1);
            }
        })
        .collect()
}

fn u8_benchmark_train_counts(args: &Args) -> Vec<usize> {
    args.train_counts
        .split(',')
        .map(|s| {
            s.trim().parse::<usize>().unwrap_or_else(|_| {
                eprintln!("invalid train_counts entry: {s:?}");
                std::process::exit(1);
            })
        })
        .inspect(|&n| {
            if n == 0 {
                eprintln!("each train count must be >= 1, got {n}");
                std::process::exit(1);
            }
        })
        .collect()
}

fn u8_benchmark_weight_modes(args: &Args) -> Vec<WeightMode> {
    args.u8_weight_modes
        .split(',')
        .map(|s| match s.trim() {
            "i8" => WeightMode::I8,
            "binary" => WeightMode::Binary,
            other => {
                eprintln!("invalid u8_weight_modes entry: {other:?}; use i8 and/or binary");
                std::process::exit(1);
            }
        })
        .collect()
}

fn u8_benchmark_splits(args: &Args) -> Vec<TrainSplit> {
    let mut splits: Vec<TrainSplit> = u8_benchmark_train_fracs(args)
        .into_iter()
        .map(TrainSplit::Frac)
        .collect();
    splits.extend(
        u8_benchmark_train_counts(args)
            .into_iter()
            .map(TrainSplit::Count),
    );
    splits
}

fn run_u8_benchmark(args: &Args) {
    let splits = u8_benchmark_splits(args);
    let weight_modes = u8_benchmark_weight_modes(args);

    println!("=== u8 linear benchmark ({}) ===\n", args.u8_weight_modes);
    println!(
        "epochs={} splits={} seeds={}\n",
        args.epochs,
        splits.len(),
        args.n_seeds
    );

    let path = args.out_dir.join("u8_linear_benchmark.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "n_inputs,n_samples,split_kind,train_frac,train_n,weight_mode,variant,train_acc,train_std,test_acc,test_std,mean_epochs,mean_fit_ns,mean_predict_ns"
    )
    .unwrap();

    println!(
        "{:>4} {:>7} {:>6} {:>7} {:>5} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "n_in", "samples", "split", "tr_frac", "tr_n", "weights", "variant", "train", "tr_std",
        "test", "te_std", "fit_ns", "pred_ns"
    );
    println!("{}", "-".repeat(108));

    for split in splits {
        for n_inputs in u8_input_range(args) {
            let n_samples = pool_size_for(n_inputs);
            for &weight_mode in &weight_modes {
                for variant in [U8Variant::Perceptron, U8Variant::Mem, U8Variant::Sym] {
                    let results: Vec<U8RunMetrics> = (0..args.n_seeds)
                        .into_par_iter()
                        .map(|i| {
                            run_seed_with_split(
                                i as u64 + 1,
                                n_inputs,
                                split,
                                args.epochs,
                                weight_mode,
                                variant,
                            )
                        })
                        .collect();
                    let summary = summarize_u8(&results);
                    let train_frac = split.frac_value().unwrap_or(f64::NAN);
                    let split_kind = split.kind();
                    let actual_train_n = results[0].train_n;
                    println!(
                        "{:>4} {:>7} {:>6} {:>7} {:>5} {:>7} {:>10} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.0} {:>10.0}",
                        n_inputs,
                        n_samples,
                        split_kind,
                        if split.frac_value().is_some() {
                            format!("{train_frac:.2}")
                        } else {
                            "-".to_string()
                        },
                        actual_train_n,
                        weight_mode.as_str(),
                        variant.as_str(),
                        summary.mean_train_acc,
                        summary.std_train_acc,
                        summary.mean_test_acc,
                        summary.std_test_acc,
                        summary.mean_fit_ns,
                        summary.mean_predict_ns,
                    );
                    let frac_csv = split
                        .frac_value()
                        .map(|f| format!("{f:.4}"))
                        .unwrap_or_default();
                    writeln!(
                        f,
                        "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.1},{:.1},{:.1}",
                        n_inputs,
                        n_samples,
                        split_kind,
                        frac_csv,
                        results[0].train_n,
                        weight_mode.as_str(),
                        variant.as_str(),
                        summary.mean_train_acc,
                        summary.std_train_acc,
                        summary.mean_test_acc,
                        summary.std_test_acc,
                        summary.mean_epochs,
                        summary.mean_fit_ns,
                        summary.mean_predict_ns,
                    )
                    .unwrap();
                }
            }
        }
    }
    println!("\nWrote {}", path.display());
}

fn print_u8_summary(results: &[U8RunMetrics]) {
    let summary = summarize_u8(results);
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "train_acc", "train_std", "test_acc", "test_std"
    );
    println!("{}", "-".repeat(52));
    println!(
        "{:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        summary.mean_train_acc,
        summary.std_train_acc,
        summary.mean_test_acc,
        summary.std_test_acc,
    );
    println!("mean epochs: {:.1}", summary.mean_epochs);
    println!("mean fit:    {:.0} ns", summary.mean_fit_ns);
    println!("mean predict:{:.0} ns", summary.mean_predict_ns);
}

fn write_u8_csv(path: &PathBuf, results: &[U8RunMetrics]) {
    let mut f = File::create(path).expect("create csv");
    writeln!(
        f,
        "seed,n_inputs,n_samples,weight_mode,variant,train_acc,test_acc,train_errors,test_errors,epochs,fit_ns,predict_ns"
    )
    .unwrap();
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{:.6},{:.6},{},{},{},{},{}",
            r.seed,
            r.n_inputs,
            r.n_samples,
            r.weight_mode.as_str(),
            r.variant.as_str(),
            r.train_accuracy,
            r.test_accuracy,
            r.train_errors,
            r.test_errors,
            r.epochs_run,
            r.timings.fit_ns,
            r.timings.predict_ns,
        )
        .unwrap();
    }
}

fn run_diag(args: &Args) {
    let topo: TopologyMode = args.topology.into();
    println!("=== Wire learner diagnostic ===\n");
    for seed in 1..=args.n_seeds {
        let report = run_wire_diagnostic(
            seed as u64,
            args.d,
            args.depth,
            args.width,
            args.stream_len,
            args.warmup_len,
            topo,
            args.diag_eval_samples,
        );
        print_wire_diag_summary(&report);
        println!();
    }
}

fn run_quick(args: &Args) {
    let gates = args.depth * args.width;
    let topo: TopologyMode = args.topology.into();
    println!("=== Quick run ===\n");
    println!(
        "d={} depth={} width={} gates={} T={} warmup={} seeds={} topology={:?}\n",
        args.d, args.depth, args.width, gates, args.stream_len, args.warmup_len, args.n_seeds, topo
    );

    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}",
        "learner", "acc", "acc_std", "last100", "l100_std", "steps95", "predict_ns", "observe_ns"
    );
    println!("{}", "-".repeat(80));

    let mut rows: Vec<(String, Vec<RunMetrics>)> = Vec::new();
    for learner in args.learner.kinds() {
        let results = run_for_learner(args, topo, learner);
        let summary = summarize(&results);
        println!(
            "{:<8} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.1} {:>12.0} {:>12.0}",
            learner.as_str(),
            summary.mean_acc,
            summary.std_acc,
            summary.mean_last100,
            summary.std_last100,
            summary.mean_steps95.unwrap_or(f64::NAN),
            summary.mean_predict_ns,
            summary.mean_observe_ns,
        );
        rows.push((learner.as_str().to_string(), results));
    }

    let path = args.out_dir.join("quick.csv");
    write_csv(&path, &rows);
    println!("\nWrote {}", path.display());
}

fn run_experiment_c(args: &Args) {
    let ds = [16usize, 32];
    let depths = [2usize, 4];
    let widths = [16usize, 32];
    let topo: TopologyMode = args.topology.into();
    let learners = args.learner.kinds();

    let path = args.out_dir.join("exp_c.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "seed,d,depth,width,gates,T,warmup,topology,learner,accuracy_after_warmup,accuracy_last_100,steps_to_95,mean_predict_ns,mean_observe_ns"
    )
    .unwrap();

    for &d in &ds {
        for &depth in &depths {
            for &width in &widths {
                if !valid_config(d, depth, width) {
                    continue;
                }
                let gates = depth * width;
                for &learner in &learners {
                    let results: Vec<RunMetrics> = (0..args.n_seeds)
                        .into_par_iter()
                        .map(|i| {
                            run_single(
                                i as u64 + 1,
                                d,
                                depth,
                                width,
                                args.stream_len,
                                args.warmup_len,
                                topo,
                                learner,
                            )
                        })
                        .collect();
                    for r in &results {
                        writeln!(
                            f,
                            "{},{},{},{},{},{},{},{:?},{},{:.6},{:.6},{},{:.1},{:.1}",
                            r.seed,
                            r.d,
                            r.depth,
                            r.width,
                            r.num_gates,
                            r.stream_len,
                            r.warmup_len,
                            r.topology,
                            r.learner.as_str(),
                            r.accuracy_after_warmup,
                            r.accuracy_last_n,
                            r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                            r.mean_predict_ns,
                            r.mean_observe_ns,
                        )
                        .unwrap();
                    }
                    eprintln!(
                        "done d={d} depth={depth} width={width} gates={gates} learner={}",
                        learner.as_str()
                    );
                }
            }
        }
    }
    println!("Wrote {}", path.display());
}

fn run_experiment_a(args: &Args) {
    let ds = [16usize, 32];
    let depths = [4usize, 8];
    let widths = [32usize, 64];
    let stream_lens = [16_384u64, 65_536, 262_144];
    let topologies = [TopologyMode::Independent, TopologyMode::Matched];
    let learners = args.learner.kinds();

    let path = args.out_dir.join("exp_a.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "seed,d,depth,width,gates,T,warmup,topology,learner,accuracy_after_warmup,accuracy_last_100,steps_to_95,mean_predict_ns,mean_observe_ns"
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
                        for &learner in &learners {
                            let results: Vec<RunMetrics> = (0..args.n_seeds)
                                .into_par_iter()
                                .map(|i| {
                                    run_single(
                                        i as u64 + 1,
                                        d,
                                        depth,
                                        width,
                                        t,
                                        args.warmup_len,
                                        topo,
                                        learner,
                                    )
                                })
                                .collect();
                            for r in &results {
                                writeln!(
                                    f,
                                    "{},{},{},{},{},{},{},{:?},{},{:.6},{:.6},{},{:.1},{:.1}",
                                    r.seed,
                                    r.d,
                                    r.depth,
                                    r.width,
                                    r.num_gates,
                                    r.stream_len,
                                    r.warmup_len,
                                    r.topology,
                                    r.learner.as_str(),
                                    r.accuracy_after_warmup,
                                    r.accuracy_last_n,
                                    r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                                    r.mean_predict_ns,
                                    r.mean_observe_ns,
                                )
                                .unwrap();
                            }
                            eprintln!(
                                "done d={d} depth={depth} width={width} gates={gates} T={t} topo={topo:?} learner={}",
                                learner.as_str()
                            );
                        }
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
    let learners = args.learner.kinds();
    let t = args.stream_len;

    let path = args.out_dir.join("exp_b.csv");
    let mut f = File::create(&path).expect("create csv");
    writeln!(
        f,
        "seed,d,depth,width,gates,T,warmup,topology,learner,accuracy_after_warmup,accuracy_last_100,steps_to_95,mean_predict_ns,mean_observe_ns"
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
                    for &learner in &learners {
                        let results: Vec<RunMetrics> = (0..args.n_seeds)
                            .into_par_iter()
                            .map(|i| {
                                run_single(
                                    i as u64 + 1,
                                    d,
                                    depth,
                                    width,
                                    t,
                                    args.warmup_len,
                                    topo,
                                    learner,
                                )
                            })
                            .collect();
                        for r in &results {
                            writeln!(
                                f,
                                "{},{},{},{},{},{},{},{:?},{},{:.6},{:.6},{},{:.1},{:.1}",
                                r.seed,
                                r.d,
                                r.depth,
                                r.width,
                                r.num_gates,
                                r.stream_len,
                                r.warmup_len,
                                r.topology,
                                r.learner.as_str(),
                                r.accuracy_after_warmup,
                                r.accuracy_last_n,
                                r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                                r.mean_predict_ns,
                                r.mean_observe_ns,
                            )
                            .unwrap();
                        }
                        eprintln!(
                            "done d={d} depth={depth} width={width} gates={gates} topo={topo:?} learner={}",
                            learner.as_str()
                        );
                    }
                }
            }
        }
    }
    println!("Wrote {}", path.display());
}

struct Summary {
    mean_acc: f64,
    std_acc: f64,
    mean_last100: f64,
    std_last100: f64,
    mean_steps95: Option<f64>,
    mean_predict_ns: f64,
    mean_observe_ns: f64,
}

fn summarize(results: &[RunMetrics]) -> Summary {
    let n = results.len() as f64;
    let mean_acc = results.iter().map(|r| r.accuracy_after_warmup).sum::<f64>() / n;
    let var = results
        .iter()
        .map(|r| (r.accuracy_after_warmup - mean_acc).powi(2))
        .sum::<f64>()
        / n;
    let mean_last100 = results.iter().map(|r| r.accuracy_last_n).sum::<f64>() / n;
    let var_last100 = results
        .iter()
        .map(|r| (r.accuracy_last_n - mean_last100).powi(2))
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
        mean_last100,
        std_last100: var_last100.sqrt(),
        mean_steps95,
        mean_predict_ns: results.iter().map(|r| r.mean_predict_ns).sum::<f64>() / n,
        mean_observe_ns: results.iter().map(|r| r.mean_observe_ns).sum::<f64>() / n,
    }
}

fn write_csv(path: &PathBuf, rows: &[(String, Vec<RunMetrics>)]) {
    let mut f = File::create(path).expect("create csv");
    writeln!(
        f,
        "learner,seed,d,depth,width,gates,T,warmup,accuracy_after_warmup,accuracy_last_100,steps_to_95,mean_predict_ns,mean_observe_ns"
    )
    .unwrap();
    for (label, results) in rows {
        for r in results {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{:.1},{:.1}",
                label,
                r.seed,
                r.d,
                r.depth,
                r.width,
                r.num_gates,
                r.stream_len,
                r.warmup_len,
                r.accuracy_after_warmup,
                r.accuracy_last_n,
                r.steps_to_95pct.map(|s| s.to_string()).unwrap_or_default(),
                r.mean_predict_ns,
                r.mean_observe_ns,
            )
            .unwrap();
        }
    }
}
