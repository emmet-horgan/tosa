//! `tosa`: command-line TOSA toolkit.
//!
//! - `tosa run`: execute a `.tosa` graph on NumPy inputs (or a conformance
//!   case directory) and write or summarize the outputs.
//! - `tosa dump`: print a graph in readable form.
//! - `tosa check`: validate a graph and show its execution plan.
//! - `tosa convert`: lower a model in another format to TOSA.
//! - `tosa frontends`: list the conversion frontends built in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use tosa_convert::{ConvertOptions, FrontendSet, analyze};
use tosa_interp::stats::OpInfo;
use tosa_interp::{FlopCounter, Interpreter, InterpreterHook, OpTrace, StatsCollector, Tensor};
use tosa_ir::{DType, Graph, npy};

#[derive(Parser)]
#[command(name = "tosa", version, about = "TOSA toolkit: run, inspect, check and convert graphs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute a graph.
    Run(RunArgs),
    /// Print a graph in readable form.
    Dump {
        /// A `.tosa` file.
        model: PathBuf,
    },
    /// Validate a graph and show its execution plan.
    Check {
        /// A `.tosa` file.
        model: PathBuf,
    },
    /// Lower a model in another format to TOSA.
    Convert(ConvertArgs),
    /// List the conversion frontends built into this binary.
    Frontends,
}

#[derive(Args)]
struct RunArgs {
    /// A `.tosa` file. Alternatively pass a conformance case directory
    /// (containing `desc.json`) with `--case`.
    model: Option<PathBuf>,
    /// A test case directory in the conformance format: the graph and inputs
    /// are taken from its `desc.json`.
    #[arg(long, conflicts_with = "model")]
    case: Option<PathBuf>,
    /// An input tensor as `NAME=FILE.npy`; repeat for every graph input.
    #[arg(short, long = "input", value_name = "NAME=FILE")]
    inputs: Vec<String>,
    /// Write each output as `<dir>/<name>.npy`.
    #[arg(short, long, value_name = "DIR")]
    output_dir: Option<PathBuf>,
    /// Print operator counts, timings, FLOP estimates and peak memory.
    #[arg(long)]
    stats: bool,
    /// Print every operator as it executes.
    #[arg(long)]
    trace: bool,
}

#[derive(Args)]
struct ConvertArgs {
    /// The model to convert.
    model: PathBuf,
    /// Where to write the `.tosa` graph (default: the model path with a
    /// `.tosa` extension).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Which frontend to use (default: detected from the file).
    #[arg(short, long)]
    frontend: Option<String>,
    /// Only report which operators of the model are supported.
    #[arg(long)]
    analyze: bool,
    /// Skip the cleanup passes (transpose cancellation, constant folding,
    /// dead-code elimination).
    #[arg(long)]
    no_cleanup: bool,
    /// Print the converted graph.
    #[arg(long)]
    dump: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Run(args) => run(args),
        Command::Dump { model } => dump(&model),
        Command::Check { model } => check(&model),
        Command::Convert(args) => convert(args),
        Command::Frontends => {
            let set = tosa_convert::default_frontends();
            if set.names().is_empty() {
                println!("no frontends are built into this binary");
            }
            for f in set.iter() {
                println!("{:<10} extensions: {}", f.name(), f.extensions().join(", "));
            }
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn load_graph(path: &Path) -> Result<(Vec<u8>, Graph)> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let graph = Graph::from_flatbuffer(&bytes)?;
    Ok((bytes, graph))
}

/// Declared types of the main block's tensors.
fn declared_dtypes(graph: &Graph) -> HashMap<String, DType> {
    graph.main_block().map(|b| b.tensors.iter().map(|t| (t.name.clone(), t.dtype)).collect()).unwrap_or_default()
}

fn read_input(name: &str, file: &Path, dtypes: &HashMap<String, DType>) -> Result<Tensor> {
    let dtype = *dtypes.get(name).ok_or_else(|| format!("the graph declares no tensor '{name}'"))?;
    let arr = npy::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let data = arr.to_tensor_data(dtype).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Tensor::new(name.to_string(), arr.shape, data))
}

fn run(args: RunArgs) -> Result<()> {
    let (model, mut inputs): (PathBuf, Vec<(String, PathBuf)>) = match (&args.case, &args.model) {
        (Some(case), _) => {
            let desc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(case.join("desc.json"))?)?;
            let tosa_file = desc["tosa_file"].as_str().ok_or("desc.json: missing tosa_file")?;
            let names = desc["ifm_name"].as_array().ok_or("desc.json: missing ifm_name")?;
            let files = desc["ifm_file"].as_array().ok_or("desc.json: missing ifm_file")?;
            let inputs = names
                .iter()
                .zip(files)
                .map(|(n, f)| (n.as_str().unwrap_or("").to_string(), case.join(f.as_str().unwrap_or(""))))
                .collect();
            (case.join(tosa_file), inputs)
        }
        (None, Some(model)) => (model.clone(), Vec::new()),
        (None, None) => return Err("pass a model file or --case DIR".into()),
    };
    for spec in &args.inputs {
        let (name, file) = spec.split_once('=').ok_or_else(|| format!("--input {spec}: expected NAME=FILE"))?;
        inputs.push((name.to_string(), PathBuf::from(file)));
    }

    let (bytes, graph) = load_graph(&model)?;
    let dtypes = declared_dtypes(&graph);
    let mut tensors = HashMap::new();
    for (name, file) in &inputs {
        tensors.insert(name.clone(), read_input(name, file, &dtypes)?);
    }

    let mut interp = Interpreter::from_flatbuffer(&bytes)?;
    let mut hooks = CliHooks {
        stats: args.stats.then(StatsCollector::new),
        flops: args.stats.then(FlopCounter::new),
        trace: args.trace.then(OpTrace::new),
    };
    let start = std::time::Instant::now();
    let outputs = interp.run(tensors, &mut hooks)?;
    let elapsed = start.elapsed();

    let block = graph.main_block().ok_or("no main block")?;
    for name in &block.outputs {
        let t = &outputs[name];
        println!("{name}: {:?} {:?}{}", t.dtype(), t.shape, preview(t));
        if let Some(dir) = &args.output_dir {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!("{name}.npy"));
            std::fs::write(&path, npy::write(&t.shape, &t.data))?;
            println!("  wrote {}", path.display());
        }
    }
    println!("ran {} operators in {elapsed:.3?}", block.operators.len());

    if let Some(trace) = &hooks.trace {
        println!("\ntrace:");
        for e in &trace.entries {
            println!(
                "  {:indent$}{} #{} {:?} -> {:?} ({:.3?})",
                "",
                e.op,
                e.op_index,
                e.input_shapes,
                e.output_shapes,
                e.elapsed,
                indent = 2 * e.depth
            );
        }
    }
    if let Some(stats) = &hooks.stats {
        println!("\n{}", stats.summary());
    }
    if let Some(flops) = &hooks.flops {
        println!("\nestimated FLOPs: {}", flops.total_flops);
        let mut by_op: Vec<_> = flops.flops_by_op.iter().collect();
        by_op.sort_by_key(|(name, _)| **name);
        for (op, n) in by_op {
            println!("  {op:<16} {n}");
        }
    }
    Ok(())
}

/// The optional collectors of `tosa run`.
#[derive(Default)]
struct CliHooks {
    stats: Option<StatsCollector>,
    flops: Option<FlopCounter>,
    trace: Option<OpTrace>,
}

impl CliHooks {
    fn each(&mut self, mut f: impl FnMut(&mut dyn InterpreterHook)) {
        if let Some(h) = &mut self.stats {
            f(h);
        }
        if let Some(h) = &mut self.flops {
            f(h);
        }
        if let Some(h) = &mut self.trace {
            f(h);
        }
    }
}

impl InterpreterHook for CliHooks {
    fn pre_op(&mut self, op: &OpInfo, inputs: &[&Tensor]) {
        self.each(|h| h.pre_op(op, inputs));
    }
    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        self.each(|h| h.post_op(op, inputs, outputs));
    }
    fn on_tensor_alloc(&mut self, name: &str, shape: &[usize], dtype: DType, bytes: usize) {
        self.each(|h| h.on_tensor_alloc(name, shape, dtype, bytes));
    }
    fn on_tensor_free(&mut self, name: &str, bytes: usize) {
        self.each(|h| h.on_tensor_free(name, bytes));
    }
}

/// The first few elements of a tensor, for a glance at the result.
fn preview(t: &Tensor) -> String {
    let text = format!("{:?}", t.data);
    let open = text.find('[').unwrap_or(0);
    let body = &text[open..];
    let shown: Vec<&str> = body.trim_matches(|c| c == '[' || c == ']').split(", ").take(6).collect();
    let more = if t.data.len() > 6 { ", ..." } else { "" };
    format!(" [{}{more}]", shown.join(", "))
}

fn dump(model: &Path) -> Result<()> {
    let (_, graph) = load_graph(model)?;
    print!("{graph}");
    Ok(())
}

fn check(model: &Path) -> Result<()> {
    let (bytes, graph) = load_graph(model)?;
    let interp = Interpreter::from_flatbuffer(&bytes)?;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, block) in graph.blocks() {
        for op in &block.operators {
            *counts.entry(op.op.name()).or_default() += 1;
        }
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    println!("{}: valid", model.display());
    println!("version {:?}", graph.version);
    for (_, block) in graph.blocks() {
        let plan = interp.plan(&block.name).ok_or("missing plan")?;
        println!(
            "block '{}': {} inputs, {} outputs, {} tensors, {} operators; execution order {:?}",
            block.name,
            block.inputs.len(),
            block.outputs.len(),
            block.tensors.len(),
            block.operators.len(),
            plan.order
        );
        if !plan.unused_inputs.is_empty() {
            println!("  unused inputs: {:?}", plan.unused_inputs);
        }
    }
    println!("operators:");
    for (op, n) in counts {
        println!("  {op:<24} {n}");
    }
    Ok(())
}

fn convert(args: ConvertArgs) -> Result<()> {
    let set: FrontendSet = tosa_convert::default_frontends();
    let bytes = std::fs::read(&args.model).map_err(|e| format!("cannot read {}: {e}", args.model.display()))?;
    let frontend = set.select(args.frontend.as_deref(), &args.model, &bytes)?;
    let source = frontend.load(&bytes)?;
    let report = analyze(&source, frontend.lowerings());
    println!(
        "{} ({} frontend): {} nodes, {} tensors",
        args.model.display(),
        frontend.name(),
        source.nodes.len(),
        source.tensors.len()
    );
    for (op, n) in &report.supported {
        println!("  supported    {op:<32} {n}");
    }
    for (op, n) in &report.unsupported {
        println!("  UNSUPPORTED  {op:<32} {n}");
    }
    if args.analyze {
        return if report.is_complete() { Ok(()) } else { Err("some operators are unsupported".into()) };
    }
    let options = ConvertOptions { cleanup: !args.no_cleanup, ..Default::default() };
    let conversion = tosa_convert::lower_graph(&source, frontend.lowerings(), &options)?;
    if args.dump {
        print!("{}", conversion.graph);
    }
    let output = args.output.unwrap_or_else(|| args.model.with_extension("tosa"));
    std::fs::write(&output, conversion.graph.to_flatbuffer()?)?;
    let block = conversion.graph.main_block().ok_or("no main block")?;
    println!("wrote {} ({} operators)", output.display(), block.operators.len());
    Ok(())
}
