import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Zap,
  Gauge,
  FlaskConical,
  LineChart,
  Download,
  Sliders,
  Cpu,
  CheckCircle2,
  XCircle,
  Terminal,
  GitBranch,
  Activity,
} from "lucide-react";

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      { title: "GPU Energy Bench — pytest for GPU energy" },
      {
        name: "description",
        content:
          "A Streamlit test bench that measures GPU energy and performance for reproducible kernels — with thresholds, power-cap sweeps, telemetry export, and pytest-style PASS/FAIL.",
      },
      { property: "og:title", content: "GPU Energy Bench" },
      {
        property: "og:description",
        content: "Pytest for GPU energy — measure, test, and reduce GPU power.",
      },
    ],
  }),
});

const features = [
  {
    icon: Activity,
    title: "Accurate energy measurement",
    body: "Background NVML sampler integrates power over time (J = ∫P·dt) using trapezoidal integration around a synchronized timed region.",
  },
  {
    icon: FlaskConical,
    title: "Pytest-style tests",
    body: "Declare tests in tests.yaml with thresholds like min_gflops_per_s or max_energy_per_gflop. Each run renders a clear PASS/FAIL panel.",
  },
  {
    icon: Sliders,
    title: "Power-cap sweep",
    body: "Run the same kernel under multiple nvidia-smi power caps and auto-plot the time vs energy Pareto front.",
  },
  {
    icon: LineChart,
    title: "Telemetry & history",
    body: "Per-run power/util/temp time series, plus an SQLite history of every run for cross-config comparisons.",
  },
  {
    icon: Download,
    title: "Export everything",
    body: "Download metrics CSV/JSON, telemetry CSV, and a self-contained Plotly HTML for sharing — one click per run.",
  },
  {
    icon: GitBranch,
    title: "Pluggable kernels",
    body: "Register a new kernel with one function call. Built-in matmul ships with FP32/FP16/BF16 and configurable size + reps.",
  },
];

const tabs = [
  "GPU info",
  "Matrix benchmark",
  "Test registry",
  "Telemetry",
  "Power limits",
  "History",
  "Export",
];

function Landing() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Hero */}
      <header className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 -z-10 opacity-60 [background:radial-gradient(60%_60%_at_50%_0%,oklch(0.488_0.243_264.376/0.18),transparent_70%)]" />
        <div className="mx-auto max-w-6xl px-6 pb-20 pt-16 sm:pt-24">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              gpu_energy_bench
            </span>
          </div>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-6xl">
            Pytest, but for{" "}
            <span className="bg-gradient-to-r from-primary via-chart-4 to-chart-1 bg-clip-text text-transparent">
              GPU energy
            </span>
            .
          </h1>
          <p className="mt-6 max-w-2xl text-base text-muted-foreground sm:text-lg">
            Run small, reproducible GPU benchmarks. Measure joules, not just
            FLOPs. Set thresholds. Sweep power caps. Find the sweet spot
            between performance and power — automatically.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <Button size="lg" asChild>
              <a href="#quickstart">
                <Terminal className="mr-1 h-4 w-4" />
                Quickstart
              </a>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a href="#features">See what it does</a>
            </Button>
          </div>

          <div className="mt-10 flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="secondary" className="font-mono">PyTorch</Badge>
            <Badge variant="secondary" className="font-mono">pynvml</Badge>
            <Badge variant="secondary" className="font-mono">Streamlit</Badge>
            <Badge variant="secondary" className="font-mono">SQLite</Badge>
            <Badge variant="secondary" className="font-mono">Plotly</Badge>
          </div>
        </div>
      </header>

      {/* Quickstart */}
      <section id="quickstart" className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-8 lg:grid-cols-2 lg:items-start">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Up and running in 30 seconds
            </h2>
            <p className="mt-3 text-muted-foreground">
              Requires an NVIDIA GPU with NVML/nvidia-smi accessible, plus
              Python 3.10+. The full app lives in{" "}
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                gpu_energy_bench/
              </code>
              .
            </p>
            <ul className="mt-6 space-y-3 text-sm">
              <Step n={1} title="Install dependencies">
                Tiny set: pynvml, torch, streamlit, plotly, pyyaml, pandas.
              </Step>
              <Step n={2} title="Launch the bench">
                Streamlit serves a 7-tab control panel locally.
              </Step>
              <Step n={3} title="Define tests in tests.yaml">
                Declarative thresholds. Edit and rerun — no code changes.
              </Step>
              <Step n={4} title="Sweep power caps & export">
                Pareto-optimize energy vs. time. Share results as CSV/JSON/HTML.
              </Step>
            </ul>
          </div>

          <Card className="overflow-hidden border-border bg-card">
            <div className="flex items-center gap-2 border-b border-border bg-muted/40 px-4 py-2">
              <span className="h-2.5 w-2.5 rounded-full bg-destructive/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-chart-4/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-chart-2/70" />
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                terminal
              </span>
            </div>
            <pre className="overflow-x-auto p-4 font-mono text-xs leading-relaxed text-foreground">
{`# 1. install
pip install -r gpu_energy_bench/requirements.txt

# 2. run the bench (opens in browser)
streamlit run gpu_energy_bench/streamlit_app.py

# 3. add a test in gpu_energy_bench/tests.yaml
- name: matmul_medium_fp32
  kernel: matmul
  params:  { size: 4096, repetitions: 10, dtype: float32 }
  thresholds:
    min_gflops_per_s:    2000
    max_energy_per_gflop: 0.05
    max_temp_c:           85`}
            </pre>
          </Card>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="border-y border-border bg-muted/20">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            What's in the box
          </h2>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            Six things you'd otherwise glue together yourself.
          </p>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((f) => (
              <Card
                key={f.title}
                className="group border-border bg-card p-6 transition-colors hover:border-primary/40"
              >
                <f.icon className="h-5 w-5 text-primary" />
                <h3 className="mt-4 font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground">{f.body}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Tabs preview + thresholds */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Seven tabs. One workflow.
            </h2>
            <p className="mt-2 text-muted-foreground">
              The Streamlit app is structured so each step of "measure → test →
              optimize → share" gets its own surface.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              {tabs.map((t) => (
                <span
                  key={t}
                  className="rounded-md border border-border bg-card px-3 py-1.5 font-mono text-xs text-foreground"
                >
                  {t}
                </span>
              ))}
            </div>

            <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Stat icon={Cpu} label="NVML metrics" value="14+" />
              <Stat icon={Gauge} label="Sample rate" value="up to 50 Hz" />
              <Stat icon={Activity} label="Energy method" value="∫ P · dt" />
            </div>
          </div>

          <Card className="border-border bg-card p-6">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Test result</h3>
              <Badge className="bg-chart-2/20 text-chart-2 hover:bg-chart-2/30">
                matmul_medium_fp32
              </Badge>
            </div>

            <div className="mt-4 rounded-md border border-chart-2/40 bg-chart-2/10 p-3 text-sm font-medium text-chart-2">
              ✅ ALL THRESHOLDS PASSED
            </div>

            <div className="mt-4 grid gap-2">
              <Check
                pass
                name="min_gflops_per_s"
                actual="2,431"
                op="≥"
                threshold="2,000"
              />
              <Check
                pass
                name="max_energy_per_gflop"
                actual="0.041"
                op="≤"
                threshold="0.050"
              />
              <Check
                pass={false}
                name="max_temp_c"
                actual="87.2"
                op="≤"
                threshold="85"
              />
            </div>

            <p className="mt-4 text-xs text-muted-foreground">
              Each threshold becomes its own card so failures are unmissable.
            </p>
          </Card>
        </div>
      </section>

      {/* Power-cap sweep callout */}
      <section className="border-t border-border bg-muted/20">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_1fr] lg:items-center">
            <div>
              <Badge variant="outline" className="font-mono">
                experiment mode
              </Badge>
              <h2 className="mt-4 text-2xl font-semibold tracking-tight sm:text-3xl">
                Find the energy sweet spot — automatically
              </h2>
              <p className="mt-3 text-muted-foreground">
                Pick a test, a min/max wattage and a number of steps. The bench
                applies each cap via <code className="font-mono text-xs">nvidia-smi -pl</code>
                , reruns the kernel under stable conditions, and plots the
                time-vs-energy Pareto front. Original cap restored on exit.
              </p>
              <ul className="mt-6 space-y-2 text-sm">
                <Bullet>Cap vs. elapsed time + total energy</Bullet>
                <Bullet>Energy per GFLOP across caps</Bullet>
                <Bullet>Every sweep run saved to history</Bullet>
              </ul>
            </div>

            <Card className="border-border bg-card p-6">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-mono">power_sweep.csv</span>
                <Download className="h-3.5 w-3.5" />
              </div>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="border-b border-border text-muted-foreground">
                    <tr>
                      <Th>cap_w</Th>
                      <Th>elapsed_s</Th>
                      <Th>energy_j</Th>
                      <Th>J/GFLOP</Th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    <Tr>
                      <Td>150</Td>
                      <Td>2.41</Td>
                      <Td>312.7</Td>
                      <Td>0.046</Td>
                    </Tr>
                    <Tr>
                      <Td>200</Td>
                      <Td>1.88</Td>
                      <Td>340.2</Td>
                      <Td>0.050</Td>
                    </Tr>
                    <Tr highlight>
                      <Td>250</Td>
                      <Td>1.62</Td>
                      <Td>388.9</Td>
                      <Td>0.057</Td>
                    </Tr>
                    <Tr>
                      <Td>300</Td>
                      <Td>1.55</Td>
                      <Td>447.1</Td>
                      <Td>0.066</Td>
                    </Tr>
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                A 150 W cap traded ~55% more time for{" "}
                <span className="text-chart-2">30% less energy</span>.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Add a kernel */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-8 lg:grid-cols-2 lg:items-start">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Add your own kernel
            </h2>
            <p className="mt-3 text-muted-foreground">
              A kernel is any function that does its own warmup +{" "}
              <code className="font-mono text-xs">torch.cuda.synchronize()</code>
              and returns a <code className="font-mono text-xs">KernelResult</code>
              . Register it once — it shows up in the test registry and history
              automatically.
            </p>
          </div>
          <Card className="overflow-hidden border-border bg-card">
            <div className="border-b border-border bg-muted/40 px-4 py-2 font-mono text-xs text-muted-foreground">
              gpu_energy_bench/kernels.py
            </div>
            <pre className="overflow-x-auto p-4 font-mono text-xs leading-relaxed">
{`def _my_conv(device, size=512, channels=64, repetitions=20):
    import torch, time
    x = torch.randn(8, channels, size, size, device=device)
    w = torch.randn(channels, channels, 3, 3, device=device)

    for _ in range(2): torch.nn.functional.conv2d(x, w)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repetitions):
        y = torch.nn.functional.conv2d(x, w)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops = 2 * channels * channels * 9 * size * size * 8 * repetitions
    return KernelResult(elapsed_s=elapsed, flops=flops,
                        repetitions=repetitions)

register("my_conv", _my_conv)`}
            </pre>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-4 px-6 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary" />
            <span className="font-mono">gpu_energy_bench</span>
          </div>
          <p>Measure joules. Set thresholds. Reduce waste.</p>
        </div>
      </footer>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Small presentational helpers                                               */
/* -------------------------------------------------------------------------- */

function Step({
  n,
  title,
  children,
}: {
  n: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <li className="flex gap-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-xs text-primary">
        {n}
      </span>
      <div>
        <p className="font-medium text-foreground">{title}</p>
        <p className="text-muted-foreground">{children}</p>
      </div>
    </li>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
}) {
  return (
    <Card className="border-border bg-card p-4">
      <Icon className="h-4 w-4 text-primary" />
      <p className="mt-2 text-lg font-semibold">{value}</p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </Card>
  );
}

function Check({
  pass,
  name,
  actual,
  op,
  threshold,
}: {
  pass: boolean;
  name: string;
  actual: string;
  op: string;
  threshold: string;
}) {
  return (
    <div
      className={
        "flex items-start gap-2 rounded-md border p-3 text-xs " +
        (pass
          ? "border-chart-2/40 bg-chart-2/5"
          : "border-destructive/40 bg-destructive/5")
      }
    >
      {pass ? (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-chart-2" />
      ) : (
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
      )}
      <div className="font-mono">
        <p className="font-semibold text-foreground">{name}</p>
        <p className="text-muted-foreground">
          actual <span className="text-foreground">{actual}</span> {op}{" "}
          threshold <span className="text-foreground">{threshold}</span>
        </p>
      </div>
    </div>
  );
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-2 text-muted-foreground">
      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
      <span>{children}</span>
    </li>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-2 py-1.5 text-left font-medium">{children}</th>;
}
function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-2 py-1.5">{children}</td>;
}
function Tr({
  children,
  highlight,
}: {
  children: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <tr
      className={
        "border-b border-border/60 last:border-0 " +
        (highlight ? "bg-primary/5 text-foreground" : "")
      }
    >
      {children}
    </tr>
  );
}
