import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Cpu, Github, Terminal, Check } from "lucide-react";
import { useState } from "react";

const REPO_URL = "https://github.com/raffrant/gpu-watt-watcher";

const INSTALL_SNIPPET = `git clone https://github.com/raffrant/gpu-watt-watcher.git
cd gpu-watt-watcher
python -m venv .venv
source .venv/bin/activate      # .venv\\Scripts\\Activate.ps1 on Windows
pip install -r requirements.txt
streamlit run streamlitapp.py`;

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      { title: "GPU Watt Watcher — Run on your GPU" },
      {
        name: "description",
        content:
          "A small app that shows how much energy your GPU uses for basic compute, memory, and AI-style workloads.",
      },
      { property: "og:title", content: "GPU Watt Watcher — Run on your GPU" },
      {
        property: "og:description",
        content:
          "A small app that shows how much energy your GPU uses for basic compute, memory, and AI-style workloads.",
      },
      { property: "og:url", content: "https://gpuegy.dev/" },
      { property: "og:type", content: "website" },
    ],
    links: [{ rel: "canonical", href: "https://gpuegy.dev/" }],
  }),
});

function Landing() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL_SNIPPET);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* noop */
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Hero */}
      <section className="mx-auto max-w-3xl px-6 pt-20 pb-12 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/50 px-3 py-1 text-xs text-muted-foreground mb-6">
          <Cpu className="h-3.5 w-3.5" />
          Runs locally on your NVIDIA GPU
        </div>

        <h1 className="text-4xl md:text-6xl font-semibold tracking-tight">
          GPU Watt Watcher
          <span className="block text-2xl md:text-3xl text-muted-foreground font-normal mt-3">
            Run on your GPU
          </span>
        </h1>

        <p className="mt-6 text-lg text-muted-foreground leading-relaxed max-w-xl mx-auto">
          A small app that shows how much energy your GPU uses for basic
          compute, memory, and AI-style workloads.
        </p>

        <div className="mt-8 flex flex-col sm:flex-row gap-3 justify-center">
          <Button asChild size="lg" className="text-base">
            <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
              <Github className="mr-1" />
              Run on my GPU
            </a>
          </Button>
          <Button asChild size="lg" variant="outline" className="text-base">
            <a href="#install">
              <Terminal className="mr-1" />
              See install steps
            </a>
          </Button>
        </div>
      </section>

      {/* Install snippet */}
      <section id="install" className="mx-auto max-w-3xl px-6 pb-12">
        <Card className="overflow-hidden border-border">
          <div className="flex items-center justify-between border-b border-border bg-muted/40 px-4 py-2">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Terminal className="h-3.5 w-3.5" />
              bash
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={copy}
              className="h-7 px-2 text-xs"
            >
              {copied ? (
                <>
                  <Check className="mr-1 h-3.5 w-3.5" /> Copied
                </>
              ) : (
                "Copy"
              )}
            </Button>
          </div>
          <pre className="overflow-x-auto px-4 py-4 text-sm leading-relaxed font-mono text-foreground/90">
            <code>{INSTALL_SNIPPET}</code>
          </pre>
        </Card>
        <p className="mt-3 text-sm text-muted-foreground text-center">
          Requires an NVIDIA GPU with CUDA and PyTorch installed.
        </p>
      </section>

      {/* What you can learn */}
      <section className="mx-auto max-w-3xl px-6 pb-24">
        <h2 className="text-xl font-semibold tracking-tight mb-6">
          What you can learn
        </h2>
        <ul className="space-y-4">
          {[
            "Is your workload compute-bound or memory-bound?",
            "Does changing batch size or precision (FP32 vs FP16) reduce Joules per token?",
            "How does your GPU's energy usage change with different workloads?",
          ].map((item) => (
            <li
              key={item}
              className="flex gap-3 rounded-lg border border-border bg-card/40 p-4"
            >
              <Check className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <span className="text-foreground/90">{item}</span>
            </li>
          ))}
        </ul>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-3xl px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-muted-foreground">
          <span>Open source · MIT</span>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
          >
            <Github className="h-4 w-4" />
            raffrant/gpu-watt-watcher
          </a>
        </div>
      </footer>
    </main>
  );
}
