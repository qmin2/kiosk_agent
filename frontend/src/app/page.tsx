"use client";

import React from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import { useCoAgent } from "@copilotkit/react-core";
import {
  Smartphone,
  Terminal,
  Activity,
  Eye,
  ChevronRight,
  Loader2
} from "lucide-react";

export default function Home() {
  const { state, running } = useCoAgent({
    name: "kiosk_agent",
  });

  // Process screenshot URL
  // The agent state includes paths like '/abs/path/to/screenshots/name.png'
  const getScreenshotUrl = (path: any) => {
    if (!path || typeof path !== 'string') return null;
    const filename = path.split('/').pop();
    return `http://localhost:8000/screenshots/${filename}`;
  };

  const screenshotUrl = getScreenshotUrl(state?.post_action_path || state?.pre_action_path);
  const history = state?.history || [];
  const latestStatus = state?.status || "Idle";

  return (
    <div className="flex h-screen w-full bg-[#0a0a0c] text-slate-200 overflow-hidden font-sans">
      {/* Sidebar: Chat Interface */}
      <div className="w-[400px] h-full border-r border-white/10 flex flex-col bg-[#0f0f12]">
        <div className="p-4 border-b border-white/10 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-indigo-400" />
          <h1 className="font-semibold text-lg tracking-tight">Kiosk Terminal</h1>
        </div>
        <div className="flex-1 overflow-hidden">
          <CopilotChat
            className="h-full"
            labels={{
              title: "Agent Controller",
              initial: "I'm ready to control the kiosk. What should I do?",
            }}
          />
        </div>
      </div>

      {/* Main Content: Kiosk Dashboard */}
      <div className="flex-1 flex flex-col overflow-hidden bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-indigo-950/20 via-slate-950 to-slate-950">
        {/* Header */}
        <header className="h-16 border-b border-white/10 flex items-center justify-between px-8 bg-black/20 backdrop-blur-md">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-green-400 animate-pulse" />
              <span className="text-sm font-medium text-slate-400">Status:</span>
              <span className="text-sm font-semibold text-indigo-300 uppercase tracking-wider">{latestStatus}</span>
            </div>
            <div className="h-4 w-px bg-white/10" />
            <div className="flex items-center gap-2">
              <Smartphone className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-medium text-slate-400">Device:</span>
              <span className="text-sm font-semibold text-slate-300">Android Kiosk</span>
            </div>
          </div>

          {running && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
              <Loader2 className="w-3 h-3 text-indigo-400 animate-spin" />
              <span className="text-xs font-medium text-indigo-400 uppercase tracking-tighter">Processing</span>
            </div>
          )}
        </header>

        {/* Dashboard Grid */}
        <div className="flex-1 p-8 grid grid-cols-12 gap-8 overflow-y-auto">

          {/* Left: Device Mirror */}
          <div className="col-span-12 lg:col-span-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500 flex items-center gap-2">
                <Eye className="w-4 h-4" /> Device Mirror
              </h2>
            </div>

            <div className="relative group p-4 rounded-3xl bg-white/5 border border-white/10 shadow-2xl backdrop-blur-sm self-center">
              <div className="absolute -inset-0.5 bg-gradient-to-b from-indigo-500/20 to-purple-500/20 rounded-3xl blur opacity-30 group-hover:opacity-100 transition duration-1000"></div>
              <div className="relative bg-[#1a1a21] rounded-2xl overflow-hidden border border-white/20 aspect-[9/16] w-[300px] flex items-center justify-center">
                {screenshotUrl ? (
                  <img
                    src={screenshotUrl}
                    alt="Kiosk Screen"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-3 text-slate-600">
                    <Smartphone className="w-12 h-12 opacity-20" />
                    <span className="text-xs font-medium">Waiting for signal...</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Thought Process & History */}
          <div className="col-span-12 lg:col-span-7 flex flex-col gap-6">

            {/* Thought Bubble */}
            <div className="p-6 rounded-2xl bg-indigo-500/5 border border-indigo-500/20">
              <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-3">Current Reasoning</h3>
              <p className="text-slate-300 leading-relaxed italic text-sm">
                {state?.thought || "Standby. Waiting for instructions to begin kiosk operations."}
              </p>
            </div>

            {/* Execution Logs */}
            <div className="flex-1 flex flex-col gap-4">
              <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">Operation History</h3>
              <div className="space-y-3">
                {history.slice(-5).reverse().map((entry: any, i: number) => (
                  <div key={i} className="group p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/20 transition-all flex items-start gap-4">
                    <div className="mt-1 w-6 h-6 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                      <ChevronRight className="w-3 h-3 text-indigo-400 group-hover:translate-x-0.5 transition-transform" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-bold text-slate-500 uppercase">Step {entry.iteration}</span>
                        <span className="text-[10px] text-slate-600 tabular-nums">ID: {entry.screen_id?.slice(0, 8)}</span>
                      </div>
                      <p className="text-sm text-slate-300 line-clamp-2 mb-2">{entry.thought}</p>
                      <div className="flex flex-wrap gap-2">
                        {entry.adb_commands?.map((cmd: string[], ci: number) => (
                          <code key={ci} className="px-2 py-0.5 rounded text-[10px] bg-black/40 text-emerald-400 border border-emerald-500/20 font-mono">
                            {cmd.join(" ")}
                          </code>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
                {history.length === 0 && (
                  <div className="p-12 text-center text-slate-600 border border-dashed border-white/5 rounded-2xl text-sm italic">
                    No operations recorded yet.
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
