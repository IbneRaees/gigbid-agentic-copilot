import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GigBid — Agentic AI Freelancer Copilot",
  description:
    "Multi-agent LangGraph freelancer copilot with hybrid RAG, claim verification, MCP guardrails, and human-in-the-loop approvals.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased selection:bg-indigo-500/30">{children}</body>
    </html>
  );
}
