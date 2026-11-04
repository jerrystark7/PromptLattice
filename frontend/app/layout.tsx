import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "PromptLattice — 提示词评测工作台", description: "测试、评估和迭代优化提示词的工作台" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="zh-CN"><body>{children}</body></html>; }
