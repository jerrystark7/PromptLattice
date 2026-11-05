import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "PromptLattice — 提示词评测工作台", description: "测试、评估和迭代优化提示词的工作台" };
const themeBootstrap = `(()=>{try{const p=localStorage.getItem('promptlattice.theme');const d=p==='dark'||(p!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.dataset.theme=d?'dark':'light'}catch{document.documentElement.dataset.theme=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}})()`;
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN" suppressHydrationWarning>
    <head>
      <link rel="preconnect" href="https://fonts.googleapis.com"/>
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous"/>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet"/>
      <script dangerouslySetInnerHTML={{ __html: themeBootstrap }}/>
    </head>
    <body>{children}</body>
  </html>;
}
