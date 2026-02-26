import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BidFlow - AI RFP Analyzer",
  description: "AI 기반 입찰 제안요청서 분석 시스템",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased flex h-screen bg-[#F6FBEF] dark:bg-gray-900 dark:text-gray-100`}
        suppressHydrationWarning
      >
        <Sidebar />
        <main className="flex-1 overflow-auto p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
